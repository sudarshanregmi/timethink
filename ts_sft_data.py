import torch
import numpy as np
import pandas as pd
import datasets
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class TSSFTDataset(Dataset):
    """
    Time Series SFT Dataset for FSDPSFTTrainer.
    Handles Multi-modal (Time Series) inputs via a Processor.
    
    This dataset loads data in the same format as TSRLHFDataset:
    - prompt: list of message dicts (chat format)
    - response: ground truth string
    - timeseries: list of timeseries arrays
    """

    def __init__(
        self,
        parquet_files: str | ListConfig,
        tokenizer,
        config,
        processor=None,
        max_samples: int = -1
    ):
        if processor is None:
            raise ValueError("TSSFTDataset requires a 'processor' to handle time series data.")
        
        self.processor = processor
        
        # Config extraction
        self.prompt_key = config.get("prompt_key", "prompt")
        self.response_key = config.get("response_key", "response")
        self.ts_key = config.get("timeseries_key", "timeseries")
        self.max_length = config.get("max_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.use_shm = config.get("use_shm", False)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        
        # Handle list/tuple keys (take first element if needed)
        if isinstance(self.prompt_key, (list, tuple, ListConfig)):
            self.prompt_key = self.prompt_key[0]
        if isinstance(self.response_key, (list, tuple, ListConfig)):
            self.response_key = self.response_key[0]
        if isinstance(self.ts_key, (list, tuple, ListConfig)):
            self.ts_key = self.ts_key[0]
        
        assert self.truncation in ["error", "left", "right"]
        
        if not isinstance(parquet_files, (list, ListConfig)):
            parquet_files = [parquet_files]
        self.parquet_files = list(parquet_files)
        self.max_samples = max_samples
        
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        
        self._download()
        self._read_files()

    def _download(self):
        """Download parquet files to local if needed."""
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files(self):
        """
        Read parquet files using HuggingFace datasets (same as TSRLHFDataset).
        This ensures consistent data loading behavior.
        """
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        total = len(self.dataframe)
        print(f"TSSFTDataset len: {len(self.dataframe)}")
        
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"Selected {self.max_samples} random samples out of {total}")

    def __len__(self):
        return len(self.dataframe)

    def _get_timeseries_data(self, doc: Dict[str, Any]) -> List[np.ndarray]:
        """
        Extract timeseries data as list of numpy arrays.
        Same logic as TSRLHFDataset.
        """
        ts_data = doc.get(self.ts_key)
        if ts_data is None:
            return []
        
        if isinstance(ts_data, list):
            return [np.array(ts) if not isinstance(ts, np.ndarray) else ts for ts in ts_data]
        elif isinstance(ts_data, np.ndarray):
            return [ts_data]
        return []

    def _build_prompt_messages(self, example: dict) -> List[Dict]:
        """
        Extract prompt messages from example.
        Returns the prompt as list of message dicts (for chat template).
        """
        prompt = example.get(self.prompt_key)
        if prompt is None:
            return []
        # Handle numpy array wrapping (can happen with parquet loading)
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        return prompt

    def _build_full_messages(self, example: dict) -> List[Dict]:
        """
        Build full conversation (prompt + response) for SFT training.
        """
        prompt_messages = self._build_prompt_messages(example)
        response = example.get(self.response_key, "")
        
        # Create a copy and append assistant response
        full_messages = list(prompt_messages)
        full_messages.append({"role": "assistant", "content": response})
        return full_messages

    def __getitem__(self, item):
        """
        Process a single item for SFT training.
        
        Returns dict with:
        - input_ids: full sequence (prompt + response)
        - attention_mask: attention mask
        - position_ids: position ids
        - loss_mask: 0 for prompt tokens, 1 for response tokens (excluding last)
        - multi_modal_inputs: timeseries embeddings and related tensors
        """
        # Get row as dict (HuggingFace Dataset indexing)
        row_dict: dict = dict(self.dataframe[item])
        
        # Extract data
        prompt_messages = self._build_prompt_messages(row_dict)
        full_messages = self._build_full_messages(row_dict)
        raw_ts_data = self._get_timeseries_data(row_dict)
        
        if not raw_ts_data:
            logger.warning(f"TSSFTDataset[{item}]: no timeseries data found (key='{self.ts_key}')")

        # === Apply chat template ===
        # Full text (prompt + response) - for training
        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            **self.apply_chat_template_kwargs
        )
        
        # Prompt only text - to determine where response starts
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.apply_chat_template_kwargs
        )
        
        # === Process with multimodal processor ===
        # Full sequence
        full_inputs = self.processor(
            text=[full_text],
            timeseries=raw_ts_data if raw_ts_data else None,
            return_tensors="pt"
        )
        
        # Prompt only (to get prompt length after tokenization with TS tokens)
        prompt_inputs = self.processor(
            text=[prompt_text],
            timeseries=raw_ts_data if raw_ts_data else None,
            return_tensors="pt"
        )
        
        # Extract tensors (remove batch dimension)
        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        prompt_length = prompt_inputs["input_ids"].shape[1]
        
        sequence_length = input_ids.shape[0]
        
        # === Create loss mask BEFORE padding/truncation ===
        # Loss mask: 0 for prompt, 1 for response
        loss_mask = torch.ones_like(attention_mask)
        
        # Mask out prompt tokens
        if prompt_length > 0:
            loss_mask[:min(prompt_length, sequence_length)] = 0
        
        # === Handle truncation ===
        if sequence_length > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Sequence length {sequence_length} exceeds max_length {self.max_length}"
                )
            sequence_length = self.max_length
        
        # === Handle padding (right padding for SFT) ===
        if sequence_length < self.max_length:
            pad_length = self.max_length - sequence_length
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            # Pad input_ids
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype)
            ])
            
            # Pad attention_mask with 0
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_length, dtype=attention_mask.dtype)
            ])
            
            # Pad loss_mask with 0 (don't compute loss on padding)
            loss_mask = torch.cat([
                loss_mask,
                torch.zeros(pad_length, dtype=loss_mask.dtype)
            ])
        
        # Mask out the last token in the valid sequence (standard SFT practice)
        valid_length = attention_mask.sum().item()
        if valid_length > 0:
            loss_mask[int(valid_length) - 1] = 0
        
        # === Compute position IDs ===
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]
        
        # === Extract multi-modal inputs (keep as-is, matching TSRLHFDataset) ===
        multi_modal_inputs = {}
        for key, value in full_inputs.items():
            if key not in ["input_ids", "attention_mask"]:
                multi_modal_inputs[key] = value
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "multi_modal_inputs": multi_modal_inputs,
        }