import json
import torch
import numpy as np
import logging
import traceback
from typing import Optional, List, Dict, Any
from verl.utils.dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

logger = logging.getLogger(__name__)


class TSRLHFDataset(RLHFDataset):
    """
    Custom dataset for Time Series LLMs.
    Key insight: vLLM will call the processor internally when we pass
    multi_modal_data. We just need to pass the raw timeseries arrays.
    """

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        self.ts_key = config.get("timeseries_key", "timeseries")
        super().__init__(data_files, tokenizer, config, processor, max_samples)
        if self.processor is None:
            logger.warning("TSRLHFDataset initialized without a processor.")

    def _build_messages(self, example: dict):
        """Extract messages from example. Unlike base class, doesn't pop."""
        return example.get(self.prompt_key)

    def _get_timeseries_data(self, doc: Dict[str, Any]) -> List[np.ndarray]:
        """Extract timeseries data as list of numpy arrays."""
        ts_data = doc.get(self.ts_key)
        if ts_data is None:
            return []
        if isinstance(ts_data, list):
            return [np.array(ts) if not isinstance(ts, np.ndarray) else ts for ts in ts_data]
        elif isinstance(ts_data, np.ndarray):
            return [ts_data]
        return []

    def maybe_filter_out_long_prompts(self, dataframe):
        if not self.filter_overlong_prompts:
            return dataframe

        print(f"Filtering overlong prompts using TS Processor logic...")

        def doc2len(doc) -> int:
            try:
                doc = dict(doc)  # Make a copy to avoid mutation
                messages = self._build_messages(doc)
                ts_data = self._get_timeseries_data(doc)

                raw_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                    **self.apply_chat_template_kwargs
                )

                inputs = self.processor(
                    text=[raw_prompt],
                    timeseries=ts_data,
                    return_tensors="pt"
                )
                return inputs["input_ids"].shape[-1]
            except Exception:
                logger.error("Error processing sample for length check:")
                traceback.print_exc()
                return self.max_prompt_length + 1

        dataframe = dataframe.filter(
            lambda doc: doc2len(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts > {self.max_prompt_length} tokens",
        )
        return dataframe

    def __getitem__(self, item):
        row_dict: dict = dict(self.dataframe[item])  # Make a copy
        messages = self._build_messages(row_dict)
        raw_ts_data = self._get_timeseries_data(row_dict)

        # === PART A: For Actor/Training ===
        raw_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            **self.apply_chat_template_kwargs
        )

        model_inputs = self.processor(
            text=[raw_prompt],
            timeseries=raw_ts_data,
            return_tensors="pt"
        )

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        # === PART B: Multi-modal inputs for actor ===
        if self.return_multi_modal_inputs:
            multi_modal_inputs = dict(model_inputs)
            multi_modal_inputs.pop("input_ids", None)
            multi_modal_inputs.pop("attention_mask", None)
            row_dict["multi_modal_inputs"] = multi_modal_inputs

        # === PART C: For vLLM Rollout ===
        multi_modal_data = {}
        if raw_ts_data:
            multi_modal_data["timeseries"] = raw_ts_data
        else:
            logger.warning(f"TSRLHFDataset[{item}]: no timeseries data found (key='{self.ts_key}')")
        row_dict["multi_modal_data"] = multi_modal_data

        # === PART D: Standard Post-processing ===
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        # Raw prompt IDs with proper truncation handling
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )
        row_dict["raw_prompt_ids"] = raw_prompt_ids

        # Optional outputs based on config
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        # Metadata — extra_info is stored as a JSON string in parquet
        extra_info = row_dict.get("extra_info")
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        elif extra_info is None:
            extra_info = {}
        row_dict["extra_info"] = extra_info
        row_dict["index"] = extra_info.get("index", 0)
        row_dict["tools_kwargs"] = extra_info.get("tools_kwargs", {})
        row_dict["interaction_kwargs"] = extra_info.get("interaction_kwargs", {})

        # Cleanup original keys
        row_dict.pop(self.ts_key, None)
        row_dict.pop(self.prompt_key, None)

        return row_dict
