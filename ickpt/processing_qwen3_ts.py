import numpy as np
from typing import List, Union, Tuple, Optional
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy

def sp_encoding(timeseries: np.ndarray) -> Tuple[np.ndarray, str, dict]:
    timeseries = np.array(timeseries)
    mean = np.mean(timeseries)
    scaled_timeseries = timeseries - mean
    scale_factor = 1.0
    if np.any(np.abs(scaled_timeseries) >= 3.0):
        scale_factor = np.max(np.abs(scaled_timeseries)) / 3.0
        scaled_timeseries /= scale_factor

    prompt = f"[offset={-mean:.4f}|scaling={scale_factor:.4f}]<ts>"

    result_timeseries = np.stack([scaled_timeseries, np.ones_like(scaled_timeseries)], axis=-1).reshape(-1, 1)
    return result_timeseries, prompt, {"offset": float(-mean), "scale_factor": float(scale_factor)}

class Qwen3TSProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer=None, chat_template=None, patch_size=8, **kwargs):
        if chat_template is None and tokenizer is not None and tokenizer.chat_template is not None:
            chat_template = tokenizer.chat_template
        self.chat_template = chat_template
        self.patch_size = patch_size
        self.image_processor = None
        super().__init__(tokenizer=tokenizer, chat_template=chat_template, **kwargs)

    def __call__(self, text: Union[str, List[str]], timeseries: Optional[List[np.ndarray]] = None, padding: Union[bool, str, PaddingStrategy] = False, padding_side: str = 'left', vllm_flag: bool = False, tokenize: bool = True, **kwargs) -> BatchFeature:
        if isinstance(text, str): text = [text]
        if timeseries is None: timeseries = []

        encoded_ts_arrays = []
        ts_tokens = []
        reconstructed_prompts = []

        if vllm_flag:
            reconstructed_prompts = text
            ts_tokens = []
            for ts in timeseries:
                encoded_ts, ts_prompt, _ = sp_encoding(ts)
                if self.tokenizer is not None:
                    tokens = self.tokenizer.encode(ts_prompt, add_special_tokens=False)
                    ts_tokens.append(tokens)
                encoded_ts_arrays.append(encoded_ts[None, ...])
            
            kwargs.pop("images", None)
            kwargs.pop("videos", None)
            tokenizer_outputs = self.tokenizer(reconstructed_prompts, padding=padding, padding_side=padding_side, **kwargs) if tokenize else {"text": reconstructed_prompts}
            outputs = tokenizer_outputs
            outputs["timeseries"] = list(zip(ts_tokens, encoded_ts_arrays))
            return BatchFeature(data=outputs)

        else:
            total_ts_cnt = 0
            for idx, prompt in enumerate(text):
                last_ts_cnt = total_ts_cnt
                prompt_segments = prompt.split("<ts>")
                total_ts_cnt += len(prompt_segments) - 1
                reconstructed_prompt = prompt_segments[0]

                for i, ts in enumerate(timeseries[last_ts_cnt:total_ts_cnt]):
                    encoded_ts, ts_prompt, _ = sp_encoding(ts)

                    num_patches = (len(ts) + self.patch_size - 1) // self.patch_size
                    placeholder_expansion = "<ts>" * (num_patches - 1)
                    
                    reconstructed_prompt += ts_prompt + placeholder_expansion + prompt_segments[i + 1]
                    encoded_ts_arrays.append(encoded_ts[None, ...])

                reconstructed_prompts.append(reconstructed_prompt)

            concatenated_ts = None
            if len(encoded_ts_arrays) > 0:
                max_length = max(ts.shape[1] for ts in encoded_ts_arrays)
                padded_ts_arrays = [np.pad(ts, ((0, 0), (0, max_length - ts.shape[1]), (0, 0)), mode="constant") for ts in encoded_ts_arrays]
                concatenated_ts = torch.from_numpy(np.concatenate(padded_ts_arrays, axis=0)).half()

            kwargs.pop("images", None)
            kwargs.pop("videos", None)
            tokenizer_outputs = self.tokenizer(reconstructed_prompts, padding=padding, padding_side=padding_side, **kwargs) if tokenize else {"text": reconstructed_prompts}
            outputs = tokenizer_outputs
            if concatenated_ts is not None: outputs["timeseries"] = concatenated_ts
            return BatchFeature(data=outputs)

    def decode(self, *args, **kwargs): return self.tokenizer.decode(*args, **kwargs)
    def batch_decode(self, *args, **kwargs): return self.tokenizer.batch_decode(*args, **kwargs)
    def encode_timeseries(
        self,
        timeseries: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        if timeseries is None:
            timeseries = []

        concatenated_ts = None
        encoded_ts_arrays = []

        for i, ts in enumerate(timeseries):
            encoded_ts, _, _ = sp_encoding(ts)
            # Ensure time series shape [1, seq_len, feature_dim] for batch concatenation
            encoded_ts_arrays.append(encoded_ts[None, ...])

        if len(encoded_ts_arrays) > 0:
            # Pad time series to the same length
            max_length = max(ts.shape[1] for ts in encoded_ts_arrays)
            padded_ts_arrays = [
                np.pad(ts, ((0, 0), (0, max_length - ts.shape[1]), (0, 0)), mode="constant", constant_values=0.0)
                for ts in encoded_ts_arrays
            ]
            concatenated_ts = np.concatenate(padded_ts_arrays, axis=0)  # Shape: [batch_size, max_length, feature_dim]

            # Convert to torch
            concatenated_ts = torch.from_numpy(concatenated_ts).half()

        return concatenated_ts

