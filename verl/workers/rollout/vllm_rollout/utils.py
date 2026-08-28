# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, the smallest `max_lora_rank` is 8, and allowed values are (8, 16, 32, 64, 128, 256, 320, 512)
    This function automatically adjusts the `max_lora_rank` to the nearest allowed value.

    Reference: https://github.com/vllm-project/vllm/blob/8a297115e2367d463b781adb86b55ac740594cf6/vllm/config/lora.py#L27
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0 to invoke this function, get {lora_rank}"
    vllm_max_lora_ranks = [8, 16, 32, 64, 128, 256, 320, 512]
    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank

    raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")

import os
import torch
from vllm.v1.sample.logits_processor import LogitsProcessor, BatchUpdate
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

class ThinkGreedyLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        
        # Read the token ID exported by the master server, default to Qwen's 151668
        self.think_token_id = int(os.environ.get("VERL_THINK_TOKEN_ID", "151668"))
        self.req_info = {}

    def is_argmax_invariant(self) -> bool:
        # We modify argmax behavior (force greedy), so this must be False
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        # process_dict_updates expects new_state to be a callable:
        #   (params, prompt_tok_ids, output_tok_ids) -> Optional[state]
        # We store the output_tok_ids reference as state so apply() can check for </think>
        def _new_state_fn(params, prompt_tok_ids, output_tok_ids):
            return output_tok_ids

        process_dict_updates(self.req_info, batch_update, _new_state_fn)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for req_idx, output_tokens in self.req_info.items():
            if self.think_token_id in output_tokens:
                req_logits = logits[req_idx]
                
                # Force greedy selection by zeroing out everything except the max logit
                max_idx = torch.argmax(req_logits)
                new_logits = torch.full_like(req_logits, -float("inf"))
                new_logits[max_idx] = req_logits[max_idx]
                
                logits[req_idx] = new_logits

        return logits
