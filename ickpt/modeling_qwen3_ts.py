from typing import Callable, Optional, Union, Any, Dict
import torch
from torch import nn
from dataclasses import dataclass

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel, Qwen3Model
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, logging, ModelOutput
from .configuration_qwen3_ts import Qwen3TSConfig

logger = logging.get_logger(__name__)

class TimeSeriesEmbedding(nn.Module):
    def __init__(self, config):
        super(TimeSeriesEmbedding, self).__init__()
        self.patch_size = config['patch_size']
        self.num_layers = config['num_layers']
        self.hidden_size = config['hidden_size']
        self.num_features = config['num_features']
        self.max_sequence_length = config['max_sequence_length']
        self.use_position_embedding = config.get('use_position_embedding', False)
        self.use_position_idx = config.get('use_position_idx', False)
        self.use_layer_norm = config.get('use_layer_norm', False)
        self.embedding_dim = config.get('embedding_dim', 16)
        
        if self.use_position_embedding:
            self.position_embedding = nn.Embedding(self.max_sequence_length + 1, self.embedding_dim)
            self.padding_idx = self.max_sequence_length
            input_size = 1 * self.patch_size + self.embedding_dim * self.patch_size
        elif self.use_position_idx:
            input_size = 2 * self.patch_size
        else:
            input_size = 1 * self.patch_size
        
        layers = []
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(input_size, self.hidden_size))
            layers.append(nn.GELU())
            input_size = self.hidden_size

        layers.append(nn.Linear(input_size, self.hidden_size))
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(self.hidden_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.num_features)
        mask = x[:, :, -1].long()
        valid_lengths = mask.sum(dim=1).long()
        patch_cnt = (valid_lengths + self.patch_size - 1) // self.patch_size

        patches_list = []
        all_position_indices = []
        patch_info_list = []
        
        for i in range(batch_size):
            vl = valid_lengths[i].item()
            pc = patch_cnt[i].item()
            if pc == 0: continue
            
            xi = x[i, :vl, :1]
            total_padded_length = pc * self.patch_size
            padding_length = total_padded_length - vl
            position_indices = torch.arange(vl, device=x.device)
            
            if padding_length > 0:
                last_value = xi[-1:, :]
                padding = last_value.repeat(padding_length, 1)
                xi = torch.cat([xi, padding], dim=0)
                padding_positions = torch.full((padding_length,), self.padding_idx, device=x.device)
                position_indices = torch.cat([position_indices, padding_positions], dim=0)

            xi = xi.reshape(pc, self.patch_size)
            position_indices = position_indices.reshape(pc, self.patch_size)

            if self.use_position_embedding:
                all_position_indices.append(position_indices)
                patch_info_list.append({'xi': xi, 'pc': pc, 'sample_idx': i})
            elif self.use_position_idx:
                pos_indices = torch.arange(vl, device=x.device).unsqueeze(1)
                pos_indices = pos_indices / max(1, valid_lengths.max().item() - 1)
                if padding_length > 0:
                    padding_indices = torch.full((padding_length, 1), -1, device=x.device)
                    pos_indices = torch.cat([pos_indices, padding_indices], dim=0)
                xi_combined = torch.cat([xi.reshape(-1, 1), pos_indices], dim=1)
                patch_input = xi_combined.reshape(pc, self.patch_size * 2)
                patches_list.append(patch_input)
            else:
                patches_list.append(xi)

        if self.use_position_embedding and all_position_indices:
            batch_position_indices = torch.cat(all_position_indices, dim=0)
            batch_pos_emb = self.position_embedding(batch_position_indices)
            emb_start_idx = 0
            for patch_info in patch_info_list:
                xi = patch_info['xi']
                pc = patch_info['pc']
                pos_emb = batch_pos_emb[emb_start_idx:emb_start_idx + pc]
                emb_start_idx += pc
                xi = xi.unsqueeze(-1)
                patch_input = torch.cat([xi.flatten(1), pos_emb.flatten(1)], dim=1)
                patches_list.append(patch_input)

        if patches_list:
            x_patches = torch.cat(patches_list, dim=0)
            x = self.mlp(x_patches)
        else:
            x = torch.empty(0, self.hidden_size, device=x.device)
        return x, patch_cnt

@dataclass
class Qwen3TSCausalLMOutputWithPast(CausalLMOutputWithPast):
    attention_mask: Optional[torch.FloatTensor] = None
    labels: Optional[torch.LongTensor] = None

class Qwen3TSGenerationMixin(GenerationMixin):
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, timeseries=None, **kwargs):
        has_ts = timeseries is not None and len(timeseries) > 0
        if has_ts and past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = past_key_values.seen_tokens if hasattr(past_key_values, 'seen_tokens') else past_key_values.get_seq_length()
            else:
                past_length = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
            if past_length > 0:
                input_ids = input_ids[:, -1:]
                timeseries = None
        
        model_inputs = super().prepare_inputs_for_generation(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, cache_position=cache_position, **kwargs)
        model_inputs["timeseries"] = timeseries
        return model_inputs
    
    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        if hasattr(outputs, "attention_mask") and outputs.attention_mask is not None:
            model_kwargs["attention_mask"] = outputs.attention_mask
        return super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder, num_new_tokens)
    
    def generate(self, inputs=None, timeseries=None, generation_config=None, **kwargs):
        if timeseries is not None: kwargs["timeseries"] = timeseries
        return super().generate(inputs=inputs, generation_config=generation_config, **kwargs)
    
    def _validate_model_kwargs(self, model_kwargs):
        timeseries = model_kwargs.pop("timeseries", None)
        super()._validate_model_kwargs(model_kwargs)
        if timeseries is not None: model_kwargs["timeseries"] = timeseries

class Qwen3TSPreTrainedModel(Qwen3PreTrainedModel):
    config_class = Qwen3TSConfig

@auto_docstring
class Qwen3TSForCausalLM(Qwen3TSPreTrainedModel, Qwen3TSGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.ts_encoder = TimeSeriesEmbedding(config.ts)
        self.ts_placeholder_token_id = getattr(config, "ts_token_start_index", None)
        self.post_init()
    
    def get_input_embeddings(self): return self.model.embed_tokens
    def set_input_embeddings(self, value): self.model.embed_tokens = value
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, new_embeddings): self.lm_head = new_embeddings

    @can_return_tuple
    def forward(self, input_ids=None, timeseries=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, cache_position=None, logits_to_keep=0, **kwargs):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        assert timeseries is not None, "Timeseries data is not detected!"

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if timeseries is not None:
                if isinstance(timeseries, list):
                    timeseries = torch.cat(timeseries, dim=0)
                if timeseries.shape[0] > 0:
                    ts_features, patch_cnt_ = self.ts_encoder(timeseries)

                    placeholder_mask = (input_ids == self.ts_placeholder_token_id)
                    num_placeholders = placeholder_mask.sum().item()
                    num_features_produced = ts_features.shape[0]
                    assert num_placeholders == num_features_produced, f"EMPTY TOKENS {num_placeholders} != {num_features_produced} TS FEATURES"
                    full_mask = placeholder_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_scatter(full_mask, ts_features.to(inputs_embeds.device, inputs_embeds.dtype))

            if cache_position is not None and attention_mask is not None:
                cache_position = torch.arange(attention_mask.size(-1) - inputs_embeds.size(1), attention_mask.size(-1), device=inputs_embeds.device)

        outputs = self.model(attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, cache_position=cache_position, **kwargs)

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return Qwen3TSCausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            attention_mask=attention_mask, labels=labels
        )

__all__ = ["Qwen3TSForCausalLM"]