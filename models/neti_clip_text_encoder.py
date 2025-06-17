import ipdb
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextConfig, CLIPTextModel, CLIPEncoder
from transformers.models.clip.modeling_clip import CLIPTextTransformer, _expand_mask

from models.net_clip_text_embedding import NeTICLIPTextEmbeddings
from utils.types import NeTIBatch


class NeTICLIPTextModel(CLIPTextModel):
    """ Modification of CLIPTextModel to use our NeTI mapper for computing the embeddings of the concept. """

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = NeTICLIPTextTransformer(config)
        self.post_init()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                batch: Optional[NeTIBatch] = None,
                layer_idx: Optional[int] = -1,
                projection_matrix_base: Optional[torch.Tensor] = None,
                projection_matrix_bypass: Optional[torch.Tensor] = None,
                ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return self.text_model.forward(batch=batch, input_ids=input_ids,
                                       projection_matrix_base=projection_matrix_base,
                                       projection_matrix_bypass=projection_matrix_bypass)


class NeTICLIPTextTransformer(CLIPTextTransformer):
    """ Modification of CLIPTextTransformer to use our NeTI mapper for computing the embeddings of the concept. """

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config=config)
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = NeTICLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.projection_matrix_base = None
        self.projection_matrix_bypass = None


    def set_projections(self, projection_matrix_base, projection_matrix_bypass):
        self.projection_matrix_base = projection_matrix_base
        self.projection_matrix_bypass = projection_matrix_bypass


    def forward(self, input_ids: Optional[torch.Tensor] = None,
                batch: Optional[NeTIBatch] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                projection_matrix_base: Optional[torch.Tensor] = None,
                projection_matrix_bypass: Optional[torch.Tensor] = None,
                ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """ 
        layer_idx is for debugging only 
        """
        if input_ids is not None:  # Regular embedding logic
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # that's means inference mode - and input_ids are actually the negative prompt
            # Regular embedding logic
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            hidden_states, v_bypass = self.embeddings(input_ids=input_ids, position_ids=position_ids)

            bsz, seq_len = input_shape
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
                hidden_states.device)

            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

            encoder_outputs = self.encoder(inputs_embeds=hidden_states,
                                           attention_mask=attention_mask,
                                           causal_attention_mask=causal_attention_mask,
                                           output_attentions=output_attentions,
                                           output_hidden_states=output_hidden_states,
                                           return_dict=return_dict, )
            last_hidden_state = encoder_outputs[0]
            v_base = self.final_layer_norm(last_hidden_state)
            v_bypass = None
        else:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            ###############################################
            v_base, v_bypass = self.embeddings(batch=batch)
            ###############################################

            ###############################################
            # Apply projection matrix
            v_base = torch.matmul(v_base, self.projection_matrix_base.to(v_base.device))
            v_bypass = torch.matmul(v_bypass, self.projection_matrix_bypass.to(v_base.device))
            ###############################################

            v_base = v_base.unsqueeze(1)
            v_bypass = v_bypass.unsqueeze(1)

            bsz, seq_len = v_base.shape[:2]
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, v_base.dtype).to(v_base.device)

            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(attention_mask, v_base.dtype)

            v_base = self.encoder(inputs_embeds=v_base,
                                  attention_mask=attention_mask,
                                  causal_attention_mask=causal_attention_mask,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states,
                                  return_dict=return_dict, )

            v_base = v_base[0]
            v_base = self.final_layer_norm(v_base)
            # v_bypass = self.final_layer_norm(v_bypass)
            # normalize v_bypass to be as the same as v_base norm:
            v_bypass = v_bypass / v_bypass.norm(dim=2, keepdim=True) * v_base.norm(dim=2, keepdim=True)

            v_base = v_base + 0.2 * v_bypass

        return v_base, v_bypass
