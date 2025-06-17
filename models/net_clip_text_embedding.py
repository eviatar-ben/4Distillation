import ipdb
from typing import Optional, Tuple, Dict

import torch
from torch import nn
from transformers import CLIPTextConfig

from models.neti_mapper import NeTIMapper
from utils.types import NeTIBatch, MapperOutput
from constants import PRETRAINED_MODE0_OBJ


class NeTICLIPTextEmbeddings(nn.Module):
    """ Modification of CLIPTextEmbedding to allow for the use of a NeTIMapper to overwrite the concept token. """

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.mapper_time = None

    def set_mapper(self, mapper_time: NeTIMapper, device='cuda'):
        self.mapper_time = mapper_time

    def forward(self,
                position_ids = None,
                input_ids = None,
                batch: Optional[NeTIBatch] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        v_bypass = None
        if batch is not None:
            mapper_time_outputs = self.mapper_time(timestep=batch.timesteps.float(),
                                                   normalized_frames_gap=batch.normalized_frames_gap.float(),
                                                   unet_layer=batch.unet_layers.float(),
                                                   truncation_idx=batch.truncation_idx,
                                                   human_action=None)
            v_base = mapper_time_outputs.word_embedding

            v_bypass = mapper_time_outputs.bypass_output

        else:
            # Regular embedding logic
            seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]
            if inputs_embeds is None:
                inputs_embeds = self.token_embedding(input_ids)
            position_embeddings = self.position_embedding(position_ids)
            v_base = inputs_embeds + position_embeddings
        return v_base, v_bypass
