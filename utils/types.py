import enum
from dataclasses import dataclass
from typing import Optional, List

import torch


@dataclass
class NeTIBatch:
    timesteps: torch.Tensor
    normalized_frames_gap: torch.Tensor
    unet_layers: torch.Tensor
    truncation_idx: Optional[int] = None
    relative_diffusion_frame_number : Optional[torch.Tensor] = None
    human_action : Optional[torch.Tensor] = None #for norm normalization
@dataclass
class PESigmas:
    sigma_t: float
    sigma_l: float
    sigma_theta: Optional[float] = float
    sigma_time: Optional[float] = float
    sigma_r: Optional[float] = float
    sigma_dtu12: Optional[float] = float

@dataclass 
class MapperOutput:
   word_embedding: torch.Tensor
   bypass_output: torch.Tensor
   bypass_unconstrained: bool
   output_bypass_alpha: float
   normalized_word_embedding: Optional[torch.Tensor] = None
