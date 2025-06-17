import checkpoint_handler
from pathlib import Path
from accelerate import Accelerator
from training.validate import ValidationHandler
import ipdb
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor
import os
import random

sys.path.append(".")
sys.path.append("..")

import constants
from models.neti_clip_text_encoder import NeTICLIPTextModel
from models.neti_mapper import NeTIMapper
from prompt_manager import PromptManager
from sd_pipeline_call import sd_pipeline_call
from models.xti_attention_processor import XTIAttenProc
from checkpoint_handler import CheckpointHandler
from utils import vis_utils
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from training.coach import Coach
from training.validate import ValidationHandler
from training.config import RunConfig
from training.dataset import TextualInversionDataset
from accelerate import Accelerator

from ip_adapter.resampler import Resampler
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


MAPPER_PATH = '/sci/labs/sagieb/eviatar/Distillation4D/results/save_linear_projection/mappers/mapper-steps-29500_time.pt'


# todo: Eviatar: dtype is sets to torch.float32 but that's need to dynamically changes as in coach.py : get_dtype()

def load_sd_model_components(train_cfg, device="cuda"):
    """
    Load the SD model components separately. This does not load the special
    tokens.
    """
    text_encoder = NeTICLIPTextModel.from_pretrained(train_cfg.model.pretrained_model_name_or_path,
                                                     subfolder="text_encoder",
                                                     revision=train_cfg.model.revision, ).to(device)

    tokenizer = CLIPTokenizer.from_pretrained(train_cfg.model.pretrained_model_name_or_path,
                                              subfolder="tokenizer")

    vae = AutoencoderKL.from_pretrained(train_cfg.model.pretrained_model_name_or_path,
                                        subfolder="vae",
                                        revision=train_cfg.model.revision)

    unet = UNet2DConditionModel.from_pretrained(train_cfg.model.pretrained_model_name_or_path,
                                                subfolder="unet",
                                                revision=train_cfg.model.revision)

    noise_scheduler = DDPMScheduler.from_pretrained(train_cfg.model.pretrained_model_name_or_path,
                                                    subfolder="scheduler")

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(train_cfg.model.image_encoder_path).to(device)

    return text_encoder, tokenizer, vae, unet, noise_scheduler, image_encoder


def init_ip_adapter(train_cfg, unet, image_encoder, accelerator):
    from models import ip_adapter

    num_queries = train_cfg.model.num_image_tokens
    # num_queries = len(self.clip_images) # look at config.py num_image_tokens

    # ip-adapter-plus
    image_proj_model = Resampler(dim=unet.config.cross_attention_dim,
                                 depth=4,
                                 dim_head=64,
                                 heads=12,
                                 num_queries=num_queries,
                                 embedding_dim=image_encoder.config.hidden_size,
                                 output_dim=unet.config.cross_attention_dim,
                                 ff_mult=4)
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {"to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                       "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"], }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                               scale=train_cfg.model.ip_hidden_states_scale, num_tokens=num_queries)
            attn_procs[name].load_state_dict(weights)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    ip_adapter = ip_adapter.IPAdapter(unet, image_proj_model, adapter_modules,
                                      image_encoder_path=train_cfg.model.image_encoder_path,
                                      ip_adapter_subset_size=train_cfg.data.ip_adapter_subset_size,
                                      ckpt_path=train_cfg.model.ip_adapter_path,
                                      device=accelerator.device,
                                      dtype=torch.float32)
    return ip_adapter



def get_random_images(N=1000):
    import os
    import random
    base_dir = '/sci/labs/sagieb/eviatar/data/UCF-101-Frames'

    # List all the action folders
    action_folders = [os.path.join(base_dir, folder) for folder in os.listdir(base_dir) if
                      os.path.isdir(os.path.join(base_dir, folder))]

    # List all subfolders (video folders) in the action folders
    video_folders = []
    for action_folder in action_folders:
        video_folders.extend([os.path.join(action_folder, subfolder) for subfolder in os.listdir(action_folder) if
                              os.path.isdir(os.path.join(action_folder, subfolder))])

    # Now, randomly select N paths
    random_paths = []
    while len(random_paths) < N:
        # Pick a random video folder
        video_folder = random.choice(video_folders)

        # Get all image files in this video folder
        image_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.jpg')]

        # Pick a random image from this video folder
        if image_files:
            random_image = random.choice(image_files)
            if random_image not in random_paths:  # Ensure uniqueness
                random_paths.append(random_image)

    return random_paths

def get_generated_videos_from_paths(paths=None, n_frames=64):

    train_cfg, mapper, projection_matrix_base, projection_matrix_bypass = \
        checkpoint_handler.CheckpointHandler.load_mapper(mapper_path=Path(MAPPER_PATH),
                                                         embedding_type="time")

    projection_matrix_base, projection_matrix_bypass = projection_matrix_base.to("cuda"), projection_matrix_bypass.to(
        "cuda")

    accelerator = Accelerator()
    text_encoder, tokenizer, vae, unet, noise_scheduler, image_encoder = load_sd_model_components(train_cfg,
                                                                                                  device=accelerator.device)

    ip_adapter = init_ip_adapter(train_cfg, unet, image_encoder, accelerator)

    # use the code from the validation handler to run DTU inference
    validator = ValidationHandler(cfg=train_cfg,
                                  weights_dtype=torch.float32,
                                  relative_tokens=True)
    if paths is None:
        conditioned_images = get_random_images(N=100)
        conditioned_images_paths = None
    else:
        # paths is a directory of images randomlly choose an image:
        conditioned_images_paths = [path +"/"+ random.choice(os.listdir(path)) for path in paths]
        conditioned_images = [Image.open(path) for path in conditioned_images_paths]

    results = validator.basic_inference(n_frames=n_frames,
                                        loaded_time_mapper=mapper,
                                        conditioned_images_paths= conditioned_images_paths,
                                        conditioned_images=conditioned_images,
                                        accelerator=accelerator,
                                        tokenizer=tokenizer,
                                        text_encoder=text_encoder,
                                        ip_adapter=ip_adapter,
                                        unet=unet,
                                        vae=vae,
                                        projection_matrix_base=projection_matrix_base,
                                        projection_matrix_bypass=projection_matrix_bypass)

def main():
    get_generated_videos_from_paths()

if __name__ == "__main__":
    main()