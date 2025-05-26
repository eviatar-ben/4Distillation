import ipdb
import os
from pathlib import Path
from typing import Tuple, List, Literal

import pyrallis
import torch
from accelerate import Accelerator
from torch import nn
from transformers import CLIPTokenizer

from models.neti_clip_text_encoder import NeTICLIPTextModel
from models.neti_mapper import NeTIMapper
from models.positional_encoding import NeTIPositionalEncoding, BasicEncoder, FourierPositionalEncoding
from training.config import RunConfig


class CheckpointHandler:

    def __init__(self, cfg: RunConfig, save_root: Path):

        self.cfg = cfg
        self.save_root = save_root
        # aggregate the tokens

    def save_model(self,
                   text_encoder: NeTICLIPTextModel,
                   projection_matrix_base: torch.Tensor,
                   projection_matrix_bypass: torch.Tensor,
                   mapper_save_name: str,
                   projection_save_name: str):
        # self.save_learned_embeds(text_encoder, accelerator, embeds_save_name)
        self.save_mapper(text_encoder, mapper_save_name)
        self.save_projection_matrix(projection_matrix_base, projection_matrix_bypass, projection_save_name)

    def save_mapper(self, text_encoder: NeTICLIPTextModel, save_name: str):
        """ Save the mapper and config to be used at inference. """
        cfg_ = RunConfig(**self.cfg.__dict__.copy())

        mapper_time = text_encoder.text_model.embeddings.mapper_time

        if mapper_time is not None:
            state_dict = {
                "cfg": pyrallis.encode(cfg_),
                'mappers': {
                    'dummy_key': {
                        "state_dict": mapper_time.state_dict(),
                        "encoder": mapper_time.encoder,
                        'placeholder_object_token': "dummy",
                    }
                }
            }
            if hasattr(mapper_time, "encode_phi"):
                state_dict.update({"encode_phi": mapper_time.encode_phi})

            fname = os.path.join(self.save_root, Path(save_name).stem + "_time" + Path(save_name).suffix)
            torch.save(state_dict, fname)

    def save_projection_matrix(self, projection_matrix_base: torch.Tensor,
                               projection_matrix_bypass: torch.Tensor,
                               projection_save_name: str):
        """ Save the projection matrix to be used at inference. """
        torch.save(projection_matrix_base, os.path.join(self.save_root,
                                                        Path(projection_save_name).stem + "_base" + Path(
                                                            projection_save_name).suffix))

        torch.save(projection_matrix_bypass, os.path.join(self.save_root,
                                                          Path(projection_save_name).stem + "_bypass" + Path(
                                                              projection_save_name).suffix))

    @staticmethod
    def clean_config_dict(cfg_dict):
        """
        If you run pyrallis.decode() to recreate a config object from a saved
        config from a pretrained model, then there are some clashes that happen.
        This is a hacky way to fix them.
        """
        # todo : in not sure whats going on here i need to recheck that
        # if 'placeholder_time_tokens' in cfg_dict['data'].keys():
        if 'placeholder_view_tokens' in cfg_dict['data'].keys():
            del cfg_dict['data']['placeholder_view_tokens']

        for k in ['target_norm_object', 'target_norm_view', 'pretrained_view_mapper', 'pretrained_view_mapper_key']:
            # for k in ['target_norm_object', 'target_norm_time', 'pretrained_time_mapper', 'pretrained_time_mapper_key']:
            if k in cfg_dict['model'].keys():
                if cfg_dict['model'][k] is None:
                    del cfg_dict['model'][k]

        for k in ["validation_time_tokens", "eval_placeholder_object_tokens"]:
            if k in cfg_dict['eval'].keys():
                if cfg_dict['eval'][k] is None:
                    del cfg_dict['eval'][k]

        for k in ['placeholder_object_tokens', 'train_data_subsets']:
            if k in cfg_dict['data'].keys():
                if cfg_dict['data'][k] is None:
                    del cfg_dict['data'][k]

        return cfg_dict

    @staticmethod
    def load_mapper(mapper_path: Path, embedding_type: Literal["object", "time"] = "object",
                    placeholder_time_tokens: List[str] = None, placeholder_time_token_ids: List[int] = None,
                    placeholder_object_tokens: List[str] = None, placeholder_object_token_ids: List[int] = None,
                    ) -> Tuple[RunConfig, NeTIMapper]:
        """ """
        mapper_ckpt = torch.load(mapper_path, map_location="cpu")
        cfg_dict = CheckpointHandler.clean_config_dict(mapper_ckpt['cfg'])
        mapper_ckpt['cfg']['model']['target_norm_time'] = 0.2  # todo thats a bypass for now
        cfg = pyrallis.decode(RunConfig, mapper_ckpt['cfg'])

        # handle the special case of getting the view token_ids from the tokenizer
        if embedding_type == "time":
            output_bypass = cfg.model.output_bypass_view
            target_norm = cfg.model.target_norm_view
        else:
            output_bypass = cfg.model.output_bypass_object
            placeholder_time_tokens, placeholder_time_token_ids = None, None
            target_norm = cfg.model.target_norm_object
            if target_norm is None and cfg.model.normalize_object_mapper_output:
                raise ValueError(
                    "need a target norm to pass to pretrained object mapper")

        # load this option that was added later
        bypass_unconstrained = False
        if 'bypass_unconstrained_object' in mapper_ckpt['cfg']['model'].keys():
            if embedding_type == "object":
                bypass_unconstrained = mapper_ckpt['cfg']['model']['bypass_unconstrained_object']
                output_bypass_alpha = mapper_ckpt['cfg']['model'].get('output_bypass_alpha_object', 0.2)
            else:
                bypass_unconstrained = mapper_ckpt['cfg']['model']['bypass_unconstrained_time']
                output_bypass_alpha = mapper_ckpt['cfg']['model'].get('output_bypass_alpha_object', 0.2)

        # Save to dict. Objects must be in this format because we can have
        # multiple object-mappers.
        neti_mapper_lookup = {}
        for k in mapper_ckpt['mappers'].keys():
            state_dict = mapper_ckpt['mappers'][k]['state_dict']
            encoder = mapper_ckpt['mappers'][k]['encoder']
            token = mapper_ckpt['mappers'][k]['placeholder_object_token']

            if embedding_type == "time":
                placeholder_object_token_id = "dummy"  # will be ignored anyway

            else:
                lookup_token_to_token_id = dict(zip(placeholder_object_tokens, placeholder_object_token_ids))
                placeholder_object_token_id = lookup_token_to_token_id[token]

            neti_mapper = NeTIMapper(embedding_type=embedding_type,
                                     placeholder_time_tokens=placeholder_time_tokens,
                                     placeholder_time_token_ids=placeholder_time_token_ids,
                                     output_dim=cfg.model.word_embedding_dim,
                                     arch_mlp_hidden_dims=cfg.model.arch_mlp_hidden_dims,
                                     use_nested_dropout=cfg.model.use_nested_dropout,
                                     nested_dropout_prob=cfg.model.nested_dropout_prob,
                                     norm_scale=target_norm,
                                     use_positional_encoding=cfg.model.use_positional_encoding_object,
                                     num_pe_time_anchors=cfg.model.num_pe_time_anchors,
                                     pe_sigmas=cfg.model.pe_sigmas,
                                     arch_time_net=cfg.model.arch_time_net,
                                     arch_time_mix_streams=cfg.model.arch_time_mix_streams,
                                     arch_time_disable_tl=cfg.model.arch_time_disable_tl,
                                     original_ti=cfg.model.original_ti,
                                     output_bypass=output_bypass,
                                     output_bypass_alpha=output_bypass_alpha,
                                     placeholder_object_token=token,
                                     bypass_unconstrained=bypass_unconstrained)

            neti_mapper.load_state_dict(state_dict, strict=True)

            # note that the encoder is only used in arch_view <= 14
            if isinstance(encoder, NeTIPositionalEncoding):  # todo recheck that
                encoder.w = nn.Parameter(mapper_ckpt['encoder'].w.cuda())
                neti_mapper.encoder = encoder.cuda()
            elif isinstance(encoder, BasicEncoder):
                encoder.normalized_timesteps = mapper_ckpt['encoder'].normalized_timesteps.cuda()
                encoder.normalized_unet_layers = mapper_ckpt['encoder'].normalized_unet_layers.cuda()
                neti_mapper.encoder = encoder.cuda()
            neti_mapper.cuda()
            neti_mapper.eval()
            neti_mapper_lookup[placeholder_object_token_id] = neti_mapper

        # if view, then return the only mapper; if object, return the dict of objects.
        mapper_out = neti_mapper_lookup['dummy'] if embedding_type == "time" else neti_mapper_lookup

        projection_base_path = Path(mapper_path.parent.parent / "projections",
                                    mapper_path.stem.replace("_time", "")
                                    .replace("mapper", "projection") + "_base" + mapper_path.suffix)

        projection_bypass_path = Path(mapper_path.parent.parent / "projections",
                                      mapper_path.stem.replace("_time", "")
                                      .replace("mapper", "projection") + "_bypass" + mapper_path.suffix)
        # load projection matrix
        projection_matrix_base = torch.load(projection_base_path)
        projection_matrix_bypass = torch.load(projection_bypass_path)

        return cfg, mapper_out, projection_matrix_base, projection_matrix_bypass
