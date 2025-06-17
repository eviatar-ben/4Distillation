import ipdb
import random
from typing import Optional, List, Literal

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from constants import UNET_LAYERS
from models.positional_encoding import NeTIPositionalEncoding, BasicEncoder, PositionalEncoding, \
    FourierPositionalEncoding, FourierPositionalEncodingNDims
from utils.types import PESigmas, MapperOutput
from torch.nn.utils.parametrizations import spectral_norm
from models.mlp import MLP
from utils.utils import num_to_string, string_to_num
from training.dataset import TextualInversionDataset
from constants import TIME_TOKEN
from constants import PROJECTION_DIMENSION

class NeTIMapper(nn.Module):
    """ Main logic of our NeTI mapper. """

    def __init__(
            self,
            embedding_type: Literal['object', 'time'],
            output_dim: int = 768,
            unet_layers: List[str] = UNET_LAYERS,
            arch_mlp_hidden_dims: int = 128,
            use_nested_dropout: bool = True,
            nested_dropout_prob: float = 0.5,
            norm_scale: Optional[torch.Tensor] = None,
            use_positional_encoding=1,
            num_pe_time_anchors: int = 10,
            # todo sigma_time should be explored
            pe_sigmas: PESigmas = PESigmas(sigma_t=0.03, sigma_l=1.0, sigma_time=0.05),
            output_bypass: bool = True,
            placeholder_time_tokens: List[str] = None,
            placeholder_time_token_ids: torch.Tensor = None,
            arch_time_net: int = 0,
            arch_time_mix_streams: int = 0,
            arch_time_disable_tl: bool = True,  # todo : why is that sets to True?
            original_ti_init_embed=None,
            original_ti: bool = False,
            bypass_unconstrained: bool = True,
            output_bypass_alpha: float = 0.2,
            placeholder_object_token: str = None,

    ):
        """
        Args:
        embedding_type: whether the Neti-mapper should learn object or view
            control. View-control will condition on camera pose as well. MLP
            architecture is also different.
        placeholder_view_tokens: all possible view_tokens used for training.
            Ignored if embedding_type=='object'.
        placeholder_view_tokens_ids: token ids for `placeholder_view_tokens`
        arch_view_disable_tl: do not condition on timestep and unet layer (t,l)
        original_ti: run the 'original TI'
        bypass_unconstrained: passed through in the output
        """
        super().__init__()
        self.embedding_type = embedding_type
        self.arch_time_net = arch_time_net
        self.use_nested_dropout = use_nested_dropout
        self.nested_dropout_prob = nested_dropout_prob
        self.arch_mlp_hidden_dims = arch_mlp_hidden_dims
        self.norm_scale = norm_scale
        self.original_ti = original_ti
        self.arch_time_disable_tl = arch_time_disable_tl
        self.original_ti_init_embed = original_ti_init_embed
        self.output_bypass_alpha = output_bypass_alpha
        self.num_unet_layers = len(unet_layers)
        if original_ti and output_bypass:
            raise ValueError(f"If doing cfg.model.original_ti=[True]",
                             f" then you cannot have cfg.model.original_ti=[True]")
        output_dim = PROJECTION_DIMENSION
        print(f"#########################################{output_dim}##############################################")
        self.output_bypass = output_bypass
        if self.output_bypass:
            output_dim *= 2  # Output two vectors


        # set up positional encoding. For older experiments (arch_time_net<14),
        # use the legacy (t,l) conditioning. For later exps, call func for setup
        self.deg_freedom = self.embedding_type
        self.pe_sigmas = pe_sigmas
        self._set_positional_encoding()

        # define architecture
        if self.embedding_type == "view":
            self.arch_time_net = arch_time_net
            self.arch_time_mix_streams = arch_time_mix_streams

            if self.arch_time_disable_tl:
                self.input_dim = 0  # set_net_view functions will increase it

            self.set_net_time(num_unet_layers=len(unet_layers), num_time_anchors=num_pe_time_anchors,
                              output_dim=output_dim)
        elif self.embedding_type == "time":
            self.arch_time_net = arch_time_net
            self.arch_time_mix_streams = arch_time_mix_streams

            if self.arch_time_disable_tl:
                self.input_dim = 0  # set_net_view functions will increase it

            self.set_net_time(num_unet_layers=len(unet_layers), num_time_anchors=num_pe_time_anchors,
                              output_dim=output_dim)

        self.name = placeholder_object_token if embedding_type == "view" else 'time'  # for debugging


        self.embedding_norms_list = []

        if 0:
            v = next(self.parameters())
            v.register_hook(lambda x: print(f"Computed backward in mapper [{self.name}]"))


    def set_input_layer(self, num_unet_layers: int, num_time_anchors: int) -> nn.Module:
        if self.use_positional_encoding:
            input_layer = nn.Linear(self.encoder.num_w * 2, self.input_dim)
            input_layer.weight.data = self.encoder.init_layer(
                num_time_anchors, num_unet_layers)
        else:
            input_layer = nn.Identity()
        return input_layer

    def forward(self,
                timestep: torch.Tensor,
                unet_layer: torch.Tensor,
                normalized_frames_gap: torch.Tensor,
                human_action: torch.Tensor = None,
                truncation_idx: int = None) -> MapperOutput:
        """
        Args:
        input_ids_placeholder_view: If embedding_type=='object', ignored. If
            embedding_type=='view', use the token id to condition on the view
            parameters for that token.
        """

        embedding = self.extract_hidden_representation(timestep, unet_layer, normalized_frames_gap)

        if self.use_nested_dropout:
            embedding = self.apply_nested_dropout(embedding, truncation_idx=truncation_idx)

        norm_log = unet_layer[0].item() == 0 # for embeddings norm logging
        output = self.get_output(embedding, norm_log=norm_log, human_action=human_action)

        return output

    def get_encoded_input(self, timestep: torch.Tensor, unet_layer: torch.Tensor) -> torch.Tensor:
        """ Encode the (t,l) params """
        encoded_input = self.encoder.encode(timestep, unet_layer, )  # (bs,2048)
        return self.input_layer(encoded_input)  # (bs, 160)

    @staticmethod
    def scale_m1_1(x, xmin, xmax):
        """ scale a tensor to (-1,1). If xmin==xmax, do nothing"""
        if type(xmin) is not torch.Tensor:
            if xmin == xmax:
                return x
        return (x - xmin) / (xmax - xmin) * 2 - 1

    def get_time_params_from_frames_number(self, frames_number, device, dtype, norm=True):
        time_params = {'times': frames_number}
        return time_params


    def mix_encoding_and_views(self, encoded_tl: torch.Tensor, encoded_views: torch.Tensor):
        if self.arch_time_mix_streams == 0:
            encoded_input_mixed = torch.cat((encoded_tl, encoded_views), dim=1)
        elif self.arch_time_mix_streams == 1:
            assert encoded_tl.shape == encoded_views.shape
            encoded_input_mixed = encoded_tl + encoded_views
        else:
            raise NotImplementedError

        return encoded_input_mixed

    def extract_hidden_representation(self, timestep: torch.Tensor, unet_layer: torch.Tensor,
                                      frames_number: torch.Tensor) -> torch.Tensor:

        # for backcompatibility, this is how the old experiments were handled
        encoded_input = self.do_positional_encoding(timestep, unet_layer, frames_number)
        embedding = self.net(encoded_input)

        return embedding

    def apply_nested_dropout(self, embedding: torch.Tensor, truncation_idx: int = None) -> torch.Tensor:
        if self.training:
            if random.random() < self.nested_dropout_prob:
                dropout_idxs = torch.randint(low=0, high=embedding.shape[1], size=(embedding.shape[0],))
                for idx in torch.arange(embedding.shape[0]):
                    embedding[idx][dropout_idxs[idx]:] = 0
        if not self.training and truncation_idx is not None:
            for idx in torch.arange(embedding.shape[0]):
                embedding[idx][truncation_idx:] = 0
        return embedding

    def get_output(self, embedding: torch.Tensor, norm_log=False, human_action = None) -> torch.Tensor:
        embedding = self.output_layer(embedding)

        # split word embedding and output bypass (if enabled) and save to object
        if not self.output_bypass:
            output = MapperOutput(word_embedding=embedding,
                                  bypass_output=None,
                                  bypass_unconstrained=False,
                                  output_bypass_alpha=self.output_bypass_alpha)
        else:
            dim = embedding.shape[1] // 2
            output = MapperOutput(
                word_embedding=embedding[:, :dim],
                bypass_output=embedding[:, dim:],
                bypass_unconstrained=False,
                output_bypass_alpha=self.output_bypass_alpha)

        # apply norm scaling to the word embedding (if enabled)
        if self.norm_scale is not None:
            # important : the norm_scale should be calculated according to the CLIP output
            output.word_embedding = F.normalize(output.word_embedding, dim=-1) * self.norm_scale

        if human_action is not None:
            if self.norm_scale is not None:
                raise ValueError("norm_scale and human_action_norm cannot be used together")
            output.normalized_word_embedding = F.normalize(output.word_embedding, dim=-1) * human_action

        if norm_log:
            embedding_norms = torch.norm(output.word_embedding, dim=-1)
            average_embedding_norm = torch.mean(embedding_norms)
            self.embedding_norms_list.append(average_embedding_norm.item())

            # normalized_by_action_word_embedding_norms = torch.norm(output.normalized_word_embedding, dim=-1)
            # self.normalize_by_action_embedding_norms_list.append(average_embedding_norm.mean().item())

        # log this to wand b:
        # wandb.log({"average_embedding_norm": average_embedding_norm})

        return output

    def _set_positional_encoding(self):
        """ Set up the Fourier features positional encoding for t,l, and pose
        params.
        There are two ways to combine encodings: (i) adds the frequencies of the
        different params (like in the Fourier Fetaures paper). (ii) computes a
        fourier feature for each term independently, and then concats them, with
        an optional normalization.

        Warning: the `seed` argument to `FourierPositionalEncodingNDims` is important
        to reloading old
        """
        # set up the variance of the random fourier feature frequencies
        sigmas = [self.pe_sigmas.sigma_t, self.pe_sigmas.sigma_l]
        if self.embedding_type == "object":
            pass

        elif self.embedding_type == "time":

            # todo: that's sounds important:
            # warning: order of sigmas must match do_positional_encoding implementation
            # Handling individual time points for given frames
            if self.deg_freedom == "time":
                sigmas += [self.pe_sigmas.sigma_time]  # Encode each frame's time point

            # Optionally, handle intervals between frames if you decide to implement this
            # elif self.deg_freedom == "frame_interval":
            #     sigmas += [
            #         self.pe_sigmas.sigma_start_time,  # Start time of the interval
            #         self.pe_sigmas.sigma_end_time  # End time of the interval
            #     ]
            else:
                raise NotImplementedError()

        # lookup the positional encoding dimension
        self.pose_encode_dim = {'15': {'object': 64, 'time': 64}}[str(self.arch_time_net)][self.embedding_type]

        # generate the positional encoder
        if self.arch_time_net in (15, 18, 20, 22):
            self.positional_encoding_method = "add_freqs"
            self.input_dim = self.pose_encode_dim
            self.encoder = FourierPositionalEncodingNDims(dim=self.pose_encode_dim, sigmas=sigmas, seed=0)

        elif self.arch_time_net in (16, 17, 19, 21):
            self.positional_encoding_method = "concat_features"
            self.input_dim = self.pose_encode_dim * len(sigmas)
            self.normalize = {
                '16': False,
                '17': True,
                '19': False,
                '21': False
            }[str(self.arch_time_net)]
            self.encoder = [FourierPositionalEncodingNDims(dim=self.pose_encode_dim, sigmas=[sigma], seed=i,
                                                           normalize=self.normalize) for i, sigma in enumerate(sigmas)]

        else:
            raise NotImplementedError(
                "Need to define the pos encoding combination method for ",
                f"for arch_view_net=[self.arch_view_net] (in this function)")

    def do_positional_encoding(self, timestep, unet_layer, frames_number):
        """ new methods for getting positional encoding for self.arch_view>=14"""
        # put timestep and unet_layer in range [-1,1]
        timestep = timestep / 1000 * 2 - 1
        unet_layer = unet_layer / self.num_unet_layers * 2 - 1
        data = torch.stack((timestep, unet_layer), dim=1)

        # if it's a view-mapper, then add view data
        if self.embedding_type == "time":
            device, dtype = timestep.device, timestep.dtype
            # view_params = self.get_view_params_from_token(frames_number, device, dtype)
            time_params = self.get_time_params_from_frames_number(frames_number, device, dtype)

            if self.deg_freedom == "time":
                data = torch.cat((data, time_params['times'].unsqueeze(-1)), dim=1)
            # elif self.deg_freedom == "theta-phi":
            #     data = torch.cat((data, view_params['thetas'], view_params['phis']), dim=1)
            # elif self.deg_freedom == "dtu-12d":
            #     data = torch.cat((data, view_params['cam_matrix']), dim=1)
            else:
                raise NotImplementedError()

        # do pos encoding (explanation in docstring for _set_positional_encoding)
        if self.positional_encoding_method == "add_freqs":
            encoding = self.encoder(data)

        elif self.positional_encoding_method == "concat_features":
            encoding = torch.cat([self.encoder[i](data[:, i]) for i in range(data.shape[1])], dim=1)
        else:
            raise ValueError()

        return encoding

    def set_net_time(self, num_unet_layers: int, num_time_anchors: int, output_dim: int = 768):
        # Original-TI (also has arch-code-1)
        if self.original_ti or self.arch_time_net == 1:
            # baseline - TI baseline, which is one thing no matter what.
            assert self.original_ti_init_embed is not None
            if self.output_bypass:
                raise
            self.ti_embeddings = self.original_ti_init_embed.unsqueeze(0).repeat(len(self.placeholder_time_token_ids),
                                                                                 1)
            self.ti_embeddings = torch.nn.parameter.Parameter(self.ti_embeddings.clone(), requires_grad=True)
            # self.ti_embeddings.register_hook(lambda x: print(x))
            self.lookup_ti_embedding = dict(
                zip(self.placeholder_time_token_ids, torch.arange(len(self.placeholder_time_token_ids))))
            self.output_layer = nn.Identity()  # the MLP aready does projection


        # this architecture key 15 is the final model used in the paper
        elif self.arch_time_net in (15,):
            h_dim = 64
            self.net = nn.Sequential(nn.Linear(self.input_dim, h_dim),
                                     nn.LayerNorm(h_dim), nn.LeakyReLU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.LayerNorm(h_dim), nn.LeakyReLU())
            self.output_layer = nn.Sequential(nn.Linear(h_dim, output_dim))

        else:
            raise NotImplementedError()

