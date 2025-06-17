import ipdb
from typing import Optional, Dict, Tuple, List, Union

import diffusers
import itertools
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import CLIPTokenizer
from pathlib import Path
import pyrallis
from PIL import Image

from checkpoint_handler import CheckpointHandler
from constants import UNET_LAYERS
from models.neti_clip_text_encoder import NeTICLIPTextModel
from models.neti_mapper import NeTIMapper
from models.xti_attention_processor import XTIAttenProc
from training.config import RunConfig
from training.dataset import TextualInversionDataset
from training.logger import CoachLogger
from training.validate import ValidationHandler
from utils.types import NeTIBatch
from utils.utils import parameters_checksum
from utils import vis_utils
import torchvision.transforms as T

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor
from ip_adapter.resampler import Resampler
from ip_adapter.utils import is_torch2_available
import wandb

MAX_FRAMES = 793

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

from constants import PROJECTION_DIMENSION


class Coach:

    def __init__(self, cfg: RunConfig):
        self._should_create_video_state = 0
        self.cfg = cfg
        self.logger = CoachLogger(cfg=cfg)

        # Initialize some basic stuff
        self.accelerator = self._init_accelerator()
        if self.cfg.optim.seed is not None:
            set_seed(self.cfg.optim.seed)
        if self.cfg.optim.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True


        # todo: Eviatar : image encoder might be unnecessary
        self.tokenizer, self.noise_scheduler, self.text_encoder, self.image_encoder, self.vae, self.unet = self._init_sd_models()
        self.ip_adapter = self._init_ip_adapter()

        # Add projection matrix
        self.projection_matrix_base = torch.nn.Parameter(torch.randn(PROJECTION_DIMENSION, 768))
        self.projection_matrix_bypass = torch.nn.Parameter(torch.randn(PROJECTION_DIMENSION, 768))
        # Add positional embedding for IP tokens
        self.positional_embeddings_for_ip_tokens = torch.nn.Parameter(torch.randn(MAX_FRAMES, self.ip_adapter.num_tokens, 768))

        self.text_encoder.text_model.set_projections(self.projection_matrix_base, self.projection_matrix_bypass)
        self.ip_adapter.set_positional_embeddings(self.positional_embeddings_for_ip_tokens)

        # Initialize dataset and dataloader
        self.train_dataset = self._init_dataset()
        self.train_dataloader = self._init_dataloader(dataset=self.train_dataset)
        self.train_data_subsets = self.cfg.data.train_data_subsets

        # ip-adapter:
        # todo : Eviatar :  IPAdapter
        if self.cfg.optim.allow_tf32:
            self.placeholder_tokens = self.train_dataset.placeholder_tokens

        # Initilize neti mapping objects, and finish preparing all the models
        neti_mapper_time, self.loaded_iteration = self._init_neti_mapper()  # todo Eviatar: IPAdapter
        self.time_mapper = neti_mapper_time
        self.text_encoder.text_model.embeddings.set_mapper(neti_mapper_time)
        self._freeze_all_modules()

        # Initialize optimizer and scheduler

        self.optimizer = self._init_optimizer()
        self.lr_scheduler = self._init_scheduler(optimizer=self.optimizer)

        # Prepare everything with accelerator
        self.ip_adapter, self.text_encoder, self.image_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler = \
            self.accelerator.prepare(self.ip_adapter, self.text_encoder, self.image_encoder,
                                     self.optimizer, self.train_dataloader, self.lr_scheduler)

        # Reconfigure some parameters that we'll need for training
        self.weight_dtype = self._get_weight_dtype()
        self._set_model_weight_dtypes(weight_dtype=self.weight_dtype)
        self._init_trackers()  # todo:

        # self._init_clip_processed_images()  # todo :Eviatar - IPAdapter

        self.validator = ValidationHandler(cfg=self.cfg,
                                           weights_dtype=self.weight_dtype,
                                           relative_tokens=self.cfg.relative_tokens)

        self.checkpoint_handler = CheckpointHandler(cfg=self.cfg,
                                                    save_root=self.cfg.log.exp_dir)

        clip_embeddings_image_encoder_path = self.cfg.model.image_encoder_path
        clip_embeddings_weight_dtype = self.weight_dtype
        self.clip_embeddings_ip_adapter_clip_image_processor = CLIPImageProcessor(resample=Image.Resampling.LANCZOS)
        self.clip_embeddings_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            clip_embeddings_image_encoder_path).to(torch.device('cuda'), dtype=clip_embeddings_weight_dtype)

        self.device = torch.device('cuda')

    def train(self):
        total_batch_size = self.cfg.optim.train_batch_size * self.accelerator.num_processes * \
                           self.cfg.optim.gradient_accumulation_steps
        self.logger.log_start_of_training(total_batch_size=total_batch_size, num_samples=len(self.train_dataset))

        global_step = self._set_global_step()
        progress_bar = tqdm(range(global_step, self.cfg.optim.max_train_steps),
                            disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        while global_step < self.cfg.optim.max_train_steps:

            self.text_encoder.train()
            for step, batch in enumerate(self.train_dataloader):
                if 'UCF' in str(self.cfg.data.train_data_dir):
                    self.train_dataset.reset_sampled_action_g_c()

                with self.accelerator.accumulate(self.text_encoder):
                    ## following commented code is to check that weights are updating
                    # cksm_object = parameters_checksum(list(self.text_encoder.text_model.embeddings.mapper_object_lookup.values())[0])
                    # cksm_view = parameters_checksum(self.time_mapper)
                    # print(f"checksums object {None} | view {cksm_view}")

                    # Convert images to latent space
                    latent_batch = batch["pixel_values"].to(dtype=self.weight_dtype)
                    latents = self.vae.encode(latent_batch).latent_dist.sample().detach()
                    latents = latents * self.vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(bsz,),
                                              device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    _hs = self.get_conditioning(timesteps=timesteps,
                                                normalized_frames_gap=batch['normalized_frames_gap'],
                                                original_ti=self.cfg.model.original_ti,
                                                device=latents.device,
                                                human_action=None)  # todo : for now normalize with super categpry to

                    # todo: IPAdapter: ImageClip
                    # todo: Eviatar :per batch all the examples' images for the IPAdapter are from the same subset
                    # current_data_subset_idx = batch["current_data_subset_idx"]
                    # image_embeds_idxs = batch["image_embeds_idxs_for_ipadapter"]
                    # image_embeds = self.get_image_embeds(current_data_subset_idx=current_data_subset_idx,
                    #                                      image_embeds_idxs=image_embeds_idxs)  # batch["image_idx"])
                    conditioned_ipadapter_frame_clip_paths = batch['conditioned_ipadapter_frame_clip_paths']

                    image_embeds = self.retrieve_image_embeds(conditioned_ipadapter_frame_clip_paths)
                    model_pred = self.ip_adapter(noisy_latents, timesteps, _hs,
                                                 image_embeds=image_embeds,
                                                 conditioned_frames_idxs=batch["conditioned_ipadapter_frame"])

                    # Predict the noise residual
                    # model_pred = self.unet(noisy_latents, timesteps, _hs).sample

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    self.accelerator.backward(loss)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    self.logger.update_step(step=global_step)
                    if self._should_save(global_step=global_step):
                        self.checkpoint_handler.save_model(text_encoder=self.text_encoder,
                                                           projection_matrix_base=self.projection_matrix_base,
                                                           projection_matrix_bypass=self.projection_matrix_bypass,
                                                           mapper_save_name=f"mapper-steps-{global_step}.pt",
                                                           projection_save_name=f"projection-steps-{global_step}.pt")
                    if self._should_eval(global_step=global_step):
                        create_video, high_quality = self._should_create_video()
                        self.plot_norm(self.time_mapper)
                        self.train_dataset.save_histogram(exp_dir=self.cfg.log.exp_dir)

                        self.validator.infer(create_video=create_video,
                                             high_quality=high_quality,
                                             dataset=self.train_dataset,
                                             accelerator=self.accelerator,
                                             tokenizer=self.tokenizer,
                                             text_encoder=self.text_encoder,
                                             ip_adapter=self.ip_adapter,
                                             unet=self.unet,
                                             vae=self.vae,
                                             num_images_per_prompt=self.cfg.eval.num_validation_images,
                                             seeds=self.cfg.eval.validation_seeds,
                                             step=global_step,
                                             projection_matrix_base=self.projection_matrix_base,
                                             projection_matrix_bypass=self.projection_matrix_bypass)

                logs = {"total_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.cfg.optim.max_train_steps:
                    break

        # Save the final model
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.checkpoint_handler.save_model(text_encoder=self.text_encoder, accelerator=self.accelerator,
                                               embeds_save_name=f"learned_embeds-final.bin",
                                               mapper_save_name=f"mapper-final.pt")
        self.accelerator.end_training()

    def get_conditioning(self, timesteps: torch.Tensor,
                         normalized_frames_gap,
                         device: torch.device,
                         human_action: torch.Tensor,
                         original_ti: bool = False) -> Union[Dict, torch.Tensor]:
        """ Compute the text conditioning for the current batch of images using our text encoder over-ride.
        If original_ti, then just return the last layer directly
        """
        _hs = {"this_idx": 0}
        for layer_idx, unet_layer in enumerate(UNET_LAYERS):
            neti_batch = NeTIBatch(timesteps=timesteps,
                                   normalized_frames_gap=normalized_frames_gap,
                                   unet_layers=torch.tensor(layer_idx, device=device).repeat(timesteps.shape[0]),
                                   human_action=human_action, )
            v_base, v_bypass = self.text_encoder(batch=neti_batch,
                                                 projection_matrix_base=self.projection_matrix_base,
                                                 projection_matrix_bypass=self.projection_matrix_bypass)
            v_base = v_base.to(dtype=self.weight_dtype)

            _hs[f"CONTEXT_TENSOR_{layer_idx}"] = v_base

            v_bypass = None  # ignore v_bypass vector
            if v_bypass is not None:
                v_bypass = v_bypass.to(dtype=self.weight_dtype)
                _hs[f"CONTEXT_TENSOR_BYPASS_{layer_idx}"] = v_bypass

            if "shape" not in _hs:
                _hs["shape"] = v_base.shape
            else:
                assert _hs["shape"] == v_base.shape

            # if running original ti, only run this
            if original_ti:
                return v_base

        return _hs

    def _set_global_step(self) -> int:
        global_step = 0
        if self.loaded_iteration is not None:
            global_step = self.loaded_iteration
        self.logger.update_step(step=global_step)
        return global_step

    def save_dataset_images(self) -> None:
        n_max = 100
        fnames = self.train_dataset.image_paths_flattened
        if len(fnames) > n_max:
            fnames = fnames[:n_max]
            save_fname = self.cfg.log.exp_dir / 'dataset_first_100.png'
        else:
            save_fname = self.cfg.log.exp_dir / 'dataset.png'

        if self.cfg.learnable_mode == 3:
            # for video in self.train_dataset.image_paths_flattened.values():
            #     images = [Image.open(f) for f in fnames]
            #     image_in_columns = vis_utils.get_image_column(images)
            #     image_in_columns = vis_utils.downsample_image(image_in_columns, 0.2)
            # Initialize a list to store the columns
            columns = []

            # Loop to generate and store each image column
            for key, video in self.train_dataset.image_paths_flattened.items():
                images = [Image.open(f) for f in video]
                image_in_columns = vis_utils.get_image_column(images)
                image_in_columns = vis_utils.downsample_image(image_in_columns, 0.2)
                columns.append(image_in_columns)
                if key == "test":
                    # for test images add some caption "test" and red border
                    columns[-1] = vis_utils.add_red_border_and_caption(columns[-1], "test")

            # add

            # Determine the total width and maximum height of the final concatenated image
            total_width = sum(column.width for column in columns)
            max_height = max(column.height for column in columns)

            # Create a blank canvas with the computed dimensions
            final_image = Image.new("RGB", (total_width, max_height))

            # Paste each column onto the final image
            x_offset = 0
            for column in columns:
                final_image.paste(column, (x_offset, 0))
                x_offset += column.width

            # Save or display the final concatenated image
            final_image.save(self.cfg.log.exp_dir / 'flattened_dataset.png')
            final_image.show()
        else:
            images = [Image.open(f) for f in fnames]
            grid = vis_utils.get_image_grid(images)
            grid = vis_utils.downsample_image(grid, 0.2)
            # grid.save(save_fname)

            image_in_columns = vis_utils.get_image_column(images)
            image_in_columns = vis_utils.downsample_image(image_in_columns, 0.2)
            image_in_columns.save(self.cfg.log.exp_dir / 'dataset_in_columns.png')

    def _init_neti_mapper(self) -> Tuple[NeTIMapper, Optional[int]]:
        neti_mapper_time = None
        loaded_iteration = None
        # this loading func is from prior codebase.
        if self.cfg.model.mapper_checkpoint_path:
            raise NotImplementedError("Check this implementation is right")
            # This isn't 100% resuming training since we don't save the optimizer, but it's close enough
            _, neti_mapper = CheckpointHandler.load_mapper(self.cfg.model.mapper_checkpoint_path)
            loaded_iteration = int(self.cfg.model.mapper_checkpoint_path.stem.split("-")[-1])

        if self.cfg.learnable_mode in (1, 2, 3):

            neti_mapper_time = NeTIMapper(embedding_type="time",
                                          output_dim=self.cfg.model.word_embedding_dim,
                                          arch_mlp_hidden_dims=self.cfg.model.arch_mlp_hidden_dims,
                                          use_nested_dropout=self.cfg.model.use_nested_dropout,
                                          nested_dropout_prob=self.cfg.model.nested_dropout_prob,
                                          norm_scale=self.cfg.model.target_norm_time,
                                          use_positional_encoding=self.cfg.model.use_positional_encoding_time,
                                          num_pe_time_anchors=self.cfg.model.num_pe_time_anchors,
                                          pe_sigmas=self.cfg.model.pe_sigmas,
                                          arch_time_net=self.cfg.model.arch_time_net,
                                          arch_time_mix_streams=self.cfg.model.arch_time_mix_streams,
                                          arch_time_disable_tl=self.cfg.model.arch_time_disable_tl,
                                          output_bypass=self.cfg.model.output_bypass_time,
                                          original_ti=self.cfg.model.original_ti,
                                          output_bypass_alpha=self.cfg.model.output_bypass_alpha_time,
                                          bypass_unconstrained=self.cfg.model.bypass_unconstrained_time)

        elif self.cfg.learnable_mode in (4, 5):
            # todo: implement this for time and not for view
            # load pretrained view mapperi
            cfg_pretrained_view_mapper, neti_mapper_time = CheckpointHandler.load_mapper(
                self.cfg.model.pretrained_view_mapper,
                "view",
                placeholder_time_tokens=self.placeholder_time_tokens,
                placeholder_time_token_ids=self.placeholder_time_token_ids)
            # save the pretrained config
            fname_cfg_pretrain = self.cfg.log.exp_dir / "config_view_pretrained.yaml"
            with (fname_cfg_pretrain).open('w') as f:
                pyrallis.dump(cfg_pretrained_view_mapper, f)

        return neti_mapper_time, loaded_iteration

    def _init_sd_models(self):
        tokenizer = self._init_tokenizer()
        noise_scheduler = self._init_noise_scheduler()
        text_encoder = self._init_text_encoder()
        image_encoder = self._init_image_encoder()  # todo: Eviatar: later on should be changed to video_encoder
        vae = self._init_vae()
        unet = self._init_unet()
        return tokenizer, noise_scheduler, text_encoder, image_encoder, vae, unet

    def _init_tokenizer(self) -> CLIPTokenizer:
        tokenizer = CLIPTokenizer.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")
        return tokenizer

    def _init_noise_scheduler(self) -> DDPMScheduler:
        noise_scheduler = DDPMScheduler.from_pretrained(self.cfg.model.pretrained_model_name_or_path,
                                                        subfolder="scheduler")
        return noise_scheduler

    def _init_text_encoder(self) -> NeTICLIPTextModel:
        text_encoder = NeTICLIPTextModel.from_pretrained(self.cfg.model.pretrained_model_name_or_path,
                                                         subfolder="text_encoder", revision=self.cfg.model.revision, )
        return text_encoder

    def _init_image_encoder(self):
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.cfg.model.image_encoder_path)
        return image_encoder

    def _init_vae(self) -> AutoencoderKL:
        vae = AutoencoderKL.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="vae",
                                            revision=self.cfg.model.revision)
        return vae

    def _init_unet(self) -> UNet2DConditionModel:
        unet = UNet2DConditionModel.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="unet",
                                                    revision=self.cfg.model.revision)
        return unet

    def _freeze_all_modules(self):

        # Freeze vae, unet, text model
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        # Freeze all parameters except for the mapper in text encoder
        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        # Freeze image encoder
        self.image_encoder.requires_grad_(False)

        # Freeze IP-Adapter components
        self.ip_adapter.image_proj_model.requires_grad_(False)
        self.ip_adapter.adapter_modules.requires_grad_(False)
        self.ip_adapter.image_encoder.requires_grad_(False)

        # Train the mapper
        def enable_mapper_training(mapper):
            mapper.requires_grad_(True)
            mapper.train()

        enable_mapper_training(self.text_encoder.text_model.embeddings.mapper_time)

        # train the projection matrices
        self.projection_matrix_base.requires_grad_(True)
        self.projection_matrix_bypass.requires_grad_(True)
        #train the positional embedding for IP tokens
        self.positional_embeddings_for_ip_tokens.requires_grad_(True)



        if self.cfg.optim.gradient_checkpointing:
            # Keep unet in train mode if we are using gradient checkpointing to save memory.
            # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
            self.unet.train()
            self.text_encoder.gradient_checkpointing_enable()
            self.unet.enable_gradient_checkpointing()

    def _set_attn_processor(self):
        # self.unet.set_attn_processor(XTIAttenProc())  # todo: Eviatar : IPAdapter  using set_attn_processor as well
        pass

    def _init_dataset(self) -> TextualInversionDataset:
        dataset = TextualInversionDataset(uniform_sampling=self.cfg.data.uniform_sampling,
                                          learnable_mode=self.cfg.learnable_mode,
                                          relative_tokens=self.cfg.relative_tokens,
                                          data_root=self.cfg.data.train_data_dir,
                                          train_data_subsets=self.cfg.data.train_data_subsets,
                                          test_data_subsets=self.cfg.data.test_data_subsets,
                                          tokenizer=self.tokenizer,
                                          text_encoder=self.text_encoder,
                                          size=self.cfg.data.resolution,
                                          repeats=self.cfg.data.repeats,
                                          center_crop=self.cfg.data.center_crop,
                                          augmentation_key=self.cfg.data.augmentation_key,
                                          ip_adapter_subset_size=self.cfg.data.ip_adapter_subset_size,
                                          set="train")
        return dataset

    def _init_dataloader(self, dataset: Dataset) -> torch.utils.data.DataLoader:

        def custom_collate_fn(batch):
            collated_batch = {}

            for key in batch[0]:  # Iterate over keys in the example dictionary
                values = [item[key] for item in batch]  # Collect values for this key

                if isinstance(values[0], torch.Tensor):
                    collated_batch[key] = torch.stack(values)  # Stack tensors normally
                elif isinstance(values[0], list):
                    if isinstance(values[0][0], str):
                        # Handle lists of strings (like file paths) correctly
                        collated_batch[key] = [v for v in values]
                    else:
                        collated_batch[key] = list(values)
                else:
                    collated_batch[key] = values  # Default for other types

            return collated_batch

        # Correctly pass the custom_collate_fn without lambda
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg.optim.train_batch_size, shuffle=True,
                                                 collate_fn=custom_collate_fn,  # Directly pass the custom_collate_fn
                                                 num_workers=self.cfg.data.dataloader_num_workers)
        return dataloader

    def _init_optimizer(self) -> torch.optim.Optimizer:
        if self.cfg.optim.scale_lr:
            self.cfg.optim.learning_rate = (self.cfg.optim.learning_rate * self.cfg.optim.gradient_accumulation_steps
                                            * self.cfg.optim.train_batch_size * self.accelerator.num_processes)

        ## choose the optimizable params depending on the learning mode
        learnable_params = [
            self.text_encoder.text_model.embeddings.mapper_time.parameters(),
            # Add projection matrices to learnable parameters
            [self.projection_matrix_base, self.projection_matrix_bypass],
            # Add positional embeddings to learnable params
            [self.positional_embeddings_for_ip_tokens],
        ]
        learnable_params_ = itertools.chain.from_iterable(learnable_params)
        optimizer = torch.optim.AdamW(learnable_params_,  # neti-mappers only
                                      lr=self.cfg.optim.learning_rate,
                                      betas=(self.cfg.optim.adam_beta1, self.cfg.optim.adam_beta2),
                                      weight_decay=self.cfg.optim.adam_weight_decay,
                                      eps=self.cfg.optim.adam_epsilon, )
        return optimizer

    def _init_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler:
        lr_scheduler = get_scheduler(self.cfg.optim.lr_scheduler, optimizer=optimizer,
                                     num_warmup_steps=self.cfg.optim.lr_warmup_steps *
                                                      self.cfg.optim.gradient_accumulation_steps,
                                     num_training_steps=self.cfg.optim.max_train_steps *
                                                        self.cfg.optim.gradient_accumulation_steps, )
        return lr_scheduler

    def _init_accelerator(self) -> Accelerator:
        accelerator_project_config = ProjectConfiguration(total_limit=self.cfg.log.checkpoints_total_limit)
        accelerator = Accelerator(gradient_accumulation_steps=self.cfg.optim.gradient_accumulation_steps,
                                  mixed_precision=self.cfg.optim.mixed_precision,
                                  log_with=self.cfg.log.report_to,
                                  # logging_dir=self.cfg.log.logging_dir, #todo: ive changed this from logging_dir to project_dir
                                  project_dir=self.cfg.log.logging_dir,
                                  project_config=accelerator_project_config, )
        self.logger.log_message(accelerator.state)
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        return accelerator

    def _set_model_weight_dtypes(self, weight_dtype: torch.dtype):
        self.unet.to(self.accelerator.device, dtype=weight_dtype)
        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        # todo: EVIATAR : IPAdapter - why the encoders are not set to the same device and dtype?
        # self.text_encoder.to(self.accelerator.device)
        # self.image_encoder.to(self.accelerator.device)

    def _get_weight_dtype(self) -> torch.dtype:
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        return weight_dtype

    def _init_trackers(self):
        config = pyrallis.encode(self.cfg)

        # tensorboard only accepts dicts with entries having (int, float, str, bool, or torch.Tensor)
        config_tensorboard = {**config['log'], **config['model']}
        if config_tensorboard is not None and 'pe_sigmas' in config_tensorboard.keys(
        ):
            del config_tensorboard['pe_sigmas']  # (doesn't like dictionaries)
            if 'pe_sigmas_view' in config_tensorboard.keys():
                del config_tensorboard[
                    'pe_sigmas_view']  # (doesn't like dictionaries)

        # give wandb the full logging dictionary bc it knows how to parse it.
        init_kwargs = {
            'wandb': {
                'config': config,
                'name': config['log']['exp_name'],
            },
        }

        # init trackers
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("view-neti",
                                           config=config_tensorboard,
                                           init_kwargs=init_kwargs)

    def _should_save(self, global_step: int) -> bool:
        return global_step % self.cfg.log.save_steps == 0

    def _should_eval(self, global_step: int) -> bool:
        return global_step % self.cfg.eval.validation_steps == 0

    def _init_clip_processed_images(self):
        # todo: IPAdapter
        # todo: Eviatar we are assuming ordered frames it is critic to double check that's this method keep them sorted
        images = {}
        processed_images_for_clip = {}
        subset_clip_embeds_images = {}
        for action_name, action in self.train_dataset.map_action_2_frame.items():
            for g in action.values():
                for c in g.values():
                    for frame_number, frame_path in c.items():
                        images[frame_number] = Image.open(frame_path)
                        processed_images_for_clip[frame_number] = self.ip_adapter.clip_image_processor(
                            images=images[frame_number], return_tensors="pt").pixel_values
                        with torch.no_grad():
                            cur_subset_clip_embeds_images = \
                                self.image_encoder(
                                    processed_images_for_clip[frame_number].to(self.accelerator.device,
                                                                               dtype=self.weight_dtype),
                                    output_hidden_states=True).hidden_states[-2]
                        subset_clip_embeds_images[frame_number] = cur_subset_clip_embeds_images

        self.ip_adapter.set_images(images, processed_images_for_clip, subset_clip_embeds_images)
        return

    def _init_ip_adapter(self):
        from models import ip_adapter

        num_queries = self.cfg.model.num_image_tokens
        # num_queries = len(self.clip_images) # look at config.py num_image_tokens

        # ip-adapter-plus
        image_proj_model = Resampler(dim=self.unet.config.cross_attention_dim,
                                     # todo: Eviatar: whats the implication on the Resampler if VideoClip is added instead of IMAGEClip
                                     depth=4,
                                     dim_head=64,
                                     heads=12,
                                     num_queries=num_queries,
                                     embedding_dim=self.image_encoder.config.hidden_size,
                                     output_dim=self.unet.config.cross_attention_dim,
                                     ff_mult=4)
        # init adapter modules
        attn_procs = {}
        unet_sd = self.unet.state_dict()
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {"to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                           "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"], }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                   scale=self.cfg.model.ip_hidden_states_scale, num_tokens=num_queries)
                attn_procs[name].load_state_dict(weights)

        self.unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())
        ip_adapter = ip_adapter.IPAdapter(self.unet, image_proj_model, adapter_modules,
                                          image_encoder_path=self.cfg.model.image_encoder_path,
                                          ip_adapter_subset_size=self.cfg.data.ip_adapter_subset_size,
                                          ckpt_path=self.cfg.model.ip_adapter_path,
                                          device=self.accelerator.device,
                                          dtype=self._get_weight_dtype())
        return ip_adapter

    def clip_encoder(self, conditioned_ipadapter_frame_clip_path):
        """
        Placeholder for the actual CLIP encoding function.
        This function should take the path of an image and return its embedding as a numpy array.
        """
        image = Image.open(conditioned_ipadapter_frame_clip_path)
        processed_frame_for_clip = self.clip_embeddings_ip_adapter_clip_image_processor(images=image,
                                                                                        return_tensors="pt").pixel_values

        with torch.no_grad():
            clip_embeds_frame = \
                self.clip_embeddings_image_encoder(processed_frame_for_clip.to(self.device, dtype=self.weight_dtype),
                                                   output_hidden_states=True).hidden_states[-2]
        return clip_embeds_frame

    def retrieve_image_embeds(self, conditioned_ipadapter_frame_clip_paths):
        import numpy as np
        import torch
        from pathlib import Path

        all_conditioned_ipadapter_frame_clips = []

        for batch_idx, batch_clip_paths in enumerate(conditioned_ipadapter_frame_clip_paths):
            # This will hold the clips for the current batch item (example)
            conditioned_ipadapter_frame_clips = []

            for path in batch_clip_paths:
                # Transform the clip path to frame path
                clip_path = Path(path)
                if 'UCF' in str(self.cfg.data.train_data_dir):
                    frame_path = str(clip_path).replace('/UCF-101-CLIP/', '/UCF-101-Frames/').replace('.npy', '.jpg')
                elif 'dnerf' in str(self.cfg.data.train_data_dir):
                    frame_path = str(clip_path).replace('/dnerf-CLIP/', '/dnerf/').replace('.npy', '.png')
                conditioned_ipadapter_frame_image_path = Path(frame_path)

                try:
                    # Attempt to load the clip
                    conditioned_ipadapter_frame_clip = np.load(clip_path)
                    conditioned_ipadapter_frame_clips.append(conditioned_ipadapter_frame_clip)
                except Exception as e:
                    # print error:
                    print(f"Error loading the clip from {clip_path}: {e}")
                    # Log the error and process the clip
                    print(f"Generating the clip from {conditioned_ipadapter_frame_image_path}: {e}")
                    conditioned_ipadapter_frame_clip = self.clip_encoder(conditioned_ipadapter_frame_image_path)

                    # Ensure the parent directory exists
                    clip_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save the tensor to a NumPy file (convert to CPU first)
                    np.save(clip_path, conditioned_ipadapter_frame_clip.cpu().numpy())

                    conditioned_ipadapter_frame_clips.append(conditioned_ipadapter_frame_clip.cpu().numpy())

            # After processing each batch, append the result (clip list) to the final list
            all_conditioned_ipadapter_frame_clips.append(
                torch.tensor(np.stack(conditioned_ipadapter_frame_clips), device='cuda'))

        # Return the list of batches (each batch contains a tensor of clips)
        # Removes the singleton dimension at index 2
        return torch.stack(all_conditioned_ipadapter_frame_clips, dim=0).squeeze(dim=2)

    def plot_norm(self, time_mapper):
        import matplotlib.pyplot as plt

        # Get the norms of the time mapper
        norms = time_mapper.embedding_norms_list

        # Ensure the plot is cleared before drawing
        plt.clf()  # Clears the current figure

        # Plot a single black line
        plt.plot(norms, label='Norm', color='black')

        plt.xlabel('Training Step')
        plt.ylabel('Norm')
        plt.title('Word Embedding Norm')

        # Save and close the figure
        plt.savefig(self.cfg.log.exp_dir / 'time_mapper_norms.png')
        plt.close()


    def _should_create_video(self):
        """
        return
        """
        self._should_create_video_state += 1
        if self._should_create_video_state == 10:
            self._should_create_video_state = 0
            return True, True
        if self._should_create_video_state == 4:
            return True, False
        return False, False
