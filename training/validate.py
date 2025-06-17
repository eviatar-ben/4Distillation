import ipdb
from typing import List

import numpy as np
from requests.exceptions import ConnectionError
import torch
from PIL import Image
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import is_wandb_available
from tqdm import tqdm
from transformers import CLIPTokenizer

import matplotlib.pyplot as plt
from training.config import RunConfig
from models.neti_clip_text_encoder import NeTICLIPTextModel
from models.xti_attention_processor import XTIAttenProc
from prompt_manager import PromptManager
from sd_pipeline_call import sd_pipeline_call
from torchvision.utils import make_grid
from training import inference_dtu
from constants import TIME_TOKEN
from utils import vis_utils

if is_wandb_available():
    import wandb

from models import ip_adapter


class ValidationHandler:

    def __init__(self, cfg: RunConfig, weights_dtype: torch.dtype, max_rows: int = 14, relative_tokens: bool = False):

        self.cfg = cfg
        self.weight_dtype = weights_dtype
        self.max_rows = max_rows
        self.relative_tokens = relative_tokens
        # self.width, self.height = 512, 512
        self.width, self.height = 256, 256

    def basic_inference(self,
                        n_frames: int,
                        loaded_time_mapper,
                        conditioned_images_paths: List[Path],
                        conditioned_images: List[Image.Image],
                        accelerator: Accelerator,
                        tokenizer: CLIPTokenizer,
                        text_encoder: NeTICLIPTextModel,
                        ip_adapter: ip_adapter.IPAdapter,
                        unet: UNet2DConditionModel,
                        vae: AutoencoderKL,
                        projection_matrix_base: torch.Tensor = None,
                        projection_matrix_bypass: torch.Tensor = None):

        inference_dir = self.cfg.log.exp_dir / "inference_images"
        inference_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

        pipeline = self.load_stable_diffusion_model_and_set_mapper_and_projection_layers(accelerator, tokenizer,
                                                                                         text_encoder, ip_adapter,
                                                                                         unet, vae,
                                                                                         projection_matrix_base=projection_matrix_base,
                                                                                         projection_matrix_bypass=projection_matrix_bypass,
                                                                                         time_mapper=loaded_time_mapper)
        prompt_manager = PromptManager(tokenizer=pipeline.tokenizer,
                                       text_encoder=pipeline.text_encoder,
                                       timesteps=pipeline.scheduler.timesteps)
        diffused_image_idxs = np.linspace(-1, 1, n_frames)
        all_frames_for_current_conditioned_image = []
        for i, conditioned_image_path in tqdm(enumerate(conditioned_images_paths), total=len(conditioned_images_paths),
                                         desc="Generating videos"):
            conditioned_image = Image.open(conditioned_image_path)
            conditioned_image_embeds = pipeline.ip_adapter.get_image_embeds(conditioned_image)[0]
            all_frames_for_current_conditioned_image = []
            for j, diffused_image_idx in enumerate(diffused_image_idxs):
                diffused_image_idx = torch.tensor(diffused_image_idx).view(1, 1).to(pipeline.device)
                time_condition_clip_embeddings = self.compute_embeddings(prompt_manager=prompt_manager,
                                                                         relative_diffusion_frame_number=diffused_image_idx,
                                                                         projection_matrix_base=projection_matrix_base,
                                                                         projection_matrix_bypass=projection_matrix_bypass)
                generator = torch.Generator(device='cuda').manual_seed(0)
                generated_images = sd_pipeline_call(pipeline,
                                                    guidance_scale=5.0,
                                                    prompt_embeds=time_condition_clip_embeddings,
                                                    conditioned_image_embeds=conditioned_image_embeds,
                                                    generator=generator,
                                                    num_inference_steps=self.cfg.eval.num_denoising_steps,
                                                    num_images_per_prompt=1,
                                                    height=self.height,
                                                    width=self.width).images
                all_frames_for_current_conditioned_image.extend(generated_images)


                save_path = inference_dir / f"{Path(conditioned_image_path).parent.name}"
                save_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

                #create dir: conditioned_image_path.stem
                generated_images[0].save(inference_dir / f"{Path(conditioned_image_path).parent.name}/ {Path(conditioned_image_path).stem}_{j}.png")

                if j == 0:
                    conditioned_image.save(inference_dir / f"{Path(conditioned_image_path).parent.name}/ {Path(conditioned_image_path).stem}.png")


            all_frames_for_current_conditioned_image = Image.fromarray(
                np.concatenate(all_frames_for_current_conditioned_image, axis=1))

            all_frames_for_current_conditioned_image.save(inference_dir / f"inference_frames_for_video{i}.png")
            # # todo: Eviatar : those embeddings need to get positional encoding as wel (as in train)

        del pipeline
        torch.cuda.empty_cache()
        if text_encoder.text_model.embeddings.mapper_time is not None:
            text_encoder.text_model.embeddings.mapper_time.train()

        return all_frames_for_current_conditioned_image

    def infer(self,
              dataset,
              accelerator: Accelerator,
              tokenizer: CLIPTokenizer,
              text_encoder: NeTICLIPTextModel,
              ip_adapter: ip_adapter.IPAdapter,
              unet: UNet2DConditionModel,
              vae: AutoencoderKL,
              num_images_per_prompt: int,
              seeds: List[int],
              step: int,
              create_video=False,
              high_quality=False,
              prompts: List[str] = None,
              projection_matrix_base: torch.Tensor = None,
              projection_matrix_bypass: torch.Tensor = None
              ):
        """ Runs inference during our training scheme. """

        try:
            pipeline = self.load_stable_diffusion_model(accelerator, tokenizer, text_encoder, ip_adapter, unet, vae)
        except ConnectionError as e:
            try:
                sleep(60 * 5)
                pipeline = self.load_stable_diffusion_model(accelerator, tokenizer, text_encoder, ip_adapter, unet, vae)
            except ConnectionError as e:
                print("Connection error, resuming")
                print(e)
                return None

        prompt_manager = PromptManager(tokenizer=pipeline.tokenizer,
                                       text_encoder=pipeline.text_encoder,
                                       timesteps=pipeline.scheduler.timesteps)

        prompts = [-120, -100, -60, 0, 20, 60, 100, 140, 160]
        if create_video and high_quality:
            prompts = list(range(-len(dataset.image_paths), len(dataset.image_paths), 1))
        if create_video and not high_quality:
            prompts = list(range(-len(dataset.image_paths), len(dataset.image_paths), 5))
        joined_images = []
        images_for_seed_0 = []
        images_for_seed_1 = []
        for prompt in prompts:
            images = self.infer_on_prompt(pipeline=pipeline, prompt_manager=prompt_manager,
                                          diffused_image_idx=prompt, num_images_per_prompt=num_images_per_prompt,
                                          seeds=seeds,
                                          dataset=dataset,
                                          projection_matrix_base=projection_matrix_base,
                                          projection_matrix_bypass=projection_matrix_bypass)
            # fig, ax = plt.subplots(1, 20, figsize=(2, 40))

            images = vis_utils.overlay_text_on_images(images, str(prompt), self.cfg.log.exp_dir)

            prompt_image = Image.fromarray(np.concatenate(images, axis=1))

            images_for_seed_0.append(images[0])
            images_for_seed_1.append(images[1])

            joined_images.append(prompt_image)
        if create_video:
            # pop out from final_image every fourth image
            images_to_save_as_image = joined_images[::4]
        else:
            images_to_save_as_image = joined_images

        final_image = Image.fromarray(np.concatenate(images_to_save_as_image, axis=0))
        final_image.save(self.cfg.log.exp_dir / f"val-image-{step}.png")

        # create video and save:
        if create_video:
            vis_utils.create_video_from_images(joined_images, self.cfg.log.exp_dir / f"val-video-{step}.mp4")
        try:
            self.log_with_accelerator(accelerator, joined_images, step=step, prompts=prompts)
        except:
            pass
        del pipeline
        torch.cuda.empty_cache()
        if text_encoder.text_model.embeddings.mapper_time is not None:
            text_encoder.text_model.embeddings.mapper_time.train()
            # todo: Eviatar: projection layers should be eval() and train() after as-well

        if self.cfg.optim.seed is not None:
            set_seed(self.cfg.optim.seed)
        return final_image

    def infer_on_prompt(self,
                        pipeline: StableDiffusionPipeline,
                        prompt_manager: PromptManager,
                        diffused_image_idx,
                        seeds: List[int],
                        num_images_per_prompt: int = 1,
                        projection_matrix_base: torch.Tensor = None,
                        projection_matrix_bypass: torch.Tensor = None,
                        dataset=None) -> List[Image.Image]:

        conditioned_images, conditioned_image_idx, len_action_test_images = dataset.get_random_test_images()

        conditioned_image = conditioned_images[0]
        conditioned_image_embeds = pipeline.ip_adapter.get_image_embeds(conditioned_image)[0]
        if self.relative_tokens:
            # todo: Eviatar: the diffused image minus the IPAdapter image
            diffused_image_idx = diffused_image_idx - conditioned_image_idx[0]

        diffused_image_idx = torch.tensor(diffused_image_idx).view(1).to(pipeline.device)
        # normalize to [-1,1]
        diffused_image_idx = ((diffused_image_idx + len_action_test_images) / len_action_test_images - 1)

        time_condition_clip_embeddings = self.compute_embeddings(prompt_manager=prompt_manager,
                                                                 relative_diffusion_frame_number=diffused_image_idx,
                                                                 projection_matrix_base=projection_matrix_base,
                                                                 projection_matrix_bypass=projection_matrix_bypass)
        # todo: Eviatar : those embeddings need to get positional encoding as wel (as in train)

        all_images = []
        for test_scene_idx in tqdm(range(num_images_per_prompt)):
            generator = torch.Generator(device='cuda').manual_seed(seeds[test_scene_idx])
            generated_images = sd_pipeline_call(pipeline,
                                                guidance_scale=5.0,
                                                prompt_embeds=time_condition_clip_embeddings,
                                                conditioned_image_embeds=conditioned_image_embeds,
                                                generator=generator,
                                                num_inference_steps=self.cfg.eval.num_denoising_steps,
                                                num_images_per_prompt=1,
                                                height=self.height,
                                                width=self.width).images
            all_images.extend(generated_images)
        return all_images

    @staticmethod
    def compute_embeddings(prompt_manager: PromptManager, relative_diffusion_frame_number,
                           projection_matrix_base, projection_matrix_bypass) -> torch.Tensor:
        with torch.autocast("cuda"):
            with torch.no_grad():
                prompt_embeds = prompt_manager.embed_prompt(relative_diffusion_frame_number,
                                                            projection_matrix_base=projection_matrix_base,
                                                            projection_matrix_bypass=projection_matrix_bypass)
        return prompt_embeds

    def load_stable_diffusion_model(self, accelerator: Accelerator,
                                    tokenizer: CLIPTokenizer,
                                    text_encoder: NeTICLIPTextModel,
                                    ip_adapter: ip_adapter.IPAdapter,
                                    unet: UNet2DConditionModel,
                                    vae: AutoencoderKL) -> StableDiffusionPipeline:
        """ Loads SD model given the current text encoder and our mapper. """
        pipeline = StableDiffusionPipeline.from_pretrained(self.cfg.model.pretrained_model_name_or_path,
                                                           text_encoder=accelerator.unwrap_model(text_encoder),
                                                           ip_adapter=ip_adapter,
                                                           tokenizer=tokenizer, unet=unet,
                                                           vae=vae,
                                                           torch_dtype=self.weight_dtype)
        pipeline.ip_adapter = ip_adapter
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.scheduler.set_timesteps(self.cfg.eval.num_denoising_steps, device=pipeline.device)
        # pipeline.unet.set_attn_processor(XTIAttenProc())
        if text_encoder.text_model.embeddings.mapper_time is not None:
            text_encoder.text_model.embeddings.mapper_time.eval()
        # todo: Eviatar  add linear_projections.eval()
        # and train() after inference
        return pipeline

    def log_with_accelerator(self, accelerator: Accelerator, images: List[Image.Image], step: int, prompts: List[str]):
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {"validation": [wandb.Image(image, caption=f"{i}: {prompts[i]}") for i, image in
                                    enumerate(images)]})

    def load_stable_diffusion_model_and_set_mapper_and_projection_layers(self,
                                                                         accelerator: Accelerator,
                                                                         tokenizer: CLIPTokenizer,
                                                                         text_encoder: NeTICLIPTextModel,
                                                                         ip_adapter: ip_adapter.IPAdapter,
                                                                         unet: UNet2DConditionModel,
                                                                         vae: AutoencoderKL,
                                                                         time_mapper,
                                                                         projection_matrix_base,
                                                                         projection_matrix_bypass
                                                                         ) -> StableDiffusionPipeline:

        """ Loads SD model given the current text encoder and our mapper. """
        pipeline = StableDiffusionPipeline.from_pretrained(self.cfg.model.pretrained_model_name_or_path,
                                                           text_encoder=accelerator.unwrap_model(text_encoder),
                                                           ip_adapter=ip_adapter,
                                                           tokenizer=tokenizer, unet=unet,
                                                           vae=vae,
                                                           torch_dtype=self.weight_dtype)
        pipeline.ip_adapter = ip_adapter
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.scheduler.set_timesteps(self.cfg.eval.num_denoising_steps, device=pipeline.device)
        # pipeline.unet.set_attn_processor(XTIAttenProc())
        assert text_encoder.text_model.embeddings.mapper_time is None
        text_encoder.text_model.embeddings.set_mapper(time_mapper)
        assert text_encoder.text_model.embeddings.mapper_time is not None
        text_encoder.text_model.embeddings.mapper_time.eval()

        assert text_encoder.text_model.projection_matrix_base is None
        assert text_encoder.text_model.projection_matrix_bypass is None
        text_encoder.text_model.set_projections(projection_matrix_base, projection_matrix_bypass)
        assert text_encoder.text_model.projection_matrix_base is not None
        assert text_encoder.text_model.projection_matrix_bypass is not None

        return pipeline
