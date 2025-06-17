import torch
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image

if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor, \
        CNAttnProcessor2_0 as CNAttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor, CNAttnProcessor


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""

    def __init__(self, unet, image_proj_model, adapter_modules, image_encoder_path,
                 ip_adapter_subset_size,
                 resample=Image.Resampling.LANCZOS, ckpt_path=None, device="cuda", dtype=torch.float16):
        super().__init__()
        self.ip_adapter_subset_size = ip_adapter_subset_size
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.device = device
        self.is_plus = None
        self.images = None
        self.test_images = None
        self.clip_processed_images = None
        self.clip_images = None
        self.clip_embeds_images = None
        self.dtype = dtype  # todo: EVIATAR - im not sure whats that should be float16 or float32 - to compatible with time_neti Ive set it to 32

        # set image encoder #todo :those can be the same as the coach's models (to propegate them as parameters to the init method)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(self.device,
                                                                                                  dtype=self.dtype)
        self.clip_image_processor = CLIPImageProcessor(resample=resample)

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)
        self.num_tokens = 16 if self.is_plus else 4

        self._positional_embeddings = None

    def forward(self, noisy_latents, timesteps, encoder_hidden_states,
                image_embeds, conditioned_frames_idxs=None, cross_attention_kwargs=None):

        # reshape the ip_tokens to (batch_size*num_images, num_tokens, embedding_size)
        reshaped_image_embeds = image_embeds.reshape(-1, *image_embeds.shape[2:])
        ip_tokens = self.image_proj_model(reshaped_image_embeds)

        # reshape the ip_tokens back to the original shape -(batch_size, num_images, num_tokens, embedding_size)
        batch_size = noisy_latents.shape[0]
        ip_tokens = ip_tokens.reshape(batch_size, -1, *ip_tokens.shape[1:])

        # Retrieve the correct positional embeddings based on conditioned_frames_idxs
        # Shape: (batch_size, num_images, num_tokens, embedding_size)
        pos_embeds = self._positional_embeddings[conditioned_frames_idxs]

        # Add the positional embeddings to ip_tokens
        ip_tokens = ip_tokens + pos_embeds

        encoder_hidden_states["ip_tokens"] = ip_tokens
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states,
                               cross_attention_kwargs=cross_attention_kwargs).sample
        return noise_pred

    def set_positional_embeddings(self, positional_embeddings):
        self._positional_embeddings = positional_embeddings

    def load_from_checkpoint(self, ckpt_path: str):

        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.is_plus = "latents" in state_dict["image_proj"]

        # Check if 'latents' exists in both the saved state_dict and the current model's state_dict
        strict_load_image_proj_model = True
        if "latents" in state_dict["image_proj"] and "latents" in self.image_proj_model.state_dict():
            # Check if the shapes are mismatched
            if state_dict["image_proj"]["latents"].shape != self.image_proj_model.state_dict()["latents"].shape:
                raise Exception  # todo IPAdapter : EVIATAR - for now raise an error
                print(f"Shapes of 'image_proj.latents' in checkpoint {ckpt_path} and current model do not match.")
                print("Removing 'latents' from checkpoint and loading the rest of the weights.")
                del state_dict["image_proj"]["latents"]
                strict_load_image_proj_model = False

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=strict_load_image_proj_model)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # todo: IPAdapter
        # Move the model components to the target device
        self.image_proj_model.to(self.device)
        self.adapter_modules.to(self.device)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    @torch.inference_mode()
    def get_image_embeds(self, images, negative_images=None):
        clip_image = self.clip_image_processor(images=images, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=self.dtype)

        if not self.is_plus:
            clip_image_embeds = self.image_encoder(clip_image).image_embeds
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            if negative_images is not None:
                negative_clip_image = self.clip_image_processor(images=negative_images,
                                                                return_tensors="pt").pixel_values
                negative_clip_image = negative_clip_image.to(self.device, dtype=self.dtype)
                negative_image_prompt_embeds = self.image_encoder(negative_clip_image).image_embeds
            else:
                negative_image_prompt_embeds = torch.zeros_like(clip_image_embeds)
            negative_image_prompt_embeds = self.image_proj_model(
                negative_image_prompt_embeds)  # todo:Eviatar embeds are already projected
        else:
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            if negative_images is not None:
                negative_clip_image = self.clip_image_processor(images=negative_images,
                                                                return_tensors="pt").pixel_values
                negative_clip_image = negative_clip_image.to(self.device, dtype=self.dtype)
                negative_clip_image_embeds = \
                    self.image_encoder(negative_clip_image, output_hidden_states=True).hidden_states[-2]
            else:
                negative_clip_image_embeds = \
                    self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2]
            negative_image_prompt_embeds = self.image_proj_model(negative_clip_image_embeds)

        num_tokens = image_prompt_embeds.shape[0] * self.num_tokens
        self.set_tokens(num_tokens)

        return image_prompt_embeds, negative_image_prompt_embeds

    @torch.inference_mode()
    def get_prompt_embeds(self, images, negative_images=None, prompt=None, negative_prompt=None, weight=[]):
        prompt_embeds, negative_prompt_embeds = self.get_image_embeds(images, negative_images=negative_images)

        if any(e != 1.0 for e in weight):
            weight = torch.tensor(weight).unsqueeze(-1).unsqueeze(-1)
            weight = weight.to(self.device)
            prompt_embeds = prompt_embeds * weight

        if prompt_embeds.shape[0] > 1:
            prompt_embeds = torch.cat(prompt_embeds.chunk(prompt_embeds.shape[0]), dim=1)
        if negative_prompt_embeds.shape[0] > 1:
            negative_prompt_embeds = torch.cat(negative_prompt_embeds.chunk(negative_prompt_embeds.shape[0]), dim=1)

        text_embeds = (None, None, None, None)
        if prompt is not None:
            text_embeds = self.pipe.encode_prompt(prompt,
                                                  negative_prompt=negative_prompt,
                                                  device=self.device,
                                                  num_images_per_prompt=1,
                                                  do_classifier_free_guidance=True)

            prompt_embeds = torch.cat((text_embeds[0], prompt_embeds), dim=1)
            negative_prompt_embeds = torch.cat((text_embeds[1], negative_prompt_embeds), dim=1)

        output = (prompt_embeds, negative_prompt_embeds)

        if self.is_sdxl:
            output += (text_embeds[2], text_embeds[3])

        return output

    def set_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def set_tokens(self, num_tokens):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.num_tokens = num_tokens  # todo: Eviatar whats that? to handle multiple images? why is that equals to 16 and then in infeence change to 6*16 = 96

    def set_images(self, images, clip_processed_images, clip_embeds_images, test_images=None):
        self.images = images
        self.clip_processed_images = clip_processed_images
        self.clip_embeds_images = clip_embeds_images
        self.test_images = test_images
