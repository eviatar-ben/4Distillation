<h1 align="center">4Distillation</h1>
<h3 align="center">Joint Time and View Control via 4D-to-2D Distillation of Spatiotemporal Implicit Representations</h3>


This repository presents **4Distillation**, a novel method for distilling knowledge from 4D reconstruction models into a 2D diffusion model.
Our approach enables **joint control over time and viewpoint** in text-to-image generation.

---

## Method

We introduce a distillation-based framework that transfers temporal and viewpoint understanding from 4D data into a 2D generative model (e.g., Stable Diffusion). 

The overview in the figure focuses on the temporal dimension, but the viewpoint dimension is learned in a fully analogous manner—using a View Mapper instead of a Time Mapper, and conditioning on the relative camera pose difference instead of the frame index difference.


---

###  Step 1: Sampling Temporal Pairs from Videos

<p align="center">
  <img src="assets/figures/Figure1.png" alt="Figure 1" />
</p>

**Figure 1**. We begin by sampling a video and randomlly selecting two frames, \(i\) and \(j\), with \( j > i \).

The **temporal displacement** is defined as:

\[
d = j - i
\]

This displacement \(d\) reflects the motion between frames and forms the basis for learning time-dependent generation.

---

###  Step 2: Time Mapper — Encoding Motion
![Figure 2](assets/figures/Figure2.png)

**Figure 2**. We pass the temporal gap \(d\), the diffusion timestep \(t\), and U-Net layer index \(l\) through a positional encoder, followed by a lightweight **MLP-based Time Mapper**.

The mapper outputs two vectors:
- \( \mathbf{v}_{\text{base}} \): aligned with CLIP’s embedding space.
- \( \mathbf{v}_{\text{bypass}} \): controls the fidelity–editability trade-off.

These are combined as:

\[
\mathbf{v}^* = \text{CLIP}(\mathbf{v}_{\text{base}}) + \lambda \cdot \frac{\mathbf{v}_{\text{bypass}}}{\|\mathbf{v}_{\text{bypass}}\|}
\]

This trade-off allows the condition vector \( \mathbf{v}^* \) to capture temporal motion while staying grounded in the pretrained CLIP space.

---

### Step 3: Temporal Knowledge Distillation into the Time Mapper via the 2D Diffusion Model
![Figure 3](assets/figures/Figure3.png)

**Figure 3**. The diffusion model is conditioned on:
- \( \mathbf{v}^* \) (as "text" input),
- Frame \(j\) (as a visual input via **decoupled cross-attention**).

The model is trained to reconstruct frame \(i\), thus learning to synthesize temporal changes between frames.

---

## 🏋️ Training Procedure

The model is trained using the standard DDPM loss. 
A simplified version of the training loop:

```python
# Encode target image into latent space
latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

# Generate condition vector using Time Mapper
v_star = self.get_conditioning(timesteps, normalized_frames_gap, ...)

# Predict noise using diffusion model + cross-attention to frame j
model_pred = self.ip_adapter(noisy_latents, timesteps, v_star, image_embeds)

# Compute MSE loss
loss = F.mse_loss(model_pred.float(), target.float())
self.accelerator.backward(loss)
