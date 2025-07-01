# 🌀 4Distillation: Time and View Control

This repository implements a method for distilling 4D reconstruction model knowledge into a 2D diffusion model, enabling **joint time and view control** in text-to-image generation.

## 📌 Overview

We introduce a lightweight neural network, **TimeMapper**, to encode temporal variations between frames. The same design is applied analogously to view-based transformations using a **ViewMapper**.

## ⏱ Temporal Conditioning via TimeMapper

### Training Process

At each training step:

1. A video `v` is sampled from a pretrained 4D reconstruction model.
2. Two random frames, `f_i` and `f_j`, are selected. 
   The temporal distance is:  
   `d = |i - j|`
3. A triplet `[t, l, d]` is formed where:  
   - `t`: diffusion timestep  
   - `l`: U-Net layer index  
   - `d`: frame distance  
4. This `[t, l, d]` is embedded with positional encoding and passed to the `TimeMapper`.

### TimeMapper Outputs

The TimeMapper predicts two vectors:
- `v_base`: added to the **initial CLIP text embedding** (key in cross-attention)
- `v_bypass`: added (after ℓ₂ normalization and scaling) to the **CLIP output** (value in cross-attention)

The final text condition used for generation is:
```python
v_star = CLIP(v_base) + λ * normalize(v_bypass)  # λ = 0.2
```

This setup enables a tradeoff between:
- **Editability** (via `v_base`)
- **Fidelity** (via `v_bypass`)

### Inference

- `v*` is used as the **text condition**.
- Frame `f_i` is provided to the pretrained diffusion model as an **image prompt** via pretrained decoupled cross-attention (IPAdapter) to reconstruct frame `f_j`.
- The loss between the predicted frame and `f_j` is used to train the `TimeMapper`.

### 🔄 ViewMapper

A separate mapper is used for handling **viewpoint changes** using a fully analogous design.

## 📂 Project Structure

```
├── time_mapper/         # Core implementation of TimeMapper
├── train.py             # Training loop
├── configs/             # Configuration files
├── scripts/             # Utility scripts
└── README.md            # This file
```

## 🛠 Setup

```bash
conda create -n 4distillation python=3.10
conda activate 4distillation
pip install -r requirements.txt
```

## 🚀 Training

```bash
python train.py --config configs/train_time.yaml
```

## 📄 License

This project is for research purposes only.
