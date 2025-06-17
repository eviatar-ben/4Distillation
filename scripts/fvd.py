import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data

from scripts.pytorch_i3d import InceptionI3d
import os

from sklearn.metrics.pairwise import polynomial_kernel
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import sys

MAX_BATCH = 16
FVD_SAMPLE_SIZE = 2048
TARGET_RESOLUTION = (224, 224)


def preprocess(videos, target_resolution):
    # videos in {0, ..., 255} as np.uint8 array
    b, t, h, w, c = videos.shape
    all_frames = torch.FloatTensor(videos).flatten(end_dim=1)  # (b * t, h, w, c)
    all_frames = all_frames.permute(0, 3, 1, 2).contiguous()  # (b * t, c, h, w)
    resized_videos = F.interpolate(all_frames, size=target_resolution,
                                   mode='bilinear', align_corners=False)
    resized_videos = resized_videos.view(b, t, c, *target_resolution)
    output_videos = resized_videos.transpose(1, 2).contiguous()  # (b, c, t, *)
    scaled_videos = 2. * output_videos / 255. - 1  # [-1, 1]
    return scaled_videos


def get_fvd_logits(videos, i3d, device):
    videos = preprocess(videos, TARGET_RESOLUTION)
    embeddings = get_logits(i3d, videos, device)
    return embeddings


def load_fvd_model(device):
    i3d = InceptionI3d(400, in_channels=3).to(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    i3d_path = os.path.join(current_dir, 'i3d_pretrained_400.pt')
    i3d.load_state_dict(torch.load(i3d_path, map_location=device))
    i3d.eval()
    return i3d


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1)  # unbiased estimate
    m_center = m - torch.mean(m, dim=1, keepdim=True)
    mt = m_center.t()  # if complex: mt = m.t().conj()
    return fact * m_center.matmul(mt).squeeze()


def frechet_distance(x1, x2):
    x1 = x1.flatten(start_dim=1)
    x2 = x2.flatten(start_dim=1)
    m, m_w = x1.mean(dim=0), x2.mean(dim=0)
    sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)

    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
    trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    mean = torch.sum((m - m_w) ** 2)
    fd = trace + mean
    return fd


def polynomial_mmd(X, Y):
    m = X.shape[0]
    n = Y.shape[0]
    # compute kernels
    K_XX = polynomial_kernel(X)
    K_YY = polynomial_kernel(Y)
    K_XY = polynomial_kernel(X, Y)
    # compute mmd distance
    K_XX_sum = (K_XX.sum() - np.diagonal(K_XX).sum()) / (m * (m - 1))
    K_YY_sum = (K_YY.sum() - np.diagonal(K_YY).sum()) / (n * (n - 1))
    K_XY_sum = K_XY.sum() / (m * n)
    mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum
    return mmd


def get_logits(i3d, videos, device):
    assert videos.shape[0] % MAX_BATCH == 0
    with torch.no_grad():
        logits = []
        for i in range(0, videos.shape[0], MAX_BATCH):
            batch = videos[i:i + MAX_BATCH].to(device)
            logits.append(i3d(batch))
        logits = torch.cat(logits, dim=0)
        return logits


def compute_fvd(real, samples, i3d, device=torch.device('cpu')):
    # real, samples are (N, T, H, W, C) numpy arrays in np.uint8
    real, samples = preprocess(real, (224, 224)), preprocess(samples, (224, 224))
    first_embed = get_logits(i3d, real, device)
    second_embed = get_logits(i3d, samples, device)

    return frechet_distance(first_embed, second_embed)


def get_video_folders(base_dir):
    """Get all video folders from the UCF-101-Frames dataset."""
    video_folders = []

    # First level: action class folders
    action_classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Second level: video folders
    for action in action_classes:
        action_path = os.path.join(base_dir, action)
        videos = [os.path.join(action_path, v) for v in os.listdir(action_path)
                  if os.path.isdir(os.path.join(action_path, v))]
        video_folders.extend(videos)

    return video_folders


def get_random_video_folders(base_dir, n_videos):
    """Sample n random video folders without loading all paths first"""
    # Get action class folders
    action_classes = [d for d in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, d))]

    sampled_folders = []
    while len(sampled_folders) < n_videos:
        # Pick a random action class
        action = random.choice(action_classes)
        action_path = os.path.join(base_dir, action)

        # Get video folders in this action class
        video_folders = [v for v in os.listdir(action_path)
                         if os.path.isdir(os.path.join(action_path, v))]

        if video_folders:
            # Pick a random video folder
            video = random.choice(video_folders)
            video_path = os.path.join(action_path, video)

            # Avoid duplicates
            if video_path not in sampled_folders:
                sampled_folders.append(video_path)

    return sampled_folders


def extract_numbers(filename):
    import re
    """Extract numbers from a filename to use as sorting keys."""
    return [int(num) if num.isdigit() else num for num in re.findall(r'\d+', filename)]


def load_video_frames_sequential(video_folder, target_frames=32, remove_conditioned_image=False):
    """Load a sequential block of frames from a video folder"""
    frames = [f for f in os.listdir(video_folder) if f.endswith(('.jpg', '.png'))]
    sorted_frames = sorted(frames, key=extract_numbers)
    if remove_conditioned_image:
        sorted_frames = sorted_frames[1:]
        assert 'inference_images' in video_folder

    # If not enough frames, duplicate the last frame
    if len(sorted_frames) < target_frames:
        last_frame = sorted_frames[-1] if sorted_frames else None
        if last_frame:
            sorted_frames.extend([last_frame] * (target_frames - len(sorted_frames)))
        else:
            return None

    # If more than enough frames, take a sequential block
    if len(sorted_frames) > target_frames:
        # Randomly select a starting point that ensures enough frames
        max_start_idx = len(sorted_frames) - target_frames
        start_idx = random.randint(0, max_start_idx)
        sorted_frames = sorted_frames[start_idx:start_idx + target_frames]

    # Load the sequential frames block
    frames = []
    for frame_file in sorted_frames:
        frame_path = os.path.join(video_folder, frame_file)
        try:
            with Image.open(frame_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                frames.append(np.array(img))
        except Exception as e:
            print(f"Error loading {frame_path}: {e}")
            return None

    if len(frames) == target_frames:
        return np.stack(frames)
    return None


def load_real_videos(base_dir, video_folders, num_videos=128, frames_per_video=32):
    """Randomly select and load a specific number of real videos."""

    if len(video_folders) < num_videos:
        print(f"Warning: Requested {num_videos} videos but only {len(video_folders)} are available.")
        num_videos = len(video_folders)

    # Randomly select video folders
    selected_folders = random.sample(video_folders, num_videos)

    # Load videos
    videos = []
    print(f"Loading {num_videos} videos...")
    for folder in tqdm(selected_folders):
        # video = load_video_frames(folder, frames_per_video)
        video = load_video_frames_sequential(folder, frames_per_video)
        if video is not None:
            videos.append(video)

    # Ensure we have enough videos
    if len(videos) < num_videos:
        print(f"Warning: Only {len(videos)} valid videos were loaded.")

    # Stack to create batch
    if videos:
        return np.stack(videos).astype(np.uint8)
    return None


def load_generated_videos(generated_videos_paths, frames_per_video=32):
    """
    Load generated videos that match the real video folder names

    Args:
        generated_videos_paths: Base directory containing generated video frames
        real_video_folders: List of real video folders to match against
        frames_per_video: Number of frames to load per video

    Returns:
        Numpy array of videos with shape (n_videos, frames_per_video, height, width, 3)
    """
    generated_videos = []
    found_videos = 0
    missing_videos = 0

    print(f"Loading generated videos from {generated_videos_paths}...")

    for generated_video_path in tqdm(generated_videos_paths):

        if os.path.isdir(generated_video_path):
            # Load video frames using existing function
            frames = load_video_frames_sequential(generated_video_path, frames_per_video, remove_conditioned_image=True)
            if frames is not None:
                generated_videos.append(frames)
                found_videos += 1
        else:
            missing_videos += 1

    print(f"Found {found_videos} generated videos, missing {missing_videos}")

    if generated_videos:
        return np.stack(generated_videos).astype(np.uint8)
    return None


def generate_videos(video_folders, generated_video_path, frames_per_video):
    import inference
    inference.get_generated_videos_from_paths(paths=video_folders, n_frames=frames_per_video)


def get_generated_video_folders(generated_video_path, num_videos=128):
    import os
    import re
    inference_path = os.path.join(generated_video_path, "inference_images")

    # Define UCF-101 action video regex pattern
    # ucf_pattern = re.compile(r"v_[A-Za-z]+_g\d{2}_c\d{2}")
    ucf_pattern = re.compile(r"v_.*")
    # Get all subdirectories that match the pattern
    ucf_dirs = [os.path.join(inference_path, d)
                for d in os.listdir(inference_path)
                if os.path.isdir(os.path.join(inference_path, d)) and ucf_pattern.match(d)]
    ucf_dirs = ucf_dirs[:num_videos]
    return ucf_dirs

def resize_videos(videos, target_size=(256, 256)):
    import numpy as np
    from PIL import Image
    num_videos, frames_per_video, height, width, channels = videos.shape
    resized_videos = np.zeros((num_videos, frames_per_video, *target_size, channels), dtype=videos.dtype)

    for v in range(num_videos):
        for f in range(frames_per_video):
            frame = videos[v, f]  # Extract frame (shape: 240x320x3)
            image = Image.fromarray(frame)  # Convert to PIL Image
            resized_frame = image.resize(target_size, resample=Image.Resampling.BICUBIC)  # Resize
            resized_videos[v, f] = np.array(resized_frame)  # Convert back to numpy array

    return resized_videos


def main(generate_video_as_random_real_videos=False):
    print(os.getcwd())
    # Configuration
    dataset_path = "/sci/labs/sagieb/eviatar/data/UCF-101-Frames"
    generated_video_path = "/sci/labs/sagieb/eviatar/Distillation4D/results/save_linear_projection"
    num_videos =  2 * MAX_BATCH  # Number of videos to sample - at least 10 videos multiple of 16
    frames_per_video = 64  # Number of frames per video -
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    random_video_folders = get_random_video_folders(dataset_path, num_videos)
    # Load real videos
    print("Loading real videos...")
    real_videos = load_real_videos(dataset_path, random_video_folders, num_videos, frames_per_video)

    if real_videos is None or len(real_videos) < 16:
        print("Error: Failed to load enough real videos.")
        return
    # resizes the videos to (num_videos, frames_per_video, 240 , 320, 3) to (num_videos, frames_per_video, 256, 256, 3)
    real_videos = resize_videos(real_videos)
    if generate_video_as_random_real_videos:
        print("Generating videos")
        generate_videos(random_video_folders, generated_video_path, frames_per_video)
    print("Loading generated videos...")
    pre_generated_videos_paths = get_generated_video_folders(generated_video_path, num_videos)
    generated_videos = load_generated_videos(pre_generated_videos_paths, frames_per_video)

    # Print shapes for verification
    print(f"Real videos shape: {real_videos.shape}")
    print(f"Generated videos shape: {generated_videos.shape}")

    # Load I3D model
    print("Loading I3D model...")
    i3d = load_fvd_model(device)

    # Calculate FVD
    print("Calculating FVD...")
    fvd = compute_fvd(real_videos, generated_videos, i3d, device)
    print(f"FVD Score: {fvd.item()}")

    # Add small noise to verify FVD calculation
    generated_videos_with_noise = generated_videos.copy()
    generated_videos_with_noise += np.random.randint(0, 5, size=generated_videos.shape, dtype=np.uint8)
    generated_videos_with_noise = np.clip(generated_videos_with_noise, 0, 255)

    # Now calculate FVD
    fvd = compute_fvd(real_videos, generated_videos_with_noise, i3d, device)
    print(f"FVD Score (with noise): {fvd.item()}")


def sanity_check():
    real = np.random.randint(0, 255, (128, 32, 256, 256, 3), dtype=np.uint8)
    samples = np.random.randint(0, 255, (128, 32, 256, 256, 3), dtype=np.uint8)
    i3d = load_fvd_model(torch.device('cuda'))
    fvd = compute_fvd(real, samples, i3d, torch.device('cuda'))
    print(fvd)
    print("Sanity check passed. with FVD score:", fvd)


if __name__ == '__main__':
    main(generate_video_as_random_real_videos=False)
    # sanity_check()
