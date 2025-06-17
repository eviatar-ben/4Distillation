import ipdb
import random
from pathlib import Path
from typing import Dict, Any, Union, Literal, List

import PIL
import numpy as np
import torch
import torch.nn as nn

import torch.utils.checkpoint
from PIL import Image, ImageOps
from packaging import version
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import transforms as T
from transformers import CLIPTokenizer

from constants import IMAGENET_STYLE_TEMPLATES_SMALL, IMAGENET_TEMPLATES_SMALL, TIME_TOKEN, TIME_SPLIT_IDXS
from utils.utils import num_to_string, string_to_num, filter_paths_imgs

from transformers import CLIPImageProcessor
import json
import os
import pickle
from tqdm import tqdm

# from coach import MAX_FRAMES

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {"linear": PIL.Image.Resampling.BILINEAR,
                         "bilinear": PIL.Image.Resampling.BILINEAR,
                         "bicubic": PIL.Image.Resampling.BICUBIC,
                         "lanczos": PIL.Image.Resampling.LANCZOS,
                         "nearest": PIL.Image.Resampling.NEAREST, }
else:
    PIL_INTERPOLATION = {"linear": PIL.Image.LINEAR,
                         "bilinear": PIL.Image.BILINEAR,
                         "bicubic": PIL.Image.BICUBIC,
                         "lanczos": PIL.Image.LANCZOS,
                         "nearest": PIL.Image.NEAREST, }


class TextualInversionDataset(Dataset):
    counter = 0

    def __init__(self,
                 data_root: Path,
                 tokenizer: CLIPTokenizer,
                 text_encoder: nn.Module,
                 learnable_mode: int,
                 relative_tokens: bool = True,
                 size: int = 768, repeats: int = 100,
                 interpolation: str = "bicubic",
                 flip_p: float = 0.0,
                 set: str = "train",
                 placeholder_object_token: str = "*",
                 augmentation_key: int = 0,
                 center_crop: bool = False,
                 train_data_subsets: List[Path] = None,
                 test_data_subsets: List[Path] = None,
                 ip_adapter_subset_size: int = None,
                 uniform_sampling: bool = False,
                 t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05):
        """
        fixed_object

        learnable_mode: integer key for what mode of caption we need to generate.
            In caption examples things in `<>` are learnable.
            0: object only, "A photo of a <object>".
            1: view only, "<view_x>. A photo of a {object}". Where {object} is
            predefined as either a string or a pretrained NeTI mapper.
            2: view and object learned jointly. "<view_x>. A photo of a <object>".
            3: multi-scene training. Learn one view-mapper shared accross scenes
                and an object mapper per-scene. "<view_x>. A photo of a <object_y>".
            4: use a pretrained view-mapper (probably pretrained from mode 3) and
                also train a new object mapper on one scene. Both view- and object-maper
                are learnable.
            5: same as mode 4, except the view mapper is not learnable. .

        fixed_object_token_or_path: if learnable_mode==1, then this cannot be None. Either a
            string if using a word from the existing vocabulary, or if using a pretrained
            viewNeTI-mapper, then write the path to that mapper.
        dtu_preprocess_key: what image preprocessing for dtu dataset.
        """
        self.learnable_mode = learnable_mode
        self.relative_tokens = relative_tokens
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_object_token = placeholder_object_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.uniform_sampling = uniform_sampling

        self.action_category_paths = list(self.data_root.glob("*"))

        # todo: Eviatar - for now to support unsaved clip embeddings (47 action instead of 101)
        if "UCF" in str(self.data_root):
            self.clip_embeddings_data_root = Path(str(self.data_root).replace('UCF-101-Frames', 'UCF-101-CLIP'))
        if 'dnerf' in str(self.data_root):
            self.clip_embeddings_data_root = Path(str(self.data_root).replace('dnerf', 'dnerf-CLIP'))

        self.ip_adapter_subset_size = ip_adapter_subset_size  # todo: Eviatar that's need to be a config parameter
        if 'UCF' in str(self.data_root):
            self.clip_embeddings_action_category_paths = list(self.clip_embeddings_data_root.glob("*"))
            self.clip_embeddings_action_category_names = [path.name for path in
                                                          self.clip_embeddings_action_category_paths]
            self.action_category_paths = [path for path in self.action_category_paths if
                                          path.name in self.clip_embeddings_action_category_names]
            self.num_actions = len(self.action_category_paths)
            assert self.num_actions > 0, "no actions folders found. Check the --data.train_data_dir option"
            self.current_action = None
            self.current_g = None
            self.current_c = None
            self.create_maps_from_g_and_c_to_len(load=True)
            self.create_maps_from_action_g_and_c_to_number_of_frames(load=True)
            self.reset_sampled_action_g_c()

            self.images_path, _ = self.get_frame_path(Path(data_root / self.current_action), self.current_g,
                                                      self.current_c)
            self.image_paths = list(self.images_path.glob("*"))
            self.image_paths = sorted(self.image_paths, key=lambda x: int(x.stem.split('_')[1]))

        if 'dnerf' in str(self.data_root):  # for now only one object is supported
            self.action_category_paths = Path(str(self.data_root))
            # Override for sanity check only
            self.action_category_paths = Path('/sci/labs/sagieb/eviatar/data/dnerf/time/jumpingjacks')
            self.image_paths = list(self.action_category_paths.glob('*'))
            self.image_paths = filter_paths_imgs(self.image_paths)
            self.image_paths = sorted(self.image_paths, key=lambda x: int(x.stem))
            self.num_images = len(self.image_paths)
            print(f"Running on {self.num_images} images")

        self.num_images = len(self.image_paths)
        assert self.num_images > 0, "no .png images found. Check the --data.train_data_dir option"

        if set == "train":
            self._length = self.num_images * repeats  # todo: Eviatar That's wrong

        self.interpolation = {"linear": PIL_INTERPOLATION["linear"],
                              "bilinear": PIL_INTERPOLATION["bilinear"],
                              "bicubic": PIL_INTERPOLATION["bicubic"],
                              "lanczos": PIL_INTERPOLATION["lanczos"],
                              }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.augmentation_key = augmentation_key
        self.handle_augmentation()

        self.text_encoder = text_encoder

        self.generated_frame_idx_histogram = np.zeros(1000)
        self.conditioned_ipadapter_frame_histogram = np.zeros(1000)
        self.generated_and_conditioned_frames_gap = np.zeros(1000 * 2 + 1)

        # todo: IPAdapter
        # self.i_drop_rate = i_drop_rate
        # self.t_drop_rate = t_drop_rate
        # self.ti_drop_rate = ti_drop_rate
        # self.image_root_path = image_root_path
        # self.data = json.load(open(json_file))  # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        # self.transform = transforms.Compose([transforms.Resize(self.size,
        #                                                        interpolation=transforms.InterpolationMode.BILINEAR),
        #                                      transforms.CenterCrop(self.size),
        #                                      transforms.ToTensor(),
        #                                      transforms.Normalize([0.5], [0.5]), ])
        #
        # self.clip_image_processor = CLIPImageProcessor()

    def reset_sampled_action_g_c(self):
        """
        For learnable_mode==3, we train multipe object tokens.

        We might want all the samples in the same batch (or accumulated set of
        batches) to be from the same scene so that the object-token gradients
        are not too noisy.

        To do that, we sample the object indexed by `self.current_object_idx`,
        and then we change this value only after accumuation.
        This function should be called in the train loop when the accumulation
        is done to randomly choose a new object value.
        """
        # one_action = 'JumpingJack'
        one_action = False
        if one_action:
            self.current_action_idx = [i for i, p in enumerate(self.action_category_paths) if p.name == one_action][0]
        else:
            self.current_action_idx = np.random.choice(self.num_actions)
        self.current_action = self.action_category_paths[self.current_action_idx].name
        # self.current_g = np.random.choice(list(self.map_g_in_action_2_length[self.current_action].keys()))
        l = list(self.map_g_in_action_2_length[self.current_action].keys())
        l.remove(15)  # for test on 15
        self.current_g = np.random.choice(l)
        # self.current_g = 15 # todo :warning- for sanity check
        self.current_c = np.random.choice(list(self.map_c_in_action_2_length[self.current_action].keys()))

        # todo: Eviatar - for now to
        self.current_c = 1
        # self.current_g = 15
        self.current_action = self.action_category_paths[self.current_action_idx].name

    def reset_sampled_object(self):
        # np.random.seed(int(time.time() * 1000) % 2**32)
        self.current_data_subset_idx = np.random.choice(len(self.train_data_subsets))

    @staticmethod
    def get_frame_path(action_dir: Path, g: int, c: int) -> Path:
        # Format the integers g and c as zero-padded strings (e.g., g=1 -> 'g01')
        g_str = f"g{g:02d}"
        c_str = f"c{c:02d}"

        # Construct the path to the specific frame
        frame_path = action_dir / f"v_{action_dir.name}_{g_str}_{c_str}"
        number_of_frames = len(list(frame_path.glob("*.jpg")))
        return frame_path, number_of_frames

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, i: int) -> Dict[str, Any]:
        if 'UCF' in str(self.data_root):
            example, image = self.get_UCF_item_multiple_frames_condition(i)

        if 'dnerf' in str(self.data_root):
            example, image = self.get_DNeRF_item(i)

        image = image.resize((self.size, self.size),
                             resample=self.interpolation)  # todo : Eviatar is that necessary?
        # #save image:
        # image.save(f"image{TextualInversionDataset.counter}.jpg")
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

        # todo: IPAdapter
        # item = self.data[idx]
        # image_file = item["image_file"]
        # raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        # example["clip_images"] = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values

        # # drop
        # drop_image_embed = 0
        # rand_num = random.random()
        # if rand_num < self.i_drop_rate:
        #     drop_image_embed = 1
        # elif rand_num < (self.i_drop_rate + self.t_drop_rate):
        #     text = ""
        # elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
        #     text = ""
        #     drop_image_embed = 1
        # example["drop_image_embed"] = drop_image_embed

        return example

    def get_UCF_item(self, i):
        action = self.action_category_paths[self.current_action_idx]
        g = self.current_g
        c = self.current_c
        # assert g == 15 and c == 1
        video_path, number_of_frames = self.get_frame_path(action, g, c)
        if self.uniform_sampling:
            generated_frame_number = random.randint(0, number_of_frames - 1)
        else:
            generated_frame_number = i % number_of_frames
        # Construct the path to the specific frame
        generated_image_path = video_path / f"frame_{generated_frame_number:05d}.jpg"  # Assuming frame number has a 5-digit format, e.g., "00009"
        image = Image.open(generated_image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        example = dict()
        example['generated_frame_idx'] = torch.tensor(generated_frame_number)
        conditioned_ipadapter_frame = self.sample_idxs(generated_frame_number, number_of_frames)
        example["conditioned_ipadapter_frame"] = conditioned_ipadapter_frame

        if self.relative_tokens:
            generated_and_conditioned_frames_gap = generated_frame_number - conditioned_ipadapter_frame  # todo: Eviatar: the diffused image minus the IPAdapter image
        else:
            generated_and_conditioned_frames_gap = generated_frame_number
        example['generated_and_conditioned_frames_gap'] = generated_and_conditioned_frames_gap
        example['normalized_frames_gap'] = self.normalize_relative_frame(generated_and_conditioned_frames_gap,
                                                                         number_of_frames,
                                                                         max_number_of_frames=None)
        # max_number_of_frames= 793

        conditioned_image_path = video_path / f"frame_{conditioned_ipadapter_frame.item():05d}.jpg"
        conditioned_ipadapter_frame_clip_embedding_path = Path(
            str(conditioned_image_path).replace('UCF-101-Frames', 'UCF-101-CLIP')).with_suffix('.npy')
        example['conditioned_ipadapter_frame_clip_path'] = str(conditioned_ipadapter_frame_clip_embedding_path)
        example['generated_image_path'] = str(generated_image_path)

        # self.generated_frame_idx_histogram[generated_frame_number] += 1
        # self.conditioned_ipadapter_frame_histogram[conditioned_ipadapter_frame] += 1

        # add generated_frame_number to histogram
        self.generated_frame_idx_histogram[generated_frame_number] += 1
        # add conditioned_ipadapter_frame to histogram
        self.conditioned_ipadapter_frame_histogram[conditioned_ipadapter_frame] += 1
        # add generated_and_conditioned_frames_gap
        self.generated_and_conditioned_frames_gap[generated_and_conditioned_frames_gap + 1000] += 1
        return example, image

    def get_UCF_item_multiple_frames_condition(self, i):
        action = self.action_category_paths[self.current_action_idx]
        g = self.current_g
        c = self.current_c
        # assert g == 15 and c == 1
        video_path, number_of_frames = self.get_frame_path(action, g, c)
        if self.uniform_sampling:
            generated_frame_number = random.randint(0, number_of_frames - 1)
        else:
            generated_frame_number = i % number_of_frames
        # Construct the path to the specific frame
        generated_image_path = video_path / f"frame_{generated_frame_number:05d}.jpg"  # Assuming frame number has a 5-digit format, e.g., "00009"
        image = Image.open(generated_image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        example = dict()
        example['generated_frame_idx'] = torch.tensor(generated_frame_number)
        conditioned_ipadapter_frame = self.sample_idxs(generated_frame_number, number_of_frames)
        example["conditioned_ipadapter_frame"] = conditioned_ipadapter_frame

        if self.relative_tokens:
            generated_and_conditioned_frames_gap = generated_frame_number - conditioned_ipadapter_frame  # todo: Eviatar: the diffused image minus the IPAdapter image
        else:
            generated_and_conditioned_frames_gap = generated_frame_number
        example['generated_and_conditioned_frames_gap'] = generated_and_conditioned_frames_gap
        example['normalized_frames_gap'] = self.normalize_relative_frame(generated_and_conditioned_frames_gap,
                                                                         number_of_frames,
                                                                         # max_number_of_frames= None)
                                                                         max_number_of_frames=793)

        conditioned_image_paths = [video_path / f"frame_{conditioned_ipadapter_frame[i].item():05d}.jpg" for i in
                                   range(self.ip_adapter_subset_size)]
        conditioned_ipadapter_frame_clip_embedding_paths = [
            Path(str(conditioned_image_paths[i]).replace('UCF-101-Frames', 'UCF-101-CLIP')).with_suffix('.npy') for i in
            range(self.ip_adapter_subset_size)]

        example['conditioned_ipadapter_frame_clip_paths'] = [str(conditioned_ipadapter_frame_clip_embedding_paths[i]) for i in range(self.ip_adapter_subset_size)]

        example['generated_image_path'] = str(generated_image_path)

        # self.generated_frame_idx_histogram[generated_frame_number] += 1
        # self.conditioned_ipadapter_frame_histogram[conditioned_ipadapter_frame] += 1

        # add generated_frame_number to histogram
        self.generated_frame_idx_histogram[generated_frame_number] += 1
        # add conditioned_ipadapter_frame to histogram
        self.conditioned_ipadapter_frame_histogram[conditioned_ipadapter_frame[0]] += 1
        self.conditioned_ipadapter_frame_histogram[conditioned_ipadapter_frame[1]] += 1
        # add generated_and_conditioned_frames_gap
        self.generated_and_conditioned_frames_gap[generated_and_conditioned_frames_gap + 1000] += 1
        return example, image

    def get_DNeRF_item(self, i):
        number_of_frames = self.num_images
        if self.uniform_sampling:
            generated_frame_number = random.randint(0, number_of_frames - 1)
        else:
            generated_frame_number = i % number_of_frames

        generated_image_path = self.image_paths[generated_frame_number]
        image = Image.open(generated_image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        example = dict()
        example['generated_frame_idx'] = torch.tensor(generated_frame_number)
        conditioned_ipadapter_frame = self.sample_idxs(generated_frame_number, len(self.image_paths))

        example["conditioned_ipadapter_frame"] = conditioned_ipadapter_frame
        if self.relative_tokens:
            generated_and_conditioned_frames_gap = generated_frame_number - conditioned_ipadapter_frame  # todo: Eviatar: the diffused image minus the IPAdapter image
        else:
            generated_and_conditioned_frames_gap = generated_frame_number
        example['generated_and_conditioned_frames_gap'] = generated_and_conditioned_frames_gap
        example['normalized_frames_gap'] = self.normalize_relative_frame(generated_and_conditioned_frames_gap,
                                                                         number_of_frames)
        conditioned_image_path = self.image_paths[conditioned_ipadapter_frame.item()]
        conditioned_ipadapter_frame_clip_embedding_path = Path(
            str(conditioned_image_path).replace('dnerf', 'dnerf-CLIP')).with_suffix('.npy')

        example['conditioned_ipadapter_frame_clip_path'] = str(conditioned_ipadapter_frame_clip_embedding_path)
        # add generated_frame_number to histogram
        self.generated_frame_idx_histogram[generated_frame_number] += 1
        # add conditioned_ipadapter_frame to histogram
        self.conditioned_ipadapter_frame_histogram[conditioned_ipadapter_frame] += 1
        # add generated_and_conditioned_frames_gap
        self.generated_and_conditioned_frames_gap[generated_and_conditioned_frames_gap + 1000] += 1
        return example, image

    def save_histogram(self, exp_dir):
        import matplotlib.pyplot as plt
        import numpy as np

        # First histogram
        plt.figure()
        plt.bar(range(len(self.generated_frame_idx_histogram)), self.generated_frame_idx_histogram, color='orange')
        plt.savefig(exp_dir / 'generated_frame_idx_histogram.png')
        plt.close()

        # Second histogram
        plt.figure()
        plt.bar(range(len(self.conditioned_ipadapter_frame_histogram)), self.conditioned_ipadapter_frame_histogram,
                color='blue')
        plt.savefig(exp_dir / 'conditioned_ipadapter_frame_histogram.png')
        plt.close()

        # Third histogram with generalized x-axis
        n = len(self.generated_and_conditioned_frames_gap)
        x_min, x_max = -n // 2 - 1, n // 2 + 1  # Generalized range

        plt.figure()
        x_values = np.linspace(x_min, x_max, n)  # Adjusted x values
        plt.bar(x_values, self.generated_and_conditioned_frames_gap, color='green')
        plt.xlim(x_min, x_max)  # Ensure x-axis matches the new range
        plt.savefig(exp_dir / 'generated_and_conditioned_frames_gap.png')
        plt.close()

    def sample_idxs(self, idx, number_of_frames):
        # in a chance of 0.05 return the same idx number_of_frames times
        if random.random() < 0.05:
            return torch.tensor([idx] * self.ip_adapter_subset_size)
        idxs = sorted(random.sample(range(number_of_frames), self.ip_adapter_subset_size + 1))
        if idx in idxs:
            idxs.remove(idx)
        idxs = idxs[:self.ip_adapter_subset_size]
        return torch.tensor(idxs)

    @staticmethod
    def normalize_relative_frame(diffusion_frame_number, number_of_frames, max_number_of_frames=None):
        if max_number_of_frames is not None:
            return torch.tensor(((diffusion_frame_number + max_number_of_frames) / max_number_of_frames) - 1)
        # normalize the relative frame number to be between [-1, 1]
        return torch.tensor(((diffusion_frame_number + number_of_frames) / number_of_frames) - 1)

    def generate_data_dictionary(self, pickle_path="data/data_dict.pkl", load_from_pickle=False):
        """ Generate a dictionary that maps
            actions_g_c_n to the path of frame number <n> from camera <c> from video <g> from action <action>.
            The dictionary has the following structure:
            data_dict[action][g][c][frame_idx] = frame_path
            """
        if load_from_pickle:
            return self.load_data_dictionary(pickle_path)

        # Initialize an empty dictionary
        data_dict = {}
        counter = 0

        for action_path in tqdm(self.action_category_paths, desc="Processing actions"):
            action = action_path.name  # Get the action name from the folder name
            if action not in data_dict:
                data_dict[action] = {}

            for video_path in tqdm(list(action_path.iterdir()), desc=f"Processing videos in {action}", leave=False):
                if video_path.is_dir():
                    video_name = video_path.name

                    # Extract g (video ID) and c (camera ID) from the video name
                    g, c = self._extract_video_and_camera(video_name)
                    if g not in data_dict[action]:
                        data_dict[action][g] = {}
                    if c not in data_dict[action][g]:
                        data_dict[action][g][c] = {}

                    for frame_path in sorted(video_path.iterdir()):
                        if frame_path.is_file() and frame_path.suffix in {'.jpg', '.png'}:
                            frame_idx = self._extract_frame_index(frame_path.name)
                            data_dict[action][g][c][frame_idx] = frame_path
            counter += 1
            if counter == 100:
                break

        self.save_data_dictionary(data_dict, pickle_path)
        return data_dict

    def generate_clip_data_dictionary(self, data_dict, pickle_path="data/clip_data_dict.pkl", load_from_pickle=False):
        """ Generate a dictionary that maps
            action_g_c_n to clip embedding of frame number <n> from camera <c> from video <g> from action <action>.
            The dictionary has the following structure:
            data_dict[action][g][c][frame_idx] = frame_path
            """
        if load_from_pickle:
            return self.load_data_dictionary(pickle_path)

        map_a_g_c_frame_2_images = data_dict  # map[action][g][camera][n] = [path of frame n]
        map_a_g_c_2_frame_processed_images_for_clip = {}  # map[action][g][camera][n] = [processed image of frame n]
        map_a_g_c_2_frame_clip_embeds = {}  # map[action][g][camera][n] = [clip embed of frame n]
        map_a_g_c_2_frame = {}
        for action_name, action in map_a_g_c_frame_2_images.items():
            for g in action.values():
                for c in g.values():
                    for frame_number, frame_path in c.items():
                        map_a_g_c_2_frame[frame_number] = Image.open(frame_path)
                        map_a_g_c_2_frame_processed_images_for_clip[frame_number] = \
                            self.ip_adapter_clip_image_processor(images=map_a_g_c_2_frame[frame_number],
                                                                 return_tensors="pt").pixel_values
                        with torch.no_grad():
                            cur_subset_clip_embeds_images = \
                                self.coach_image_encoder(
                                    map_a_g_c_2_frame_processed_images_for_clip[frame_number].to(
                                        self.coach_accelerator_device,
                                        dtype=self.weight_dtype),
                                    output_hidden_states=True).hidden_states[-2]
                        map_a_g_c_2_frame_clip_embeds[frame_number] = cur_subset_clip_embeds_images

        clip_data_dict = map_a_g_c_2_frame_clip_embeds

        self.save_data_dictionary(clip_data_dict, pickle_path)
        return clip_data_dict

    @staticmethod
    def _extract_video_and_camera(video_name):
        """Extract video ID (g) and camera ID (c) from the video name."""
        try:
            parts = video_name.split('_')
            g = int(parts[-2][1:])  # Convert g to an integer, e.g., g01 -> 1
            c = int(parts[-1][1:])  # Convert c to an integer, e.g., c01 -> 1
            return g, c
        except (IndexError, ValueError):
            raise ValueError(f"Unexpected video name format: {video_name}")

    @staticmethod
    def _extract_frame_index(frame_name):
        """Extract the frame index from the frame file name."""
        try:
            # Assuming frame names are like 'frame_0001.jpg'
            frame_idx = int(frame_name.split('_')[-1].split('.')[0])
            return frame_idx
        except (IndexError, ValueError):
            raise ValueError(f"Unexpected frame name format: {frame_name}")

    @staticmethod
    def save_data_dictionary(data_dict, pickle_path="data/data_dict.pkl"):
        """Save the data dictionary to a pickle file."""
        with open(pickle_path, 'wb') as f:
            pickle.dump(data_dict, f)

    @staticmethod
    def load_data_dictionary(pickle_path="data/data_dict.pkl"):
        """Load the data dictionary from a pickle file."""
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

    def get_random_test_images(self):
        # UCF-101 dataset
        # action = random.choice(self.action_category_paths) # todo: Eviatar should be excluded action
        # g = random.choice(list(action.glob("v_*")))
        # c = random.choice(list(g.glob("v_*")))
        # conditioned_image = random.choice(list(temp.glob("*.jpg")))
        if 'UCF' in str(self.data_root):
            action_g_c = 'JumpingJack/v_JumpingJack_g15_c01'
            temp = self.data_root / action_g_c
            conditioned_image_idxs = [36]
            conditioned_image = list(temp.glob("*.jpg"))[conditioned_image_idxs[0]]
            conditioned_image = Image.open(conditioned_image)
            conditioned_images = [conditioned_image]
            len_action_test_images = len(list(temp.glob("*.jpg")))
        elif 'dnerf' in str(self.data_root):
            conditioned_image = Image.open(self.image_paths[50])
            conditioned_images = [conditioned_image]
            conditioned_image_idxs = [50]
            len_action_test_images = len(self.image_paths)
        else:
            raise Exception("Not implemented")
        return conditioned_images, conditioned_image_idxs, len_action_test_images

    def handle_augmentation(self):
        if self.augmentation_key > 0:
            if self.learnable_mode == 0:
                size = (self.size, self.size)
            # elif self.dtu_preprocess_key == 0:
            #     size = (512, 512)
            # elif self.dtu_preprocess_key == 1:
            #     size = (384, 512)  # size axes are reversed compared to PIL oib
            elif True:
                #     # elif self.video_preprocess_key == 2:
                size = (self.size, self.size)
                # todo: implement this for now ive just copied the above

            if self.augmentation_key == 1:
                self.augmentations = T.Compose([
                    T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)], p=0.75),
                    T.RandomGrayscale(p=0.1),
                    T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.10),
                    T.RandomApply([T.RandomRotation(degrees=10, fill=1)], p=0.75),
                    T.RandomResizedCrop(size, scale=(0.850, 1.15)), ])
            elif self.augmentation_key == 2:
                # the same but without the geometric transforms
                self.augmentations = T.Compose([
                    T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)], p=0.75),
                    T.RandomGrayscale(p=0.1),
                    T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.10), ])
            elif self.augmentation_key == 3:
                # the same but minus the small geometric transforms
                self.augmentations = T.Compose([
                    T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)], p=0.75),
                    T.RandomGrayscale(p=0.1),
                    T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.10),
                    T.RandomApply([T.RandomRotation(degrees=10, fill=1)], p=0.75), ])
            elif self.augmentation_key == 4:
                # the same but minus the small geometric transforms
                self.augmentations = T.Compose([
                    T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)], p=0.75),
                    T.RandomGrayscale(p=0.1),
                    T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.10),
                    T.RandomResizedCrop(size, scale=(0.850, 1.15)), ])
            elif self.augmentation_key == 5:
                self.augmentations = T.Compose([
                    T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)], p=0.75),
                    # T.RandomGrayscale(p=0.1),
                    T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.25),  # a little higher (no grayscale anymore)
                    T.RandomResizedCrop(size, scale=(0.950, 1.05)), ])
                # the same but the small geometric transforms are now very small
            elif self.augmentation_key == 6:
                # the exact thing used in RealFusion
                self.augmentations = T.Compose([
                    T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)], p=0.75),
                    T.RandomGrayscale(p=0.1),
                    T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.10),
                    T.RandomApply([T.RandomRotation(degrees=10, fill=1)], p=0.75),
                    T.RandomResizedCrop(size, scale=(0.70, 1.3)), ])
            elif self.augmentation_key == 7:
                # same as RealFusion but no grayscale ... I turn up the gaussianblur probablity a bit
                # self.augmentations = T.Compose([
                #     T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)], p=0.25),
                #     T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.2),
                #     T.RandomApply([T.RandomRotation(degrees=10, fill=1)], p=0.25),
                #     T.RandomApply([T.RandomResizedCrop(size, scale=(0.70, 1.3))], p=0.25)
                #     # T.RandomResizedCrop(size, scale=(0.70, 1.3)),
                self.augmentations = T.Compose([T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)], p=0.75),
                                                T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.2),
                                                T.RandomApply([T.RandomRotation(degrees=10, fill=1)], p=0.75),
                                                T.RandomResizedCrop(size, scale=(0.70, 1.3)), ])

            elif self.augmentation_key == 8:
                # the same as 7 but without the geometric transforms. For ablations
                self.augmentations = T.Compose([T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)], p=0.75),
                                                T.RandomGrayscale(p=0.1),
                                                T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.10)])

            elif self.augmentation_key == 10:
                self.augmentations = T.Compose([
                    T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.25),  # a little higher (no grayscale anymore)
                ])

            elif self.augmentation_key == 100:
                self.augmentations = T.Compose([T.Lambda(lambda x: x)])  # No augmentation

            else:
                raise

    def create_maps_from_action_g_and_c_to_number_of_frames(self, save=True, load=True,
                                                            path="data/ucf-101/dictionaries"):
        from pathlib import Path
        if load:
            self.map_action_g_c_2_length = pickle.load(open(Path(path) / "map_action_g_c_2_length.pkl", "rb"))
            return
        maximum_len = 0

        # Assuming self.action_category_paths is already defined as a list of action directories
        self.map_action_g_c_2_length = {}

        # Iterate over each action directory
        for action in tqdm(self.action_category_paths, desc="Processing actions"):
            action_path = Path(action)
            action_name = action_path.name  # Extract action name, e.g., 'ApplyEyeMakeup'

            # Initialize the dictionary for the current action
            self.map_action_g_c_2_length[action_name] = {}
            # Iterate over the video directories in the action folder
            for video in action_path.glob("v_*"):  # Iterating over video directories like 'v_ApplyEyeMakeup_g01'
                if video.is_dir():
                    # Extract g and c numbers from the directory name
                    video_name = video.name  # Example: 'v_ApplyEyeMakeup_g01_c01'
                    parts = video_name.split("_")
                    if len(parts) >= 3:
                        g_part = parts[2]  # e.g., 'g01'
                        c_part = parts[3]  # e.g., 'c01'

                        # Extract numeric values of g and c
                        g_num = int(g_part[1:])  # Remove 'g' and convert to int
                        c_num = int(c_part[1:])  # Remove 'c' and convert to int

                        cur_len = len(list(video.glob("*.jpg")))

                        if g_num in self.map_action_g_c_2_length[action_name]:
                            if c_num in self.map_action_g_c_2_length[action_name][g_num]:
                                self.map_action_g_c_2_length[action_name][g_num][c_num] += len(
                                    list(video.glob("*.jpg")))
                            else:
                                self.map_action_g_c_2_length[action_name][g_num][c_num] = len(list(video.glob("*.jpg")))
                        else:
                            self.map_action_g_c_2_length[action_name][g_num] = {c_num: len(list(video.glob("*.jpg")))}

                        if cur_len > maximum_len:
                            maximum_len = cur_len
        print(f"########################################{maximum_len}################################################")
        self.map_action_g_c_2_length["max_len"] = maximum_len
        # save
        pickle.dump(self.map_action_g_c_2_length, open(Path(path) / "map_action_g_c_2_length.pkl", "wb"))

    def create_maps_from_g_and_c_to_len(self, save=True, load=True, path="data/ucf-101/dictionaries"):
        from pathlib import Path
        from collections import defaultdict
        if load:
            self.map_g_in_action_2_length = pickle.load(open(Path(path) / "map_g_in_action_2_length.pkl", "rb"))
            self.map_c_in_action_2_length = pickle.load(open(Path(path) / "map_c_in_action_2_length.pkl", "rb"))
            self.map_action_to_number_of_total_sub_frames = pickle.load(
                open(Path(path) / "map_action_to_number_of_total_sub_frames.pkl", "rb"))
            return

        # Assuming self.action_category_paths is already defined as a list of action directories
        self.map_g_in_action_2_length = {}
        self.map_c_in_action_2_length = {}
        self.map_action_to_number_of_total_sub_frames = {}

        # Iterate over each action directory
        for action in tqdm(self.action_category_paths, desc="Processing actions"):
            action_path = Path(action)
            action_name = action_path.name  # Extract action name, e.g., 'ApplyEyeMakeup'

            # Initialize the dictionary for the current action
            self.map_g_in_action_2_length[action_name] = {}
            self.map_c_in_action_2_length[action_name] = {}
            self.map_action_to_number_of_total_sub_frames[action_name] = 0

            # Temporary dictionaries to count occurrences of g and c
            g_count = defaultdict(int)
            c_count = defaultdict(int)

            # Iterate over the video directories in the action folder
            for video in action_path.glob("v_*"):  # Iterating over video directories like 'v_ApplyEyeMakeup_g01'
                if video.is_dir():
                    # Extract g and c numbers from the directory name
                    video_name = video.name  # Example: 'v_ApplyEyeMakeup_g01_c01'
                    parts = video_name.split("_")
                    if len(parts) >= 3:
                        g_part = parts[2]  # e.g., 'g01'
                        c_part = parts[3]  # e.g., 'c01'

                        # Extract numeric values of g and c
                        g_num = int(g_part[1:])  # Remove 'g' and convert to int
                        c_num = int(c_part[1:])  # Remove 'c' and convert to int

                        # Count occurrences of g and c
                        g_count[g_num] += 1
                        c_count[c_num] += 1
                    self.map_action_to_number_of_total_sub_frames[action_name] += len(list(video.glob("*.jpg")))

            # Store the counts in the maps
            self.map_g_in_action_2_length[action_name] = dict(g_count)
            self.map_c_in_action_2_length[action_name] = dict(c_count)

        # save
        pickle.dump(self.map_g_in_action_2_length, open(Path(path) / "map_g_in_action_2_length.pkl", "wb"))
        pickle.dump(self.map_c_in_action_2_length, open(Path(path) / "map_c_in_action_2_length.pkl", "wb"))
        pickle.dump(self.map_action_to_number_of_total_sub_frames,
                    open(Path(path) / "map_action_to_number_of_total_sub_frames.pkl", "wb"))
