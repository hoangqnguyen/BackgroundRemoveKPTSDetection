import glob
import os

import cv2
import torch
from torchvision import transforms
import time


def to_torch(np_array):
    # configs = parse_configs()
    tensor = torch.from_numpy(np_array).float()
    return torch.autograd.Variable(tensor, requires_grad=False)


def normalize(video_tensor):
    """
    Undoes mean/standard deviation normalization, zero to one scaling,
    and channel rearrangement for a batch of images.
    args:
        video_tensor: a (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return normalize(video_tensor) / 255.0

def load_pretrained_model(model, pretrained_path):
    """Load weights from the pretrained model"""
    assert os.path.isfile(pretrained_path), "=> no checkpoint found at '{}'".format(pretrained_path)
    checkpoint = torch.load(pretrained_path, map_location='cuda:0')
    pretrained_dict = checkpoint['state_dict']
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}
        # 2. overwrite entries in the existing state dict
        model_state_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.module.load_state_dict(model_state_dict)
    else:
        model_state_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}
        # Load global to local stage
        # 2. overwrite entries in the existing state dict
        model_state_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_state_dict)
    return model


def img_to_video(images, video_name, fps):
    images = glob.glob(images)
    sort_images = sorted(images)
    img_array = []
    for img_path in sort_images:
        img = cv2.imread(img_path)
        height, width, layers = img.shape
        img_array.append(img)

    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()



class Timer:
    def __init__(self, message="Elapsed time"):
        self.message = message

    def __enter__(self):
        self.start_time = time.time()
        return self  # If you need to return any object, it would be here

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.time() - self.start_time
        print(f"{self.message}: {elapsed_time:.4f} seconds")