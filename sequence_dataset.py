import os
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2
from tqdm import tqdm


class VideoFrameSlidingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        folder_path,
        transform=None,
        resize=(384, 384),
        window_size=16,
        stride=1,
        channel_first=True, # (C, T, H, W) if True, (T, C, H, W) if False
        max_cache_size=100,
    ):
        self.folder_path = folder_path
        self.resize = resize
        self.window_size = window_size
        self.stride = stride
        self.channel_first = channel_first
        self.max_cache_size = max_cache_size

        # Get sorted list of frame files
        self.frame_files = sorted(
            [
                os.path.join(folder_path, fname)
                for fname in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, fname))
            ]
        )

        self.num_frames = len(self.frame_files)
        print(f"Folder path: {self.folder_path}, Total frames: {self.num_frames}")

        # Define default transform if none is provided
        self.transform = transform or v2.Compose(
            [
                v2.Resize(self.resize, antialias=False),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Initialize cache for loaded frames with LRU eviction policy
        self.frame_cache = OrderedDict()

    def _load_frame(self, frame_path):
        """Load a single frame."""
        # Check if frame is already cached
        if frame_path in self.frame_cache:
            # Move the accessed frame to the end to mark it as recently used
            self.frame_cache.move_to_end(frame_path)
            return self.frame_cache[frame_path]

        # Read frame and get original size
        frame = read_image(frame_path)  # Loaded as (C, H, W) in uint8 format
        org_size = (frame.shape[2], frame.shape[1])  # (Width, Height)

        # Add frame to cache
        self._add_to_cache(frame_path, (frame, org_size))

        return frame, org_size

    def _add_to_cache(self, frame_path, frame_data):
        """Add a frame to the cache and maintain the cache size limit."""
        if len(self.frame_cache) >= self.max_cache_size:
            # Remove the least recently used item if cache size exceeds max_cache_size
            self.frame_cache.popitem(last=False)
        self.frame_cache[frame_path] = frame_data

    def _load_sequence(self, start_idx):
        """Load a sequence of frames starting from `start_idx`."""
        frames = []
        org_sizes = []

        for i in range(self.window_size):
            actual_idx = start_idx + i
            if actual_idx < self.num_frames:
                frame, org_size = self._load_frame(self.frame_files[actual_idx])
            else:
                # Zero-padding if not enough frames
                frame = torch.zeros(
                    (3, self.resize[1], self.resize[0]), dtype=torch.float
                )
                org_size = (self.resize[0], self.resize[1])

            frames.append(frame)
            org_sizes.append(org_size)

        return frames, org_sizes

    def __len__(self):
        # Calculate the number of sequences based on stride and window_size
        return max(1, (self.num_frames - self.window_size) // self.stride + 1)

    def __getitem__(self, idx):
        # Calculate start index for this sequence
        start_idx = idx * self.stride

        # Load sequence of frames and their original sizes
        frames, org_sizes = self._load_sequence(start_idx)

        # Stack frames into a tensor of shape (T, C, H, W)
        frames = torch.stack(frames, dim=0)

        # Apply transform to each frame independently
        frames = torch.stack([self.transform(frame) for frame in frames], dim=0)

        # Convert to (C, T, H, W) if channel_first is True
        if self.channel_first:
            frames = frames.permute(
                1, 0, 2, 3
            )  # Convert from (T, C, H, W) to (C, T, H, W)

        return {"sequence_id": idx, "frames": frames, "org_sizes": torch.tensor(org_sizes)}

    def clear_cache(self):
        """Clear the frame cache to free up memory."""
        self.frame_cache.clear()


def get_dataloader_sliding(
    folder_path,
    batch_size=1,
    resize=(384, 384),
    transform=None,
    num_workers=2,
    window_size=16,
    stride=1,
    channel_first=True,
    max_cache_size=100,
):
    dataset = VideoFrameSlidingDataset(
        folder_path,
        transform=transform,
        resize=resize,
        window_size=window_size,
        stride=stride,
        channel_first=channel_first,
        max_cache_size=max_cache_size,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    from utils.utils import Timer

    with Timer('Total time'):
        folder_path = "/mnt/d/Dataset/KOVO_Dataset/images/2324/205/s2_rally_001"
        dataloader = get_dataloader_sliding(
            folder_path, batch_size=2, window_size=16, stride=4, channel_first=False, max_cache_size=200, num_workers=4
        )

        for sequences in tqdm(dataloader):
            # print(sequences["frames"].shape)
            # print(sequences["org_sizes"])
            pass
            # break
