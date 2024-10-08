import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.io import read_video
from torchvision.transforms import v2
from tqdm import tqdm


class VideoFrameSlidingDataset(torch.utils.data.Dataset):
    """MP4 mode only works with short videos (few seconds long) due to memory constraints."""

    def __init__(
        self,
        input_data,  # Could be either a list of numpy images or a video path
        mode="list",  # "list" for numpy arrays, "mp4" for video file
        transform=None,
        window_size=16,
        stride=1,
        channel_first=False,  # (C, T, H, W) if True, (T, C, H, W) if False
        buffer_size=100,  # Only used in "mp4" mode
    ):
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.channel_first = channel_first
        self.buffer_size = buffer_size
        self.transform = transform

        if self.mode == "list":
            self.image_list = input_data
            self.num_frames = len(self.image_list)
            print(f"Total frames in list: {self.num_frames}")
        elif self.mode == "mp4":
            self.video_path = input_data
            video, _, _ = read_video(self.video_path, pts_unit="sec")
            self.frames = video.permute(0, 3, 1, 2)  # Convert to (T, C, H, W)
            self.num_frames = self.frames.size(0)
            print(f"Video path: {self.video_path}, Total frames: {self.num_frames}")

        else:
            raise ValueError("Mode must be either 'list' or 'mp4'")

    def _get_frame_from_video(self, idx):
        """Retrieve a frame directly from the loaded video tensor."""
        if idx < self.num_frames:
            frame = self.frames[idx]
            org_size = (frame.shape[2], frame.shape[1])  # (Width, Height)
            return frame, org_size
        else:
            raise KeyError(f"Frame at index {idx} could not be loaded into buffer.")

    def _load_sequence(self, start_idx):
        """Load a sequence of frames starting from `start_idx`."""
        frames = []
        org_sizes = []

        for i in range(self.window_size):
            actual_idx = start_idx + i
            if actual_idx < self.num_frames:
                if self.mode == "list":
                    # Load frame from the list of numpy arrays
                    frame = torch.tensor(
                        self.image_list[actual_idx]
                    )  # Convert to tensor
                    org_size = (frame.shape[2], frame.shape[1])  # (Width, Height)
                elif self.mode == "mp4":
                    # Load frame from video tensor
                    frame, org_size = self._get_frame_from_video(actual_idx)
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")
            else:
                # Zero-padding if not enough frames
                frame = torch.zeros(
                    (3, org_size[1], org_size[0]),
                    dtype=torch.float,
                )
                org_size = (frame.shape[2], frame.shape[1])

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

        # Apply transform to the entire sequence (all frames in the sequence)
        if self.transform:
            frames = self.transform(frames)

        # Convert to (C, T, H, W) if channel_first is True
        if self.channel_first:
            frames = frames.permute(
                1, 0, 2, 3
            )  # Convert from (T, C, H, W) to (C, T, H, W)

        return {
            "sequence_id": idx,
            "frames": frames,
            "org_sizes": torch.tensor(org_sizes),
        }


def get_dataloader_sliding(
    input_data,
    mode="list",
    batch_size=1,
    resize=None,  # If None, no resizing is done
    transform=None,
    num_workers=2,
    window_size=16,
    stride=1,
    channel_first=True,
    buffer_size=100,  # Only used in "mp4" mode
):
    # Build the transform pipeline
    transform_list = []
    if resize:
        transform_list.append(
            v2.Resize(resize, antialias=False)
        )  # Apply resizing if resize is not None
    transform_list.append(
        v2.ToDtype(torch.float32, scale=True)
    )  # Convert to float and scale
    transform_list.append(
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )  # Normalize
    composed_transform = v2.Compose(transform_list) if transform_list else None

    dataset = VideoFrameSlidingDataset(
        input_data=input_data,
        mode=mode,
        transform=composed_transform,
        window_size=window_size,
        stride=stride,
        channel_first=channel_first,
        buffer_size=buffer_size,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    from utils.utils import Timer
    import numpy as np

    # Example: Using a video file and converting it to a list of frames
    video_path = "videos/v_2324_223_s2_short.mp4"
    configs = dict(
        batch_size=2,
        window_size=16,
        stride=4,
        channel_first=False,
        resize=(384, 384),
        buffer_size=200,
        num_workers=4,
    )

    with Timer("Total time (mp4 mode)"):
        # First, load frames from an mp4
        dataloader_mp4 = get_dataloader_sliding(
            video_path,
            mode="mp4",
            **configs,
        )
        for sequences in tqdm(dataloader_mp4):
            pass

    # Example: Using those frames in "list" mode
    all_frames = dataloader_mp4.dataset.frames.cpu().numpy()

    print(f"Frames shape: {all_frames.shape}")
    with Timer("Total time (list mode)"):
        dataloader_list = get_dataloader_sliding(
            all_frames,
            mode="list",
            **configs,
        )

        for sequences in tqdm(dataloader_list):
            pass
