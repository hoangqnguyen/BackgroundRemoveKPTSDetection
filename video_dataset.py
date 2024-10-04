from tqdm import tqdm
import torch
import cv2
import os
from torch.utils.data import DataLoader
from utils.utils import to_torch, normalize


class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, transform=None, resize=(384, 384), preload=False):
        self.video_path = video_path
        self.transform = transform
        self.resize = resize
        self.preload = preload

        # If preload is enabled, load all frames during initialization
        if self.preload:
            self.frames = self._load_frames(video_path)
            self.length = len(self.frames)
            self.cap = None
            print(f"Video path: {self.video_path}, Total frames: {self.length}")
        else:
            self.frames = None
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise IOError(f"Unable to open video file: {video_path}")
            self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video path: {self.video_path}, Total frames: {self.length}")

    def _load_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Unable to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"FPS: {fps}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and convert frame to RGB
            org_size = (frame.shape[1], frame.shape[0])
            frame = cv2.resize(frame, self.resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Store the frame and its original size
            frames.append((frame, org_size))

        cap.release()
        return frames

    def _get_frame(self, idx):
        """Read a single frame from the video if preload is disabled."""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_path)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            raise IOError(
                f"Unable to read frame at index {idx} from video file: {self.video_path}"
            )

        org_size = (frame.shape[1], frame.shape[0])
        frame = cv2.resize(frame, self.resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame, org_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(
                f"Index {idx} out of bounds for video length {self.length}"
            )

        # Get frame and original size
        if self.preload:
            frame, org_size = self.frames[idx]
        else:
            frame, org_size = self._get_frame(idx)

        # Convert frame to tensor and normalize
        frame = to_torch(frame).permute(2, 0, 1)
        frame = normalize(frame)

        # Apply transformations if provided
        if self.transform:
            frame = self.transform(frame)

        return {"frame_id": idx, "frame": frame, "org_size": torch.tensor(org_size)}

    def __del__(self):
        if self.cap is not None:
            self.cap.release()


def get_dataloader(
    video_path,
    batch_size=1,
    resize=(384, 384),
    transform=None,
    num_workers=2,
    preload=True,
):
    dataset = VideoFrameDataset(
        video_path, transform=transform, resize=resize, preload=preload
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    video_path = "videos/v_2324_223_s2_short.mp4"
    dataloader = get_dataloader(video_path, batch_size=4, preload=False)

    for frames in tqdm(dataloader):
        pass
        # for k, v in frames.items():
        #     print(f"{k}: {type(v)}")
        # break
