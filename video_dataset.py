from tqdm import tqdm
import torch
import cv2
from torch.utils.data import DataLoader
from utils.utils import to_torch, normalize


class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, transform=None, resize=(384, 384), buffer_size=100):
        self.video_path = video_path
        self.transform = transform
        self.resize = resize
        self.buffer_size = buffer_size

        # Initialize video capture and length
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Unable to open video file: {video_path}")
        self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"Video path: {self.video_path}, Total frames: {self.length}")

        # Variables for caching frames in each worker
        self.worker_buffer = {}
        self.worker_start_idx = None

    def _load_frames(self, start_idx):
        """Load a window of frames starting from `start_idx`."""
        frames = {}
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        for i in range(self.buffer_size):
            actual_idx = start_idx + i
            if actual_idx >= self.length:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Resize and convert frame to RGB
            org_size = (frame.shape[1], frame.shape[0])
            frame = cv2.resize(frame, self.resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Store the frame and its original size
            frames[actual_idx] = (frame, org_size)

        cap.release()
        return frames

    def _get_frame(self, idx):
        """Retrieve a frame from the buffer or load if necessary."""
        # Check if the index is within the buffer
        if self.worker_start_idx is None or idx < self.worker_start_idx or idx >= self.worker_start_idx + self.buffer_size:
            # Load a new set of frames into the buffer
            self.worker_buffer = self._load_frames(idx)
            self.worker_start_idx = idx

        # Ensure that the frame exists in the buffer
        if idx not in self.worker_buffer:
            raise KeyError(f"Frame at index {idx} could not be loaded into buffer.")

        return self.worker_buffer[idx]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of bounds for video length {self.length}")

        # Get frame and original size
        frame, org_size = self._get_frame(idx)

        # Convert frame to tensor and normalize
        frame = to_torch(frame).permute(2, 0, 1)
        frame = normalize(frame)

        # Apply transformations if provided
        if self.transform:
            frame = self.transform(frame)

        return {"frame_id": idx, "frame": frame, "org_size": torch.tensor(org_size)}


def get_dataloader(
    video_path,
    batch_size=1,
    resize=(384, 384),
    transform=None,
    num_workers=2,
    buffer_size=100,
):
    dataset = VideoFrameDataset(
        video_path, transform=transform, resize=resize, buffer_size=buffer_size
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    video_path = "videos/v_2324_223_s2_short.mp4"
    dataloader = get_dataloader(video_path, batch_size=16, buffer_size=200, num_workers=8)

    for frames in tqdm(dataloader):
        pass
        # for k, v in frames.items():
        #     print(f"{k}: {type(v)}")
        # break
