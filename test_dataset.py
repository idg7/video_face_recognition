from torch.utils.data import Dataset, DataLoader, Subset
from glob import glob
from os import path
import numpy as np
from PIL import Image
import torch


class TestDataset(Dataset):
    def __init__(self, ds_path, frame_transform, video_transform, num_frames: int = 16, test: bool = False, step_size: int = 1):
        self.test = test
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.ds_path = ds_path
        self.ids = glob(path.join(self.ds_path, '*'))
        self.videos = glob(path.join(self.ds_path, '*', '*'))
        self.labels = [i for i in range(len(self.ids))]
        self.num_frames = num_frames
        self.step_size = step_size
        
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video = self.videos[idx]
        
        key = path.relpath(video, self.ds_path)

        path2imgs = sorted(glob(path.join(video, "*.jpg")))
        step_size = min(self.step_size, len(path2imgs) // self.num_frames)
        if self.test:
            rand_sample = 0
        start_idx = int((len(path2imgs) - self.num_frames * step_size) * rand_sample)
        path2imgs = path2imgs[start_idx : start_idx + self.num_frames * step_size : step_size]
        if self.num_frames * step_size > len(path2imgs):
            pass
        frames = []

        for p2i in path2imgs:
            frame = Image.open(p2i)
            frames.append(frame)
        frames_tr = []
        for frame in frames:
            frame = self.frame_transform(frame)
            frames_tr.append(frame)
        if len(frames_tr) > 0:
            frames_tr = torch.stack(frames_tr)
            frames_tr = self.video_transform(frames_tr)
        return frames_tr, key