from torch.utils.data import Dataset, DataLoader, Subset
from glob import glob
from os import path
import numpy as np
from PIL import Image
import torch


class VideoDataset(Dataset):
    def __init__(self, ds_path, frame_transform, video_transform, num_frames: int = 16, test: bool = False, step_size: int = 1):
        self.test = test
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.ds_path = ds_path
        self.ids = glob(path.join(self.ds_path, '*'))
        self.labels = [i for i in range(len(self.ids))]
        self.num_frames = num_frames
        self.step_size = step_size
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        videos = glob(path.join(self.ids[idx], "*"))
        filtered_videos = []
        for video in videos:
            path2imgs = glob(path.join(video, "*.jpg"))
            if len(path2imgs) >= self.num_frames:
                filtered_videos.append(video)
        
        if len(filtered_videos) <= 0:
            return self[(idx+1) % len(self)]
        else:
            chosen_video_index = int(len(filtered_videos) * np.random.uniform())

            path2imgs = sorted(glob(path.join(filtered_videos[chosen_video_index], "*.jpg")))
            rand_sample = np.random.uniform()
            step_size = min(self.step_size, len(path2imgs) // self.num_frames)
            if self.test:
                rand_sample = 0
            start_idx = int((len(path2imgs) - self.num_frames * step_size) * rand_sample)
            path2imgs = path2imgs[start_idx : start_idx + self.num_frames * step_size : step_size]
            if self.num_frames * step_size > len(path2imgs):
                pass
            label = self.labels[idx]
            frames = []

            for p2i in path2imgs:
                frame = Image.open(p2i)
                frames.append(frame)
            frames_tr = []
            for frame in frames:
                frame = self.frame_transform(frame)
                frames_tr.append(frame)
            if len(frames_tr)>0:
                frames_tr = torch.stack(frames_tr)
                frames_tr = self.video_transform(frames_tr)
            return frames_tr, label