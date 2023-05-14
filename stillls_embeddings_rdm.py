from torchvision import transforms, models
import os

import copy
from tqdm import tqdm_notebook, tqdm
from torchvision.transforms.functional import to_pil_image
import matplotlib.pylab as plt

from torch.utils.data import Dataset, DataLoader, Subset
from os import path
from glob import glob
from PIL import Image
import torch
import numpy as np
import random
from torch import optim, nn, tensor, Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from vgg16decoder import Vgg16Decoder, DecoderOnly
from video_dataset import VideoDataset
from test_dataset import TestDataset
import mlflow
import pandas as pd
from myutils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def video_mean_emb_rdm(img_encoder: nn.Module, dataset_dl: torch.utils.data.DataLoader) -> pd.DataFrame:
    running_loss = 0.0
    len_data = len(dataset_dl.dataset)
    pbar = tqdm(dataset_dl)
    time = None
    mean_embeddings = []
    keys = []
    for x, key in pbar:
        x = x.to(device)
        keys = keys + key
        
        bs, ts, c, h, w = x.shape
        time = ts
        frame_features = []

        x = x.reshape((bs * ts, c, h, w))
        features = img_encoder(x)
        frame_features = features.reshape((bs, ts, -1))
        
        mean_embeddings.append(torch.mean(frame_features, dim=1))

    print(keys)

    mean_embeddings = torch.cat(mean_embeddings)
    mean_embeddings = mean_embeddings / mean_embeddings.norm(dim=1, p=2)[:, None]
    rdm = 1 - torch.mm(mean_embeddings, mean_embeddings.transpose(0, 1))

    rdm = pd.DataFrame(rdm.cpu(), index=keys, columns=keys)
    return rdm


def next_frame_training():
    model_type = "rnn"

    timesteps = 16

    import torchvision.transforms as transforms

    train_frame_transformer = transforms.Compose([
                transforms.Resize((256,256)),transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
                
    train_video_transformer = transforms.Compose([                
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(),
                transforms.RandomCrop([224,224]),
                ])

    test_frame_transformer = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
    test_video_transformer = transforms.Compose([transforms.CenterCrop([224,224])])

    def collate_fn_rnn(batch):
        imgs_batch, label_batch = list(zip(*batch))
        imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
        label_batch = [l for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
        imgs_tensor = torch.stack(imgs_batch)
        labels_tensor = label_batch #torch.stack(label_batch)
        return imgs_tensor,labels_tensor

    base_model = models.vgg16(num_classes=8749)
    base_model.features = torch.nn.DataParallel(base_model.features)
    model_checkpoint = torch.load('/home/administrator/experiments/familiarity/pretraining/vgg16/models/119.pth')
    base_model.load_state_dict(model_checkpoint['state_dict'])
    num_features = base_model.classifier[-1].in_features
    del base_model.classifier[-1]
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()
    base_model.cuda()

    path2weights = "./models/weights.pt"
    os.makedirs("./models", exist_ok=True)

    test_ds = TestDataset(f"/home/ssd_storage/datasets/motion_seminar/mtcnn/video_frames", test_frame_transformer, test_video_transformer, num_frames=17, test=True, step_size=5)

    rdm_dl = DataLoader(test_ds, batch_size=1,
                            shuffle=True, collate_fn=collate_fn_rnn, num_workers=4)

    with torch.no_grad():
        rdm = video_mean_emb_rdm(base_model, rdm_dl)
        rdm.to_csv('/home/administrator/PycharmProjects/PyTorch-Computer-Vision-Cookbook/Chapter10/results/video_mean_emb_rdm.csv')
        mlflow.log_artifact('/home/administrator/PycharmProjects/PyTorch-Computer-Vision-Cookbook/Chapter10/results/video_mean_emb_rdm.csv')

    

if __name__ == '__main__':
    MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000' #5000
    MLFLOW_ARTIFACT_STORE = '/home/hdd_storage/mlflow/artifact_store' #'/home/hdd_storage/mlflow/artifact_store'
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("HERE")
    if mlflow.get_experiment_by_name('Video face recognition') is None:
        mlflow.create_experiment('Video face recognition', artifact_location=os.path.join(MLFLOW_ARTIFACT_STORE, 'Video face recognition'))
    mlflow.set_experiment('Video face recognition')
    print("HERE")
    run_name = 'Mean embedding RDM'
    with mlflow.start_run(run_name=run_name):
        print("HERE")
        # main()
        next_frame_training()