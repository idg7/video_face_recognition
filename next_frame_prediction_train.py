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
import mlflow
from myutils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_val_next_frame_pred(model, params):
    start_epoch=params["start_epoch"]
    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]
    train_seq=params["train_seq"]
    val_freq=params["val_freq"]
    transformer = params["transformer"]
    img_encoder = params["img_encoder"]
    
    loss_history={
        "train": [],
        "val": [],
        "val - shuffled TS": [],
        "val - shuffled ID": []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf')
    epoch = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format((epoch - start_epoch), num_epochs - 1, current_lr))
        model.train()
        train_loss = loss_epoch_next_frame_prediction(img_encoder, model,loss_func,train_dl,sanity_check,opt)
        
        loss_history["train"].append(train_loss)
        mlflow.log_metric('Train loss', train_loss, epoch)
        print("train loss: %.6f" % (train_loss))
        if ((epoch - start_epoch) % val_freq) == 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_epoch_next_frame_prediction(img_encoder, model, loss_func, val_dl, sanity_check)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), path2weights)
                print("Copied best model weights!")
            
            loss_history["val"].append(val_loss)
            mlflow.log_metric('Val loss', val_loss, epoch)
            lr_scheduler.step(val_loss)
            if current_lr != get_lr(opt):
                print("Loading best model weights!")
                model.load_state_dict(best_model_wts)
            print("train loss: %.6f, dev loss: %.6f" % (train_loss,val_loss))
            print("-"*10)
    
    model.eval()
    with torch.no_grad():
        print("TRAIN")
        train_loss = loss_epoch_next_frame_prediction(img_encoder, model, loss_func, train_dl, sanity_check)
        print("train loss: %.6f" % (train_loss))
        val_loss = loss_epoch_next_frame_prediction_shuffled_ts(img_encoder, model, loss_func, train_dl, sanity_check, mode='shuffle frame order')
        print("Shuffled frame order train loss: %.6f" % (val_loss))
        val_loss = loss_epoch_next_frame_prediction_shuffled_ts(img_encoder, model, loss_func, train_dl, sanity_check, mode='shuffle dynamic features')
        print("Shuffled dynamic features train loss: %.6f" % (val_loss))
        val_loss = loss_epoch_next_frame_prediction_shuffled_ts(img_encoder, model, loss_func, train_dl, sanity_check, mode='only dynamic features')
        print("only dynamic features train loss: %.6f" % (val_loss))
        val_loss = loss_epoch_next_frame_prediction_shuffled_ts(img_encoder, model, loss_func, train_dl, sanity_check, mode='only dynamic shuffled features')
        print("only dynamic features train loss: %.6f" % (val_loss))


        print("\n\n\nVAL")
        val_loss = loss_epoch_next_frame_prediction(img_encoder, model, loss_func, val_dl, sanity_check)
        print("dev loss: %.6f" % (val_loss))
        val_loss = loss_epoch_next_frame_prediction_shuffled_ts(img_encoder, model, loss_func, val_dl, sanity_check, mode='shuffle frame order')
        print("Shuffled frame order dev loss: %.6f" % (val_loss))
        val_loss = loss_epoch_next_frame_prediction_shuffled_ts(img_encoder, model, loss_func, val_dl, sanity_check, mode='shuffle dynamic features')
        print("Shuffled dynamic features dev loss: %.6f" % (val_loss))
        val_loss = loss_epoch_next_frame_prediction_shuffled_ts(img_encoder, model, loss_func, val_dl, sanity_check, mode='only dynamic features')
        print("only dynamic features dev loss: %.6f" % (val_loss))
        val_loss = loss_epoch_next_frame_prediction_shuffled_ts(img_encoder, model, loss_func, val_dl, sanity_check, mode='only dynamic shuffled features')
        print("only dynamic shuffled features dev loss: %.6f" % (val_loss))

        print("\n\n\nMean embedding MSE")
        val_loss = loss_epoch_next_frame_prediction_mean_feat_target(img_encoder, model, loss_func, val_dl, sanity_check)
        print("dev loss: %.6f" % (val_loss))
        val_loss = loss_epoch_next_frame_prediction_mean_feat_target(img_encoder, model, loss_func, val_dl, sanity_check, mode='shuffle frame order')
        print("Shuffled frame order dev loss: %.6f" % (val_loss))
        val_loss = loss_epoch_next_frame_prediction_mean_feat_target(img_encoder, model, loss_func, val_dl, sanity_check, mode='shuffle dynamic features')
        print("Shuffled dynamic features dev loss: %.6f" % (val_loss))
        val_loss = loss_epoch_next_frame_prediction_mean_feat_target(img_encoder, model, loss_func, val_dl, sanity_check, mode='only dynamic features')
        print("only dynamic features dev loss: %.6f" % (val_loss))
        val_loss = loss_epoch_next_frame_prediction_mean_feat_target(img_encoder, model, loss_func, val_dl, sanity_check, mode='only dynamic shuffled features')
        print("only dynamic shuffled features dev loss: %.6f" % (val_loss))
    
    model.load_state_dict(best_model_wts)
        
    return model, loss_history


def loss_epoch_next_frame_prediction(img_encoder: nn.Module, model: nn.Module, loss_func: nn.Module, dataset_dl: torch.utils.data.DataLoader, sanity_check: bool = False, opt: torch.optim.Optimizer = None):
    running_loss=0.0
    len_data = len(dataset_dl.dataset)
    pbar = tqdm(dataset_dl)
    time = None
    for x, _ in pbar:
        # if torch.cuda.is_available()
        x = x.to(device)
        
        bs, ts, c, h, w = x.shape
        time = ts
        frame_features = []
        x = x.reshape((bs * ts, c, h, w))
        features = img_encoder(x)
        frame_features = features.reshape((bs, ts, -1))

        for t in range(1, ts - 1):
            output = model(frame_features[:, : t, :])
            loss = loss_func(output, frame_features[:, t + 1])
            
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            loss_b = loss.item()
            pbar.set_description("loss: %.6f" % (loss_b))
            running_loss += loss_b
            
            if sanity_check is True:
                break
            
    loss=running_loss/float(len_data * time)
    return loss


def remove_static_features(feats: Tensor):
    id_mean = torch.mean(feats, dim=1)
    feats = feats - id_mean.unsqueeze(1)
    return feats
    
def shuffle_dynamic_features(feats: Tensor):
    id_mean = torch.mean(feats, dim=1).unsqueeze(1)
    dynamic_features = feats - id_mean
    feats = id_mean + dynamic_features[torch.randperm(id_mean.shape[0])]
    return feats

def shuffle_frame_order(feats: Tensor):
    feats = feats[:, torch.randperm(feats.shape[1])]
    return feats


def loss_epoch_next_frame_prediction_shuffled_ts(img_encoder: nn.Module, model: nn.Module, loss_func: nn.Module, dataset_dl: torch.utils.data.DataLoader, sanity_check: bool = False, opt: torch.optim.Optimizer = None, mode: str = None):
    running_loss=0.0
    len_data = len(dataset_dl.dataset)
    pbar = tqdm(dataset_dl)
    time = None
    for x, _ in pbar:
        # if torch.cuda.is_available()
        x = x.to(device)
        
        bs, ts, c, h, w = x.shape
        time = ts
        frame_features = []

        x = x.reshape((bs * ts, c, h, w))
        features = img_encoder(x)
        frame_features = features.reshape((bs, ts, -1))
        if mode == 'shuffle frame order':
            frame_features = shuffle_frame_order(frame_features)
        if mode == 'shuffle dynamic features':
            frame_features = shuffle_dynamic_features(frame_features)
        if mode == 'only dynamic features':
            frame_features = remove_static_features(frame_features)
        if mode == 'only dynamic shuffled features':
            frame_features = remove_static_features(frame_features)
            frame_features = shuffle_frame_order(frame_features)

        for t in range(1, ts - 1):
            output = model(frame_features[:, : t, :])
            loss = loss_func(output, frame_features[:, t + 1])
            
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            loss_b = loss.item()
            pbar.set_description("loss: %.6f" % (loss_b))
            running_loss += loss_b
            
            if sanity_check is True:
                break
            
    loss=running_loss/float(len_data * time)
    return loss


def loss_epoch_next_frame_prediction_mean_feat_target(img_encoder: nn.Module, model: nn.Module, loss_func: nn.Module, dataset_dl: torch.utils.data.DataLoader, sanity_check: bool = False, opt: torch.optim.Optimizer = None, mode: str = None):
    running_loss=0.0
    len_data = len(dataset_dl.dataset)
    pbar = tqdm(dataset_dl)
    time = None
    for x, _ in pbar:
        # if torch.cuda.is_available()
        x = x.to(device)
        
        bs, ts, c, h, w = x.shape
        time = ts
        frame_features = []

        x = x.reshape((bs * ts, c, h, w))
        features = img_encoder(x)
        frame_features = features.reshape((bs, ts, -1))
        if mode == 'shuffle frame order':
            frame_features = shuffle_frame_order(frame_features)
        if mode == 'shuffle dynamic features':
            frame_features = shuffle_dynamic_features(frame_features)
        if mode == 'only dynamic features':
            frame_features = remove_static_features(frame_features)
        if mode == 'only dynamic shuffled features':
            frame_features = remove_static_features(frame_features)
            frame_features = shuffle_frame_order(frame_features)

        for t in range(1, ts - 1):
            output = model(frame_features[:, : t])
            loss = loss_func(output, torch.mean(frame_features[:, : t], dim=1))
            
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            loss_b = loss.item()
            pbar.set_description("loss: %.6f" % (loss_b))
            running_loss += loss_b
            
            if sanity_check is True:
                break
            
    loss=running_loss/float(len_data * time)
    return loss


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
        label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
        imgs_tensor = torch.stack(imgs_batch)
        labels_tensor = torch.stack(label_batch)
        return imgs_tensor,labels_tensor

    batch_size = 4

    base_model = models.vgg16(num_classes=8749)
    base_model.features = torch.nn.DataParallel(base_model.features)
    model_checkpoint = torch.load('/home/administrator/experiments/familiarity/pretraining/vgg16/models/119.pth')
    base_model.load_state_dict(model_checkpoint['state_dict'])
    num_features = base_model.classifier[-1].in_features
    del base_model.classifier[-1]
    for param in base_model.parameters():
        param.requires_grad = False

    
    params_model={
        "dim": num_features,
        "dr_rate": 0.5,
        "pretrained" : True,
        "rnn_num_layers": 8,
        "rnn_hidden_size": 2048, 'train_seq': True, 'freeze_conv':True}
    
    model = DecoderOnly(params_model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    base_model = base_model.to(device)

    path2weights = "./models/weights.pt"
    os.makedirs("./models", exist_ok=True)

    start_epoch = 0
    for i, training_size in enumerate([1]): #enumerate(range(5, 0, -1)):
        train_ds = VideoDataset(f"/home/ssd_storage/datasets/processed/youtube_faces_curriculum_2/over_{training_size}_train_1_val/train", train_frame_transformer, train_video_transformer, num_frames=timesteps, step_size=5)
        test_ds = VideoDataset(f"/home/ssd_storage/datasets/processed/youtube_faces_curriculum_2/over_{training_size}_train_1_val/val", test_frame_transformer, test_video_transformer, num_frames=timesteps, test=True, step_size=5)
        print('loading model...')
        best_weights_pth = "/home/administrator/PycharmProjects/PyTorch-Computer-Vision-Cookbook/Chapter10/models/Next frame prediction - L-D - with skip frames.pt"
        state_dict = torch.load(best_weights_pth)
        model.load_state_dict(state_dict)

        print(len(train_ds))
        print(len(test_ds))

        train_dl = DataLoader(train_ds, batch_size=batch_size,
                            shuffle=True, collate_fn=collate_fn_rnn, num_workers=4)
        test_dl = DataLoader(test_ds, batch_size=2*batch_size,
                            shuffle=False, collate_fn=collate_fn_rnn, num_workers=4)
        
        loss_func = nn.MSELoss()#reduction="sum"
        opt = optim.Adam(model.parameters(), lr=1e-5)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=50,verbose=1)
        os.makedirs("./models", exist_ok=True)
        
        num_epochs = 21
        if training_size == 3:
            num_epochs = 21
        if training_size == 2:
            num_epochs = 81
        if training_size <= 1:
            num_epochs = 0#101
        mlflow.log_metric('Curriculum step', (100 * (i+1) / 5), start_epoch)
        # model.should_use_static_features(False)
        params_train={
            "start_epoch": start_epoch,
            "num_epochs": num_epochs,
            "optimizer": opt,
            "loss_func": loss_func,
            "train_dl": train_dl,
            "val_dl": test_dl,
            "sanity_check": False,
            "lr_scheduler": lr_scheduler,
            "path2weights": f"./models/{run_name}.pt",
            'train_seq': True,
            'val_freq': 5,
            'transformer': True,
            'img_encoder': base_model,
            }
        
        model, loss_hist = train_val_next_frame_pred(model, params_train)
        start_epoch += num_epochs
        # myutils.plot_loss(loss_hist, metric_hist)


if __name__ == '__main__':
    MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000' #5000
    MLFLOW_ARTIFACT_STORE = '/home/hdd_storage/mlflow/artifact_store' #'/home/hdd_storage/mlflow/artifact_store'
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("HERE")
    if mlflow.get_experiment_by_name('Video face recognition') is None:
        mlflow.create_experiment('Video face recognition', artifact_location=os.path.join(MLFLOW_ARTIFACT_STORE, 'Video face recognition'))
    mlflow.set_experiment('Video face recognition')
    print("HERE")
    run_name = 'Next frame prediction'
    run_name = 'Next frame prediction - L-D - with skip frames'
    with mlflow.start_run(run_name=run_name):
        print("HERE")
        # main()
        next_frame_training()