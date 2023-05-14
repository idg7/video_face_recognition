from torchvision import transforms, models
import os
import myutils

from torch.utils.data import Dataset, DataLoader, Subset
from os import path
from glob import glob
from PIL import Image
import torch
import numpy as np
import random
from torch import optim, nn, tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from vgg16decoder import Vgg16Decoder
from video_dataset import VideoDataset
import mlflow

# CUDA_LAUNCH_BLOCKING=1
def curriculum_training():
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


    PRETRAINED = False
    params_model={
        "num_classes": 100,
        "dr_rate": 0.5,
        "pretrained" : True,
        "rnn_num_layers": 4,
        "rnn_hidden_size": 512, 'train_seq': True, 'freeze_conv':True}
    model = Vgg16Decoder(params_model)

    # state_dict = torch.load('./models/weights_rnn.pt')
    # model.load_state_dict(state_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    path2weights = "./models/weights.pt"
    os.makedirs("./models", exist_ok=True)
    # torch.save(model.state_dict(), path2weights)
    # epochs = {5:20, 4:20, 3:40, 2:40, }
    start_epoch = 0
    for i, training_size in enumerate([1]): #enumerate(range(5, 0, -1)):
        train_ds = VideoDataset(f"/home/ssd_storage/datasets/processed/youtube_faces_curriculum_2/over_{training_size}_train_1_val/train", train_frame_transformer, train_video_transformer, num_frames=timesteps, step_size=4)
        test_ds = VideoDataset(f"/home/ssd_storage/datasets/processed/youtube_faces_curriculum_2/over_{training_size}_train_1_val/val", test_frame_transformer, test_video_transformer, num_frames=timesteps, test=True, step_size=4)
        
        print(len(train_ds))
        print(len(test_ds))
        model.replace_classifier(len(train_ds))

        # best_weights_pth = '/home/administrator/PycharmProjects/PyTorch-Computer-Vision-Cookbook/Chapter10/models/Curriculum learning - max videos regiment, pretrained-frozen - deep transformer (4 layers) with positional encodings - order fixed.pt'
        # state_dict = torch.load(best_weights_pth)
        # model.load_state_dict(state_dict)

        train_dl = DataLoader(train_ds, batch_size=batch_size,
                            shuffle=True, collate_fn=collate_fn_rnn, num_workers=4)
        test_dl = DataLoader(test_ds, batch_size=2*batch_size,
                            shuffle=False, collate_fn=collate_fn_rnn, num_workers=4)
        
        loss_func = nn.CrossEntropyLoss()#reduction="sum"
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
            }

        best_weights_pth = '/home/administrator/PycharmProjects/PyTorch-Computer-Vision-Cookbook/Chapter10/models/Max videos regiment, pretrained-frozen - deep transformer (4 layers) with positional encodings - with frame skipping.pt'

        state_dict = torch.load(best_weights_pth)
        model.load_state_dict(state_dict)
        
        model, loss_hist, metric_hist = myutils.train_val(model,params_train)
        start_epoch += num_epochs
        # myutils.plot_loss(loss_hist, metric_hist)




def main():
    model_type = "3dcnn"
    model_type = "rnn"

    timesteps = 32
    if model_type == "rnn":
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]

    import torchvision.transforms as transforms

    train_frame_transformer = transforms.Compose([
                transforms.Resize((256,256)),transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
                
    train_video_transformer = transforms.Compose([                
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(),
                transforms.RandomCrop([224,224]),
                ])
    #             transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),

    # train_ds = VideoDataset("/home/ssd_storage/datasets/processed/youtube_faces_half min_size=2_{'train': 0.5, 'val': 0.5}/train", train_frame_transformer, train_video_transformer)
    # train_ds = VideoDataset("/home/ssd_storage/datasets/processed/youtube_faces_4_half min_size=4_{'train': 0.5, 'val': 0.5}/train", train_frame_transformer, train_video_transformer)
    # train_ds = VideoDataset("/home/ssd_storage/datasets/processed/youtube_faces_3_half min_size=3_{'train': 1, 'val': 1}/train", train_frame_transformer, train_video_transformer)
    train_ds = VideoDataset("/home/ssd_storage/datasets/processed/youtube_faces_4_half min_size=3_{'train': 2, 'val': 1}/train", train_frame_transformer, train_video_transformer)
    
    print(len(train_ds))

    test_frame_transformer = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
    test_video_transformer = transforms.Compose([transforms.CenterCrop([224,224])])

    # test_ds = VideoDataset("/home/ssd_storage/datasets/processed/youtube_faces_half min_size=2_{'train': 0.5, 'val': 0.5}/val", test_frame_transformer, test_video_transformer)
    # test_ds = VideoDataset("/home/ssd_storage/datasets/processed/youtube_faces_4_half min_size=4_{'train': 0.5, 'val': 0.5}/val", test_frame_transformer, test_video_transformer)
    # test_ds = VideoDataset("/home/ssd_storage/datasets/processed/youtube_faces_3_half min_size=3_{'train': 1, 'val': 1}/val", test_frame_transformer, test_video_transformer)
    test_ds = VideoDataset("/home/ssd_storage/datasets/processed/youtube_faces_4_half min_size=3_{'train': 2, 'val': 1}/val", test_frame_transformer, test_video_transformer)
    
    print(len(test_ds))

    def collate_fn_r3d_18(batch):
        imgs_batch, label_batch = list(zip(*batch))
        imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
        label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
        imgs_tensor = torch.stack(imgs_batch)
        imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
        labels_tensor = torch.stack(label_batch)
        return imgs_tensor,labels_tensor

    def collate_fn_rnn(batch):
        imgs_batch, label_batch = list(zip(*batch))
        imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
        label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
        imgs_tensor = torch.stack(imgs_batch)
        labels_tensor = torch.stack(label_batch)
        return imgs_tensor,labels_tensor
        

    batch_size = 4
    if model_type == "rnn":
        train_dl = DataLoader(train_ds, batch_size= batch_size,
                            shuffle=True, collate_fn=collate_fn_rnn, num_workers=4)
        test_dl = DataLoader(test_ds, batch_size= 2*batch_size,
                            shuffle=False, collate_fn=collate_fn_rnn, num_workers=4)  
    else:
        train_dl = DataLoader(train_ds, batch_size= batch_size, 
                            shuffle=True, collate_fn=collate_fn_r3d_18, num_workers=4)
        test_dl = DataLoader(test_ds, batch_size= 2*batch_size, 
                            shuffle=False, collate_fn= collate_fn_r3d_18, num_workers=4)


    from torchvision import models
    from torch import nn

    # if model_type == "rnn":
    PRETRAINED = False
    params_model={
        "num_classes": len(train_ds),
        "dr_rate": 0.5,
        "pretrained" : True,
        "rnn_num_layers": 2,
        "rnn_hidden_size": 512, 'train_seq': True, 'freeze_conv':False}
    model = Vgg16Decoder(params_model)
    # state_dict = torch.load('./models/weights_rnn.pt')
    # model.load_state_dict(state_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    path2weights = "./models/weights.pt"
    os.makedirs("./models", exist_ok=True)
    # torch.save(model.state_dict(), path2weights)

    loss_func = nn.CrossEntropyLoss()#reduction="sum"
    opt = optim.Adam(model.parameters(), lr=1e-5)
    # opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=50,verbose=1)
    os.makedirs("./models", exist_ok=True)
    
    best_weights_pth = '/home/administrator/PycharmProjects/PyTorch-Computer-Vision-Cookbook/Chapter10/models/Max videos regiment, pretrained-frozen - deep transformer (4 layers) with positional encodings - with frame skipping.pt'

    state_dict = torch.load(best_weights_pth)
    model.load_state_dict(state_dict)
    
    params_train={
        "num_epochs": 0,
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
        }
    print("HERE!!!!!")
    model,loss_hist,metric_hist = myutils.train_val(model,params_train)

    myutils.plot_loss(loss_hist, metric_hist)

if __name__ == '__main__':
    MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000' #5000
    MLFLOW_ARTIFACT_STORE = '/home/hdd_storage/mlflow/artifact_store' #'/home/hdd_storage/mlflow/artifact_store'
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("HERE")
    if mlflow.get_experiment_by_name('Video face recognition') is None:
        mlflow.create_experiment('Video face recognition', artifact_location=os.path.join(MLFLOW_ARTIFACT_STORE, 'Video face recognition'))
    mlflow.set_experiment('Video face recognition')
    print("HERE")
    # run_name = '1 training video per cls'
    # run_name = '2 training video per cls'
    # run_name = '2 training video per cls, pretrained-finetuned'
    run_name = 'Curriculum learning - max videos regiment, pretrained-finetuned'
    run_name = 'Curriculum learning - max videos regiment, pretrained-frozen'
    run_name = 'Curriculum learning - max videos regiment, pretrained-frozen - deep transformer (4 layers) with positional encodings'
    run_name = 'Curriculum learning - max videos regiment, pretrained-frozen - deep transformer (4 layers) with positional encodings - order fixed'
    run_name = 'Classify movement'
    run_name = 'Max videos regiment, pretrained-frozen - deep transformer (4 layers) with positional encodings - with frame skipping'
    run_name = 'Max videos regiment, pretrained-frozen - deep transformer (4 layers) with positional encodings - with frame skipping - just tests'
    with mlflow.start_run(run_name=run_name):
        print("HERE")
        # main()
        curriculum_training()