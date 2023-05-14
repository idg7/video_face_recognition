import os
import torch
import copy
from tqdm import tqdm_notebook, tqdm
from torchvision.transforms.functional import to_pil_image
import matplotlib.pylab as plt
from tqdm import tqdm_notebook
import mlflow

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_vids(path2ajpgs):
    listOfCats = os.listdir(path2ajpgs)
    ids = []
    labels = []
    for catg in listOfCats:
        path2catg = os.path.join(path2ajpgs, catg)
        listOfSubCats = os.listdir(path2catg)
        path2subCats= [os.path.join(path2catg,los) for los in listOfSubCats]
        ids.extend(path2subCats)
        labels.extend([catg]*len(listOfSubCats))
    return ids, labels, listOfCats 

def denormalize(x_, mean, std):
    x = x_.clone()
    for i in range(3):
        x[i] = x[i]*std[i]+mean[i]
    x = to_pil_image(x)        
    return x

def train_val(model, params):
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
    
    loss_history={
        "train": [],
        "val": [],
        "val - shuffled TS": [],
        "val - shuffled ID": []
    }
    
    metric_history={
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
        if train_seq:
            train_loss, train_metric=loss_epoch_seq(model,loss_func,train_dl,sanity_check,opt)
        else:
            if transformer:
                train_loss, train_metric=loss_epoch_frame_decoder(model,loss_func,val_dl,sanity_check)
            else:
                train_loss, train_metric=loss_epoch_frame(model,loss_func,val_dl,sanity_check)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        mlflow.log_metric('Train loss', train_loss, epoch)
        mlflow.log_metric('Train accuracy', 100*train_metric, epoch)
        print("train loss: %.6f, train accuracy: %.2f" %(train_loss,100*train_metric))
        if ((epoch - start_epoch) % val_freq) == 0:
            model.eval()
            with torch.no_grad():
                if train_seq:
                    val_loss, val_metric=loss_epoch_seq(model,loss_func,val_dl,sanity_check)
                else:
                    if transformer:
                        val_loss, val_metric=loss_epoch_frame_decoder(model,loss_func,val_dl,sanity_check)
                    else:
                        val_loss, val_metric=loss_epoch_frame(model,loss_func,val_dl,sanity_check)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), path2weights)
                print("Copied best model weights!")
            
            loss_history["val"].append(val_loss)
            metric_history["val"].append(val_metric)
            mlflow.log_metric('Val loss', val_loss, epoch)
            mlflow.log_metric('Val accuracy', 100*val_metric, epoch)
            lr_scheduler.step(val_loss)
            if current_lr != get_lr(opt):
                print("Loading best model weights!")
                model.load_state_dict(best_model_wts)
            

            print("train loss: %.6f, dev loss: %.6f, train accuracy: %.2f, dev accuracy: %.2f" %(train_loss,val_loss,100*train_metric,100*val_metric))
            print("-"*10)
    model.eval()
    with torch.no_grad():
        if train_seq:
            val_loss, val_metric=loss_epoch_seq(model,loss_func,val_dl,sanity_check)
            print("Val loss: %.6f, Val acc: %.6f" %(val_loss,100*val_metric))
            model.switch_shuffle_dynamic_features()
            val_loss, val_metric=loss_epoch_seq(model,loss_func,val_dl,sanity_check)
            print("Shuffled dynamics loss: %.6f, Shuffled dynamics acc: %.6f" %(val_loss,100*val_metric))
            model.switch_shuffle_dynamic_features()
            # val_loss, val_metric=loss_epoch_seq(model,loss_func,val_dl,sanity_check)
            con_val_loss, con_val_metric = loss_epoch_seq_check_continuous_motion(model,loss_func,val_dl,sanity_check)
            print("Shuffled TS loss: %.6f, Shuffled TS acc: %.6f" %(con_val_loss,100*con_val_metric))

            model.switch_dynamic_features()
            con_val_loss, con_val_metric = loss_epoch_seq(model,loss_func,val_dl,sanity_check)
            print("No dynamics: %.6f, No dynamics: %.6f" %(con_val_loss,100*con_val_metric))
            model.switch_dynamic_features()

            model.switch_shuffle_dynamic_features()
            con_val_loss, con_val_metric = loss_epoch_seq_check_continuous_motion(model,loss_func,val_dl,sanity_check)
            print("Shuffled dynamics + TS loss: %.6f, Shuffled dynamics + TS acc: %.6f" %(con_val_loss,100*con_val_metric))
            model.switch_shuffle_dynamic_features()

            id_val_loss, id_val_metric = loss_epoch_seq_check_frame_identities(model,loss_func,val_dl,sanity_check)
            print("Shuffled ID loss: %.6f, Shuffled ID acc: %.6f" %(id_val_loss,100*id_val_metric))
        else:
            if transformer:
                val_loss, val_metric=loss_epoch_frame_decoder(model,loss_func,val_dl,sanity_check)
            else:
                val_loss, val_metric=loss_epoch_frame(model,loss_func,val_dl,sanity_check)
    
    
    print("-"*10)

    loss_history["val - shuffled TS"].append(con_val_loss)
    metric_history["val - shuffled TS"].append(con_val_metric)
    mlflow.log_metric('Val (shuffled TS) loss', con_val_loss, epoch)
    mlflow.log_metric('Val (shuffled TS) accuracy', 100*con_val_metric, epoch)

    loss_history["val - shuffled ID"].append(id_val_loss)
    metric_history["val - shuffled ID"].append(id_val_metric)
    mlflow.log_metric('Val (shuffled ID) loss', id_val_loss, epoch)
    mlflow.log_metric('Val (shuffled ID) accuracy', 100*id_val_metric, epoch)

    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history

# get learning rate 
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    with torch.no_grad():
        metric_b = metrics_batch(output,target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b
    

def loss_epoch_seq(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)
    pbar = tqdm(dataset_dl)
    for xb, yb in pbar:
        xb=xb.to(device)
        yb=yb.to(device)
        output=model(xb)
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt)
        pbar.set_description("loss: %.6f, Corrects: %.4f" % (loss_b, metric_b))
        running_loss+=loss_b
        
        if metric_b is not None:
            running_metric+=metric_b
        if sanity_check is True:
            break
    loss=running_loss/float(len_data)
    metric=running_metric/float(len_data)
    return loss, metric

def loss_epoch_seq_check_continuous_motion(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)
    pbar = tqdm(dataset_dl)
    for xb, yb in pbar:
        xb=xb.to(device)
        yb=yb.to(device)
        t_order = torch.randperm(xb.shape[1])
        xb = xb[:,t_order]
        output=model(xb)
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt)
        pbar.set_description("Loss: %.6f, Corrects: %.4f" % (loss_b, metric_b))
        running_loss+=loss_b
        
        if metric_b is not None:
            running_metric+=metric_b
        if sanity_check is True:
            break
    loss=running_loss/float(len_data)
    metric=running_metric/float(len_data)
    return loss, metric

def loss_epoch_seq_check_frame_identities(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)
    pbar = tqdm(dataset_dl)
    for xb, yb in pbar:
        xb=xb.to(device)
        yb=yb.to(device)
        xb = torch.transpose(xb, 0, 1)
        output=model(xb)
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt)
        pbar.set_description("Loss: %.6f, Corrects: %.4f" % (loss_b, metric_b))
        running_loss+=loss_b
        
        if metric_b is not None:
            running_metric+=metric_b
        if sanity_check is True:
            break
    loss=running_loss/float(len_data)
    metric=running_metric/float(len_data)
    return loss, metric


def loss_epoch_frame_decoder(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)
    pbar = tqdm(dataset_dl)
    i = 1
    for xb, yb in pbar:
        xb=xb.to(device)
        yb=yb.to(device)
        bs, ts, _, _, _ = xb.shape
        h = None
        c = None
        for t in range(ts):
            output = model(xb[:, : t + 1])
            loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
            pbar.set_description("loss: %.6f, Corrects: %.4f" % (loss_b, metric_b))
            running_loss += loss_b
            
            if metric_b is not None:
                running_metric += metric_b
            if sanity_check is True:
                break
        i += 1

    denominator = float(len_data) * float(ts)
    loss = running_loss / denominator
    metric = running_metric / denominator
    return loss, metric


def loss_epoch_frame(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)
    pbar = tqdm(dataset_dl)
    i = 1
    for xb, yb in pbar:
        xb=xb.to(device)
        yb=yb.to(device)
        bs, ts, _, _, _ = xb.shape
        h = None
        c = None
        for t in range(ts):
            output, h, c = model(xb[:, t], h, c)
            loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
            pbar.set_description("loss: %.6f, Corrects: %.4f" % (loss_b, metric_b))
            running_loss += loss_b
            
            if metric_b is not None:
                running_metric += metric_b
            if sanity_check is True:
                break
        i += 1

    denominator = float(len_data) * float(ts)
    loss = running_loss / denominator
    metric = running_metric / denominator
    return loss, metric


def plot_loss(loss_hist, metric_hist):

    num_epochs= len(loss_hist["train"])

    plt.title("Train-Val Loss")
    plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
    plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()

    plt.title("Train-Val Accuracy")
    plt.plot(range(1,num_epochs+1), metric_hist["train"],label="train")
    plt.plot(range(1,num_epochs+1), metric_hist["val"],label="val")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()

#%%
    
from torch import nn
class Resnt18Rnn(nn.Module):
    def __init__(self, params_model):
        super(Resnt18Rnn, self).__init__()
        num_classes = params_model["num_classes"]
        dr_rate= params_model["dr_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]
        
        baseModel = models.resnet18(pretrained=pretrained)
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)
    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        ii = 0
        y = self.baseModel((x[:,ii]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, ts):
            y = self.baseModel((x[:,ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:,-1])
        out = self.fc1(out) 
        return out 
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x    


from torchvision import models
from torch import nn

def get_model(num_classes, model_type="rnn"):
    if model_type == "rnn":
        params_model={
            "num_classes": num_classes,
            "dr_rate": 0.1,
            "pretrained" : True,
            "rnn_num_layers": 1,
            "rnn_hidden_size": 100,}
        model = Resnt18Rnn(params_model)        
    else:
        model = models.video.r3d_18(pretrained=True, progress=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)    
    return model


import cv2
import numpy as np
def get_frames(filename, n_frames= 1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list= np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frames.append(frame)
    v_cap.release()
    return frames, v_len


import torchvision.transforms as transforms
from PIL import Image
def transform_frames(frames, model_type="rnn"):
    if model_type == "rnn":
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]

    test_transformer = transforms.Compose([
                transforms.Resize((h,w)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]) 

    frames_tr = []
    for frame in frames:
        frame = Image.fromarray(frame)
        frame_tr = test_transformer(frame)
        frames_tr.append(frame_tr)
    imgs_tensor = torch.stack(frames_tr)    

    if model_type=="3dcnn":
        imgs_tensor = torch.transpose(imgs_tensor, 1, 0)
    imgs_tensor = imgs_tensor.unsqueeze(0)
    return imgs_tensor


def store_frames(frames, path2store):
    for ii, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        path2img = os.path.join(path2store, "frame"+str(ii)+".jpg")
        cv2.imwrite(path2img, frame)
