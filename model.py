import torch
import torch.nn as nn
import shutil
from vit_pytorch import SimpleViT
import os
from torchvision.models import resnet18, ResNet18_Weights
from torchsummary import summary

from path import *

class ViTWrapper():
    def __init__(self, name = 'vit', device = 'cpu', classes = 2, pretrain = False, pen_features : int = 0):
        self.model = SimpleViT(
            image_size = 1024,
            patch_size = 32,
            num_classes = classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            channels = 1
        )
        self.name = name
        self.cls = classes
        self.device = device
        self.model.linear_head = nn.Identity()
        self.pen_features = pen_features
        if not pretrain:
            self.load_state(s = 'vit_model_best.pth')
            self.model.linear_head = self.__set_head(classes)
        self.model = self.model.to(device)
        return
    
    def __set_head(self, cls):
        return nn.Sequential(
            nn.Linear(1024, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, cls)
        )

    def binary(self):
        self.cls = 1
        self.model.linear_head[4] = nn.Linear(100, 1)
        self.model.to(self.device)
    
    def set_csv_model(self, base):
        if self.pen_features != 0:
            self.model = ModelCSV(base, self.model, self.pen_features, self.cls, self.device)
            print("CSV Model!")
        else:
            print("Cannot use CSV Model! No Pen Features loaded.")
        return
    
    def get_model(self):
        return self.model
        
    def save_state(self, state, is_best):
        out = os.path.join(CHECKPOINTS, f'{self.name}_checkpoint.pth')
        torch.save(state, out)
        if is_best:
            shutil.copyfile(out, os.path.join(CHECKPOINTS,f'{self.name}_model_best.pth'))
    
    def load_state(self, s):
        s = torch.load(os.path.join(CHECKPOINTS, s))
        self.model.load_state_dict(s['state_dict'])
    
    def resume(self, r):
        print(f"=> loading checkpoint '{r}'")
        c = torch.load(os.path.join(CHECKPOINTS, r))
        self.load_state(c['state_dict'])
        return c['epoch'], c['best_val_loss'], c['exit_counter'], c['optimizer'], c['best_val_f1']

class ResnetWrapper():
    def __init__(self, name : str = 'resnet18', device : str = 'cpu', classes : int = 2, pretrain : bool = False, pen_features : int = 0):
        self.model = resnet18(ResNet18_Weights.DEFAULT)
        self.name = name
        self.device = device
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Identity()
        self.pen_features = pen_features
        self.cls = classes
        if not pretrain:
            self.load_state(s = 'resnet18_model_best.pth')
            self.model.fc = self.__set_head(classes)
        self.model = self.model.to(device)
        return
    
    def __set_head(self, cls):
        return nn.Sequential(
            nn.Linear(512, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, cls)
        )

    def binary(self):
        self.cls = 1
        self.model.fc[4] = nn.Linear(100, 1)
        self.model.to(self.device)

    def freeze(self):
        print("BRRRRRRRRRRRRRRRRR!")
        for name, param in self.model.named_parameters():
            if 'model.fc' in name: continue
            if 'classify' in name: continue
            param.requires_grad = False
    
    def get_model(self):
        return self.model
    
    def set_csv_model(self, base):
        if self.pen_features != 0:
            self.model = ModelCSV(base, self.model, self.pen_features, self.cls, self.device)
            print("CSV Model!")
        else:
            print("Cannot use CSV Model! No Pen Features loaded.")
        return
        
    def save_state(self, state, is_best):
        out = os.path.join(CHECKPOINTS, f'{self.name}_checkpoint.pth')
        torch.save(state, out)
        if is_best:
            shutil.copyfile(out, os.path.join(CHECKPOINTS,f'{self.name}_model_best.pth'))
    
    def load_state(self, s):
        s = torch.load(os.path.join(CHECKPOINTS, s))
        self.model.load_state_dict(s['state_dict'])
    
    def resume(self, r):
        print(f"=> loading checkpoint '{r}'")
        c = torch.load(os.path.join(CHECKPOINTS, r))
        self.load_state(c['state_dict'])
        return c['epoch'], c['best_val_loss'], c['exit_counter'], c['optimizer']

class ModelCSV(nn.Module):
    def __init__(self, name, model, pen_features, cls, device):
        super(ModelCSV, self).__init__()
        self.model = model
        self.device = device
        if name == 'resnet': 
            self.model.fc[4] = nn.Identity()
        elif name == 'vit': 
            self.model.linear_head[4] = nn.Identity()
        self.classify = nn.Linear(100 + pen_features, cls)
        self.model.to(device), self.classify.to(device)
    
    def forward(self, img, pfeat):
        x = self.model(img)
        x = torch.cat((x, pfeat), dim=1).to(self.device)
        return self.classify(x)
