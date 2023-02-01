import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ConvertImageDtype, Pad, Resize, PILToTensor
from torchvision.transforms.functional import to_pil_image

from utils import create_disgraphia_splits, get_IAM_statistics, get_base_statistics, get_bhk_features

class IAMDL(Dataset):

    def __init__(self, set : str, device):
        assert set == 'testset' or set == 'trainset' or set == 'validationset'
        self.set = f"IAM/{set}"
        self.set_samples = self.__get_set_samples()
        self.max_width, self.max_height = get_IAM_statistics()
        self.device = device
    
    def __len__(self):
        return len(self.set_samples)
    
    def __getitem__(self, index):
        return self.set_samples[index]
    
    def __get_set_samples(self):
        set_samples = []
        for author in os.listdir(self.set):
            writings = os.path.join(self.set, author)
            for png in os.listdir(writings):
                    set_samples.append(os.path.join(writings, png))
        return set_samples
    
    def get_triplet(self, sample):
        pos_aut = '/'.join(sample.split("/")[:-1])
        anc_img = sample.split("/")[-1]
        pos_img = random.choice([a for a in os.listdir(pos_aut)])
        while(pos_img == anc_img):
            pos_img = random.choice([a for a in os.listdir(pos_aut)])

        neg_aut = os.path.join(self.set, random.choice([a for a in os.listdir(self.set)]))
        while(pos_aut == neg_aut):
            neg_aut = os.path.join(self.set, random.choice([a for a in os.listdir(self.set)]))
        neg_img = random.choice([a for a in os.listdir(neg_aut)])

        anchor_img = Image.open(os.path.join(pos_aut, anc_img))
        anchor_w, anchor_h = anchor_img.size
        transform = Compose([
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Pad((0, 0, self.max_width - anchor_w, self.max_height - anchor_h), fill=1.),
            Resize((128, 1024))
        ])
        anchor = transform(anchor_img)

        positive_img = Image.open(os.path.join(pos_aut, pos_img))
        positive_w, positive_h = positive_img.size
        transform = Compose([
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Pad((0, 0, self.max_width - positive_w, self.max_height - positive_h), fill=1.),
            Resize((128, 1024))
        ])
        positive = transform(positive_img)

        negative_img = Image.open(os.path.join(neg_aut, neg_img))
        negative_w, negative_h = negative_img.size
        transform = Compose([
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Pad((0, 0, self.max_width - negative_w, self.max_height - negative_h), fill=1.),
            Resize((128, 1024))
        ])
        negative = transform(negative_img)

        return anchor, positive, negative
    
    def batch_triplets(self, samples):
        
        batch_size = len(samples)
        anchors = torch.empty(size=(batch_size, 1, 128, 1024))
        positives = torch.empty(size=(batch_size, 1, 128, 1024))
        negatives = torch.empty(size=(batch_size, 1, 128, 1024))
        
        for batch, sample in enumerate(samples):
            anchors[batch], positives[batch], negatives[batch] = self.get_triplet(sample)
        
        return anchors.to(self.device), positives.to(self.device), negatives.to(self.device)

class DisgraphiaDL(Dataset):

    def __init__(self, base :  str, set : str, device : str, use_csv : bool = False):
        assert set == 'train' or set == 'validation' or set == 'test'
        assert base == 'children' or base == 'adults'
        create_disgraphia_splits(f'/home1/gemelli/dyslexia/data/{base}/png-lines')
        self.BASE = f"data/{base}"
        self.SET = os.path.join(self.BASE, f"{set}.txt")
        self.set_samples = self.__set_samples()
        self.max_width, self.max_height = get_base_statistics(base)
        self.device = device
        self.use_csv = use_csv
        if use_csv: _, self.pen_features = get_bhk_features()
    
    def __len__(self):
        return len(self.set_samples)
    
    def __getitem__(self, index):
        img = Image.open(self.set_samples[index]).convert('L')
        transform = Compose([
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Pad((0, 0, self.max_width - img.size[0], self.max_height - img.size[1]), fill=1.),
            Resize((192, 512)) # 192, 512
        ])
        img = transform(img)
        if 'O' in self.set_samples[index].split("/")[-2]: label = torch.tensor(0)
        else: label = torch.tensor(1)

        to_pil_image(img).save('prova.png')
        if self.use_csv:
            pen_features, _ = get_bhk_features(self.set_samples[index], self.BASE.split("/")[1])
        return img.to(self.device), label.to(self.device), pen_features.to(self.device)
    
    def __set_samples(self):
        set_samples = []
        set_authors = [line.rstrip('\n') for line in open(self.SET, 'r')]
        AUTHORS = os.path.join(self.BASE, 'png-lines')
        for author in os.listdir(AUTHORS):
            if author not in set_authors: continue
            LINES = os.path.join(AUTHORS, author)
            for png in os.listdir(LINES):
                    set_samples.append(os.path.join(LINES, png))

        return set_samples
    
    def get_binary_weights(self):
        counter = [0, 0]
        for sample in self.set_samples:
            if '0' in sample.split("/")[-2]: counter[0] += 1
            else: counter[1] += 1
        return torch.tensor([min(counter) / counter[0], min(counter) / counter[1]]).to(self.device)
