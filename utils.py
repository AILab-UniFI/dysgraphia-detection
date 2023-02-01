import os
import torch
import shutil
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.transforms.functional import pil_to_tensor, resize
import random
import pandas as pd
import numpy as np
random.seed(42)

SETS = ['trainset.txt', 'validationset.txt', 'testset.txt']
CSVS = 'data/csv'

def get_IAM_statistics():
    h = []
    w = []
    # images = []
    for set in SETS:
        set_dir = f'IAM/{set.split(".")[0]}'
        for author in os.listdir(set_dir):
            WRITINGS = os.path.join(set_dir, author)
            for line in os.listdir(WRITINGS):
                img_path = os.path.join(WRITINGS, line)
                img = Image.open(img_path).convert('RGB')
                # images.append(resize(pil_to_tensor(img) / 255., (128, 1024)))
                img_size = img.size
                h.append(img_size[1])
                w.append(img_size[0])
    # images = torch.stack(images, dim=0)
    #mean = torch.mean(images)
    # std = torch.std(images)

    # print(mean, std)
    
    return max(w), max(h)

def get_base_statistics(base):
    h = []
    w = []
    lines = f'/home1/gemelli/dyslexia/data/{base}/lines'
    for author in os.listdir(lines):
        aut_dir = os.path.join(lines, author)
        for line in os.listdir(aut_dir):
            img_path = os.path.join(aut_dir, line)
            img = Image.open(img_path).convert('RGB')
            # images.append(resize(pil_to_tensor(img) / 255., (128, 1024)))
            img_size = img.size
            h.append(img_size[1])
            w.append(img_size[0])
    
    return max(w), max(h)

def create_disgraphia_splits(path):
    if os.path.isfile(os.path.join('/'.join(path.split("/")[:-1]), 'train.txt')):
        return
    else:
        print("Creating splits.")
    dis = [filename for filename in os.listdir(path) if 'X' in filename]
    not_dis = [filename for filename in os.listdir(path) if 'O' in filename]

    test_dis = random.sample(dis, 3)
    validation = [random.choice(test_dis)]
    test_dis.remove(validation[0])

    test_not_dis = random.sample(not_dis, 3)
    validation.append(random.choice(test_not_dis))
    test_not_dis.remove(validation[1])

    test_dis.extend(test_not_dis)
    train = [filename for filename in os.listdir(path) if filename not in test_dis]

    with open(os.path.join('/'.join(path.split("/")[:-1]), 'train.txt'), 'w') as f:
        for t in train:
            f.write(f"{t}\n")
    
    with open(os.path.join('/'.join(path.split("/")[:-1]), 'validation.txt'), 'w') as f:
        for t in validation:
            f.write(f"{t}\n")

    with open(os.path.join('/'.join(path.split("/")[:-1]), 'test.txt'), 'w') as f:
        for t in test_dis:
            f.write(f"{t}\n")


def create_authors_per_set():
    DATA = 'IAM/DATA'
    XML = 'IAM/xml'
    SETS_PATH = 'IAM/SETS'

    for set in SETS:
        set_dir = f'IAM/{set.split(".")[0]}'
        if not os.path.isdir(set_dir):
            os.mkdir(f'IAM/{set.split(".")[0]}')
        set = os.path.join(SETS_PATH, set)
        set_samples = [line.rstrip('\n') for line in open(set, 'r')]
        for f in set_samples:
            subdir = '-'.join(f.split("-")[:2])
            tree = ET.parse(os.path.join(XML, subdir + '.xml'))
            root = tree.getroot()
            writer = root.attrib['writer-id']
            print('author:', writer, end='\r')
            if not os.path.isdir(os.path.join(set_dir, writer)):
                os.mkdir(os.path.join(set_dir, writer))

            dir = f.split("-")[0]

            for png in os.listdir(os.path.join(DATA, dir, subdir)):
                shutil.copy(os.path.join(DATA, dir, subdir, png), 
                            os.path.join(set_dir, writer, png))

def get_bhk_features(filename = '/home1/gemelli/dyslexia/data/children/svg-lines/A01_O_1cb57/row3_O_1cb57.svg', base = 'children', bhk = 'binary'):
    assert bhk == 'binary' or bhk == 'float' or bhk == 'double'
    print(f"-> Using Smart Pen Features: {bhk}")
    author = filename.split("/")[-2]
    line = filename.split("/")[-1].split("_")[0]
    csv_path = os.path.join(CSVS, f'{base}_{bhk}.csv')
    df = pd.read_csv(csv_path, header=0, index_col=0)
    global_features = torch.tensor(df.filter(like='global').loc[author].to_numpy())
    line_features = torch.tensor(df.filter(like=line).loc[author].to_numpy())
    features = torch.cat((global_features, line_features))
    return features, features.shape[0]

get_bhk_features(bhk='double')