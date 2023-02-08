import os
import torch
import shutil
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.transforms.functional import pil_to_tensor, resize
from torch.nn.functional import pad
import random
import pandas as pd
import numpy as np
from itertools import islice
from sklearn import preprocessing
random.seed(42)

from path import *

SETS = ['trainset.txt', 'validationset.txt', 'testset.txt']

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
    # mean = torch.mean(images)
    # std = torch.std(images)

    # print(mean, std)
    
    return max(w), max(h)

def get_base_statistics(base):
    h = []
    w = []
    lines = DYSG / f'{base}/original'
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

def create_simple_splits(path):
    if os.path.isdir(os.path.join(path, 'train.txt')):
        return
    else:
        print("Creating Simple splits.")
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

def create_multiple_splits(path, labels):
    if not os.path.isdir(os.path.join(path, 'splits')):
        os.mkdir(os.path.join(path, 'splits'))
    columns = ['CERTIFIED', 'EXPERT', 'PROFESSORS']
    labels = pd.read_csv(labels, header=0, index_col=0, sep=";")

    for column in columns:
        if os.path.isdir(os.path.join(path, 'splits', column)):
            continue
        else:
            print(f'Creating {column} split.')
            os.mkdir(os.path.join(path, 'splits', column))
        column_dis = [name for name in labels.loc[labels[column] >= 0.5].index.tolist()]
        column_not_dis = [name for name in labels.loc[labels[column] < 0.5].index.tolist()]
        if column == 'CERTIFIED': lenght_splits = [int(len(column_dis) / 4), int(len(column_dis) / 4), int(len(column_dis) / 4), len(column_dis) - int(len(column_dis) / 4)*3]
        elif column == 'EXPERT': lenght_splits = [4, 4, 4, 3]
        elif column == 'PROFESSORS': lenght_splits = [5, 5, 5, 4]
        else: break
        random.shuffle(column_dis)
        it_column_dis = iter(column_dis)
        tests = [list(islice(it_column_dis, elem)) for elem in lenght_splits]
        for t, l in enumerate(lenght_splits):
            # selection = column_not_dis.pop(column_not_dis.index(random.sample(column_not_dis, l)))
            selection = random.sample(column_not_dis, l)
            [column_not_dis.remove(s) for s in selection]
            tests[t].extend(selection)
        
        for t, test in enumerate(tests):
            split = os.path.join(path, 'splits', column, f'split{t}')
            os.mkdir(split)
            training = []
            [training.extend(tt) for i, tt in enumerate(tests) if i != t]
            column_not_dis_copy = column_not_dis.copy()
            validation = [random.sample(training, 1)[0], random.sample(column_not_dis_copy, 1)[0]]
            training.remove(validation[0]), column_not_dis_copy.remove(validation[1])
            training.extend(column_not_dis_copy)
            print("Union:",len(training), len(validation), len(test), len(training) + len(validation) + len(test))

            with open(os.path.join(split, "train.txt"), 'w') as output:
                for row in training:
                    output.write(str(row) + '\n')
            
            with open(os.path.join(split, "validation.txt"), 'w') as output:
                for row in validation:
                    output.write(str(row) + '\n')
            
            with open(os.path.join(split, "test.txt"), 'w') as output:
                for row in test:
                    output.write(str(row) + '\n')


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

def get_bhk_features(filename = os.path.join(DYSG,'children/original/A01_1cb57/row3_O_1cb57.png'), base = 'children', bhk = 'binary'):
    # read
    assert bhk == 'binary' or bhk == 'float' or bhk == 'double'
    author = filename.split("/")[-2]
    line = filename.split("/")[-1].split("_")[0]
    csv_path = CSVS / f'{base}_{bhk}.csv'
    df = pd.read_csv(csv_path, header=0, index_col=0)
    # print(csv_path)
    # normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df.values)
    norm_df = pd.DataFrame(x_scaled)
    norm_df.columns = df.columns
    norm_df.index = df.index

    # get features
    global_features = torch.tensor(norm_df.filter(like='global').loc[author].to_numpy(), dtype=torch.float32)
    line_features = torch.tensor(norm_df.filter(like=line).loc[author].to_numpy(), dtype=torch.float32)
    if line_features.shape[0] == 29: line_features = pad(line_features, (0, 7))
    features = torch.cat((global_features, line_features))
    return features, features.shape[0]

create_multiple_splits('/home1/gemelli/dysgraphia-detection/data/children', '/home1/gemelli/dysgraphia-detection/data/children/labels.csv')
