import numpy as np
import pandas as pd
import os
import random

from tqdm import tqdm
import xml.etree.ElementTree as ET

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
# from typing_extensions import Concatenate
import matplotlib.patches as patches
import matplotlib.text as text
import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
from sklearn.preprocessing import LabelEncoder


seed = 42
num_classes = 4
num_queries = 20
null_class_coef = 0.2
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 200
ANNOT_DIR = 'maskdata/annotations'
device = torch.device('cuda')

# 1. Chuẩn bị dữ liệu
def get_objects(xml_file):
    annotation = ET.parse(xml_file)
    root = annotation.getroot()
    
    name = os.path.basename(xml_file).replace('.xml','')
    size = root.find('size')
    
    objects = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        new_object = {}
        new_object['image_id'] = name
        new_object['labels'] = obj.find('name').text
        new_object['width'] = int(size.find('width').text)
        new_object['height'] = int(size.find('height').text)
        new_object['x'] = int(bbox.find('xmin').text)
        new_object['y'] = int(bbox.find('ymin').text)
        new_object['w'] = int(bbox.find('xmax').text)-int(bbox.find('xmin').text)
        new_object['h'] = int(bbox.find('ymax').text)-int(bbox.find('ymin').text)
        objects.append(new_object)
    return objects
annots = []
for xml in os.listdir(ANNOT_DIR):
    annots += get_objects(os.path.join(ANNOT_DIR,xml))
df = pd.DataFrame(annots)


for i in df.index:
    exceed_w =  df.iloc[i].x + df.iloc[i].w - df.iloc[i].width
    exceed_h =  df.iloc[i].y + df.iloc[i].h - df.iloc[i].height
    if exceed_w > 0:
        df.loc[df.index == i,'w'] -= exceed_w
    if exceed_h > 0:
        df.loc[df.index == i,'h'] -= exceed_h

encoder = LabelEncoder()
df.labels = encoder.fit_transform(df.labels)
print(df.head())

df_split = df[['image_id']].copy()
df_split['bbox_count'] = 1
df_split = df_split.groupby('image_id').sum()

g = plt.figure(figsize=(15,4))
g = sns.countplot(x='bbox_count',data=df_split)

train_split = df_split[:int(len(df_split)*.8)]
val_split = df_split[int(len(df_split)*.8):]

def train_transform():
    return A.Compose([
        A.HorizontalFlip(),
        A.RandomRotate90(),
        A.RandomBrightnessContrast(),
        A.Resize(300, 300),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco',label_fields=['labels']))

def valid_trainsform():
    return A.Compose([
        A.Resize(300,300),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco',label_fields=['labels']))

IMG_DIR = '/content/gdrive/MyDrive/maskdata/images'

class MaskDataset(Dataset):
    def __init__(self, image_ids, dataframe, transforms=None):
        self.image_ids = image_ids
        self.df = dataframe
        self.transforms = transforms
    def __len__(self) -> int:
        return self.image_ids.shape[0]
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id']==image_id]
        
        image = cv2.imread(f'{IMG_DIR}/{image_id}.png', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        boxes = records[['x', 'y', 'w', 'h']].values
        area = boxes[:,2]*boxes[:,3]
        area = torch.as_tensor(area, dtype=torch.float32)
        
        labels =  records['labels'].values
        
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']
            
        _, h, w = image.shape
        boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'],rows=h,cols=w)  
        
        target = {}
        target['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels,dtype=torch.long)
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        
        return image, target, image_id


def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = MaskDataset(image_ids=train_split.index.values,
                           dataframe=df,
                           transforms=train_transform())
train_data_loader = DataLoader(train_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=2,
                             collate_fn=collate_fn)
val_dataset = MaskDataset(image_ids=val_split.index.values,
                           dataframe=df,
                           transforms=valid_trainsform())
val_data_loader = DataLoader(val_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=2,
                             collate_fn=collate_fn)



# 2. Huấn luyện mô hình
# 3. Dự đoán