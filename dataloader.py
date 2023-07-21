from torchvision import transforms
import random
import numpy as np
from PIL import Image
import torch
import math
from mydataset import *

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class RandomErasing(object):

    def __init__(self, probability=0.5, sl=0.02, sh=0.2, r1=0.3, mean=None):
        if mean is None: mean = [0., 0., 0.]
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)

                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]

                return img

        return img


class Cutout(object):

    def __init__(self, n_holes=256, length=4, probability=0.5):
        self.n_holes = n_holes
        self.length = length
        self.probability = probability

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        holes = random.randint(64, self.n_holes)
        for n in range(holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            length = self.length

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            # h = y2 - y1
            # w = x2 - x1

            # mask[y1: y2, x1: x2] = torch.randn((h, w))
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def get_label(label_path):
    f = open(label_path)
    lines = f.readlines()
    return lines


def load_triplet_direction_gallery_probe(root, train_paths, probe_paths, signal=' ',
                                         input_size=(224, 448), warper=None):

    train_list = []
    for i in train_paths:
        tmp = get_label(i)
        train_list = train_list + tmp


    probe_list = []
    for i in probe_paths:
        tmp = get_label(i)
        probe_list = probe_list + tmp
        
    initial_transform = transforms.Resize(input_size, interpolation=3)

    train_transformer = transforms.Compose([
        transforms.Pad(10),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        RandomErasing(),
        Cutout(),
    ])

    probe_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    train_dataset = dataset_direction_triplet(root, train_list, flag=1, signal=signal, transform=train_transformer,warper=warper,initial_transform=initial_transform)
    probe_dataset = dataset_direction(root, probe_list, flag=1, signal=signal, transform=probe_transformer,warper=warper,initial_transform=initial_transform)

   

    return train_dataset,probe_dataset,train_dataset.num_classes




def load_dve_pair(root, train_paths, signal=' ',
                                  input_size=(224, 448),warper=None):

    train_list = []
    for i in train_paths:
        tmp = get_label(i)
        train_list = train_list + tmp
        
    initial_transform = transforms.Resize(input_size, interpolation=3)
    train_transformer = transforms.Compose([
        transforms.Pad(10),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        RandomErasing(),
        Cutout(),
    ])

    train_dataset = dataset_direction(root, train_list, flag=1, signal=signal,initial_transform= initial_transform,transform=train_transformer,warper=warper)


    return train_dataset

########################################################################
# load reid: train gallery probe
#
def load_direction_gallery_probe(root, train_paths, probe_paths, signal=' ',
                                  input_size=(224, 448),warper=None):

    train_list = []
    for i in train_paths:
        tmp = get_label(i)
        train_list = train_list + tmp
        
    probe_list = []
    for i in probe_paths:
        tmp = get_label(i)
        probe_list = probe_list + tmp
        
    initial_transform = transforms.Resize(input_size, interpolation=3)

    train_transformer = transforms.Compose([
        transforms.Pad(10),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        RandomErasing(),
        Cutout(),
    ])
    
    probe_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    
    


    train_dataset = dataset_direction(root, train_list, flag=1, signal=signal, transform=train_transformer,initial_transform=initial_transform,warper=warper)
    probe_dataset = dataset_direction(root, probe_list, flag=1, signal=signal, transform=probe_transformer,initial_transform=initial_transform,warper=warper)


    return train_dataset, probe_dataset,train_dataset.num_classes



# load reid: train gallery probe
#
def train_collate(batch):
    batch_size = len(batch)
    images = []
    labels = []
    directs = []
    warps = []
    metas = []
    dve_warp = len(batch[b]) > 3
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.extend(batch[b][0])
            labels.extend(batch[b][1])
            directs.extend(batch[b][2])
            if dve_warp:
                warps.extend(batch[b][3])
                metas.extend(batch[b][4])
    images = torch.stack(images, 0)
    labels = torch.from_numpy(np.array(labels))
    directs = torch.from_numpy(np.array(directs))
    if dve_warp:
        warps = torch.stack(warps, 0)
        metas =  torch.stack(metas, 0)
        return images, labels, directs, warps, metas
    return images, labels, directs


  
    
    
    