# -*- coding: utf-8 -*

# -------------------------------------------------------------------------------
# Author: LiuNing
# Contact: 2742229056@qq.com
# Software: PyCharm
# File: mydataset.py
# Time: 7/26/19 3:33 PM
# Description: 
# -------------------------------------------------------------------------------

from torch.utils.data import Dataset
import os
from PIL import Image
import random
import numpy as np
from typing import List,Tuple,Dict
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch
########################################################################
# return image and label

class dataset(Dataset):
    def __init__(self, root, label, flag=1, signal=' ', transform=None):
        self._root = root
        self._flag = flag
        self._label = label
        self._transform = transform
        self._signal = signal
        self._list_images(self._root, self._label, self._signal)

    def _list_images(self, root, label, signal):
        self.synsets = []
        self.synsets.append(root)
        self.items = []

        c = 0
        for line in label:
            cls = line.rstrip('\n').split(signal)
            fn = cls.pop(0)

            if os.path.isfile(os.path.join(root, fn)):
                self.items.append((os.path.join(root, fn), float(cls[0])))
            else:
                print(os.path.join(root, fn))
            c += 1
        print('the total image is ', c)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        img = Image.open(self.items[index][0])
        img = img.convert('RGB')
        label = self.items[index][1]
        if self._transform is not None:
            return self._transform(img), label
        return img, label


########################################################################
# return image, label, and direction(left/right)
#
class dataset_direction(Dataset):
    def __init__(self, root, label, flag=1, signal=' ',initial_transform=None, transform=None, warper=None):
        self._root = root
        self._flag = flag
        self._label = label
        self._transform = transform
        self._signal = signal
        self._list_images(self._root, self._label, self._signal)
        self._get_num_classes()
        self.warper = warper
        self._initial_transform = initial_transform
        
    def _get_num_classes(self):
        self.num_classes = len(list(set([item[1] for item in self.items])))

    def _list_images(self, root, label, signal):
        self.synsets = []
        self.synsets.append(root)
        self.items = []
        self.targets = []
        

        c = 0
        for line in label:
            cls = line.rstrip('\n').split(signal)
            fn = cls.pop(0)

            if os.path.isfile(os.path.join(root, fn)):
                self.items.append((os.path.join(root, fn), float(cls[0]), float(cls[1])))
                self.targets.append(float(cls[0]))
            else:
                print(os.path.join(root, fn))
            c += 1
        print('the total image is ', c)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        img = Image.open(self.items[index][0])
        img = img.convert('RGB')
        label = self.items[index][1]
        direction = self.items[index][2]
    
        
        if self.warper is not None:
            if self.warper.return_pairs:
                #hard coded dimension for dve training for now.
                im1 = self._initial_transform(img)
                im1 = TF.to_tensor(im1) * 255
                
                im1, im2, flow, grid, kp1, kp2 = self.warper(im1, keypts=None, crop=0)

                im1 = im1.to(torch.uint8)
                im2 = im2.to(torch.uint8)

                C, H, W = im1.shape

                im1 = TF.to_pil_image(im1)
                im2 = TF.to_pil_image(im2)

                im1 = self._transform(im1)
                im2 = self._transform(im2)
                # print("tx-2: {:.3f}s".format(time.time() - tic)) ; tic = time.time()

                C, H, W = im1.shape
                #data = torch.stack((im1, im2), 0)
                meta = {
                    #'flow': flow[0],
                    'grid': grid[0],
                    #'im1': im1,
                    #'im2': im2,
                    #'index': index
                }
                return  im1,label,direction,im2,meta
            else:
                raise NotImplementedError
        
        if self._initial_transform is not None:
            img = self._initial_transform(img)
        if self._transform is not None:
            img = self._transform(img)
            
        
        return img, label, direction
    
    


########################################################################
# triplet load data, return anchor,positive and negtive
#
class dataset_direction_triplet(Dataset):
    def __init__(self, root, label, flag=1, signal=' ',initial_transform=None, transform=None, warper=None):
        self._root = root
        self._flag = flag
        self._label = label
        self._transform = transform
        self._signal = signal
        self._list_images(self._root, self._label, self._signal)
        self._dict_train = self.get_train_dict()
        self._labels = self.get_train_label()
        self.num_classes = len(self._dict_train)
        self.warper = warper
        self._initial_transform = initial_transform

    def _list_images(self, root, label, signal):
        self.synsets = []
        self.synsets.append(root)
        self.items = []

        c = 0
        for line in label:
            cls = line.rstrip('\n').split(signal)
            fn = cls.pop(0)

            if os.path.isfile(os.path.join(root, fn)):
                self.items.append((os.path.join(root, fn), float(cls[0]), float(cls[1])))
            else:
                print(os.path.join(root, fn))
            c += 1
        print('the total image is ', c)

    def get_train_label(self):
        labels = []
        for name, label, direct in self.items:
            labels.append(label)
        return labels

    def get_train_dict(self):
        dict_train = {}
        for name, label, direct in self.items:
            if not label in dict_train.keys():
                dict_train[label] = [(name, direct)]
            else:
                dict_train[label].append((name, direct))
        return dict_train

    def get_image(self, image_name, ):
        img = Image.open(image_name)
        img = img.convert('RGB')
        if self._initial_transform is not None:
            img = self._initial_transform(img)
        if self._transform is not None:
            img =  self._transform(img)
        return img

    def __len__(self):
        return len(self.items)
    
    def get_dve_warp(self,image_name):
        img = Image.open(image_name)
        im1 = img.convert('RGB')
        if self._initial_transform is not None:
            im1 = self._initial_transform(img)
            
        im1 = TF.to_tensor(im1) * 255
        
        _, im2, _, grid, _, _ = self.warper(im1, keypts=None, crop=0)
        
        im2 = im2.to(torch.uint8)
        im2 = TF.to_pil_image(im2)
        im2 = self._transform(im2)
        meta = grid[0]
        return im2,meta
        
        

    def __getitem__(self, index):
        
        anchor_name = self.items[index][0]
        anchor_label = self.items[index][1]
        anchor_direct = self.items[index][2]
        names = self._dict_train[anchor_label]
        nums = len(names)
        
        assert nums >= 2, f'{anchor_name} {names}'
    
        positive_name, positive_direct = random.choice(list(set(names) ^ set([(anchor_name, anchor_direct)])))
        negative_label = random.choice(list(set(self._labels) ^ set([anchor_label])))
        negative_name, negative_direct = random.choice(self._dict_train[negative_label])

        positive_image = self.get_image(positive_name, self._transform, self._initial_transform)
        negative_image = self.get_image(negative_name, self._transform, self._initial_transform)
        anchor_image = self.get_image(anchor_name, self._transform,self._initial_transform)
        
        assert negative_name != anchor_name
        
        if self.warper is not None:
            if self.warper.return_pairs:
                anchor_warp,anchor_meta = self.get_dve_warp(anchor_name)
                positive_warp,positive_meta = self.get_dve_warp(positive_name)
                negative_warp,negative_meta = self.get_dve_warp(negative_name)
                
                return [anchor_image, positive_image, negative_image], \
                       [anchor_label, anchor_label, negative_label], \
                       [anchor_direct, positive_direct, negative_direct]\
                       [anchor_warp,positive_warp,negative_warp]\
                       [anchor_meta,positive_meta,negative_meta]
            else:
                raise NotImplementedError
        
        

        return [anchor_image, positive_image, negative_image], \
               [anchor_label, anchor_label, negative_label], \
               [anchor_direct, positive_direct, negative_direct]


