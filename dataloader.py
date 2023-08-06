from torchvision import transforms
import random
import numpy as np
from PIL import Image
import torch
import math
from mydataset import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from tps import Warper
import matplotlib.pyplot as plt

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


def load_gallery_probe_data(root, gallery_paths, probe_paths, resize_size=(324, 504), input_size=(288, 448),
                            batch_size=32, num_workers=0,data_transforms=None):
    gallery_list = []
    for i in gallery_paths:
        tmp = get_label(i)
        gallery_list = gallery_list + tmp

    probe_list = []
    for i in probe_paths:
        tmp = get_label(i)
        probe_list = probe_list + tmp

    # changed this to load for labeled data
    gallery_dataset = dataset_test(root, gallery_list, unlabeled=False, 
                                   transform=data_transforms)
    probe_dataset = dataset_test(root, probe_list, unlabeled=False,
                                 transform=data_transforms)

    gallery_iter = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    probe_iter = DataLoader(
        probe_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return gallery_iter, probe_iter

def filter_species(species, label):
    if species == 'tiger':
        if label <= 106:
            return True
    elif species == 'yak':
        if label <=120:
            return True
    elif species == 'elephant':
        if label <=336:
            return True
    else:
        raise ValueError('species not supported')
    return False


class JointAllBatchSampler(Sampler):
    '''
    sampler used in dataloader. method __iter__ should output the indices each time it is called
    '''
    def __init__(self, dataset, batchsize, num_other, species, *args, **kwargs):
        super(JointAllBatchSampler, self).__init__(dataset, *args, **kwargs)
        self.num_other = num_other
        self.species = species
        self.batch_size = batchsize
        self.dataset = dataset
        self.labels = np.array(dataset.targets)
        self.labels_uniq = list(set(self.labels)) 
        #self.len = len(dataset) // self.batch_size
        
        self.img_dict = {}
        filtered_indexes = [i for i, label in enumerate(self.labels) if filter_species(self.species, label)]
        self.img_dict['current'] = filtered_indexes
        self.num_cur_images = len(filtered_indexes)
        
        filtered_indexes = [i for i, label in enumerate(self.labels) if not filter_species(self.species, label)]
        self.img_dict['other'] = filtered_indexes
            
        self.iter_num = self.num_cur_images // (self.batch_size-self.num_other)
    
    def __iter__(self):
        curr_p = 0
        other_p = 0
        random.shuffle(self.img_dict['current'])
        random.shuffle(self.img_dict['other'])
        for i in range(self.iter_num):
            idx = []
            cur_inds = self.img_dict['current'][curr_p: curr_p + self.batch_size-self.num_other]
            other_inds = self.img_dict['other'][other_p: other_p + self.num_other]
            
            for j in range(self.num_other):
                idx.append(cur_inds[j])
                idx.append(other_inds[j])
            idx.extend(cur_inds[self.num_other:])
                    
            curr_p += self.batch_size-self.num_other
            other_p += self.num_other
            
            yield idx

    def __len__(self):
        return self.iter_num 
   



def load_triplet_direction_gallery_probe(root, train_paths, probe_paths, signal=' ',
                                         input_size=(224, 448), warper=None,resize_size=(256,256),cropped=True,joint_all=False):

    train_list = []
    for i in train_paths:
        tmp = get_label(i)
        train_list = train_list + tmp


    probe_list = []
    for i in probe_paths:
        tmp = get_label(i)
        probe_list = probe_list + tmp
        
    if cropped:
        initial_transform = transforms.Resize(input_size, interpolation=3)

        train_transformer = transforms.Compose([
            transforms.Pad(10),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            RandomErasing(),
            Cutout(),
        ])
    else:    
        initial_transform =  transforms.Compose([
            transforms.Resize(resize_size, interpolation=3),
            transforms.RandomCrop(input_size)])


        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            RandomErasing(),
            Cutout(),
        ])
        
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

    train_dataset = dataset_direction_triplet(root, train_list, flag=1, signal=signal, transform=train_transformer,warper=warper,initial_transform=initial_transform,joint_all=joint_all)
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
                                  input_size=(224, 224),warper=None,resize_size=(256,256),cropped=True):

    train_list = []
    for i in train_paths:
        tmp = get_label(i)
        train_list = train_list + tmp
        
    probe_list = []
    for i in probe_paths:
        tmp = get_label(i)
        probe_list = probe_list + tmp
        
    if cropped:
        initial_transform = transforms.Resize(input_size, interpolation=3)

        train_transformer = transforms.Compose([
            transforms.Pad(10),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            RandomErasing(),
            Cutout(),
        ])
    else:    
        initial_transform =  transforms.Compose([
            transforms.Resize(resize_size, interpolation=3),
            transforms.RandomCrop(input_size)])


        train_transformer = transforms.Compose([
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
    dve_images = []
    warps = []
    metas = []
    dve_warp = len(batch[0]) > 3
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.extend(batch[b][0])
            labels.extend(batch[b][1])
            directs.extend(batch[b][2])
            if dve_warp:
                dve_images.extend(batch[b][3])
                warps.extend(batch[b][4])
                metas.extend(batch[b][5])
    images = torch.stack(images, 0)
    labels = torch.from_numpy(np.array(labels))
    directs = torch.from_numpy(np.array(directs))
    if dve_warp:
        dve_images = torch.stack(dve_images, 0)
        warps = torch.stack(warps, 0)
        metas =  torch.stack(metas, 0)
        return images, labels, directs,dve_images, warps, metas
    return images, labels, directs


if __name__ == '__main__':
    h, w = 224, 224
    warper = Warper(h,w)
    train_path_dic = {'tiger':'./datalist/mytrain.txt',
                      'yak':'./datalist/yak_mytrain_aligned.txt',
                      'elephant':'./datalist/ele_train.txt',
                      'all':'./datalist/all_train_aligned.txt',
                      'tiger_all':'./datalist/tiger_train_all.txt',
                      'yak_all':'./datalist/yak_train_all.txt',
                      'elephant_all':'./datalist/ele_train_all.txt'
                      }
    probe_path_dic = {'tiger':'./datalist/myval.txt',
                        'yak':'./datalist/yak_myval_aligned.txt',
                        'elephant':'./datalist/ele_val.txt',
                        'all':'./datalist/all_val_aligned.txt'
                        }

    root = './data/Animal-Seg-V3/'
    train_paths1 = [train_path_dic['tiger_all'], ]
    train_paths2 = [train_path_dic['tiger'],]
    probe_paths = [probe_path_dic['tiger'], ]
    probe_path_dic = {'tiger':'./datalist/myval.txt',
                        'yak':'./datalist/yak_myval_aligned.txt',
                        'elephant':'./datalist/ele_val.txt',
                        'all':'./datalist/all_val_aligned.txt'
                        }
    
    train_data1, val_data, num_classes= load_direction_gallery_probe(
        root=root,
        train_paths=train_paths1,
        probe_paths=probe_paths,
        signal=' ',
        input_size=(h,w),
        warper=warper
    )
    
    train_data2, val_data, num_classes= load_direction_gallery_probe(
        root=root,
        train_paths=train_paths2,
        probe_paths=probe_paths,
        signal=' ',
        input_size=(h,w),
        warper=None
    )
    
    image_dve = train_data1[0][0]
    image = train_data2[0][0]
    print(image_dve.shape,image_dve)
    print(image.shape,image)
    plt.figure()
    #plt.imshow(image_dve.permute(1,2,0))
    plt.imshow(image.permute(1,2,0))
    plt.savefig('plot1.png') 
  
    
    
    