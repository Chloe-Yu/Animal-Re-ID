import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from utils import load_network,fliplr
from torch.autograd import Variable
from torchvision import datasets,transforms
from model import tiger_cnn5_v1
from dataloader import load_gallery_probe_data
from PIL import Image
import sys
import random
sys.path.insert(0,os.path.join(os.path.dirname(__file__), '../'))


def extract_feature(model,dataloaders,dve=None):
    #features = torch.FloatTensor()
    count = 0
    feature_dict ={}


    for iter, data in enumerate(dataloaders):
        if dve is not None:
            img,dve_img,image_names, label = data
        else:
            img,image_names, label = data

        n, c, h, w = img.size()
        count += n
        print(count)
       
        ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
                if dve is not None:
                    dve_img = fliplr(dve_img)
                    
            input_img = Variable(img.cuda())
            f_dves = None
            if dve is not None:
                input_dve_img = Variable(dve_img.cuda())
                f_dves = dve(input_dve_img)
                      
            if dve is not None:
                outputs = model(input_img, f_dves) 
            else:
                outputs = model(input_img)
                
            ff += outputs[1]
                    
        # norm feature
    
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        
        
        #features = torch.cat((features,ff.data.cpu()), 0)
        for i,name in enumerate(image_names):
            feature_dict[name] = ff[i]
        
    return feature_dict


parser = argparse.ArgumentParser(description='Test')
parser.add_argument("-mt", "--model_type", required=False, help="it can be tiger,yak or elephant,all", default='tiger')
parser.add_argument("-d", "--dataset_type", required=False, help="it can be tiger,s_yak,h_yak or elephant", default='tiger')
parser.add_argument("-m", "--model_path", required=False, default=None)
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--seed', default="0", help='random seed')
#parser.add_argument("--data_type",required=True, default = 'yak',type=str)
opt = parser.parse_args()

seed = int(opt.seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

dict_nclasses = {'yak':121,'tiger':107,'elephant':337,'all':565}
opt.nclasses = dict_nclasses[opt.model_type]
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

h, w = 224, 224

use_gpu = torch.cuda.is_available()

data_transforms = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
    ])

gallery_path_dic = {'tiger':'./datalist/mytest.txt',
                's_yak':'./datalist/yak_gallery_simple.txt',
                'h_yak':'./datalist/yak_gallery_hard.txt',
                'yak':'./datalist/yak_gallery_simple.txt',
                'elephant':'./datalist/ele_new_test_gallery.txt',
                'debug':'./datalist/debug_ele_train.txt',
                'yak_train':'./datalist/yak_mytrain_aligned.txt'}
probe_path_dic = {'tiger':'./datalist/mytest.txt',
                    's_yak':'./datalist/yak_probe_simple.txt',
                    'h_yak':'./datalist/yak_probe_hard.txt',
                    'yak':'./datalist/yak_probe_simple.txt',
                    'elephant':'./datalist/ele_new_test_probe.txt',
                    'debug':'./datalist/debug_ele_train.txt',
                    'yak_train':'./datalist/yak_myval_aligned.txt'}

data_dirs = {
        'tiger_ori':'./data/tiger/test_original',
        'tiger_seg':'./data/tiger/tiger_test_isnet_seg',
        'elephant_seg':'./data/ele_test_v3_seg',
        'elephant_ori':'./data/elephant',
        'h_yak_seg':'./data/yak_test_seg_isnet_pp',
        'yak_seg':'./data/yak_test_seg_isnet_pp',
        'h_yak_ori':'./data/val',
        'yak_ori':'./data/val',
        'yak_train_ori':'./data/Animal-2/'
    }


gallery_paths = [gallery_path_dic[opt.dataset_type], ]
probe_paths = [probe_path_dic[opt.dataset_type], ]
root = data_dirs[opt.dataset_type+'_ori']
remove_closest = True
if 'yak' in opt.dataset_type :
    remove_closest = False
    
gallery_iter, probe_iter = load_gallery_probe_data(
    root=root,
    gallery_paths=gallery_paths,
    probe_paths=probe_paths,
    batch_size=opt.batchsize,
    num_workers=2,
    data_transforms=data_transforms
)
dataloaders = {'gallery':gallery_iter,'query':probe_iter}


model_structure = tiger_cnn5_v1(opt.nclasses, use_posture=True)

if opt.model_path is not None:
    model = load_network(model_structure,opt.model_path,True)
    trained = 'trained_'
else:
    model = model_structure
    if use_gpu:
        model = model.cuda()
        model = nn.DataParallel(model)

        
    trained = 'pretrained_'

model = model.eval()



with torch.no_grad():
    feature_dict = extract_feature(model,dataloaders['gallery'])
    feature_dict1 = extract_feature(model,dataloaders['query'])
    torch.save(feature_dict,'pgcfl_gallery'+opt.model_type+'_'+trained+opt.dataset_type+'_'+opt.dataset_type+"_feature_dict.pt")
    torch.save(feature_dict1,'pgcfl_probe'+opt.model_type+'_'+trained+opt.dataset_type+'_'+opt.dataset_type+"_feature_dict.pt")
