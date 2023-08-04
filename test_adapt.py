import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import os
import random
from PIL import Image
from model import  ft_net_swin, tiger_cnn5_64,tiger_cnn5_v1,seresnet_dve_1,ft_net_64,seresnet_dve_1_5,seresnet_dve_2
from tiger_eval import evaluate_tiger
import json
import sys
from utils import load_network,fliplr
from dataloader import load_gallery_probe_data
from metric import evaluate_CMC_per_query,evaluate_rerank
sys.path.insert(0,os.path.join(os.path.dirname(__file__), '../'))


def extract_feature_gallery(model,dataloaders,linear_num,batchsize,concat,dve=None,probe_dve=None,probe_xl2=None):
    count = 0
    names = []

    linear_num = linear_num*2 if concat else linear_num
    for iter, data in enumerate(dataloaders):
        img,image_names, label = data

        n, c, h, w = img.size()
        count += n
        
        ff = torch.FloatTensor(n,linear_num).zero_().cuda()

        if concat:
            input_img = Variable(img.cuda())
            flip_inputs = fliplr(img)
            flip_inputs = Variable(flip_inputs.cuda())
            
            if dve is not None:
                f_dves = dve(input_img)[0]
                flip_f_dves = dve(flip_inputs)[0]
                
                feature = model.test_adapt(input_img, f_dves, probe_dve[0], opt.shuffle, opt.simple,probe_xl2[0])[1] 
                flip_features = model.test_adapt(flip_inputs,flip_f_dves,probe_dve[1],opt.shuffle, opt.simple,probe_xl2[1])[1]
                
            else:
                feature = model(input_img)[1]
                flip_features = model(flip_inputs)[1]
                
            if opt.simple:
                score = torch.sum(feature, flip_features)
            else:
                ff += torch.cat((feature, flip_features), dim=1)
            
        else:
            for i in range(2):
                if(i==1):
                    img = fliplr(img)
                        
                input_img = Variable(img.cuda())
                f_dves = None
                if dve is not None:
                    f_dves = dve(input_img)[0]
                        
                
                if dve is not None:
                    outputs = model(input_img, f_dves) 
                else:
                    outputs = model(input_img)
                
                ff += outputs[1]
        
        if not opt.simple:            
            # norm feature
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        
        if iter == 0:
            
            labels = torch.LongTensor(len(dataloaders.dataset))
            if opt.simple:
                scores = torch.FloatTensor(len(dataloaders.dataset))
            else:
                features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
            
        start = iter*batchsize
        end = min( (iter+1)*batchsize, len(dataloaders.dataset))
        if opt.simple:
            scores[start:end] = score
        else:
            features[ start:end, :] = ff
        labels[start:end] = label
        
        names.extend(list(image_names))
    if opt.simple:
        return scores,labels,names
    
    return features,labels,names


def extract_feature_probe(model,dataloaders,linear_num,batchsize,concat,dve=None):
    count = 0
    names = []

    linear_num = linear_num*2 if concat else linear_num
    for iter, data in enumerate(dataloaders):
        img,image_names, label = data

        n, c, h, w = img.size()
        count += n
        
        ff = torch.FloatTensor(n,linear_num).zero_().cuda()

        if concat:
            input_img = Variable(img.cuda())
            flip_inputs = fliplr(img)
            flip_inputs = Variable(flip_inputs.cuda())
            
            if dve is not None:
                f_dves = dve(input_img)[0]
                flip_f_dves = dve(flip_inputs)[0]
                feature = model(input_img, f_dves)[1] 
                flip_features = model(flip_inputs,flip_f_dves)[1]
                
            else:
                feature = model(input_img)[1]
                flip_features = model(flip_inputs)[1]
            
            ff += torch.cat((feature, flip_features), dim=1)
            
        else:
            input_img = Variable(img.cuda())
            _,f_dves,posture_logits = model(input_img)
            #_,preds_posture = torch.max(posture_logits.data, 1)
            ff += f_dves
                    
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        
        if iter == 0:
            #postures = torch.LongTensor(len(dataloaders.dataset))
            features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
            labels = torch.LongTensor(len(dataloaders.dataset))
            
        start = iter*batchsize
        end = min( (iter+1)*batchsize, len(dataloaders.dataset))

        features[ start:end, :] = ff
        labels[start:end] = label
        #postures[start:end] = preds_posture
        names.extend(list(image_names))
    return features,labels,names

def eval_on_one(model,dve,dataset_type,data_transforms,linear_num,concat,seg=True,batch_size=32):
    gallery_path_dic = {'tiger':'./datalist/mytest.txt',
                    's_yak':'./datalist/yak_gallery_simple.txt',
                    'h_yak':'./datalist/yak_gallery_hard.txt',
                    'yak':'./datalist/yak_gallery_hard.txt',
                    'elephant':'./datalist/ele_new_test_gallery.txt',
                    'debug':'./datalist/debug_ele_train.txt'}
    probe_path_dic = {'tiger':'./datalist/mytest.txt',
                        's_yak':'./datalist/yak_probe_simple.txt',
                        'h_yak':'./datalist/yak_probe_hard.txt',
                        'yak':'./datalist/yak_probe_hard.txt',
                        'elephant':'./datalist/ele_new_test_probe.txt',
                        'debug':'./datalist/debug_ele_train.txt'}
    
    data_dirs = {
            'tiger_ori':'./data/tiger/test_original',
            'tiger_seg':'./data/tiger/tiger_test_isnet_seg',
            'elephant_seg':'./data/ele_test_v3_seg',
            'elephant_ori':'./data/elephant',
            'h_yak_seg':'./data/yak_test_seg_isnet_pp',
            'yak_seg':'./data/yak_test_seg_isnet_pp',
            'h_yak_ori':'./data/val',
            'yak_ori':'./data/val'
        }

    
    gallery_paths = [gallery_path_dic[dataset_type], ]
    probe_paths = [probe_path_dic[dataset_type], ]
    root = data_dirs[dataset_type+seg]
    remove_closest = True
    if 'yak' in dataset_type :
        remove_closest = False
        
    gallery_iter, probe_iter = load_gallery_probe_data(
        root=root,
        gallery_paths=gallery_paths,
        probe_paths=probe_paths,
        batch_size=batch_size,
        num_workers=2,
        data_transforms=data_transforms
    )
    dataloaders = {'gallery':gallery_iter,'query':probe_iter}
    
    with torch.no_grad():
        query_features,dve_features, query_labels,query_names = extract_feature_probe(model,dataloaders['query'],linear_num,batch_size,concat,dve)
        CMC = torch.IntTensor(len(dataloaders['gallery'])).zero_()
        ap = 0.0
        my_result = []
        
        for i in range(dve_features.shape[0]):
            curr_query = query_features[i] if opt.simple else None
            curr_query_dve = dve_features[i]
            gallery_features,gallery_labels,gallery_names = extract_feature_gallery(model,dataloaders['gallery'],linear_num,batch_size,concat,dve,curr_query_dve,curr_query)
            
            if opt.simple:
                scores = gallery_features
                ap_tmp, CMC_tmp = evaluate_rerank(scores,query_labels[i],gallery_labels,remove_closest)
            else:
                ap_tmp, CMC_tmp,scores = evaluate_CMC_per_query(query_features[i], query_labels[i],gallery_features, gallery_labels,remove_closest)
            
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            if opt.data_type == 'tiger':
                tmp = {}
                tmp['query_id'] = int(query_names[i].rstrip('.jpg'))
                index =np.argsort(scores)
                gallery_tmp = [(int(gallery_names[j].rstrip('.jpg'))) for j in index[1:]]
                tmp['ans_ids'] = gallery_tmp
                my_result.append(tmp)
                
        
        CMC = CMC.float()
        CMC = CMC / len(query_labels) 
        ap = ap/len(query_labels)




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("-mt", "--model_type", required=False, help="it can be tiger,s_yak,h_yak or elephant,all", default='tiger')
    parser.add_argument("-d", "--dataset_type", required=False, help="it can be tiger,s_yak,h_yak or elephant or all", default='tiger')
    parser.add_argument("-m", "--model_path", required=False, default=None)
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
    parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
    parser.add_argument('--use_swin', action='store_true',help='swin is used')
    parser.add_argument('--use_ori', action='store_true', help='use original image' )
    parser.add_argument('--way1_dve', action='store_true', help='use method1 for combining dve with re-id' )
    parser.add_argument('--name', default='ft_ResNet50', type=str, help='model name')
    parser.add_argument('--concat', action='store_true', help='concat flipped feature' )
    parser.add_argument('--ori_dim', action='store_true', help='use original input image dimension' )
    parser.add_argument('--ori_stride', action='store_true', help='use original stride at layer 2' )
    parser.add_argument('--transform_ori', action='store_true', help='use original eval transform' )
    parser.add_argument('-r', '--resume', default=None, type=str,help='path to dve checkpoint (default: None)')
    parser.add_argument('--seed', default="0", help='random seed')
    parser.add_argument('--joint', action='store_true', help='trained joint or joint all' )
    parser.add_argument('--stacked', action='store_true', help='stack last 3 layers of backbone for dve loss.' )
    parser.add_argument('--shuffle', action='store_true', help='test adapation strategey' )
    parser.add_argument('--simple', action='store_true', help='test adapation strategey' )
    opt = parser.parse_args()