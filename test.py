from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import os
import random
from PIL import Image
from model import  tiger_cnn5_v1,ft_net_vit,ft_net
from tiger_eval import evaluate_tiger
import sys
from utils import load_network,fliplr
from dataloader import load_gallery_probe_data
from metric import evaluate_CMC
sys.path.insert(0,os.path.join(os.path.dirname(__file__), '../'))


def extract_feature(model,dataloaders,linear_num,batchsize,concat):
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
            
            feature = model(input_img)[1]
            flip_features = model(flip_inputs)[1]
            ff += torch.cat((feature, flip_features), dim=1)
            
        else:
            for i in range(2):
                if(i==1):
                    img = fliplr(img)
                        
                input_img = Variable(img.cuda())
                outputs = model(input_img)
                
                ff += outputs[1]
                    
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        
        if iter == 0:
            features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
            labels = torch.LongTensor(len(dataloaders.dataset))
            
        start = iter*batchsize
        end = min( (iter+1)*batchsize, len(dataloaders.dataset))

        features[ start:end, :] = ff
        labels[start:end] = label
        names.extend(list(image_names))
    return features,labels,names





def eval_on_one(model,dataset_type,linear_num,concat,seg=True,batch_size=32):
    if dataset_type == 'tiger':
        data_transforms = transforms.Compose([
            # transforms.Resize((324,504), Image.BILINEAR),
            # transforms.CenterCrop((288,448)),
            transforms.Resize((256,256), Image.BILINEAR),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        data_transforms = transforms.Compose([
                transforms.Resize((224, 224), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
               
        ])
    
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
            'tiger_ori':'./data/tiger/test',
            'tiger_seg':'./data/tiger/tiger_test_masked',
            'elephant_seg':'./data/elephant_test_masked',
            'elephant_ori':'./data/elephant_test',
            'h_yak_seg':'./data/yak_test_masked',
            'yak_seg':'./data/yak_test_masked',
            'h_yak_ori':'./data/yak_test',
            'yak_ori':'./data/yak_test'
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
        gallery_feature,gallery_label,gallery_names = extract_feature(model,dataloaders['gallery'],linear_num,batch_size,concat)
        query_feature, query_label,query_names = extract_feature(model,dataloaders['query'],linear_num,batch_size,concat)
    
    CMC,ap,q_g_dist = evaluate_CMC(query_feature, query_label, gallery_feature, gallery_label,remove_closest,'cos',True)
    
     
    if dataset_type == 'tiger':
        my_result = []
    
        for i in range(len(query_names)):
            tmp = {}
            image_name = query_names[i]

            index =np.argsort(q_g_dist[i, :])

            tmp['query_id'] = int(image_name.rstrip('.jpg'))
            p = 0
            gallery_tmp = []
            for j in index:
                if p == 0:
                    p += 1
                    continue
                current_name = gallery_names[j]
                gallery_tmp.append(int(current_name.rstrip('.jpg')))
            tmp['ans_ids'] = gallery_tmp
            my_result.append(tmp)
        
        metric = evaluate_tiger(my_result,'plain',path=False)
    else:
        metric = {'Rank@1':CMC[0].numpy().tolist(),'Rank@5':CMC[4].numpy().tolist(),'Rank@10':CMC[9].numpy().tolist(),'mAP':ap / len(query_label)}
    
    del query_feature
    del gallery_feature
    torch.cuda.empty_cache()
    
    return metric
    
    
def evaluate(model,opt):
    metrics = {}
    if opt.dataset_type == 'all':
        for dataset_type in ['tiger','h_yak','elephant']:   
            metrics[dataset_type]=eval_on_one(model,dataset_type,opt.linear_num,
                                              opt.concat,seg,opt.batchsize)
    else:
        metrics[opt.dataset_type]=eval_on_one(model,opt.dataset_type,opt.linear_num,
                                              opt.concat,seg,opt.batchsize)
    return metrics
        
    

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
    parser.add_argument('--name', default='ft_ResNet50', type=str, help='model name')
    parser.add_argument('--concat', action='store_true', help='concat flipped feature' )
    parser.add_argument('--ori_dim', action='store_true', help='use original input image dimension' )
    parser.add_argument('--ori_stride', action='store_true', help='use original stride at layer 2' )
    parser.add_argument('--transform_ori', action='store_true', help='use original eval transform' )
    parser.add_argument('-r', '--resume', default=None, type=str,help='path to dve checkpoint (default: None)')
    parser.add_argument('--seed', default="0", help='random seed')
    parser.add_argument('--joint', action='store_true', help='trained joint or joint all' )
    parser.add_argument('--stacked', action='store_true', help='stack last 3 layers of backbone for dve loss.' )
    parser.add_argument('--no_posture', action='store_true', help='posture data wa not used for training' )

    opt = parser.parse_args()
    ###load config###
    
    opt_dic = {'model_type':opt.model_type,'dataset_type':opt.dataset_type,'model_path':opt.model_path,'gpu_ids':opt.gpu_ids,
               'batchsize':opt.batchsize,'linear_num':opt.linear_num,'use_swin':opt.use_swin,'use_ori':opt.use_ori,
               'name':opt.name,'concat':opt.concat,'ori_dim':opt.ori_dim,'ori_stride':opt.ori_stride,
               'transform_ori':opt.transform_ori,'resume':opt.resume,'seed':opt.seed,'joint':opt.joint}

    seed = int(opt.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    dict_nclasses = {'yak':121,'tiger':107,'elephant':337,'all':565}
    nclasses = dict_nclasses[opt.model_type]
    
    name = opt.name
    
    use_gpu = torch.cuda.is_available()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    
    # if opt.ori_dim:
    #     h, w = 288, 448
    # else:
    #     h, w = 224, 224
        
    
        
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
            'tiger_ori':'./data/tiger/test',
            'tiger_seg':'./data/tiger/tiger_test_masked',
            'elephant_seg':'./data/elephant_test_masked',
            'elephant_ori':'./data/elephant_test',
            'h_yak_seg':'./data/yak_test_masked',
            'h_yak_ori':'./data/yak_test'
        }
    
    seg = '_ori' if opt.use_ori else '_seg'
    
    use_posture = False if opt.no_posture else True
    
    if name == 'tiger_cnn5_v1':
        model_structure = tiger_cnn5_v1(nclasses, linear_num=opt.linear_num, circle=True,use_posture=use_posture,dve=opt.joint,stackeddve=opt.stacked,smallscale = not opt.ori_stride)
    elif name == 'vit':
        model_structure = ft_net_vit(nclasses, use_posture = use_posture)
    elif name == 'resnet':
        model_structure = ft_net(nclasses,circle=True, use_posture = use_posture)
    else:
        print('unsupported model'+name)
        exit()
    
    if opt.model_path is not None:
        model = load_network(model_structure,opt.model_path,use_gpu)
    else:
        model = model_structure
        if use_gpu:
            model = model.cuda()
            model = nn.DataParallel(model)
    
    model = model.eval()


    metric = evaluate(model,opt)

    con = 'concat_' if opt.concat else ''
    to = 'to_' if opt.transform_ori else ''
    
    res_name = seg+ con+to+ name+str(seed)+'.txt'
    
    if opt.model_path is not None:
        res_name = opt.dataset_type+'_'+opt.model_path.split('/')[-2]+'_'+opt.model_path.split('/')[-1][:-4]+'_'+res_name
    else:
        res_name = opt.dataset_type+'_untrained_'+res_name
    
    metric['opt'] = opt_dic
    result_metric = open('./result/'+res_name,'w')
    
    metric_str = "\n".join([k+': '+str(v) for k,v in metric.items()])
    result_metric.write(metric_str)
    result_metric.close()
  
 