import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
from tqdm import tqdm
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
                if isinstance(model, torch.nn.DataParallel):
                    feature = model.module.test_adapt(input_img, f_dves, probe_dve[0].cuda(), opt.shuffle, opt.simple,probe_xl2[0],opt.alpha)
                    flip_features = model.module.test_adapt(flip_inputs,flip_f_dves,probe_dve[1].cuda(),opt.shuffle, opt.simple,probe_xl2[1],opt.alpha)
                else:
                    feature = model.test_adapt(input_img, f_dves, probe_dve[0], opt.shuffle, opt.simple,probe_xl2[0],opt.alpha)
                    flip_features = model.test_adapt(flip_inputs,flip_f_dves,probe_dve[1],opt.shuffle, opt.simple,probe_xl2[1],opt.alpha)
                    
            else:
                feature = model(input_img)[1]
                flip_features = model(flip_inputs)[1]
                
            if opt.simple:
                score = feature+flip_features
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

    
    for iter, data in enumerate(dataloaders):
        img,image_names, label = data

        n, c, h, w = img.size()
        count += n
        

        if concat:
            input_img = Variable(img.cuda())
            flip_inputs = fliplr(img)
            flip_inputs = Variable(flip_inputs.cuda())
            
            if dve is not None:
                f_dves = dve(input_img)[0]
                flip_f_dves = dve(flip_inputs)[0]
                
            if isinstance(model, torch.nn.DataParallel):
                feature = model.module.get_probe_feature(input_img,opt.simple)
                flip_feature = model.module.get_probe_feature(flip_inputs,opt.simple)
            else:
                feature = model.get_probe_feature(input_img,opt.simple)
                flip_feature = model.get_probe_feature(flip_inputs,opt.simple)
            
            
            
        else:
            raise NotImplementedError
            # input_img = Variable(img.cuda())
            # _,f_dves,posture_logits = model(input_img)
            # #_,preds_posture = torch.max(posture_logits.data, 1)
            # ff += f_dves
                    
        # norm feature,if simple is true, shape is [B,512,56,56], otherwise shape is [B,512]
        # featurenorm = torch.norm(feature, p=2, dim=1, keepdim=True)
        # feature = feature.div(featurenorm.expand_as(feature))
        # flipfeaturenorm = torch.norm(flip_feature, p=2, dim=1, keepdim=True)
        # flip_feature = flip_feature.div(flipfeaturenorm.expand_as(flip_feature))
        
        f_dves = f_dves.reshape(f_dves.shape[0],f_dves.shape[1],-1) #[B,64,56,56]->[B,64,3136]
        f_dves_normed = f_dves.div(torch.norm(f_dves, p=2, dim=1, keepdim=True).expand_as(f_dves))
        flip_f_dves = flip_f_dves.reshape(flip_f_dves.shape[0],flip_f_dves.shape[1],-1) 
        flip_f_dves_normed = flip_f_dves.div(torch.norm(flip_f_dves, p=2, dim=1, keepdim=True).expand_as(flip_f_dves))
        
        if iter == 0:
            #postures = torch.LongTensor(len(dataloaders.dataset))
            if opt.simple:
                features = torch.FloatTensor( len(dataloaders.dataset), feature.shape[1],feature.shape[2],feature.shape[3])
                flip_features = torch.FloatTensor( len(dataloaders.dataset), feature.shape[1],feature.shape[2],feature.shape[3])
            else:
                features = torch.FloatTensor( len(dataloaders.dataset), feature.shape[1])
                flip_features = torch.FloatTensor( len(dataloaders.dataset), flip_feature.shape[1])
                
            labels = torch.LongTensor(len(dataloaders.dataset))
            dve_features = torch.FloatTensor( len(dataloaders.dataset), f_dves_normed.shape[1], f_dves_normed.shape[2])
            flip_dve_features = torch.FloatTensor( len(dataloaders.dataset), f_dves_normed.shape[1], f_dves_normed.shape[2])
            
        start = iter*batchsize
        end = min( (iter+1)*batchsize, len(dataloaders.dataset))

        features[ start:end, :] = feature
        flip_features[ start:end, :] = flip_feature 
        dve_features[ start:end, :] = f_dves_normed
        flip_dve_features[ start:end, :] = flip_f_dves_normed
        labels[start:end] = label
        #postures[start:end] = preds_posture
        names.extend(list(image_names))
    return features,dve_features,labels,names,flip_features,flip_dve_features

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
        query_features,dve_features, query_labels,query_names,query_flip_features,flip_dve_features = extract_feature_probe(model,dataloaders['query'],linear_num,batch_size,concat,dve)
        CMC = torch.IntTensor(len(dataloaders['gallery'].dataset)).zero_()
        
        ap = 0.0
        my_result = []
        
        #for i in tqdm(range(dve_features.shape[0])):
        for i in tqdm(range(10,40)):
            curr_query = [query_features[i],query_flip_features[i]] if opt.simple else [None,None]
            curr_query_dve = [dve_features[i],flip_dve_features[i]]
            gallery_features,gallery_labels,gallery_names = extract_feature_gallery(model,dataloaders['gallery'],linear_num,batch_size,concat,dve,curr_query_dve,curr_query)
            
            invalid_index=None
            if remove_closest:             
                curr_query_name = query_names[i]
                #print(curr_query_name)
                invalid_index = gallery_names.index(curr_query_name)
            
            if opt.simple:
                scores = gallery_features
                ap_tmp, CMC_tmp = evaluate_rerank(scores,query_labels[i],gallery_labels,remove_closest,invalid_index)
            else:
                feat =torch.cat((query_features[i], query_flip_features[i]) )
                
                CMC_tmp,ap_tmp,scores = evaluate_CMC_per_query(feat, query_labels[i],gallery_features, gallery_labels,remove_closest,invalid_index)
                
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            if opt.dataset_type == 'tiger':
                tmp = {}
                tmp['query_id'] = int(query_names[i].rstrip('.jpg'))
                index =np.argsort(scores)
                gallery_tmp = [(int(gallery_names[j].rstrip('.jpg'))) for j in index[1:]]
                tmp['ans_ids'] = gallery_tmp
                my_result.append(tmp)
                
        
        CMC = CMC.float()
        CMC = CMC / len(query_labels) 
        ap = ap/len(query_labels)
    if opt.dataset_type == 'tiger':
        metric = evaluate_tiger(my_result,'plain',path=False)
    else:
        metric = {'Rank@1':CMC[0].numpy().tolist(),'Rank@5':CMC[4].numpy().tolist(),'Rank@10':CMC[9].numpy().tolist(),'mAP':ap}
    return metric

def evaluate(model,dve,opt):
    metrics = {}
    if opt.dataset_type == 'all':
        for dataset_type in ['tiger','h_yak','elephant']:   
            metrics[dataset_type]=eval_on_one(model,dve,dataset_type,data_transforms,opt.linear_num,
                                              opt.concat,seg,opt.batchsize)
    else:
        metrics[opt.dataset_type]=eval_on_one(model,dve,opt.dataset_type,data_transforms,opt.linear_num,
                                              opt.concat,seg,opt.batchsize)
    return metrics


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("-mt", "--model_type", required=False, help="it can be tiger,s_yak,h_yak or elephant,all", default='tiger')
    parser.add_argument("-d", "--dataset_type", required=False, help="it can be tiger,s_yak,h_yak or elephant or all", default='tiger')
    parser.add_argument("-m", "--model_path", required=False, default=None)
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--alpha',default=1.0, type=float,help='alpha for attention')
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
    parser.add_argument('--shuffle', action='store_true', help='test adapation strategey' )
    parser.add_argument('--simple', action='store_true', help='test adapation strategey' )
    parser.add_argument('--way1_dve', action='store_true', help='use method1 for combining dve with re-id' )
    opt = parser.parse_args()
    
    opt_dic = {'model_type':opt.model_type,'dataset_type':opt.dataset_type,'model_path':opt.model_path,'gpu_ids':opt.gpu_ids,
               'batchsize':opt.batchsize,'linear_num':opt.linear_num,'use_swin':opt.use_swin,'use_ori':opt.use_ori,
               'way1_dve':opt.way1_dve,'name':opt.name,'concat':opt.concat,'ori_dim':opt.ori_dim,'ori_stride':opt.ori_stride,
               'transform_ori':opt.transform_ori,'resume':opt.resume,'seed':opt.seed,'joint':opt.joint,
               'stacked':opt.stacked,'shuffle':opt.shuffle,'simple':opt.simple}

    
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
    
    if opt.ori_dim:
        h, w = 288, 448
    else:
        h, w = 224, 224
        
    if opt.transform_ori:
        data_transforms = transforms.Compose([
            transforms.Resize((324,504), Image.BILINEAR),
            transforms.CenterCrop((288,448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        data_transforms = transforms.Compose([
                transforms.Resize((h, w), interpolation=3),
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
            'tiger_ori':'./data/tiger/test_original',
            'tiger_seg':'./data/tiger/tiger_test_isnet_seg',
            'elephant_seg':'./data/ele_test_v3_seg',
            'elephant_ori':'./data/elephant',
            'h_yak_seg':'./data/yak_test_seg_isnet_pp',
            'h_yak_ori':'./data/val'
        }
    seg = '_ori' if opt.use_ori else '_seg'
    
    
    use_posture = True if opt.model_path is not None and ('posture' in opt.model_path) else False
    
    if name == 'tiger_cnn5_v1':
        model_structure = tiger_cnn5_v1(nclasses, linear_num=opt.linear_num, circle=True,use_posture=use_posture,dve=opt.joint,stackeddve=opt.stacked,smallscale = not opt.ori_stride)
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
    
    
    dve = ft_net_64(stride=1)
    dve = dve.cuda()
    dve = nn.DataParallel(dve)
    if opt.resume is not None:
        checkpoint = torch.load(opt.resume)
        dve.load_state_dict(checkpoint['state_dict'])
        print('Successfully loaded DVE from '+opt.resume)
    
    dve = dve.eval()


    metric = evaluate(model,dve,opt)

    con = 'concat_' if opt.concat else ''
    to = 'to_' if opt.transform_ori else ''
    shuffle = 'shuffle_' if opt.shuffle else ''
    simple = 'simple_' if opt.simple else ''
    res_name = 'adapted_'+str(opt.alpha)+shuffle+simple+seg+ con+to+ name+str(seed)+'.txt'
    
    if opt.model_path is not None:
        res_name = opt.dataset_type+'_'+opt.model_path.split('/')[-2]+'_'+opt.model_path.split('/')[-1][:-4]+'_'+res_name
        if opt.way1_dve and opt.resume is None:
            res_name = 'untrained_dve_'+res_name
    elif opt.way1_dve and opt.resume is None:
        res_name = opt.dataset_type+'_untrained_both_'+res_name
    elif opt.way1_dve and opt.resume is not None:
        res_name = opt.dataset_type+'_untrained_backbone_'+res_name
    else:
        res_name = opt.dataset_type+'_untrained_'+res_name
    
    metric['opt'] = opt_dic
    result_metric = open('./result/sample_'+res_name,'w')
    #json.dump(metric, open('./result/'+res_name[:-3]+'json','w'))
    
    metric_str = "\n".join([k+': '+str(v) for k,v in metric.items()])
    result_metric.write(metric_str)
    result_metric.close()
  
 