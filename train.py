import argparse
from model import tiger_cnn5_v1,ft_net,ft_net_vit
from utils import fliplr,init_log,save_network
import os
from shutil import copyfile
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
from dataloader import train_collate,load_direction_gallery_probe,load_triplet_direction_gallery_probe
from tps import Warper
import time
from label_smoothing import LabelSmoothingCrossEntropy
from circle_loss import CircleLoss, convert_label_to_similarity
from dve_utils import dense_correlation_loss_dve,LossWrapper
from metric import evaluate_CMC
version =  torch.__version__
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import losses, miners
from test import eval_on_one


def train(model, criterion, optimizer, scheduler,dataloaders, num_epochs=25,writer=None):
    best_ap = 0.0
    warm_up = 0.01 # We start from the 0.01*lrRate, to be consistent with PGCFL
    dataset_sizes = {'train': len(dataloaders['train'].dataset),'val': len(dataloaders['val'].dataset) }
    if opt.triplet_sampler:
        iter_per_epoch = dataset_sizes['train']//(opt.batch_size//3)
    else:
        iter_per_epoch = dataset_sizes['train']//opt.batch_size
        
    samples_per_epoch = iter_per_epoch*opt.batch_size
    samples_per_epoch_cur = iter_per_epoch*opt.batch_size
    warm_iteration = iter_per_epoch*opt.warm_epoch
    
    if opt.circle:
        criterion_circle = CircleLoss(m=0.25, gamma=32) # gamma = 64 may lead to a better result.
    if opt.triplet:
        miner = miners.MultiSimilarityMiner()
        criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    if opt.joint:
        if isinstance(model, torch.nn.DataParallel):
            dve_loss = torch.nn.DataParallel(
                LossWrapper(dense_correlation_loss_dve),
                device_ids=[f'cuda:{id}' for id in opt.device.split(',')]
            )
        else:
            dve_loss = dense_correlation_loss_dve
    
    
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
                
                if epoch < opt.warm_epoch:               
                    if isinstance(model, torch.nn.DataParallel):
                        model.module.fix_params(is_training=False)
                    else:
                        model.fix_params(is_training=False)
                elif epoch == opt.warm_epoch:
                    if isinstance(model, torch.nn.DataParallel):
                         model.module.fix_params(is_training=True)
                    else:
                         model.module.fix_params(is_training=True)
                   
            else:
                model.train(False)  # Set model to evaluate mode
                
            running_loss_ent = 0.0
            running_loss_pos = 0.0
            running_loss_circle = 0.0
            running_loss_dve = 0.0
            running_loss_tri = 0.0
            
            running_corrects = 0.0
            running_corrects_posture = 0.0
            query_features = torch.FloatTensor(dataset_sizes['val'],opt.linear_num)
            query_labels =  torch.LongTensor(dataset_sizes['val'])
            gallery_features = torch.FloatTensor(samples_per_epoch,opt.linear_num)
            gallery_labels =  torch.LongTensor(samples_per_epoch)
            
            # Iterate over data.
            batch_i=0
            for _, data in enumerate(dataloaders[phase]):
                # get the inputs
                if opt.joint:
                    inputs_ori, labels, direction,dve_img,warped_inputs,meta = data
                else:
                    inputs_ori, labels, direction = data
                labels = labels.long()
                direction = direction.long()

                now_batch_size,c,h,w = inputs_ori.shape
                if now_batch_size<opt.batch_size and phase == 'train': # skip the last batch
                    continue
                
                inputs = inputs_ori
                if random.uniform(0, 1) > 0.5:
                    inputs = fliplr(inputs_ori)
                    direction = 1 - direction

                if phase == 'val':
                    query_labels[batch_i:min(batch_i+now_batch_size,dataset_sizes['val'])] = labels
                else:
                    gallery_labels[batch_i:min(batch_i+now_batch_size,samples_per_epoch)] = labels
                    
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                    direction = Variable(direction.cuda().detach())
                    if opt.joint and epoch >= opt.dve_start:
                        warped_inputs = Variable(warped_inputs.cuda().detach())
                        dve_img = Variable(dve_img.cuda().detach())
                else:
                    inputs, labels, direction = Variable(inputs), Variable(labels), Variable(directions)
                    if opt.joint and epoch >= opt.dve_start:
                        warped_inputs = Variable(warped_inputs)
                        dve_img = Variable(dve_img)
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                if opt.joint and epoch >= opt.dve_start:
                    if phase == 'val':
                        with torch.no_grad():
                            outputs = model(inputs)
                            if isinstance(model, torch.nn.DataParallel):
                                dve_outputs = model.module.dve_forward(dve_img)
                                dve_warped_outputs = model.module.dve_forward(warped_inputs)
                            else:
                                dve_outputs = model.dve_forward(dve_img)
                                dve_warped_outputs = model.dve_forward(warped_inputs)
                    else:
                        outputs = model(inputs)
                        if isinstance(model, torch.nn.DataParallel):
                            dve_outputs = model.module.dve_forward(dve_img)
                            dve_warped_outputs = model.module.dve_forward(warped_inputs)
                        else:
                            dve_outputs = model.dve_forward(dve_img)
                            dve_warped_outputs = model.dve_forward(warped_inputs)
                else:
                    if phase == 'val':
                        with torch.no_grad():
                            outputs = model(inputs)
                    else:
                        outputs = model(inputs)
                        
                if return_feature:
                    logits, ff, posture_logits = outputs          
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))
                    if phase =='val':
                        query_features[batch_i:min(batch_i+now_batch_size,dataset_sizes['val']),:] = ff.cpu()
                    else:
                        gallery_features[batch_i:min(batch_i+now_batch_size,samples_per_epoch),:] = ff.cpu()
                    
                
                    labels_cur = labels
                    
                    
                    loss = 0.0
                    if opt.ent_cls:
                        ent_loss = criterion(logits, labels_cur)
                        loss += ent_loss
                        _, preds = torch.max(logits.data, 1)
                        running_loss_ent += ent_loss.item() * logits.shape[0]
                    if opt.use_posture:
                        pos_loss = criterion(posture_logits, direction)
                        loss +=  pos_loss
                        _,preds_posture = torch.max(posture_logits.data, 1)
                        running_loss_pos += pos_loss.item() * posture_logits.shape[0]
                    if opt.circle:
                        cir_loss = criterion_circle(*convert_label_to_similarity( ff, labels))
                        running_loss_circle +=  cir_loss.item() 
                        loss += (opt.circle_loss_scale*cir_loss)/ff.shape[0]
                    if opt.triplet:
                        hard_pairs = miner(ff, labels)
                        trip_loss = criterion_triplet(ff, labels, hard_pairs) #/now_batch_size
                        loss += trip_loss
                        running_loss_tri += trip_loss.item() * ff.shape[0]
                    if opt.joint and epoch >= opt.dve_start:
                        if isinstance(model, torch.nn.DataParallel):
                            dloss = dve_loss(dve_outputs,dve_warped_outputs,meta,normalize_vectors=False).mean()
                        else:
                            dloss = dve_loss(dve_outputs,dve_warped_outputs,meta,normalize_vectors=False)
                        loss += opt.dve_loss_scale* dloss
                        running_loss_dve += dloss.item() * dve_outputs.shape[0]
            
                else:
                    raise NotImplementedError
                
                del inputs
                del inputs_ori
                if opt.joint and epoch >= opt.dve_start:
                    del dve_img
                    del warped_inputs
                    del dve_outputs
                    del dve_warped_outputs
                    
                torch.cuda.empty_cache()
                
                    
                # backward + optimize only if in training phase
                
                if epoch<opt.warm_epoch and phase == 'train': 
                    warm_up = min(1.0, warm_up + 0.99 / warm_iteration)# changed from 0.9 to 0.99 to be consistent with PGCFL
                    loss = loss*warm_up

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                if opt.ent_cls:
                    running_corrects += float(torch.sum(preds == labels_cur.data))
                if opt.use_posture:
                    running_corrects_posture += float(torch.sum(preds_posture == direction.data))
                batch_i+=now_batch_size
                
            if phase == 'train':
                epoch_ent_loss = running_loss_ent/samples_per_epoch_cur
                epoch_pos_loss = running_loss_pos/samples_per_epoch_cur
                epoch_cir_loss = running_loss_circle/samples_per_epoch_cur
                epoch_dve_loss = running_loss_dve/samples_per_epoch
                epoch_tri_loss = running_loss_tri/samples_per_epoch
                
                if opt.ent_cls:
                    epoch_acc = running_corrects / samples_per_epoch_cur
                if opt.use_posture:
                    epoch_acc_posture = running_corrects_posture / samples_per_epoch_cur
            else:
                epoch_ent_loss = running_loss_ent/dataset_sizes[phase]
                epoch_pos_loss = running_loss_pos/dataset_sizes[phase]
                epoch_cir_loss = running_loss_circle/dataset_sizes[phase]
                epoch_dve_loss = running_loss_dve/dataset_sizes[phase]
                epoch_tri_loss = running_loss_tri/dataset_sizes[phase]
                if opt.ent_cls:
                    epoch_acc = running_corrects / dataset_sizes[phase]
                if opt.use_posture:
                    epoch_acc_posture = running_corrects_posture / dataset_sizes[phase]

            if phase == 'val':
                CMC,ap,_ = evaluate_CMC(query_features, query_labels, gallery_features, gallery_labels,
                                      remove_closest=False, distance='euclidean')
                
                
                writer.add_scalar('val_Rank@1',CMC[0].numpy(), epoch)
                writer.add_scalar('val_Rank@5',CMC[4].numpy(), epoch)
                writer.add_scalar('val_Rank@10',CMC[9].numpy(), epoch)
                writer.add_scalar('val_mAP',ap/dataset_sizes['val'], epoch)
             
                
                if epoch in [99,199,229,259,289,309]:
                    save_network(model,epoch,name)
                    
                if opt.ent_cls:
                    writer.add_scalar('val_ent_loss',epoch_ent_loss, epoch)
                if opt.use_posture:
                    writer.add_scalar('val_posture_loss',epoch_pos_loss, epoch)
                if opt.circle:
                    writer.add_scalar('val_circle_loss',epoch_cir_loss, epoch)
                if opt.triplet:
                    writer.add_scalar('val_triplet_loss',epoch_tri_loss, epoch)
                if opt.joint:
                    writer.add_scalar('val_dve_loss',epoch_dve_loss, epoch)
                if opt.ent_cls:
                    writer.add_scalar('val_acc',epoch_acc, epoch)
                if opt.use_posture:
                    writer.add_scalar('val_acc_posture',epoch_acc_posture, epoch)
                    
                    
                if opt.test and epoch % opt.test_freq == 0:
                    seg = '_seg' if ('Seg' in root) else '_ori'
                    test_metric = eval_on_one(model,opt.data_type,
                                              opt.linear_num,concat=True,seg=seg)
                    if opt.data_type == 'tiger':
                        test_metric_map = test_metric["result"][0]["public_split"]['mmAP']
                        test_metric_top1 = test_metric["result"][0]["public_split"]['top-1(cross_cam)']
                        test_metric_top1_s = test_metric["result"][0]["public_split"]['top-1(single_cam)']
                        writer.add_scalar('test_Rank@1_single',test_metric_top1_s, epoch)
                    else:
                        test_metric_map = test_metric['mAP']
                        test_metric_top1 = test_metric['Rank@1']
                    writer.add_scalar('test_Rank@1',test_metric_top1, epoch)
                    writer.add_scalar('test_mAP',test_metric_map, epoch)
                    if opt.test_freq == 1 and test_metric_map > best_ap:
                        best_ap = test_metric_map
                        save_network(model,epoch,name,best=True)
                    
                if opt.test_transfer and epoch % opt.test_freq == 0:
                    seg = '_seg' if ('Seg' in root) else '_ori'
                    for datatype in ['tiger','yak','elephant']:
                        if datatype != opt.data_type:
                            test_metric = eval_on_one(model,datatype,
                                                    opt.linear_num,concat=True,seg=seg)
                            if datatype == 'tiger':
                                test_metric_map = test_metric["result"][0]["public_split"]['mmAP']
                                test_metric_top1 = test_metric["result"][0]["public_split"]['top-1(cross_cam)']
                                test_metric_top1_s = test_metric["result"][0]["public_split"]['top-1(single_cam)']
                                writer.add_scalar(datatype+'_test_transfer_Rank@1_single',test_metric_top1_s, epoch)
                            else:
                                test_metric_map = test_metric['mAP']
                                test_metric_top1 = test_metric['Rank@1']
                            writer.add_scalar(datatype+'_test_transfer_Rank@1',test_metric_top1, epoch)
                            writer.add_scalar(datatype+'_test_transfer_mAP',test_metric_map, epoch)
                    
            if phase == 'train':
                scheduler.step()
                if opt.ent_cls:
                    writer.add_scalar('train_ent_loss',epoch_ent_loss, epoch)
                    writer.add_scalar('train_acc',epoch_acc, epoch)
                if opt.use_posture:
                    writer.add_scalar('train_posture_loss',epoch_pos_loss, epoch)
                    writer.add_scalar('train_acc_posture',epoch_acc_posture, epoch)
                if opt.circle:
                    writer.add_scalar('train_circle_loss',epoch_cir_loss, epoch)
                if opt.triplet:
                    writer.add_scalar('train_triplet_loss',epoch_tri_loss, epoch)
                if opt.joint:
                    writer.add_scalar('train_dve_loss',epoch_dve_loss, epoch)

    save_network(model,epoch,name,last=True)        


    

    

if __name__ =='__main__':
    
    assert int(version[0])>0 or int(version[2]) > 3 # for the new version like 0.4.0, 0.5.0 and 1.0.0
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--device',default='0,1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model folder name')
    parser.add_argument("-m", "--model_path", required=False, default=None, help='pretrained dve model path')
    parser.add_argument('--joint', action='store_true', help='jointly training dve and re-id' )
    parser.add_argument('--test', action='store_true', help='log metric on test data' )
    parser.add_argument('--test_transfer', action='store_true', help='log metric on test data of other species' )
    parser.add_argument('--test_freq',default='1',type=int, help='test frequency' )
    parser.add_argument('--seed', default=0, type=int, help='seed')
    # data
    parser.add_argument("--data_type",required=True, default = 'yak',type=str)
    parser.add_argument('--batch_size', default=32, type=int, help='batchsize')
    parser.add_argument('--triplet_sampler', action='store_true', help='making sure training batch has enough pos and neg.' )
    parser.add_argument('--use_posture', action='store_true', help='use the posture data for supervision' )
    parser.add_argument('--background', action='store_true', help='includes background in data' )
    parser.add_argument('--ori_dim', action='store_true', help='use original input dimension' )
    # optimizer
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay. More Regularization Smaller Weight.')
    parser.add_argument('--total_epoch', default=60, type=int, help='total training epoch')
    parser.add_argument('--circle_loss_scale', default=1.0, type=float, help='circle_loss_scale')
    parser.add_argument('--dve_loss_scale', default=1.0, type=float, help='dve_loss_scale')
    parser.add_argument('--dve_start', default=0, type=int, help='epoch to start adding dve loss')
    # backbone
    parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
    parser.add_argument('--stride', default=1, type=int, help='stride')
    parser.add_argument('--ori_stride', action='store_true', help='no modification to layer 2 of se-resnet' )
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
    parser.add_argument('--use_vit', action='store_true', help='use vit transformer 224x224' )
    parser.add_argument('--use_cnn5_v1', action='store_true', help='use tiger_cnn5_v1' )
    # loss
    parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--label_smoothing', action='store_true', help='adds label smoothing' )
    parser.add_argument('--circle', action='store_true', help='use Circle loss' )
    parser.add_argument('--triplet', action='store_true', help='use triplet loss' )
    parser.add_argument('--ent_cls', action='store_true', help='use Classification loss for enetity' )
    
    opt = parser.parse_args()
    #initialization
    use_gpu = torch.cuda.is_available()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    
    seed = int(opt.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    name = opt.name
    dir_name = os.path.join('./model',name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    #record every run
    copyfile('./train.py', dir_name+'/train.py')
    copyfile('./model.py', dir_name+'/model.py')

    # save opts
    logging = init_log(dir_name)
    _print = logging.info
    writer = SummaryWriter(log_dir=dir_name)
    _print(opt)
    
    
    # Load Data
    if opt.ori_dim:
        h,w= 288, 448
    else:
        h, w = 224, 224
        
    
    warper = None
    if opt.joint:
        # joint dve and reid on one species
        warper = Warper(h,w)
        
    train_path_dic = {'tiger':'./datalist/mytrain.txt',
                      'yak':'./datalist/yak_mytrain_aligned.txt',
                      'elephant':'./datalist/ele_train.txt',
                      'all':'./datalist/all_train_aligned.txt'
                      }

    probe_path_dic = {'tiger':'./datalist/myval.txt',
                      'yak':'./datalist/yak_myval_aligned.txt',
                      'elephant':'./datalist/ele_val.txt',
                      'all':'./datalist/all_val_aligned.txt'
                    }
    if opt.background:
        root = './data/Animal/'
    else:
        root = './data/Animal-masked/'


    train_paths = [train_path_dic[opt.data_type], ]
    probe_paths = [probe_path_dic[opt.data_type], ]
    
    print(train_paths)
    
    #vanilla triplet sampling
    if opt.triplet_sampler:
        train_data, val_data, num_classes = load_triplet_direction_gallery_probe(
            root=root,
            train_paths=train_paths,
            probe_paths=probe_paths,
            signal=' ',
            input_size=(h,w),
            warper = warper,
            cropped=opt.data_type != 'tiger'
        )
        collate_fn=train_collate
    else:
        train_data, val_data, num_classes= load_direction_gallery_probe(
                root=root,
                train_paths=train_paths,
                probe_paths=probe_paths,
                signal=' ',
                input_size=(h,w),
                warper=warper,
                cropped=opt.data_type != 'tiger'
            )
        collate_fn=torch.utils.data.dataloader.default_collate
    
    
    dict_nclasses = {'yak':121,'tiger':107,'elephant':337,'all':565}
    numClasses = dict_nclasses[opt.data_type]
    assert num_classes == numClasses
    
    
    image_datasets={'train':train_data,'val':val_data}
    
    
    batchsize = opt.batch_size
    if opt.joint and opt.triplet_sampler:
        batchsize  = opt.batch_size//3
    dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=batchsize, collate_fn=collate_fn,drop_last=True,
                                                shuffle=True, num_workers=2, pin_memory=True,
                                                prefetch_factor=2, persistent_workers=True) # 8 workers may work faster
                ,'val': DataLoader(image_datasets['val'], batch_size=opt.batch_size, collate_fn=torch.utils.data.dataloader.default_collate,drop_last=True,
                                                shuffle=True, num_workers=2, pin_memory=True,
                                                prefetch_factor=2, persistent_workers=True) # 8 workers may work faster
                }

        
    since = time.time()
    if opt.joint:
        inputs, classes, directions, _,_,_ = next(iter(dataloaders['train']))
    else:
        inputs, classes, directions = next(iter(dataloaders['train']))
        
    print(time.time()-since)
    
    # set model
    
    return_feature = True # for our re-id purpose, we always need feature
    
    if opt.use_cnn5_v1:
        model = tiger_cnn5_v1(numClasses,stride = opt.stride,linear_num=opt.linear_num,circle=return_feature,use_posture=opt.use_posture
                              ,dve=opt.joint, smallscale=not opt.ori_stride, model_path=opt.model_path)
    elif opt.use_vit:
        model = ft_net_vit(numClasses, opt.droprate, return_feature = return_feature, linear_num=opt.linear_num, use_posture=opt.use_posture)
    else:
        #resnet50
        model = ft_net(numClasses, opt.droprate, circle = return_feature, linear_num=opt.linear_num, use_posture=opt.use_posture)
    print(model)
    
    
    # optimization
    optim_name = optim.SGD
    
    if opt.use_cnn5_v1:
        base_params = list(map(id, model.model.backbone.parameters()))
        extra_params = list(filter(lambda p: id(p) not in base_params, model.parameters()))
        base_params = model.model.backbone.parameters()
    else:
        base_params = list(map(id, model.model.parameters()))
        extra_params = list(filter(lambda p: id(p) not in base_params, model.parameters()))
        base_params = model.model.parameters()
    
        
    optimizer_ft = optim_name([
                    {'params': base_params, 'lr': 0.1*opt.lr},
                    {'params': extra_params, 'lr': opt.lr}
                ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
    
    
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=opt.total_epoch*2//3, gamma=0.1)
    
    if opt.label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
        
    
    model = model.cuda()
    model = nn.DataParallel(model)
    train(model,criterion,optimizer_ft,exp_lr_scheduler,dataloaders,opt.total_epoch,writer)