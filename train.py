import argparse
from model import  ft_net_swin,seresnet_dve_1,tiger_cnn5_v1,tiger_cnn5_64,ft_net_64,seresnet_dve_2,seresnet_dve_1_5,seresnet_dve_att
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
from dataloader import train_collate,load_direction_gallery_probe,load_triplet_direction_gallery_probe,load_dve_pair,JointAllBatchSampler
from tps import Warper
import time
import sys
from label_smoothing import LabelSmoothingCrossEntropy
from circle_loss import CircleLoss, convert_label_to_similarity
from dve_utils import dense_correlation_loss_dve,LossWrapper
from metric import evaluate_CMC
version =  torch.__version__
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import losses, miners


def train(model, criterion, optimizer, scheduler,dataloaders, num_epochs=25,writer=None,way1_dve=None):
    best_ap = 0.0
    warm_up = 0.01 # We start from the 0.01*lrRate, to be consistent with PGCFL
    dataset_sizes = {'train': len(dataloaders['train'].dataset),'val': len(dataloaders['val'].dataset) }
    if opt.triplet_sampler:
        iter_per_epoch = dataset_sizes['train']//(opt.batch_size//3)
    elif opt.joint_all:
        iter_per_epoch = len(dataloaders['train'].batch_sampler)
    else:
        iter_per_epoch = dataset_sizes['train']//opt.batch_size
        
    samples_per_epoch = iter_per_epoch*opt.batch_size
    samples_per_epoch_cur = iter_per_epoch*(opt.batch_size-opt.num_other)
    warm_iteration = iter_per_epoch*opt.warm_epoch
    
    if opt.circle:
        criterion_circle = CircleLoss(m=0.25, gamma=32) # gamma = 64 may lead to a better result.
    if opt.triplet:
        miner = miners.MultiSimilarityMiner()
        criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    if opt.joint or opt.joint_all:
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
                num_other = opt.num_other
                   
            else:
                model.train(False)  # Set model to evaluate mode
                num_other = 0
                
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
                if opt.joint or opt.joint_all:
                    inputs_ori, labels, direction,warped_inputs,meta = data
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
                    if opt.joint or opt.joint_all:
                        warped_inputs = Variable(warped_inputs.cuda().detach())
                else:
                    inputs, labels, direction = Variable(inputs), Variable(labels), Variable(directions)
                    if opt.joint or opt.joint_all:
                        warped_inputs = Variable(warped_inputs)
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                if opt.joint or opt.joint_all:
                    if phase == 'val':
                        with torch.no_grad():
                            outputs,dve_outputs = model(inputs,dve=True)
                            if isinstance(model, torch.nn.DataParallel):
                                dve_warped_outputs = model.module.dve_forward(warped_inputs)
                            else:
                                dve_warped_outputs = model.dve_forward(warped_inputs)
                    else:
                        outputs,dve_outputs = model(inputs,dve=True)
                        if isinstance(model, torch.nn.DataParallel):
                            dve_warped_outputs = model.module.dve_forward(warped_inputs)
                        else:
                            dve_warped_outputs = model.dve_forward(warped_inputs)
                elif opt.way1_dve:
                    if phase == 'val':
                        with torch.no_grad():
                            dve_outputs = way1_dve(inputs)
                            outputs = model(inputs,dve_outputs[0])
                    else:
                        dve_outputs = way1_dve(inputs)
                        outputs = model(inputs,dve_outputs[0])
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
                    
                    if opt.joint_all:
                        # excluding images of other species for ent_loss and pos_loss & circle loss
                        positions = torch.tensor([i for i in range(logits.size(0)) if (i%2==0) or i>=2*opt.num_other]).cuda()
                        direction = direction[positions]
                        labels = labels[positions]
                        logits = logits[positions]
                        posture_logits = posture_logits[positions]
                        ff = ff[positions]
                    
                    
                    loss = 0.0
                    if opt.ent_cls:
                        ent_loss = criterion(logits, labels)
                        loss += ent_loss
                        _, preds = torch.max(logits.data, 1)
                        running_loss_ent += ent_loss.item() * (now_batch_size-num_other)
                    if opt.use_posture:
                        pos_loss = criterion(posture_logits, direction)
                        loss +=  pos_loss
                        _,preds_posture = torch.max(posture_logits.data, 1)
                        running_loss_pos += pos_loss.item() * (now_batch_size-num_other)
                    if opt.circle:
                        cir_loss = criterion_circle(*convert_label_to_similarity( ff, labels))
                        running_loss_circle +=  cir_loss.item() 
                        loss += (opt.circle_loss_scale*cir_loss)/now_batch_size
                    if opt.triplet:
                        hard_pairs = miner(ff, labels)
                        trip_loss = criterion_triplet(ff, labels, hard_pairs) #/now_batch_size
                        loss += trip_loss
                        running_loss_tri += trip_loss.item() * now_batch_size

                    if opt.joint or opt.joint_all:
                        if isinstance(model, torch.nn.DataParallel):
                            dloss = dve_loss(dve_outputs,dve_warped_outputs,meta,normalize_vectors=False).mean()
                        else:
                            dloss = dve_loss(dve_outputs,dve_warped_outputs,meta,normalize_vectors=False)
                        loss += opt.dve_loss_scale* dloss
                        running_loss_dve += dloss.item() * now_batch_size
            
                else:
                    raise NotImplementedError
                
                del inputs
                del inputs_ori
                torch.cuda.empty_cache()
                
                    
                # backward + optimize only if in training phase
                
                if epoch<opt.warm_epoch and phase == 'train': 
                    warm_up = min(1.0, warm_up + 0.99 / warm_iteration)# changed from 0.9 to 0.99 to be consistent with PGCFL
                    loss = loss*warm_up

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                if opt.ent_cls:
                    running_corrects += float(torch.sum(preds == labels.data))
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
                # last_model_wts = model.module.state_dict()
                CMC,ap,_ = evaluate_CMC(query_features, query_labels, gallery_features, gallery_labels,
                                      remove_closest=False, distance='euclidean')
                if ap > best_ap:
                    best_ap = ap
                    save_network(model,epoch,name,best=True)
                
                writer.add_scalar('Rank@1',CMC[0].numpy(), epoch)
                writer.add_scalar('Rank@5',CMC[4].numpy(), epoch)
                writer.add_scalar('Rank@10',CMC[9].numpy(), epoch)
                writer.add_scalar('mAP',ap/dataset_sizes['val'], epoch)
             
                
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
                if opt.joint or opt.joint_all:
                    writer.add_scalar('val_dve_loss',epoch_dve_loss, epoch)
                if opt.ent_cls:
                    writer.add_scalar('val_acc',epoch_acc, epoch)
                if opt.use_posture:
                    writer.add_scalar('val_acc_posture',epoch_acc_posture, epoch)
            if phase == 'train':
                scheduler.step()
                if opt.ent_cls:
                    writer.add_scalar('train_ent_loss',epoch_ent_loss, epoch)
                if opt.use_posture:
                    writer.add_scalar('train_posture_loss',epoch_pos_loss, epoch)
                if opt.circle:
                    writer.add_scalar('train_circle_loss',epoch_cir_loss, epoch)
                if opt.triplet:
                    writer.add_scalar('train_triplet_loss',epoch_tri_loss, epoch)
                if opt.joint or opt.joint_all:
                    writer.add_scalar('train_dve_loss',epoch_dve_loss, epoch)
                
                if opt.ent_cls:
                    writer.add_scalar('train_acc',epoch_acc, epoch)
                if opt.use_posture:
                    writer.add_scalar('train_acc_posture',epoch_acc_posture, epoch)
       


    save_network(model,epoch,name,last=True)        


    

    

if __name__ =='__main__':
    
    assert int(version[0])>0 or int(version[2]) > 3 # for the new version like 0.4.0, 0.5.0 and 1.0.0
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--device',default='0,1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model folder name')
    parser.add_argument("-m", "--model_path", required=False, default=None)
    parser.add_argument('--joint', action='store_true', help='jointly training dve and re-id' )
    parser.add_argument('--joint_att', action='store_true', help='use attention while jointly trianing.' )
    parser.add_argument('--stacked',action='store_true', help='stack last 3 layer of backbone to train with dve' )
    parser.add_argument('--joint_all', action='store_true', help='jointly training dve(3 species) and re-id' )
    parser.add_argument('--ori_dim', action='store_true', help='use original input dimension' )
    parser.add_argument('--ori_stride', action='store_true', help='no modification to layer 2 of se-resnet' )
    parser.add_argument('--way1_dve', action='store_true', help='use way1 of combining dve with re-id' )
    parser.add_argument('--version',default='1',type=str, help='version of way1 to use' )
    # data
    parser.add_argument("--data_type",required=True, default = 'yak',type=str)
    parser.add_argument('--batch_size', default=32, type=int, help='batchsize')
    parser.add_argument('--num_other', default=0, type=int, help='number of images of other species in a batch')
    parser.add_argument('--triplet_sampler', action='store_true', help='making sure training batch has enough pos and neg.' )
    parser.add_argument('--use_posture', action='store_true', help='use the posture data for supervision' )
    # optimizer
    parser.add_argument('--label_smoothing', action='store_true', help='adds label smoothing' )
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay. More Regularization Smaller Weight.')
    parser.add_argument('--total_epoch', default=60, type=int, help='total training epoch')
    # backbone
    parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
    parser.add_argument('--stride', default=1, type=int, help='stride')
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
    parser.add_argument('--use_swin', action='store_true', help='use swin transformer 224x224' )
    parser.add_argument('--use_resnet', action='store_true', help='use resnet 64 dve' )
    parser.add_argument('--use_resnet_complete', action='store_true', help='use resnet_dve_complete' )
    parser.add_argument('--use_cnn5_v1', action='store_true', help='use tiger_cnn5_v1' )
    # loss
    parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
    #parser.add_argument('--freeze_dve', default=0, type=int, help='the first few epoch that needs to freeze dve')
    #parser.add_argument('--freeze_always', action='store_true', help='always keep the first layers for dve' )
    parser.add_argument('--circle', action='store_true', help='use Circle loss' )
    parser.add_argument('--triplet', action='store_true', help='use triplet loss' )
    parser.add_argument('--ent_cls', action='store_true', help='use Classification loss for enetity' )
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--circle_loss_scale', default=1.0, type=float, help='circle_loss_scale')
    parser.add_argument('--dve_loss_scale', default=1.0, type=float, help='dve_loss_scale')

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
    if opt.joint or opt.joint_all:
        # joint dve and reid on one species
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

    if opt.joint_all:
        train_paths = [train_path_dic[opt.data_type+'_all'], ]
        probe_paths = [probe_path_dic[opt.data_type], ]
    else:
        train_paths = [train_path_dic[opt.data_type], ]
        probe_paths = [probe_path_dic[opt.data_type], ]
    
    print(train_paths)
    
    if opt.triplet_sampler:#classic triplet sampling
        #DOES NOT support joint all with all species for now.
        train_data, val_data, num_classes = load_triplet_direction_gallery_probe(
            root=root,
            train_paths=train_paths,
            probe_paths=probe_paths,
            signal=' ',
            input_size=(h,w),
            warper = warper
        )
        collate_fn=train_collate
    else:
        train_data, val_data, num_classes= load_direction_gallery_probe(
                root=root,
                train_paths=train_paths,
                probe_paths=probe_paths,
                signal=' ',
                input_size=(h,w),
                warper=warper
            )
        collate_fn=torch.utils.data.dataloader.default_collate
    
    
    dict_nclasses = {'yak':121,'tiger':107,'elephant':337,'all':565}
    numClasses = dict_nclasses[opt.data_type]
    if opt.joint_all:
        assert num_classes == numClasses+2
    else:
        assert num_classes == numClasses
    
    
    image_datasets={'train':train_data,'val':val_data}
    
    if opt.joint_all:
        assert opt.data_type != 'all' # this is for training re-id with only one species,please use --joint instead
        dataloaders={'train':DataLoader(image_datasets['train'],
                                        batch_sampler = JointAllBatchSampler(image_datasets['train'],opt.batch_size,
                                                                             opt.num_other,opt.data_type),num_workers=2),
                    'val': DataLoader(image_datasets['val'], batch_size=opt.batch_size, collate_fn=torch.utils.data.dataloader.default_collate,drop_last=True,
                                                    shuffle=True, num_workers=2, pin_memory=True,
                                                    prefetch_factor=2, persistent_workers=True) # 8 workers may work faster
                    }
        

    else:
        dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=opt.batch_size, collate_fn=collate_fn,drop_last=True,
                                                    shuffle=True, num_workers=2, pin_memory=True,
                                                    prefetch_factor=2, persistent_workers=True) # 8 workers may work faster
                    ,'val': DataLoader(image_datasets['val'], batch_size=opt.batch_size, collate_fn=torch.utils.data.dataloader.default_collate,drop_last=True,
                                                    shuffle=True, num_workers=2, pin_memory=True,
                                                    prefetch_factor=2, persistent_workers=True) # 8 workers may work faster
                    }
  
        
    since = time.time()
    if opt.joint or opt.joint_all:
        inputs, classes, directions, _,_ = next(iter(dataloaders['train']))
    else:
        inputs, classes, directions = next(iter(dataloaders['train']))
        
    print(time.time()-since)
    
    
    
    
    
    # set model
    
    return_feature = True # for our re-id purpose, we always need feature
    
    if opt.use_cnn5_v1:
        model = tiger_cnn5_v1(numClasses,stride = opt.stride,linear_num=opt.linear_num,circle=return_feature,use_posture=opt.use_posture
                              ,dve=opt.joint or opt.joint_all,stackeddve=opt.stacked, smallscale=not opt.ori_stride,attn=opt.joint_att)
    elif opt.use_swin:
        model = ft_net_swin(numClasses, opt.droprate, return_feature = return_feature, linear_num=opt.linear_num, use_posture=opt.use_posture)
    elif opt.way1_dve:
        if opt.version == '1':
            model = seresnet_dve_1(numClasses, opt.droprate, circle = return_feature, linear_num=opt.linear_num,dve_dim=64,use_posture=opt.use_posture)
        elif opt.version == '2':
            model = seresnet_dve_2(numClasses, opt.droprate, circle = return_feature, linear_num=opt.linear_num,dve_dim=64,use_posture=opt.use_posture)
        elif opt.version == '1_5':
            model = seresnet_dve_1_5(numClasses, opt.droprate, circle = return_feature, linear_num=opt.linear_num,dve_dim=64,use_posture=opt.use_posture)
        elif opt.version == 'att':
            model = seresnet_dve_att(numClasses, opt.droprate, circle = return_feature, linear_num=opt.linear_num,dve_dim=64,use_posture=opt.use_posture)
        else:
            sys.exit('way1 model is not specified.')
    else:
        sys.exit('model is not specified.')
    print(model)
    
    
    # optimization
    optim_name = optim.SGD
    
    if opt.use_cnn5_v1:
        base_params = list(map(id, model.model.backbone.parameters()))
        extra_params = list(filter(lambda p: id(p) not in base_params, model.parameters()))
        base_params = model.model.backbone.parameters()
    elif opt.way1_dve:
        base_params = list(map(id, model.get_base_params()))
        extra_params = list(filter(lambda p: id(p) not in base_params, model.parameters()))
        base_params = model.get_base_params()
    elif opt.use_swin:
        base_params = list(map(id, model.model.parameters()))
        extra_params = list(filter(lambda p: id(p) not in base_params, model.parameters()))
        base_params = model.model.parameters()
    else:
        sys.exit('Invalid Condition.')
        
    optimizer_ft = optim_name([
                    {'params': base_params, 'lr': 0.1*opt.lr},
                    {'params': extra_params, 'lr': opt.lr}
                ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
    
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=opt.total_epoch*2//3, gamma=0.1)
    
    if opt.label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        # not implemented to use with --joint_all
        criterion = nn.CrossEntropyLoss()
        
    
    way1_dve=None
    if opt.way1_dve:
        #way1_dve = tiger_cnn5_64(stride=1)
        way1_dve = ft_net_64(stride=1)
        way1_dve = way1_dve.cuda()
        way1_dve = nn.DataParallel(way1_dve)
        
        if opt.model_path is not None:
            checkpoint = torch.load(opt.model_path)
            way1_dve.load_state_dict(checkpoint['state_dict'])
            print('Successfully loaded DVE from '+opt.model_path)
        way1_dve = way1_dve.eval()

        #freeze the network
        for _, para in way1_dve.named_parameters():
            para.requires_grad = False
    model = model.cuda()
    model = nn.DataParallel(model)
    train(model,criterion,optimizer_ft,exp_lr_scheduler,dataloaders,opt.total_epoch,writer,way1_dve)