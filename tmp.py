import argparse
import torch
from torch.utils.data import DataLoader
from dataloader import train_collate,load_direction_gallery_probe,load_triplet_direction_gallery_probe,load_dve_pair,JointAllBatchSampler
from tps import Warper
version =  torch.__version__


def train(dataloaders):
    dataset_sizes1 = {'train': len(dataloaders['train'].dataset),'val': len(dataloaders['val'].dataset) }
    if opt.triplet_sampler:
        iter_per_epoch = dataset_sizes['train']//(opt.batch_size//3)
    elif opt.joint_all:
        iter_per_epoch = len(dataloaders['train'].batch_sampler)
    else:
        iter_per_epoch = dataset_sizes['train']//opt.batch_size
        
    print(dataset_sizes1,dataset_sizes)  
    
  
    

    

if __name__ =='__main__':
    
    assert int(version[0])>0 or int(version[2]) > 3 # for the new version like 0.4.0, 0.5.0 and 1.0.0
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--device',default='0,1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
    parser.add_argument("-m", "--model_path", required=False, default=None)
    parser.add_argument('--joint', action='store_true', help='jointly training dve and re-id' )
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

    opt = parser.parse_args()

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
    root = '/home/yinyu/Thesis/data/Animal-Seg-V3/'
    
    if opt.joint_all:
        train_paths = [train_path_dic['all'], ]
        probe_paths = [probe_path_dic[opt.data_type], ]
    else:
        train_paths = [train_path_dic[opt.data_type], ]
        probe_paths = [probe_path_dic[opt.data_type], ]
    
    
    
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
    
    dataset_sizes = {'train': len(train_data),'val': len(val_data) }
    dict_nclasses = {'yak':121,'tiger':107,'elephant':337,'all':565}
    numClasses = dict_nclasses[opt.data_type]
    if opt.joint_all:
        assert num_classes == 565
    else:
        assert num_classes == numClasses
    
    
    image_datasets={'train':train_data,'val':val_data}
    
    if opt.joint_all:
        assert opt.data_type != 'all' # this is for training re-id with only one species,please use --joint instead
        dataloaders={'train':DataLoader(image_datasets['train'],
                                        batch_sampler = JointAllBatchSampler(image_datasets['train'],opt.batch_size,
                                                                             opt.num_other,opt.data_type),num_workers=2),
                    'val': DataLoader(image_datasets['val'], batch_size=opt.batch_size, collate_fn=torch.utils.data.dataloader.default_collate,
                                                    shuffle=True, num_workers=2, pin_memory=True,
                                                    prefetch_factor=2, persistent_workers=True) # 8 workers may work faster
                    }
        

    else:
        dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=opt.batch_size, collate_fn=collate_fn, drop_last=True,
                                                    shuffle=True, num_workers=2, pin_memory=True,
                                                    prefetch_factor=2, persistent_workers=True) # 8 workers may work faster
                    ,'val': DataLoader(image_datasets['val'], batch_size=opt.batch_size, collate_fn=torch.utils.data.dataloader.default_collate,
                                                    shuffle=True, num_workers=2, pin_memory=True,
                                                    prefetch_factor=2, persistent_workers=True) # 8 workers may work faster
                    }
  

   
    train(dataloaders)