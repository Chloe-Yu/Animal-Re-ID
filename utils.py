import logging
import torch
import torch.nn as nn
import os
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label,name,best=False,last=False):
    if last:
        save_filename = 'net_last.pth'
    elif best:
        save_filename = 'net_best.pth'
    else:
        save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.module.state_dict(), save_path)
    
def load_network(network,model_path,use_gpu):
    #save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    checkpoint = torch.load(model_path)
    if "module" == list(checkpoint.keys())[0][:6]:
        if use_gpu:
            network = network.cuda()
            network = nn.DataParallel(network)
        else:
            checkpoint = {k[7:]:v for k,v in checkpoint.items()}
        network.load_state_dict(checkpoint)
    elif 'net_state_dict' in checkpoint.keys():
        network.load_state_dict(checkpoint['net_state_dict'],strict=False)
        if use_gpu:
            network = network.cuda()
            network = nn.DataParallel(network)
    else:
        network.load_state_dict(checkpoint)
        if use_gpu:
            network = network.cuda()
            network = nn.DataParallel(network)
    
    
    print('Successfully loaded model')
    return network