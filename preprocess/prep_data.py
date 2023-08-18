import pandas as pd
import os
from data_utils import copyfile
#####################
#Train & val data folder: '../data/Animal/train' and '../data/Animal/val'
#Test data folder: '../data/elephant_test/'
#####################
path = os.getcwd()
parent = os.path.dirname(path)
def prep_elephant(train_save_path,val_save_path):
    filtered_df = pd.read_csv('./re_mapped_filtered_anno_elephant.csv')
    train_ents = sorted(list(filtered_df[filtered_df['train']==True]['new_id'].unique())) 
    download_path = parent+"/data/elephant/"
    train_path = download_path
    if not os.path.isdir(parent+"/data/Animal"):
        os.mkdir(parent+"/data/Animal")
        
    # train and validation data
    for ent in train_ents:
        imgs = filtered_df[filtered_df['new_id']==ent]
        val = len(imgs)>2
        for _,img in imgs.iterrows():
            old_name = img['old_name']
            new_name = img['new_name']
            
            src_path = train_path + old_name
            dst_path = train_save_path + '/' + str(ent+500)
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
                if val:
                    dst_path = val_save_path + '/' + str(ent+500)  #first image is used as val image
                    os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + new_name)
    
    #test data
    test_imgs = filtered_df[filtered_df['train'] == False]
    dst_path = parent+'/data/elephant_test/'
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    for _,img in test_imgs.iterrows():
        old_name = img['old_name']
        new_name = img['new_name']
        
        src_path = train_path + old_name
        copyfile(src_path, dst_path + '/' + new_name)


#####################
#Train & val data folder: '../data/Animal/train' and '../data/Animal/val'
#Test data folder: '../data/tiger/test'
#####################
def prep_tiger(train_save_path,val_save_path):
    train_imgs = pd.read_csv(parent+'/data/tiger/atrw_anno_reid_train/reid_list_train.csv',header=None)
    train_imgs.rename(columns={0:'id',1:'imgname'},inplace=True)
    ents = list(train_imgs['id'].unique())
    #m = {ent:i for i,ent in enumerate(sorted(ents))}
    #train_imgs['new_id'] = train_imgs['id'].map(m)
    
    train_path = parent+"/data/tiger/atrw_reid_train/train/"

    for ent in ents:
        imgs = list(train_imgs[train_imgs['id']==ent]['imgname'])
        for img in imgs:
            src_path = train_path + img
            dst_path = train_save_path + '/' + str(ent)
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
                dst_path = val_save_path + '/' + str(ent)  #first image is used as val image
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + img)

#####################
#Train & val data folder: '../data/Animal/train' and '../data/Animal/val'
#Test data folder: '../data/yak_test'
#####################         
def prep_yak(train_save_path,val_save_path):
    path = os.getcwd()
    parent = os.path.dirname(path)
    for dir in os.listdir(parent+'/data/yak/train'):
        if dir!='.DS_Store':
            images = os.listdir(parent+'/data/yak/train/'+dir)
            val = len(images)>2
            ent = int(dir)+300
            
            src_path = parent+'/data/yak/train/' + dir + '/'
            
            for file in images:
                dst_path = train_save_path + '/' + str(ent)+'/'
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                    if val:
                        dst_path = val_save_path + '/' + str(ent)+'/'  #first image is used as val image
                        os.mkdir(dst_path)
                copyfile(src_path+file,dst_path+file)

def prep_all():
    if not os.path.isdir(parent+"/data/Animal"):
        os.mkdir(parent+"/data/Animal")

    train_save_path = parent+"/data/Animal/train"
    val_save_path = parent+"/data/Animal/val"

    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)
        os.mkdir(val_save_path)
    
    prep_tiger(train_save_path,val_save_path)
    prep_yak(train_save_path,val_save_path)
    prep_elephant(train_save_path,val_save_path)



if __name__ == '__main__':
    prep_all()