import os
import cv2
from PIL import Image
from rembg import new_session, remove
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch
import shutil

"""images are proportionally resized with maximum height 1500 and maximum width 2250"""
def resize_imgs_in_dir(dir_path):
    dirs = os.listdir(dir_path)
    for name in dirs:
        if name[0]!='.':
            save_path = dir_path+name+'/'
            for img_name in os.listdir(dir_path + name):
                image = cv2.imread(dir_path+name+'/'+img_name)
                height, width, channels = image.shape
                if height<=1500 and width <=2250:
                    continue
                if height>1500:
                    # resize width,height
                    width = int(1500*width/height)
                    image = cv2.resize(image, (width,1500))
                if width>2250:
                    image = cv2.resize(image, (2250,int(2250*min(height,1500)/width)))
                cv2.imwrite(save_path+img_name, image)


def get_isnet_mask(entity,img_name,input_path,mask_dir,session,output_path=None):

    image = Image.open(input_path+entity+'/'+img_name)

    m = remove(image, session=session, only_mask=True,post_process_mask = True)
    bg_mask = np.logical_not(m)
    
    if mask_dir is not None:
        mask_path = mask_dir+entity+'/'
        if not os.path.isdir(mask_path):
            os.mkdir(mask_path)

        np.save(mask_path+img_name[:-3]+'npy',bg_mask)
    
    
    if output_path is not None:
        image_cv = cv2.imread(input_path+entity+'/'+img_name)
        image_cv[bg_mask] = 255.0
        save_path = output_path+entity+'/'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        cv2.imwrite(save_path+img_name, image_cv)


        
def get_isnet_masks():
    model_name = "isnet-general-use"
    session = new_session(model_name)
    path = os.getcwd()
    parent = os.path.dirname(path)
    if not os.path.isdir(parent+"/data/ISNetMask/"):
        os.mkdir(parent+"/data/ISNetMask/")
        
    for split in ['train/','val/']:
        input_path = parent+"/data/Animal/"+split
        mask_dir = parent+"/data/ISNetMask/"+split

        if not os.path.isdir(mask_dir):
            os.mkdir(mask_dir)
        
        for dir in os.listdir(input_path):
            #get isnet masks for tiger and elephant.
            if dir[0]!='.':
                for img_name in os.listdir(input_path+'/'+ dir):
                    if img_name[0]!='.' and img_name.endswith('.jpg'):
                            get_isnet_mask(dir,img_name,input_path,mask_dir,session)
    #elephant test
    input_path = parent+"/data/elephant_test/"
    mask_dir = parent+"/data/elephant_isnet_mask/"
    if not os.path.isdir(mask_dir):
        os.mkdir(mask_dir)
        
    for name in os.listdir(input_path):
        if name[0]!='.' and name.endswith('.jpg'):
            get_isnet_mask('',name,input_path,mask_dir,session)
    # tiger test
    input_path = parent+"/data/tiger/test/"
    mask_dir = parent+"/data/tiger_isnet_mask/"
    if not os.path.isdir(mask_dir):
        os.mkdir(mask_dir)
        
    for name in os.listdir(input_path):
        if name[0]!='.' and name.endswith('.jpg'):
            get_isnet_mask('',name,input_path,mask_dir,session)
            
    #yak test
    input_path = parent+"/data/yak_test/"
    output_path = parent+"/data/yak_test_masked/"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        
    dirs = os.listdir(input_path)
    for name in dirs:
        if name[0]!='.':
            for img_name in os.listdir(input_path  + name):
                if img_name[0]!='.' and img_name.endswith('.jpg'):
                    get_isnet_mask(name,img_name,input_path,None,session,output_path=output_path)



def get_iou(mask1,mask2):
    return np.sum(np.logical_and(mask1,mask2))/np.sum(np.logical_or(mask1,mask2))


def get_tiger_mask(entity,img_name,input_path,mask_dir,output_path,mask_generator):
    image = cv2.imread(input_path+entity+'/'+img_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prev_mask = np.logical_not(np.load(mask_dir+entity+'/'+img_name[:-3]+'npy'))
    h,w = prev_mask.shape
    ratio = np.sum(prev_mask)/(h*w)

    anns = mask_generator.generate(image)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    m = sorted_anns[0]['segmentation']

    iou = get_iou(m, prev_mask)
    if ratio >0.1 and iou < 0.3:
        m = np.logical_not(m)

    save_path = output_path+entity+'/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    image[np.logical_not(m)] = 255.0

    cv2.imwrite(save_path+img_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    del anns
    del sorted_anns
    del image
    del m
    del prev_mask
    torch.cuda.empty_cache()


def get_tiger_joint_masks(mask_generator):
    path = os.getcwd()
    parent = os.path.dirname(path)
    #test data
    input_path = parent+"/data/tiger/test/"
    output_path = parent+"/data/tiger/tiger_test_masked/"
    mask_dir = parent+"/data/tiger_isnet_mask/"
    
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        
    dirs = os.listdir(input_path)
    for name in dirs:
        if name[0]!='.' and name[-5]!=')':
            get_tiger_mask('',name,input_path,mask_dir,output_path,mask_generator)
    #train data
    for split in ['train','val']:
        input_path = parent+"/data/Animal/"+split+"/"
        output_path = parent+"/data/Animal-masked/"+split+"/"
        mask_dir = parent+"/data/ISNetMask/"+split+"/"
        
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        
        dirs = os.listdir(input_path)
        for name in dirs:
            if name[0]!='.' and int(name)<300:
                for img_name in os.listdir(input_path+name):
                    get_tiger_mask(name,img_name,input_path,mask_dir,output_path,mask_generator)


def filter_mask(mask,m,threshold=0.85):
    sam_m = mask['segmentation']
    overlap = np.logical_and(sam_m,m)
    ratio = np.sum(overlap)/np.sum(sam_m)
    if ratio > threshold:
        return True


def get_ele_mask(entity,img_name,mask_generator,input_path,mask_dir,output_path,threshold=0.85):

    image = cv2.imread(input_path+entity+'/'+img_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    m = np.load(mask_dir+entity+'/'+img_name[:-3]+'npy')
    m = np.logical_not(m)

    masks = mask_generator.generate(image)
    masks_filtered = [mask['segmentation'] for mask in masks if filter_mask(mask,m, threshold)]

    final_mask = masks_filtered[0]
    for i in range(1,len(masks_filtered)):
        final_mask = np.logical_or(masks_filtered[i],final_mask)

    save_path = output_path+entity+'/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
      
    image[np.logical_not(final_mask)] = 255.0
    cv2.imwrite(save_path+img_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    del masks
    del masks_filtered
    del image
    del m
    del final_mask
    torch.cuda.empty_cache()


def get_ele_joint_masks(mask_generator):
    path = os.getcwd()
    parent = os.path.dirname(path)
    
    #train data
    for split in ['train','val']:
        input_path = parent+"/data/Animal/"+split+"/"
        output_path = parent+"/data/Animal-masked/"+split+"/"
        mask_dir = parent+"/data/ISNetMask/"+split+"/"
        dirs = os.listdir(input_path)
        for name in dirs:
            if name[0]!='.' and int(name)>=500:
                for img_name in os.listdir(input_path+name):
                    get_ele_mask(name,img_name,mask_generator,input_path,mask_dir,output_path,threshold=0.5)
    #test data          
    input_path =  parent+"/data/elephant_test/"
    output_path = parent+"/data/elephant_test_masked/"
    mask_dir = parent+"/data/elephant_isnet_mask/"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        
    dirs = os.listdir(input_path)
    for name in dirs:
        if name[0]!='.' and name[-5]!=')':
            get_ele_mask('',name,mask_generator,input_path,mask_dir,output_path,threshold=0.5)


def get_yak_joint_masks(predictor):
    path = os.getcwd()
    parent = os.path.dirname(path)
    for split in ['train','val']:
        input_path = parent+"/data/Animal/"+split+"/"
        dirs = os.listdir(input_path)
        for name in dirs:
            if name[0]!='.' and int(name)<500 and int(name)>=300:
                for img_name in os.listdir(input_path+name):
                    get_yak_mask(name,img_name,split,predictor)
    
def get_yak_mask(entity,img_name,split,predictor,filter_mask=False):
    path = os.getcwd()
    parent = os.path.dirname(path)
    input_path = parent+'/data/Animal/'+split+'/'+entity+'/'+img_name
    mask_path = "/data/Animal-masked/"+split+'/'+entity+'/'+img_name[:-3]+'npy'
    output_img_path = parent+"/data/Animal-masked/"+split+"/"+entity+'/'
    if not os.path.isdir(output_img_path):
        os.mkdir(output_img_path)
    
    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    h,w,_ = image.shape
    input_box = np.array([0, 0, w, h])
    masks, scores, logits  = predictor.predict(
    point_coords= np.array([[w//2, h//2]]),
    point_labels= np.array([1]),
    box=input_box[None, :],
    multimask_output=filter_mask,
    )
    # when we set multimask_output=True
    if filter_mask:
        mask_prev = np.logical_not(np.load(mask_path))
        
        #ious = []
        best_iou=0.0
        for mask in masks:
            iou = get_iou(mask,mask_prev)
            if iou > best_iou:
                best_mask = mask
                best_iou=iou
    else:
        best_mask = masks
    image[np.logical_not(best_mask)] = 255.0
    cv2.imwrite(output_img_path+img_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))



    
    
    

def get_joint_masks():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    
    path = os.getcwd()
    parent = os.path.dirname(path)
    
    output_path = parent+"/data/Animal-masked/"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("Getting masks for tigers...")
    get_tiger_joint_masks(mask_generator)
    print("Getting masks for elephants...")
    get_ele_joint_masks(mask_generator)
    del mask_generator
    torch.cuda.empty_cache()

    print("Getting masks for yaks...")
    predictor = SamPredictor(sam)
    get_yak_joint_masks(predictor)
    
    #clean temporary masks"
    
    shutil.rmtree(parent+"/data/ISNetMask/")
    shutil.rmtree(parent+"/data/tiger_isnet_mask/")
    shutil.rmtree(parent+"/data/elephant_isnet_mask/")
    
    
    



def get_masked_imgs():
    path = os.getcwd()
    parent = os.path.dirname(path)
    resize_imgs_in_dir(parent+'/data/Animal/train/')
    resize_imgs_in_dir(parent+'/data/Animal/val/')
    get_isnet_masks()
    get_joint_masks()
    
if __name__ == '__main__':
    get_masked_imgs()