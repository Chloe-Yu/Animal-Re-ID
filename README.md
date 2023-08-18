# Animal Re-ID 



## Getting Started

### Dependencies

Tested under python 3.8. Dependent packages can be found in requirements.txt

### Data Preparation
#### ELPephants

Please contact the authors of [ELPephants](https://openaccess.thecvf.com/content_ICCVW_2019/html/CVWC/Korschens_ELPephants_A_Fine-Grained_Dataset_for_Elephant_Re-Identification_ICCVW_2019_paper.html) for access to the dataset.  

We used the [official implementation of YOLOv7](https://github.com/WongKinYiu/yolov7) and yolov7.pt to detect elephants and get the biggest bounding box per image for cropping. The cropped images are used below. To get the same bounding box, move preprocess/cut.py to yolov7 repo and run
```
python3 cut.py --source path/ELPephant (elephant ID system)-selected/images --weights yolov7.pt
```
and move the cropped images into data/elephant.


#### ATRW

The train and test dataset can be found [here](https://cvwc2019.github.io/challenge.html).

Please put the four folders atrw_reid_train/, test/, atrw_anno_reid_train/, atrw_anno_reid_test/ under data/tiger/.


#### YakReID-103
Please contact the authors of [YakReID-103](https://ieeexplore.ieee.org/abstract/document/9484341) for access to the dataset.

Move IJCB 2021/hard-test/val under data/ and rename it yak_test/. Move IJCB 2021/train under data/yak/


The the data folder structure should be 
```
  {repo_root}
  ├── data
  |   ├── elephant
  |       ├── 0.jpg
  │       ├── 1.jpg
  │       ├── 2.jpg
  |       ├── ...
  |   ├── tiger
  │   │   ├── atrw_reid_train
  │   │   ├── atrw_anno_reid_train
  │   │   ├── atrw_anno_reid_test
  │   │   └── test
  │   │       ├── 000000.jpg
  │   │       ├── 000004.jpg
  │   │       ├── 000005.jpg
  │   │       ├── 000006.jpg
  │   │       ├── 000008.jpg
  │   │       └── ...
  │   ├── yak_test
  │   ├── yak
  │   └── ...
```
 
run
```
cd preprocess
python prep_data.py
```
to move the train, val, test data in place.

### Background Removal

To get the masked out images, first download the pretrained sam model [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and put it under preprocess/, then run
```
python3 mask_img.py
```
You might want to run it on GPU to save time.

### Train


To train our model on ATRW:

```
python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint --use_cnn5_v1 --data_type tiger --ent_cls --circle --use_posture --joint --batch_size 30 --lr 0.01 --total_epoch 80 -d 0,1  --warm_epoch 3 --label_smoothing --triplet_sampler --circle_loss_scale 2.0 --dve_loss_scale 0.2 
```
To train on YakReID-103, and ELPephants, change --data_type tiger to --data_type yak and --data_type elephant, respectively.



### Test

To test a model trained on tiger:

```
python test_rerank.py --concat --name tiger_cnn5_v1 -d tiger -mt tiger --gpu_ids 0,1 --joint -m {repo_root}/model/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint/net_last.pth
```
--name specifies the model type.

-d specifies the dataset to test on, it can be yak, elephant, tiger or all/

-mt specifies which species the model was trained on for re-id.

--joint indicates that the model is trained with dve loss.

-m specifies the model path.

And to disable reranking:
```
python test.py --concat --name tiger_cnn5_v1 -d tiger -mt tiger --gpu_ids 0,1 --joint -m {repo_root}/model/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint/net_last.pth
```




