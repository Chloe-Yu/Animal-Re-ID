ELPephants

Please contact xxx for access to the dataset.  

We used the [official implementation of YOLOv7](https://github.com/WongKinYiu/yolov7) and yolov7.pt to detect elephants and get the biggest bounding box per image for cropping. The cropped images are used below. To get the same bounding box, move preprocess/cut.py to yolov7 repo and run
```
python3 cut.py --source path/ELPephant (elephant ID system)-selected/images --weights yolov7.pt
```
and move the cropped images into data/elephant.


ATRW

the train and test dataset can be found [here](https://cvwc2019.github.io/challenge.html)
put the four folders atrw_reid_train/,test/, atrw_anno_reid_train/, atrw_anno_reid_test/ under data/tiger/.


YakReID-103
Please contact xxx for access.
Move IJCB 2021/hard-test/val under data/ and rename it yak_test. Move IJCB 2021/train under data/yak/


So the data folder structure is 
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
python pre_data.py
```
to move the train, val, test data in place.

To get the masked out images,run
```
python3 mask_img.py
```
You might want to run it on GPU to save time.


Finally, we have our data folders




"/data/tiger/test/"
"/data/tiger/tiger_test_masked/"

"/data/elephant_test/"
"/data/elephant_test_masked/"

"/data/yak_test/"
"/data/yak_test_masked/"