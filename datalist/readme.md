## DataSet
[2019.8.1]

* mytrain.txt, myval.txt: official train-set split so that each entity has one image in validation set.
There should be 1780 in train, 107 in val, however, only 1824 heading direction annotations are provided in PGCFL. so 1718 train images are actually used.

* test.txt: official test-set, non-labeled.

### train.txt  
the image path is relative path.

| image_path | class_name | left(0)/right(1)|
| :--------: | :--------: | :-------------: |
| train/0/000384.jpg | 0 | 0 |
| train/0/000584.jpg | 0 | 0 |
| ... | ... | ... |

### gallery.txt
the image path is relative path.

| image_path | class_name |
| :--------: | :--------: |
| train/0/000384.jpg | 0 |
| train/0/000584.jpg | 0 |
| ... | ... |

### probe.txt
the image path is relative path.

| image_path | class_name |
| :--------: | :--------: |
| val/0/0002.jpg | 107 |
| val/0/0003.jpg | 107 |
| ... | ... |


### test.txt
the image path is relative path.

| image_path |
| :--------: |
| test/002107.jpg |
| test/001133.jpg |
| ... |
