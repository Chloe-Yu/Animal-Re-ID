## DataSet

* mytrain.txt, myval.txt: official train-set split so that each entity has one image in validation set.
There should be 1780 in train, 107 in val, however, only 1824 heading direction annotations are provided in PGCFL. so 1718 train images are actually used.

* test.txt: official test-set, non-labeled.

### mytrain.txt  
the image path is relative path.

| image_path | class_name | left(0)/right(1)|
| :--------: | :--------: | :-------------: |
| train/0/000384.jpg | 0 | 0 |
| train/0/000584.jpg | 0 | 0 |
| ... | ... | ... |



### mytest.txt
the image path is relative path.

| image_path |
| :--------: |
| test/002107.jpg |
| test/001133.jpg |
| ... |
