To jointly train with dve:

`python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint --ent_cls --joint --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 100 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --seed 0`