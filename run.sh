/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name swin_triplet_posture_segv3_ls_tiger --batch_size 32 --ent_cls --triplet --triplet_sampler --use_posture --use_swin --lr 0.01 --total_epoch 2   -d 0,1 --data_type tiger  --warm_epoch 3 --label_smoothing >./slurm/swin_triplet_posture_segv3_ls_tiger.out 2>&1 

/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_dve_joint --ent_cls --joint --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 2 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_dve_joint.out 2>&1

/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak --ent_cls --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 2 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_yak.out 2>&1

/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all --ent_cls --joint_all --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 2 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all.out 2>&1

/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name seresnet_dve_1_circle_posture_segv3_ls_ele  --batch_size 32 --ent_cls --circle --way1_dve --lr 0.01 --total_epoch 2 -d 0,1 --data_type elephant --use_posture --warm_epoch 3  --label_smoothing --model_path /home/yinyu/Thesis/cnn5-64-epoch120.pth >./slurm/seresnet_dve_1_circle_posture_segv3_ls_ele.out 2>&1