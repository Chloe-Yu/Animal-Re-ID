#!/bin/bash

#SBATCH --chdir /home/yinyu/Thesis/animal-reid
#SBATCH --nodes 1
#SBATCH --ntasks 1

#SBATCH --time 6:00:00
#SBATCH --partition gpu

#SBATCH --cpus-per-task 1
#SBATCH --mem 40G
#SBATCH --gres gpu:2


#SBATCH --mail-user=yingxue.yu@epfl.ch 
#SBATCH --mail-type=ALL

#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name swin_triplet_posture_segv3_ls_tiger --batch_size 32 --ent_cls --triplet --triplet_sampler --use_posture --use_swin --lr 0.01 --total_epoch 2   -d 0,1 --data_type tiger  --warm_epoch 3 --label_smoothing >./slurm/swin_triplet_posture_segv3_ls_tiger.out 2>&1 

# 267 270 271 272
# /home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_dve_joint --ent_cls --joint --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_dve_joint.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak --ent_cls --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_yak.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all --ent_cls --joint_all --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name seresnet_dve_1_circle_posture_segv3_ls_yak  --batch_size 32 --ent_cls --circle --way1_dve --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing --model_path /home/yinyu/Thesis/cnn5-64-epoch120.pth >./slurm/seresnet_dve_1_circle_posture_segv3_ls_yak.out 2>&1

#268 273 274 275
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint --ent_cls --joint --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 100 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger --ent_cls --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 100 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all --ent_cls --joint_all --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 100 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name seresnet_dve_1_circle_posture_segv3_ls_tiger  --batch_size 32 --ent_cls --circle --way1_dve --lr 0.01 --total_epoch 100 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --model_path /home/yinyu/Thesis/cnn5-64-epoch120.pth >./slurm/seresnet_dve_1_circle_posture_segv3_ls_tiger.out 2>&1

#269 276 277 278
# /home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_elephant_dve_joint --ent_cls --joint --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 200 -d 0,1 --data_type elephant --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_elephant_dve_joint.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_elephant --ent_cls --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 200 -d 0,1 --data_type elephant --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_elephant.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_elephant_dve_joint_all --ent_cls --joint_all --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 200 -d 0,1 --data_type elephant --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_elephant_dve_joint_all.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name seresnet_dve_1_circle_posture_segv3_ls_elephant  --batch_size 32 --ent_cls --circle --way1_dve --lr 0.01 --total_epoch 200 -d 0,1 --data_type elephant --use_posture --warm_epoch 3  --label_smoothing --model_path /home/yinyu/Thesis/cnn5-64-epoch120.pth >./slurm/seresnet_dve_1_circle_posture_segv3_ls_ele.out 2>&1

#279 280
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_666 --ent_cls --joint --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 100 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --seed 666 >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_666 --ent_cls --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 100 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --seed 666 >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger.out 2>&1

#139 140
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name seresnet_dve_1_circle_posture_segv3_ls_tiger_mod  --batch_size 32 --ent_cls --circle --way1_dve --lr 0.01 --total_epoch 100 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --model_path /home/yinyu/Thesis/resnet-64-epoch120.pth >./slurm/seresnet_dve_1_circle_posture_segv3_ls_tiger_md.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name seresnet_dve_1_circle_posture_segv3_ls_yak_mod  --batch_size 32 --ent_cls --circle --way1_dve --lr 0.01 --total_epoch 100 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing --model_path /home/yinyu/Thesis/resnet-64-epoch120.pth >./slurm/seresnet_dve_1_circle_posture_segv3_ls_yak_md.out 2>&1

#170 171
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name seresnet_dve_2_circle_posture_segv3_ls_yak --version 2  --batch_size 32 --ent_cls --circle --way1_dve --lr 0.01 --total_epoch 100 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing --model_path /home/yinyu/Thesis/resnet-64-epoch120.pth >./slurm/seresnet_dve_2_circle_posture_segv3_ls_yak.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name seresnet_dve_2_circle_posture_segv3_ls_tiger --version 2  --batch_size 32 --ent_cls --circle --way1_dve --lr 0.01 --total_epoch 100 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --model_path /home/yinyu/Thesis/resnet-64-epoch120.pth >./slurm/seresnet_dve_2_circle_posture_segv3_ls_yak.out 2>&1
#172 173
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name seresnet_dve_1_5_circle_posture_segv3_ls_yak --version 1_5  --batch_size 32 --ent_cls --circle --way1_dve --lr 0.01 --total_epoch 100 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing --model_path /home/yinyu/Thesis/resnet-64-epoch120.pth >./slurm/seresnet_dve_1_5_circle_posture_segv3_ls_yak.out 2>&1
/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name seresnet_dve_1_5_circle_posture_segv3_ls_tiger --version 1_5  --batch_size 32 --ent_cls --circle --way1_dve --lr 0.01 --total_epoch 100 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --model_path /home/yinyu/Thesis/resnet-64-epoch120.pth >./slurm/seresnet_dve_1_5_circle_posture_segv3_ls_tiger.out 2>&1

