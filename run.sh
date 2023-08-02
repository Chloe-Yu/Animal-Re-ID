#!/bin/bash

#SBATCH --chdir /home/yinyu/Thesis/animal-reid
#SBATCH --nodes 1
#SBATCH --ntasks 1

#SBATCH --time 14:00:00
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
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name seresnet_dve_1_5_circle_posture_segv3_ls_tiger --version 1_5  --batch_size 32 --ent_cls --circle --way1_dve --lr 0.01 --total_epoch 100 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --model_path /home/yinyu/Thesis/resnet-64-epoch120.pth >./slurm/seresnet_dve_1_5_circle_posture_segv3_ls_tiger.out 2>&1

#999 000
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_mod --num_other 8 --ent_cls --joint_all --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_mod.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_mod --num_other 8 --ent_cls --joint_all --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_mod.out 2>&1
#001 002
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_mod_stack --num_other 8 --ent_cls --joint_all --stacked --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_mod_stack.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_mod_stack --num_other 8 --ent_cls --joint_all --stacked --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_mod_stack.out 2>&1

#81 80
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_stack  --ent_cls --joint --stacked --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_stack.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_stack  --ent_cls --joint --stacked --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_stack.out 2>&1
#19 20 21
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_mod1 --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 150 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_mod1.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_mod1 --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 150 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_mod1.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_ele_dve_joint_all_mod1 --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 300 -d 0,1 --data_type elephant --use_posture --warm_epoch 3  --label_smoothing >./slurm/cnn5_v1_circle_posture_segv3_ls_ele_dve_joint_all_mod1.out 2>&1

#81 82 83
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_416 --ent_cls --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --seed 416 >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger_416.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_416 --ent_cls --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing --seed 416 >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_416.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_ele_416 --ent_cls --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 200 -d 0,1 --data_type elephant --use_posture --warm_epoch 3  --label_smoothing --seed 416 >./slurm/cnn5_v1_circle_posture_segv3_ls_ele_416.out 2>&1

#74 75 76
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_scaled --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing --circle_loss_scale 2.0 --dve_loss_scale 0.2 >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_scaled.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_scaled --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --circle_loss_scale 2.0 --dve_loss_scale 0.2  >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_scaled.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_ele_dve_joint_all_scaled --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 200 -d 0,1 --data_type elephant --use_posture --warm_epoch 3  --label_smoothing --circle_loss_scale 2.0 --dve_loss_scale 0.2 >./slurm/cnn5_v1_circle_posture_segv3_ls_ele_dve_joint_all_scaled.out 2>&1

#29 30 31
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_scaled2 --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing --circle_loss_scale 5.0 --dve_loss_scale 0.5 >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_scaled2.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_scaled2 --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --circle_loss_scale 5.0 --dve_loss_scale 0.5  >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_scaled2.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_ele_dve_joint_all_scaled2 --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 200 -d 0,1 --data_type elephant --use_posture --warm_epoch 3  --label_smoothing --circle_loss_scale 5.0 --dve_loss_scale 0.5 >./slurm/cnn5_v1_circle_posture_segv3_ls_ele_dve_joint_all_scaled2.out 2>&1

#84 85 86
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_scaled_test --seed 416 --test --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing --circle_loss_scale 2.0 --dve_loss_scale 0.2 >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_scaled_test.out 2>&1
# /home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_scaled_test --seed 416 --test --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --circle_loss_scale 2.0 --dve_loss_scale 0.2 >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_scaled_test.out 2>&1
# /home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_ele_dve_joint_all_scaled_test --seed 416 --test --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 200 -d 0,1 --data_type elephant --use_posture --warm_epoch 3  --label_smoothing --circle_loss_scale 2.0 --dve_loss_scale 0.2 >./slurm/cnn5_v1_circle_posture_segv3_ls_ele_dve_joint_all_scaled_test.out 2>&1

#52 53 54
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_416_test --test --ent_cls --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --seed 416 >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger_666.out 2>&1
# /home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_416_test --test --ent_cls --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing --seed 416 >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_666.out 2>&1
# /home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_ele_416_test --test --ent_cls --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 200 -d 0,1 --data_type elephant --use_posture --warm_epoch 3  --label_smoothing --seed 416 >./slurm/cnn5_v1_circle_posture_segv3_ls_ele_666.out 2>&1

#90 91 92
# /home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_scaled_test_phased --seed 416 --test --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing --circle_loss_scale 2.0 --dve_loss_scale 0.2 --dve_start 20 >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_scaled_test.out 2>&1
# /home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_scaled_test_phased --seed 416 --test --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 150 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --circle_loss_scale 2.0 --dve_loss_scale 0.2 --dve_start 20 >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_scaled_test.out 2>&1
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_ele_dve_joint_all_scaled_test_phased --seed 416 --test --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 200 -d 0,1 --data_type elephant --use_posture --warm_epoch 3  --label_smoothing --circle_loss_scale 2.0 --dve_loss_scale 0.2 --dve_start 20 >./slurm/cnn5_v1_circle_posture_segv3_ls_ele_dve_joint_all_scaled_test.out 2>&1

#61 62 63
#/home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_scaled_test_correct --seed 416 --test --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing --circle_loss_scale 2.0 --dve_loss_scale 0.2 >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_scaled_test.out 2>&1
# /home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_scaled_test_correct --seed 416 --test --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 150 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --circle_loss_scale 2.0 --dve_loss_scale 0.2 >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_scaled_test.out 2>&1
# /home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_ele_dve_joint_all_scaled_test_correct --seed 416 --test --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 200 -d 0,1 --data_type elephant --use_posture --warm_epoch 3  --label_smoothing --circle_loss_scale 2.0 --dve_loss_scale 0.2 >./slurm/cnn5_v1_circle_posture_segv3_ls_ele_dve_joint_all_scaled_test.out 2>&1

#77 78 79
# /home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_scaled2_test_correct --seed 416 --test --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing  --dve_loss_scale 0.5  >./slurm/cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all_scaled_test.out 2>&1
# /home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_scaled2_test_correct --seed 416 --test --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 150 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing --dve_loss_scale 0.5  >./slurm/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint_all_scaled_test.out 2>&1
 /home/yinyu/miniconda3/envs/thesis/bin/python train.py --name cnn5_v1_circle_posture_segv3_ls_ele_dve_joint_all_scaled2_test_correct --seed 416 --test --num_other 8 --ent_cls --joint_all --batch_size 40 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 200 -d 0,1 --data_type elephant --use_posture --warm_epoch 3  --label_smoothing  --dve_loss_scale 0.5  >./slurm/cnn5_v1_circle_posture_segv3_ls_ele_dve_joint_all_scaled_test.out 2>&1

