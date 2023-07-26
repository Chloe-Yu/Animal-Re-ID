To jointly train with dve:

`python train.py --name cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint --ent_cls --joint --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 100 -d 0,1 --data_type tiger --use_posture --warm_epoch 3  --label_smoothing`

models, training parameters and script will be saved under directory ./model/cnn5_v1_circle_posture_segv3_ls_tiger_dve_joint/
--ent_cls --circle --use_posture   is to specify that 2 classification loss and circle loss is used for training
--joint is to have dve loss computed on only data_type
--stacked is the variation that stacks the last 3 layers of backbone for computing dve loss.


To train dve with 3 species, use --joint_all:

`python train.py --name cnn5_v1_circle_posture_segv3_ls_yak_dve_joint_all --num_other 8 --ent_cls --joint_all --batch_size 32 --circle --use_cnn5_v1 --lr 0.01 --total_epoch 120 -d 0,1 --data_type yak --use_posture --warm_epoch 3  --label_smoothing`


--num_other species how many images of other species are included in one batch of size --batch_size.

To test a trained model on 3 species:
`python test.py --concat --name tiger_cnn5_v1 -d all -mt elephant --gpu_ids 0,1 --joint -m /home/yinyu/Thesis/animal-reid/model/cnn5_v1_circle_posture_segv3_ls_elephant_dve_joint/net_best.pth`

-mt is which species the model was trained on for re-id
--joint is when model is trained with dve loss (no matther the number of species)
-m is model path
--stacked is the variation that stacks the last 3 layers of backbone for computing dve loss.

