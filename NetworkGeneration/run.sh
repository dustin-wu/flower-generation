##### Generate 3D models with GAN #####
python main.py --dataset flowers_vox_new256 --gan --checkpoint_dir checkpoint_V2

##### Train GAN with autoencoder's z vectors #####
# python main.py --dataset flowers_vox_new256 --gan --train --epoch 200 --checkpoint_dir checkpoint_V2 --learning_rate 0.00001 --beta1 0.5

##### Generate z vectors with autoencoder #####
# python main.py --dataset flowers_vox_new256 --ae --getz --checkpoint_dir checkpoint_V2

##### Produce 3D models as encoded by autoencoder #####
# python main.py --dataset flowers_vox_new256 --ae --checkpoint_dir checkpoint_V2

##### Train Autoencoder #####
# python main.py --epoch 200 --dataset flowers_vox_new256 --train --ae --learning_rate 0.00005 --beta1 0.5 --sample_vox_size 64 --checkpoint_dir checkpoint_V2