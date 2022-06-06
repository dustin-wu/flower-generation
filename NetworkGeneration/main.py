import os
import argparse
from ae_model import IMAE
from gan_model import IMGAN

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Number of epochs [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00005, type=float, help="Learning rate [0.00005]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.9, type=float, help="Adam optimizer momentum [0.9]")
parser.add_argument("--dataset", action="store", dest="dataset", default="all_vox256_img", help="The name of dataset")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Checkpoint directory [checkpoint]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="./data", help="Dataset directory [data]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/", help="Samples directory [samples]")
parser.add_argument("--sample_vox_size", action="store", dest="sample_vox_size", default=64, type=int, help="Voxel resolution for training [64]")
parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
parser.add_argument("--start", action="store", dest="start", default=0, type=int, help="Starting index for output shapes [0]")
parser.add_argument("--end", action="store", dest="end", default=16, type=int, help="Ending index for output shapes [16]")
parser.add_argument("--ae", action="store_true", dest="ae", default=False, help="True for ae [False]")
parser.add_argument("--gan", action="store_true", dest="gan", default=False, help="True for gan [False]")
parser.add_argument("--getz", action="store_true", dest="getz", default=False, help="True for getting latent codes [False]")
args = parser.parse_args()

if not os.path.exists(args.sample_dir):
    os.makedirs(args.sample_dir)

if args.ae:
    im_ae = IMAE(args)

    if args.train:
        im_ae.train()
    elif args.getz:
        im_ae.get_z_dataset(args)
    else:
        im_ae.ae_mesh_test(args)
elif args.gan:
    if args.train:
        im_gan = IMGAN(args)
        im_gan.train()
    else:
        im_gan = IMGAN(args)
        generated_z = im_gan.generate_samples(16)
        im_ae = IMAE(args)
        im_ae.ae_z_test(args, generated_z, 128)

else:
    print("Error: no model operation specified!")