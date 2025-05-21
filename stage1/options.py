import argparse
parser = argparse.ArgumentParser()


parser.add_argument('--epochs', type=int, default=40, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default='7',help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of model.')
parser.add_argument('--de_type', nargs='+', default=['denoise15', 'denoise25', 'denoise50', 'defocusblur', 'dehaze', 'stripe'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='path of txt for training set')
parser.add_argument('--denoise15_dir', type=str, default='/root/autodl-tmp/Data/Train/degrad/noise15/',
                    help='where clean images of denoising saves.')
parser.add_argument('--denoise25_dir', type=str, default='/root/autodl-tmp/Data/Train/degrad/noise25/',
                    help='where clean images of denoising saves.')
parser.add_argument('--denoise50_dir', type=str, default='/root/autodl-tmp/Data/Train/degrad/noise50/',
                    help='where clean images of denoising saves.')
parser.add_argument('--defocusdeblur_dir', type=str, default='/root/autodl-tmp/Data/Train/degrad/DefocusBlur/',
                    help='where clean images of debluring saves.')
parser.add_argument('--dehaze_dir', type=str, default='/root/autodl-tmp/Data/Train/degrad/haze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--stripe_dir', type=str, default='/root/autodl-tmp/Data/Train/degrad/stripe/',
                    help='where training images of stripe saves.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/", help='Name of the Directory where the checkpoint is to be saved')
parser.add_argument("--wblogger",type=str,default="TG-ECNet",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--use_ckpt",type=str,default=False,help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--ckpt_dir",type=str,default="",help="Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default='8', help="Number of GPUs to use for training")

options = parser.parse_args()