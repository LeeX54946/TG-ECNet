import os
import numpy as np
import argparse
import subprocess
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from utils.dataset_utils import TestDataset
from utils.image_io import save_image_tensor, gray_to_rgb
from net.TG_ECNet import TG_ECNet
import kornia
import cv2
from utils.loss_utils import *
import warnings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


class TGECNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = TG_ECNet()
        self.loss_fn = nn.L1Loss(reduction='mean')
        self.FusionLoss = Fusionloss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored, loss_b = self.net(degrad_patch)
        L1_V = self.loss_fn(restored, clean_patch)
        loss_grad, loss_laplacian = self.FusionLoss(clean_patch, restored)
        loss = L1_V.cuda('cuda:0') + 0.1 * loss_b.cuda('cuda:0') + 10 * loss_grad.cuda('cuda:0')
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_last_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=5, max_epochs=180)
        return [optimizer], [scheduler]


def test_Degradation(net, dataset, taskname, task="derain"):
    output_path = testopt.output_path + task + '/' + taskname
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, num_workers=0)

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch, data_YCbCr2, gray) in tqdm(testloader):
            degrad_patch, clean_patch, data_YCbCr2, gray = degrad_patch.cuda(), clean_patch.cuda(), data_YCbCr2.cuda(), gray.cuda()

            restored = net(degrad_patch)
            restored = restored[0]
            restore = torch.nn.functional.interpolate(restored, size=gray.cpu().squeeze().shape)
            restore = gray_to_rgb(restore, data_YCbCr2)

            save_image_tensor(restore, output_path + degraded_name[0] + '.png')


if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=6, help='6 for all-in-one')
    parser.add_argument('--stripe_path', type=str, default="/root/autodl-tmp/Data/Test/Stripe/",
                        help='save path of test stripe infrared images')
    parser.add_argument('--deblur_path', type=str, default="/root/autodl-tmp/Data/Test/DefocusBlur/",
                        help='save path of test blur visible images')
    parser.add_argument('--denoise_path', type=str, default="/root/autodl-tmp/Data/Test/denoise/",
                        help='save path of test noisy visible images')
    parser.add_argument('--dehaze_path', type=str, default="/root/autodl-tmp/Data/Test/dehaze/",
                        help='save path of test hazy visible images')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="stage1_pretrained.ckpt", help='checkpoint save path')
    testopt = parser.parse_args()
    os.makedirs(testopt.output_path, exist_ok=True)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)
    ckpt_path = "ckpt/" + testopt.ckpt_name
    denoise_splits = ["noise15/FMB/", "noise15/M3FD/", "noise15/LLVIP/", "noise15/MSRS/", "noise15/DroneRGBT/", "noise15/DroneVehicle/", 
                      "noise25/FMB/", "noise25/M3FD/", "noise25/LLVIP/", "noise25/MSRS/", "noise25/DroneRGBT/", "noise25/DroneVehicle/",
                      "noise50/FMB/", "noise50/M3FD/", "noise50/LLVIP/", "noise50/MSRS/", "noise50/DroneRGBT/", "noise50/DroneVehicle/"]
    dehaze_splits = ["M3FD/", 'FMB/', 'LLVIP/', "DroneRGBT/", "DroneVehicle/"]
    deblur_splits = ['MSRS/', 'RoadScene/', 'LLVIP/', "DroneRGBT/", "DroneVehicle/"]
    stripe_splits = ["M3FD/", "LLVIP/", "DroneRGBT/", "DroneVehicle/"]

    print("CKPT name : {}".format(ckpt_path))
    net = TGECNet.load_from_checkpoint(checkpoint_path=ckpt_path)
    model = torch.save(net.state_dict(), 'stage1.pth')
    #print(net.state_dict())
    #for key, value in net.state_dict().items():
        #print(key, value.size())
    net.cuda().eval()

    if testopt.mode == 6:

        denoise_base_path = testopt.denoise_path
        dehaze_base_path = testopt.dehaze_path
        deblur_base_path = testopt.deblur_path
        stripe_base_path = testopt.stripe_path

        print('Start testing Denoising data')
        for name in denoise_splits:
            print('Start testing denoise data of {}'.format(name))
            testopt.denoise_path = os.path.join(denoise_base_path, name)
            denoise_set = TestDataset(testopt, task="denoise")
            test_Degradation(net, denoise_set, name, task="denoise")

        print('Start testing Dehazing data')
        for name in dehaze_splits:
            print('Start testing dehaze data of {}'.format(name))
            testopt.dehaze_path = os.path.join(dehaze_base_path, name)
            dehaze_set = TestDataset(testopt, task="dehaze")
            test_Degradation(net, dehaze_set, name, task="dehaze")

        print('Start testing Deblurring data')
        for name in deblur_splits:
            print('Start testing deblur data of {}'.format(name))
            testopt.deblur_path = os.path.join(deblur_base_path, name)
            deblur_set = TestDataset(testopt, task="deblur")
            test_Degradation(net, deblur_set, name, task="deblur")

        print('Start testing stripe data')
        for name in stripe_splits:
            print('Start testing stripe data of {}'.format(name))
            testopt.stripe_path = os.path.join(stripe_base_path, name)
            stripe_set = TestDataset(testopt, task="stripe")
            test_Degradation(net, stripe_set, name, task="stripe")

