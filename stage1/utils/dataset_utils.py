import os
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch
from utils.image_utils import random_augmentation, crop_img
from utils.image_io import image_read_cv2

import cv2


class TrainDataset(Dataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__()
        self.args = args
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)
        self.de_dict = {'denoise15': 0,'denoise25': 1,'denoise50': 2, 'dehaze': 3,  'defocusdeblur': 4, 'stripe': 5}
        self._init_ids()
        self._merge_ids()
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])
        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise15' in self.de_type:
            self._init_noise15_ids()
        if 'denoise25' in self.de_type:
            self._init_noise25_ids()
        if 'denoise50' in self.de_type:
            self._init_noise50_ids()
        if 'stripe' in self.de_type:
            self._init_stripe_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'defocusdeblur' in self.de_type:
            self._init_defocusdeblur_ids()
        random.shuffle(self.de_type)

    def _init_noise15_ids(self):
        ref_file = self.args.data_file_dir + "noisy/noise15.txt"
        temp_ids = []
        temp_ids += [self.args.denoise15_dir + id_.strip() for id_ in open(ref_file)]
        name_list = os.listdir(self.args.denoise15_dir)
        self.noise15_ids = [{"clean_id": x, "de_type": 0} for x in temp_ids]
        self.num_noise15 = len(temp_ids)
        print("Total Denoise15 Ids : {}".format(self.num_noise15))
    
    def _init_noise25_ids(self):
        ref_file = self.args.data_file_dir + "noisy/noise25.txt"
        temp_ids = []
        temp_ids += [self.args.denoise25_dir + id_.strip() for id_ in open(ref_file)]
        name_list = os.listdir(self.args.denoise25_dir)
        self.noise25_ids = [{"clean_id": x, "de_type": 1} for x in temp_ids]
        self.num_noise25 = len(temp_ids)
        print("Total Denoise25 Ids : {}".format(self.num_noise25))
        
    def _init_noise50_ids(self):
        ref_file = self.args.data_file_dir + "noisy/noise50.txt"
        temp_ids = []
        temp_ids += [self.args.denoise50_dir + id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise50_dir)
        self.noise50_ids = [{"clean_id": x, "de_type": 2} for x in temp_ids]
        self.num_noise50 = len(temp_ids)
        print("Total Denoise50 Ids : {}".format(self.num_noise50))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.args.data_file_dir + "hazy/haze.txt"
        temp_ids += [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids]
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

        
    def _init_defocusdeblur_ids(self):
        temp_ids = []
        blur = self.args.data_file_dir + "blur/DefocusBlur.txt"
        temp_ids += [self.args.defocusdeblur_dir + id_.strip() for id_ in open(blur)]
        self.defocusdeblur_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]
        self.defocusdeblur_ids = self.defocusdeblur_ids
        self.deblur_counter = 0
        self.num_defocusdeblur = len(self.defocusdeblur_ids)
        print('Total DefocusBlur Ids : {}'.format(self.num_defocusdeblur))


    def _init_stripe_ids(self):
        temp_ids = []
        stripe = self.args.data_file_dir + "stripe/stripe.txt"
        temp_ids += [self.args.stripe_dir + id_.strip() for id_ in open(stripe)]
        self.stripe_ids = [{"clean_id": x, "de_type": 5} for x in temp_ids]
        self.stripe_ids = self.stripe_ids
        self.stripe_counter = 0
        self.num_stripe = len(self.stripe_ids)
        print('Total Stripe Ids : {}'.format(self.num_stripe))


    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)
        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]


        return patch_1, patch_2

    def _get_denoise15_name(self, noise_name):
        gt_name = noise_name.replace("degrad", "visible")
        return gt_name
        
    def _get_denoise25_name(self, noise_name):
        gt_name = noise_name.replace("degrad", "visible")
        return gt_name

    def _get_denoise50_name(self, noise_name):
        gt_name = noise_name.replace("degrad", "visible")
        return gt_name
                
    def _get_nonstripe_name(self, stripe_name):
        gt_name = stripe_name.replace("degrad", "infrared")
        return gt_name

        
    def _get_defocusdeblur_name(self, deblur_name):
        gt_name = deblur_name.replace("degrad", "visible")
        return gt_name
        
    def _get_nonhazy_name(self, hazy_name):
        gt_name = hazy_name.replace("degrad", "visible")
        return gt_name
    

        
    def _merge_ids(self):
        self.sample_ids = []
        if "denoise15" in self.de_type:
            self.sample_ids += self.noise15_ids
        if "denoise25" in self.de_type:
            self.sample_ids += self.noise25_ids
        if "denoise50" in self.de_type:
            self.sample_ids += self.noise50_ids
        if "stripe" in self.de_type:
            self.sample_ids += self.stripe_ids
        if "dehaze" in self.de_type:
            self.sample_ids += self.hazy_ids
        if "defocusdeblur" in self.de_type:
            self.sample_ids += self.defocusdeblur_ids
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        
        de_id = sample["de_type"]
        if de_id == 0:
            clean_name = self._get_denoise15_name(sample["clean_id"])
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('L')), base=16)
            clean_img = crop_img(np.array(Image.open(clean_name).convert('L')), base=16)
        if de_id == 1:
            clean_name = self._get_denoise25_name(sample["clean_id"])
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('L')), base=16)
            clean_img = crop_img(np.array(Image.open(clean_name).convert('L')), base=16)
        if de_id == 2:
            clean_name = self._get_denoise50_name(sample["clean_id"])
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('L')), base=16)
            clean_img = crop_img(np.array(Image.open(clean_name).convert('L')), base=16)
        elif de_id == 3:
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('L')), base=16)
            clean_name = self._get_nonhazy_name(sample["clean_id"])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('L')), base=16)
        elif de_id == 4:
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('L')), base=16)
            clean_name = self._get_defocusdeblur_name(sample["clean_id"])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('L')), base=16)
        elif de_id == 5:
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('L')), base=16)
            clean_name = self._get_nonstripe_name(sample["clean_id"])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('L')), base=16)

        degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))    
        clean_patch, degrad_patch = self.toTensor(clean_patch), self.toTensor(degrad_patch)
        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)

class TestDataset(Dataset):
    def __init__(self, args, task):
        super(TestDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args
        self.task_dict = {'denoise': 0, 'dehaze': 1, 'deblur': 2, 'stripe': 3}
        self.toTensor = ToTensor()

        self.set_dataset(task)

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.denoise_path + 'input/')
            self.ids += [self.args.denoise_path + 'input/' + id_ for id_ in name_list]
        if self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 2:
            self.ids = []
            name_list = os.listdir(self.args.deblur_path + 'input/')
            self.ids += [self.args.deblur_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 3:
            self.ids = []
            name_list = os.listdir(self.args.stripe_path + 'input/')
            self.ids += [self.args.stripe_path + 'input/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "visible")
        elif self.task_idx == 1:
            gt_name = degraded_name.replace("input", "visible")
        elif self.task_idx == 2:
            gt_name = degraded_name.replace("input", "visible")
        elif self.task_idx == 3:
            gt_name = degraded_name.replace("input", "infrared")

        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('L')), base=16)
        gray =  self.toTensor(np.array(Image.open(clean_path).convert('L')))
        degraded_name = degraded_path.split('/')[-1][:-4]
        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('L')), base=16)
        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        data_YCbCr2 = image_read_cv2(clean_path, mode='YCrCb')[np.newaxis, np.newaxis, ...].squeeze()
               
        return [degraded_name], degraded_img, clean_img, data_YCbCr2, gray

    def __len__(self):
        return self.length


