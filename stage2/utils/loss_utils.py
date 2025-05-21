import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch.fft as fft

def rgb_to_gray(images):
    weight = torch.tensor([[[[0.2989]]], 
                           [[[0.5870]]], 
                           [[[0.1140]]]], dtype=torch.float32)

    # Move the weight tensor to the same device as the images
    weight = weight.to(images.device)

    # Applying the depthwise convolution
    gray_images = F.conv2d(images, weight=weight, groups=3)

    # Since the output will have 3 separate channels, sum them up across the channel dimension
    gray_images = gray_images.sum(dim=1, keepdim=True)

    return gray_images


def RGB2YCrCb(rgb_image, with_CbCr=True):
    """
    Convert RGB format to YCrCb format.
    Used in the intermediate results of the color space conversion, because the default size of rgb_image is [B, C, H, W].
    :param rgb_image: image data in RGB format
    :param with_CbCr: boolean flag to determine if Cb and Cr channels should be returned
    :return: Y, CbCr (if with_CbCr is True), otherwise Y, Cb, Cr
    """
    R = rgb_image[:, 0:1, ::]
    G = rgb_image[:, 1:2, ::]
    B = rgb_image[:, 2:3, ::]
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.5 * B + 128/255.0
    Cr = 0.5 * R - 0.419 * G - 0.081 * B + 128/255.0

    Y = Y.clamp(0.0, 1.0)
    Cr = Cr.clamp(0.0, 1.0)
    Cb = Cb.clamp(0.0, 1.0)
    
    if with_CbCr:
        CbCr = torch.cat([Cb, Cr], dim=1)
        return Y, CbCr
    
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    """
    Convert YcrCb format to RGB format
    :param Y.
    :param Cb.
    :param Cr.
    :return.
    """
    R = Y + 1.402 * (Cr - 128/255.0)
    G = Y - 0.344136 * (Cb - 128/255.0) - 0.714136 * (Cr - 128/255.0)
    B = Y + 1.772 * (Cb - 128/255.0)
    
    RGB = torch.cat([R, G, B], dim=1)
    RGB = RGB.clamp(0,1.0)
    
    return RGB

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, image_visible, image_fused):
        # Convert RGB images to YCbCr
        ycbcr_visible = self.rgb_to_ycbcr(image_visible)
        ycbcr_fused = self.rgb_to_ycbcr(image_fused)

        # Extract CbCr channels
        cb_visible = ycbcr_visible[:, 1, :, :]
        cr_visible = ycbcr_visible[:, 2, :, :]
        cb_fused = ycbcr_fused[:, 1, :, :]
        cr_fused = ycbcr_fused[:, 2, :, :]

        # Compute L1 loss on Cb and Cr channels
        loss_cb = F.l1_loss(cb_visible, cb_fused)
        loss_cr = F.l1_loss(cr_visible, cr_fused)

        # Total color loss
        loss_color = loss_cb + loss_cr

        return loss_color

    def rgb_to_ycbcr(self, image):
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        ycbcr_image = torch.stack((y, cb, cr), dim=1)

        return ycbcr_image

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x.cuda('cuda:0'), self.weightx.cuda('cuda:0'), padding=1)
        sobely=F.conv2d(x.cuda('cuda:0'), self.weighty.cuda('cuda:0'), padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

def rgb2gray(image):
    b, c, h, w = image.size()
    if c == 1:
        return image
    image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
    image_gray = image_gray.unsqueeze(dim=1)
    return image_gray

class FFT_Loss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-3):
        super(FFT_Loss, self).__init__()
    def forward(self, x, gt):
        x = x + 1e-8
        gt = gt + 1e-8
        x_freq= torch.fft.rfftn(x, norm='backward')
        x_amp = torch.abs(x_freq)
        x_phase = torch.angle(x_freq)

        gt_freq = torch.fft.rfftn(gt, norm='backward')     
        gt_amp = torch.abs(gt_freq)
        gt_phase = torch.angle(gt_freq)

        loss_amp = torch.mean(torch.sum((x_amp - gt_amp) ** 2))
        loss_phase = torch.mean(torch.sum((x_phase - gt_phase) ** 2))
        return loss_amp, loss_phase
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class Laplacian(nn.Module):
    def __init__(self):
        super(Laplacian, self).__init__()
        kernel = [[1, 1, 1],
                  [1, -8, 1],
                  [1, 1, 1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 3, 3]
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        # Apply convolution with the Laplacian kernel
        laplacian = F.conv2d(x, self.weight, padding=1)
        return laplacian


class Fusionloss(nn.Module):
    def __init__(self,alpha,beta):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy() 
        self.laplacianconv = Laplacian()
        # self.Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')
        self.alpha = alpha  
        self.beta  = beta 

    def forward(self,generate_img,image_vis,image_ir):
        
        
        x_in_max = torch.max(image_vis,image_ir)
        loss_in  = F.l1_loss(x_in_max,generate_img)

        y_grad            = self.sobelconv(image_vis)
        ir_grad           = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint      = torch.max(y_grad,ir_grad)
        loss_grad         = F.l1_loss(x_grad_joint,generate_img_grad)
        
        y_laplacian            = self.laplacianconv(image_vis)
        ir_laplacian           = self.laplacianconv(image_ir)
        generate_img_laplacian = self.laplacianconv(generate_img)
        x_laplacian_joint      = torch.max(y_laplacian,ir_laplacian)
        loss_laplacian         = F.l1_loss(x_laplacian_joint,generate_img_laplacian)

    

        loss_total = self.alpha*loss_grad + self.beta*loss_laplacian+ loss_in
        return loss_total, loss_in, loss_grad, loss_laplacian


class PrintAccuracyAndLossCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics['train_loss']
        print(f"Epoch {trainer.current_epoch}: Train Loss {train_loss:.4f}")