import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import sys
import numpy as np
import torchvision.utils as vutils

def dark_channel(im, window_size):
    r, g, b = torch.split(im, 1, dim=1)
    dc = torch.min(torch.min(r, g), b)
    
    inverted_im = 1.0 - dc
    dark = F.max_pool2d(inverted_im, kernel_size=window_size, stride=1, padding=window_size // 2)
    dark = 1.0 - dark
    return dark

def estimate_atmosphere(im, dark, top_percent=0.001):
    b, c, h, w = im.shape
    im_flat = im.reshape(c, -1).permute(1, 0)
    dark_flat = dark.reshape(-1)

    num_top = max(int(top_percent * h * w), 1)
    top_indices = torch.argsort(dark_flat, descending=True)[:num_top]
    A = im_flat[top_indices].mean(dim=0)
    return A.unsqueeze(0)

def estimate_transmission(im, A, window_size, omega=0.95):
    A = A.reshape(1, 3, 1, 1)
    norm_im = im / A
    transmission = 1 - omega * dark_channel(norm_im, window_size)
    return transmission

def guided_filter(I, p, r, eps):
    pad = r // 2
    I_padded = F.pad(I, (pad, pad, pad, pad), mode='reflect')
    p_padded = F.pad(p, (pad, pad, pad, pad), mode='reflect')

    mean_I = F.avg_pool2d(I_padded, kernel_size=r, stride=1)
    mean_p = F.avg_pool2d(p_padded, kernel_size=r, stride=1)
    mean_Ip = F.avg_pool2d(I_padded * p_padded, kernel_size=r, stride=1)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = F.avg_pool2d(I_padded * I_padded, kernel_size=r, stride=1)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = F.avg_pool2d(F.pad(a, (pad, pad, pad, pad), mode='reflect'), kernel_size=r, stride=1)
    mean_b = F.avg_pool2d(F.pad(b, (pad, pad, pad, pad), mode='reflect'), kernel_size=r, stride=1)

    crop_h, crop_w = I.shape[2], I.shape[3]
    mean_a = mean_a[:, :, :crop_h, :crop_w]
    mean_b = mean_b[:, :, :crop_h, :crop_w]

    q = mean_a * I + mean_b
    return q

def refine_transmission(im, transmission, r=40, eps=1e-3):
    gray = (0.299 * im[:, 0, :, :] + 0.587 * im[:, 1, :, :] + 0.114 * im[:, 2, :, :]).unsqueeze(1)
    t_refined = guided_filter(gray, transmission, r, eps)
    return t_refined

def recover_image(im, t, A, t_min=0.1):
    A = A.reshape(1, 3, 1, 1)
    t = torch.clamp(t, min=t_min)
    J = (im - A) / t + A
    return torch.clamp(J, 0, 1)

def video_dehaze(input):
    window = 15
    omega  = 0.95
    r      = 40
    eps    = 1e-3
    t_min  = 0.1
    output = []
    
    for i in range(input.shape[1]//3):
        image         = input[:, i*3:(i+1)*3, :, :]
        dark          = dark_channel(image, window)
        A             = estimate_atmosphere(image, dark)
        t_rough       = estimate_transmission(image, A, window, omega)
        t_refine      = refine_transmission(image, t_rough, r, eps)
        dehazed_image = recover_image(image, t_refine, A, t_min)
        output.append(dehazed_image)
        
    output = torch.cat(output, dim=1)
    return output

# class DehazeFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         window = 15
#         omega  = 0.95
#         r      = 40
#         eps    = 1e-3
#         t_min  = 0.1
#         output = []

        
#         for i in range(input.shape[1]):
#             image         = input[:, i*3:(i+1)*3, :, :]
#             dark          = dark_channel(image, window)
#             A             = estimate_atmosphere(image, dark)
#             t_rough       = estimate_transmission(image, A, window, omega)
#             t_refine      = refine_transmission(image, t_rough, r, eps)
#             dehazed_image = recover_image(image, t_refine, A, t_min)
#             output.append(dehazed_image)
            
#         output = torch.cat(output, dim=1)
#         return output

#     @staticmethod
#     def backward(ctx, grad):
#         return grad

# class Dehaze(nn.Module):
#     def __init__(self):
#         super(Dehaze, self).__init__()

#     def forward(self, input):
#         return DehazeFunction.apply(input)