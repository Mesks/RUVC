import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositiveNegativeContrastiveLoss(nn.Module):
    def __init__(self, channel=30, temperature=3, max_pixel_value=1.0, device=None):
        super(PositiveNegativeContrastiveLoss, self).__init__()
        assert temperature > 0, "Temperature must be positive."
        self.temperature     = temperature
        self.max_pixel_value = max_pixel_value
        self.device          = device
        self.eps             = 1e-8
        self.mse             = MeanSquaredErrorLoss(channel=channel, device=device)

    def forward(self, output, positive, negative):
        assert output.shape == positive.shape and output.shape == negative.shape, print("LOSS ERROR::Network output shape is not same with positive or negative.")
        self.device  = output.device if self.device== None else self.device

        mse_positive = F.mse_loss(output, positive, reduction='none').mean(dim=[1, 2, 3])
        mse_negative = F.mse_loss(output, negative, reduction='none').mean(dim=[1, 2, 3])
        
        psnr_positive = 10 * torch.log10((self.max_pixel_value ** 2) / (mse_positive + self.eps))
        psnr_negative = 10 * torch.log10((self.max_pixel_value ** 2) / (mse_negative + self.eps))
        # print(psnr_positive, psnr_negative)
        contrastive = torch.exp(psnr_negative / self.temperature) / torch.exp(psnr_positive / self.temperature)
        # print(psnr_positive / self.temperature, torch.exp(psnr_positive / self.temperature), psnr_negative / self.temperature, torch.exp(psnr_negative / self.temperature), torch.log(contrastive))
        # sys.exit()
        # loss        = -torch.log(contrastive).mean()
        loss        = contrastive.mean()

        return loss
    
class MeanSquaredErrorLoss(nn.Module):
    def __init__(self, channel=18, device=None):
        super(MeanSquaredErrorLoss, self).__init__()
        '''old LF_constraint'''
        self.frame_num = channel//3
        self.device    = device
        
    def forward(self, video, constraint):
        loss = 0
        for i in range(self.frame_num):
            diff = video[:,3*i:3*(i+1),:,:] - constraint[:,3*i:3*(i+1),:,:]
            loss += torch.mean(torch.sum((diff) ** 2, (1, 2, 3)))
        # return loss
        return torch.log(loss) / self.frame_num
    
class MeanAbsoluteErrorLoss(nn.Module):
    def __init__(self, channel=18, device=None):
        super(MeanAbsoluteErrorLoss, self).__init__()
        '''old HF_constraint'''
        self.frame_num = channel//3
        self.device    = device
        
    def forward(self, video, constraint):
        loss = 0
        for i in range(self.frame_num):
            diff = video[:,3*i:3*(i+1),:,:] - constraint[:,3*i:3*(i+1),:,:]
            loss += torch.mean(torch.sum(torch.abs(diff), (1, 2, 3)))
        # return loss
        return torch.log(loss) / self.frame_num
        
    # def LF_loss(self, predicted_video, original_video):
    #     '''
    #         The L2 loss between the decoded image (resolution is 1/2 or others times the original image) and the constrained image.
    #     '''
    #     return F.mse_loss(predicted_video, original_video)
class SmoothMeanAbsoluteErrorLoss(nn.Module):
    def __init__(self, channel=18, device=None, need_log=False):
        super(SmoothMeanAbsoluteErrorLoss, self).__init__()
        '''old reconstruction_loss'''
        self.need_log  = need_log
        self.frame_num = channel//3
        self.device    = device
        self.eps       = 1e-6
        
    def forward(self, video, constraint):
        loss = 0
        for i in range(self.frame_num):
            diff = video[:,3*i:3*(i+1),:,:] - constraint[:,3*i:3*(i+1),:,:]
            loss += torch.mean(torch.sum(torch.sqrt(diff ** 2 + self.eps), (1, 2, 3)))
        # return loss / self.channel
        if self.need_log:
            return torch.log(loss / self.frame_num)
        else:
            return loss / self.frame_num
    # def reconstruction_loss(self, video, constraint):
    #     assert video.shape == constraint.shape, "Images must have the same dimensions"
    #     assert torch.all(torch.isfinite(video)), "video contains NaN or Inf"
    #     assert torch.all(torch.isfinite(constraint)), "constraint contains NaN or Inf"

    #     mse_loss = F.mse_loss(video, constraint)
    #     return mse_loss
    
class TotalVariationLoss(nn.Module):
    def __init__(self, channel=18, device=None):
        super(TotalVariationLoss, self).__init__()
        self.frame_num = channel//3
        self.device    = device
        self.eps       = 1e-6
        
    def forward(self, video):
        diff_x = torch.abs(video[:, :, :, 1:] - video[:, :, :, :-1])
        diff_y = torch.abs(video[:, :, 1:, :] - video[:, :, :-1, :])
        return 0.05 * torch.mean(diff_x) + torch.mean(diff_y)
        
class SimpleLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-6):
        super(SimpleLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            v = (x - target)**2
            # return torch.mean(torch.sum((x - target)**2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            v = torch.sqrt(diff * diff + self.eps)
        else:
            print("reconstruction loss type error!")
            return 0
        
        return torch.log(v.sum(-1).sum(-1).sum(-1).sum(-1))
    
class LossModel(nn.Module):
    def __init__(self, channel=18, device=None):
        super(LossModel, self).__init__()        
        self.LF_loss_computer = MeanSquaredErrorLoss(channel=channel, device=device)
        self.HF_loss_computer = MeanAbsoluteErrorLoss(channel=channel, device=device)
        self.re_loss_computer = SmoothMeanAbsoluteErrorLoss(channel=channel, device=device, need_log=True)
        
        self.LF_coefficient   = 6.
        self.HF_coefficient   = 3.
        self.re_coefficient   = 1.
        
    def LF_loss(self, video, positive, negative):
        return self.LF_loss_computer(video, positive) * self.LF_coefficient
            
    def HF_loss(self, video, constraint):
        return self.HF_loss_computer(video, constraint) * self.HF_coefficient
        
    def re_loss(self, video, constraint):
        return self.re_loss_computer(video, constraint) * self.re_coefficient