import random
import torch
import torch.nn as nn
import skvideo.io
import numpy as np
import os, sys
import time
import json
from auxiliary_modules import weights_initialization as wi
from auxiliary_modules import video_tensor_processor as vtp
from training_modules import configure as cfg

import cv2

from PIL import Image

class x265Pure(torch.autograd.Function):
    @staticmethod
    # def forward(ctx, input, CRF, rank, video_type, intermediatedir, is_train=True):
    def forward(ctx, input, CRF, rank, GOP_size, intermediatedir, is_train):
        input      = torch.clamp(input, 0.0, 1.0)
        output     = (input*255.0).round()
        bt,c,h,w   = output.size()
        frames     = output.permute(0,2,3,1)                                        # (b,h,w,c)
        frames     = frames.cpu().numpy().astype(np.uint8)
        video_name = os.path.join(intermediatedir, f"intermedia_video_{rank}.mkv")
        codec_params = f"crf={CRF}:no-info=1"
        # codec_params = f"crf={CRF}:keyint={GOP_size}:no-info=1" if is_train else f"crf={CRF}:no-info=1"
        # codec_params = f"crf={CRF}:keyint={GOP_size}:min-keyint={GOP_size}:scenecut=0:open-gop=0:no-info=1" if is_train else f"crf={CRF}:no-info=1"
        
        # encode_inputdict  = {
        #     '-s': str(w) + "x" + str(h),
        #     '-pix_fmt': 'rgb24'
        # }
        # encode_outputdict = {
        #     '-c:v'         : 'libx265',
        #     "-s"           : str(w) + "x" + str(h),
        #     '-pix_fmt'     : 'yuv420p',
        #     "-vframes"     : str(bt),
        #     "-preset"      : "veryfast",
        #     "-tune"        : "zerolatency",
        #     "-x265-params" : "crf=" + str(local_CRF) + ":no-info=1"
        # }
        encode_outputdict = {
            '-c:v'         : 'libx265',
            "-s"           : str(w) + "x" + str(h),
            '-pix_fmt'     : 'yuv420p',
            # '-pix_fmt'     : 'yuv444p',
            # "-preset"      : "veryfast",
            # "-tune"        : "zerolatency",
            "-vframes"     : str(bt),
            "-x265-params" : codec_params
        }
        T1 = time.time()
        writer = skvideo.io.FFmpegWriter(video_name, outputdict = encode_outputdict, verbosity = 0)  # inputdict use default is right.
        try:
            for i in range(bt):
                writer.writeFrame(frames[i, :, :, :])                                                # RGB write                
        except OSError as e:
            print(f"OSError encountered: {e}")
            raise
        
        writer.close()
        file_size = os.path.getsize(video_name)
        
        T1     = time.time()
        reader = skvideo.io.FFmpegReader(video_name)                               # RGB read, (h,w,c)
        decoded_frames = []                                                        # Automatically proofread the code stream header file.
        for frame in reader.nextFrame():
            decoded_frames += [torch.from_numpy(frame.copy().astype(np.float32))]
            
        
        decoded_video = torch.stack(decoded_frames,dim=0).permute(0,3,1,2) / 255.  # (b,h,w,c)->(b,c,h,w) and normalize
        return decoded_video.to(input.device), file_size
        
    @staticmethod
    def backward(ctx, grad_output, _):
        return grad_output, None, None, None, None, None                       # Keeping the number of gradients consistent with the forward in the backward is necessary


class CodecPure(nn.Module):
    def __init__(self,cfg:cfg.Configuration, sample_num:int, rank:int=0):
        super(CodecPure, self).__init__()
        self.CRF             = cfg.training_CRF
        self.GOP_size        = cfg.RUVC_GOP + 4
        self.scale_times     = cfg.rescaling_times
        self.intermediatedir = cfg.intermediatedir
        self.rank            = rank
        self.random_seed     = cfg.random_seed
        self.h265_crf_step   = sample_num // 3
        
    # def forward(self, input, video_type:str='HR', crf:float=26.0, is_train=True):
    def forward(self, input, is_train:bool=False, crf:float=26.0):
        if crf is None:
            crf = self.CRF

        # return x265Pure.apply(input, crf, self.rank, video_type, self.intermediatedir, is_train)
        return x265Pure.apply(input, crf, self.rank, self.GOP_size, self.intermediatedir, is_train)
    
    def random_CRF(self, random_seed:int=0, current_iteration:int=0):
        if self.random_seed is not None:
            random.seed(self.random_seed + random_seed)
        
        range_left  = 18.
        range_right = 23.
        if current_iteration > self.h265_crf_step and current_iteration <= self.h265_crf_step * 2:
            range_right += 5.
        elif current_iteration > self.h265_crf_step * 2 and current_iteration <= self.h265_crf_step * 3:
            range_right += 2*5.
            
        return round(random.uniform(range_left, range_right), 1)
    
class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True, INN_init=True, is_res=False):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                wi.initialize_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                wi.initialize_kaiming([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            wi.initialize_kaiming(self.conv5, 0)
        else:
            wi.initialize_xavier([self.conv1, self.conv2, self.conv3, self.conv4,self.conv5], 1)
        self.is_res = is_res
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if self.is_res:
            x5 = x5+x
        return x5

class SpaceToDepth(nn.Module):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size in {2, 4}, "Space2Depth only supports blocks size = 4 or 2"
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        S = self.block_size
        x = x.view(N, C, H // S, S, W // S, S)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * S * S, H // S, W // S)  # (N, C*bs^2, H//bs, W//bs)
        return x

    def extra_repr(self):
        return f"block_size={self.block_size}"
    
class FeatureCalapseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, scale = 4,init='xavier', gc=32, bias=True, INN_init=True, is_res=False, GOP_size=6):
        super(FeatureCalapseBlock, self).__init__()
        self.scale = scale
        self.is_res = is_res
        self.GOP_size = GOP_size
        if scale>1:
            self.ds = SpaceToDepth(scale)
            self.us = nn.PixelShuffle(scale)
        channel_in = (scale**2)*channel_in
        channel_out = (scale**2)*channel_out
        gc = (scale)*gc
        self.conv1 = nn.Conv3d(channel_in, gc, (3,3,3), 1, (1,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,3,3), 1, (1,1,1), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                wi.initialize_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                wi.initialize_kaiming([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            wi.initialize_xavier(self.conv5, 0)

    def forward(self, x,io_type="2d"):
        res = x
        if self.scale>1:
            x = self.ds(x)
        if io_type == "2d":
            bt,c,w,h = x.size()
            t = self.GOP_size
            b = bt//t
            x  = x.reshape(b,t,c,w,h).transpose(1,2)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if io_type == "2d":
            x5 = x5.transpose(1,2).reshape(bt,-1,w,h)
        if self.scale>1:
            x5 = self.us(x5)
        if self.is_res:
            x5 = x5+res
        return x5

class x265Surrogate(torch.autograd.Function):
    @staticmethod
    # def forward(ctx, input, CRF, rank, video_type, intermediatedir, is_train=True):
    def forward(ctx, input, DNN_output, CRF, rank, video_type, intermediatedir):
        input      = torch.clamp(input, 0.0, 1.0)
        output     = (input*255.0).round()
        bt,c,h,w   = output.size()
        frames     = output.permute(0,2,3,1)                                        # (b,h,w,c)
        frames     = frames.cpu().numpy().astype(np.uint8)
        video_name = os.path.join(intermediatedir, f"intermedia_video_{video_type}_{rank}.h265")
        local_CRF  = CRF
        codec_params = "crf=" + str(local_CRF) + ":no-info=1"
        inputdict  = {
            '-s': str(w) + "x" + str(h),
            '-pix_fmt': 'rgb24',
        }
        encode_outputdict = {
            '-c:v'         : 'libx265',
            "-s"           : str(w) + "x" + str(h),
            '-pix_fmt'     : 'yuv420p',
            "-vframes"     : str(bt),
            "-x265-params" : codec_params
        }
        T1 = time.time()
        writer = skvideo.io.FFmpegWriter(video_name, inputdict=inputdict, outputdict=encode_outputdict, verbosity = 0)
        try:
            for i in range(bt):
                writer.writeFrame(frames[i, :, :, :])                              # RGB write
        except OSError as e:
            print(f"OSError encountered: {e}")
            raise
        
        writer.close()
        # file_size = os.path.getsize(video_name)
        
        T1 = time.time()
        decode_outputdict = {}
        reader = skvideo.io.FFmpegReader(video_name, inputdict={}, outputdict=decode_outputdict)   # RGB read, (h,w,c)
        decoded_frames = []                                                        # skvideo == 1.1.11 the color space used is RGB
        for frame in reader.nextFrame():
            decoded_frames += [torch.from_numpy(frame.copy().astype(np.float32))]
        
        decoded_video = torch.stack(decoded_frames,dim=0).permute(0,3,1,2) / 255.0   # (b,h,w,c)->(b,c,h,w) and normalize
        ctx.save_for_backward(DNN_output, input, decoded_video)
        
        return decoded_video.to(input.device)

class CodecWithSurrogate(nn.Module):
    def __init__(self, cfg:cfg.Configuration, rank:int, sample_num:int):
        super(CodecWithSurrogate, self).__init__()
        mid_c                = 24
        self.rank            = rank
        self.h265_crf_step   = sample_num // 4
        self.lambda_corr     = 1e-5
        self.GOP_size        = cfg.RUVC_GOP
        self.random_seed     = cfg.random_seed
        self.intermediatedir = cfg.intermediatedir
        self.suggrogate_net  = nn.Sequential(
            DenseBlock(4,mid_c,INN_init=False),
            DenseBlock(mid_c,mid_c,INN_init=False,is_res=True),
            FeatureCalapseBlock(mid_c,mid_c,INN_init=True,is_res=True,GOP_size=self.GOP_size),
            FeatureCalapseBlock(mid_c,mid_c,INN_init=True,is_res=True,GOP_size=self.GOP_size),
            DenseBlock(mid_c,mid_c,INN_init=False,is_res=True),
            DenseBlock(mid_c,3,INN_init=False),
        )
        self.indicator_fuser = nn.Sequential(
            nn.Linear(2,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,1),
        )
    def forward(self, input, video_type:str='LR', crf:float=26.0):
        bt,c,h,w = input.size()
        t        = self.GOP_size
        b        = bt//t
        input_3d = input.reshape(b,t,c,h,w)
            
        temporal_indicator = torch.linspace(0,1,t).cuda(input_3d.device)
        q_indicator = torch.zeros_like(temporal_indicator).cuda(input_3d.device).fill_(crf/30)
        indicator = torch.stack([temporal_indicator,q_indicator],dim=1)
        indicator = self.indicator_fuser(indicator.unsqueeze(0))
        indicator = indicator.unsqueeze(-1).unsqueeze(-1).repeat(b,1,1,h,w)
            
        input_3d_temporal_ind    = torch.cat([input_3d,indicator],dim=2)
        sug_out = self.suggrogate_net(input_3d_temporal_ind.reshape(bt,c+1,h,w))+input
        H265_encoder_encoder_out = x265Surrogate.apply(input, sug_out, crf, self.rank, video_type, self.intermediatedir)
        x = H265_encoder_encoder_out.detach()
        y = sug_out
        mimick_loss = torch.mean((x - y)**2.0)

        vx = x - torch.mean(x,dim=0,keepdim=True)
        vy = y - torch.mean(y,dim=0,keepdim=True)

        correlation_param = torch.sum(vx * vy,dim=0,keepdim=True) / \
             (torch.sqrt(torch.sum(vx ** 2,dim=0,keepdim=True)) * torch.sqrt(torch.sum(vy ** 2,dim=0,keepdim=True))+1e-8)
        correlation_param = correlation_param.mean()
        sug_out.data = H265_encoder_encoder_out
        return sug_out, mimick_loss - self.lambda_corr*correlation_param
    
    def random_CRF(self, random_seed:int=0, current_iteration:int=0):
        if self.random_seed is not None:
            random.seed(self.random_seed + random_seed)
        
        range_left  = 18.
        range_right = 23.
        if current_iteration > self.h265_crf_step and current_iteration <= self.h265_crf_step * 2:
            range_right += 5.
        elif current_iteration > self.h265_crf_step * 2 and current_iteration <= self.h265_crf_step * 3:
            range_right += 2*5.
            
        return round(random.uniform(range_left, range_right), 1)