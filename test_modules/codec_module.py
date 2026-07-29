import random
import json
import subprocess
import torch
import torch.nn as nn
import skvideo.io
import numpy as np
import os, sys
import time
from auxiliary_modules import weights_initialization as wi
from test_modules import configure as cfg


class x265Pure(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, CRF, intermediatedir, keyint):
        input      = torch.clamp(input, 0.0, 1.0)
        output     = (input*255.0).round()
        bt,c,h,w   = output.size()
        frames     = output.permute(0,2,3,1)                                        # (b,h,w,c)
        frames     = frames.cpu().numpy().astype(np.uint8)
        video_name = os.path.join(intermediatedir, "test_intermedia_video.mkv")
        # video_name = os.path.join(intermediatedir, "test_intermedia_video.h264")
        x265_params = f"crf={CRF}:no-info=1"
        inputdict  = {
            '-s': str(w) + "x" + str(h),
            '-pix_fmt': 'rgb24',
        }
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
            "-vframes"     : str(bt),
            "-x265-params" : x265_params,
            '-metadata:s:v:0': f'RUVC_CRF={str(CRF)}'
        }
        # encode_outputdict = {
        #     '-c:v'         : 'libx264',
        #     "-s"           : str(w) + "x" + str(h),
        #     '-pix_fmt'     : 'yuv420p',
        #     "-vframes"     : str(bt),
        #     "-x265-params" : f"crf={CRF}"
        # }
        print(f"{'Encoding...':<21}", end="")
        T1 = time.time()
        # writer = skvideo.io.FFmpegWriter(video_name, inputdict=inputdict, outputdict=encode_outputdict, verbosity = 0)
        writer = skvideo.io.FFmpegWriter(video_name, outputdict=encode_outputdict, verbosity = 0)
        try:
            for i in range(bt):
                writer.writeFrame(frames[i, :, :, :])                              # RGB write
        except OSError as e:
            print(f"OSError encountered: {e}")
            raise
        print(f"Finished. Consumed {time.time() - T1:.6f} seconds")
        writer.close()
        file_size = os.path.getsize(video_name)
        
        print(f"{'Decoding...':<21}", end="")
        T1 = time.time()
        reader = skvideo.io.FFmpegReader(video_name)   # RGB read, (h,w,c)
        decoded_frames = []                                                          # skvideo == 1.1.11 the color space used is RGB
        for frame in reader.nextFrame():
            decoded_frames += [torch.from_numpy(frame.copy().astype(np.float32))]
        print(f"Finished. Consumed {time.time() - T1:.6f} seconds")
        
        decoded_video = torch.stack(decoded_frames,dim=0).permute(0,3,1,2) / 255.0   # (b,h,w,c)->(b,c,h,w) and normalize
        meta = skvideo.io.ffprobe(video_name).get("video", {}).get("tags",{}).get("tag",{})
        crf_value = next((float(tag['@value']) for tag in meta if tag.get('@key') == 'RUVC_CRF'), None)
        return decoded_video, file_size, crf_value
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None                                  # Keeping the number of gradients consistent with the forward in the backward is necessary


def _run_command(command, input_data=None):
    result = subprocess.run(
        command,
        input=input_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode(errors='replace').strip()
        raise RuntimeError(f"FFmpeg command failed: {' '.join(command)}\n{message}")
    return result.stdout


def _read_vvc_metadata(video_name):
    output = _run_command([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream_tags', '-of', 'json', video_name,
    ])
    streams = json.loads(output.decode()).get('streams', [])
    if not streams:
        raise RuntimeError('The VVC Matroska file does not contain a video stream.')
    return {key.upper(): value for key, value in streams[0].get('tags', {}).items()}


class vvencPure(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, QP, intermediatedir):
        input = torch.clamp(input, 0.0, 1.0)
        output = (input * 255.0).round()
        bt, _, h, w = output.size()
        frames = output.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        video_name = os.path.join(intermediatedir, 'test_intermedia_video.mkv')
        qp = int(QP)

        encode_command = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pixel_format', 'rgb24',
            '-video_size', f'{w}x{h}', '-framerate', '25', '-i', '-',
            '-c:v', 'libvvenc', '-qp', str(qp), '-pix_fmt', 'yuv420p10le',
            '-frames:v', str(bt),
            '-metadata:s:v:0', 'RUVC_CODEC=vvenc',
            '-metadata:s:v:0', f'RUVC_QP={qp}',
            '-metadata:s:v:0', 'RUVC_QP_MAX=63',
            '-f', 'matroska', video_name,
        ]

        print(f"{'Encoding...':<21}", end='')
        T1 = time.time()
        _run_command(encode_command, frames.tobytes())
        print(f"Finished. Consumed {time.time() - T1:.6f} seconds")
        file_size = os.path.getsize(video_name)

        print(f"{'Decoding...':<21}", end='')
        T1 = time.time()
        decoded_bytes = _run_command([
            'ffmpeg', '-loglevel', 'error', '-c:v', 'libvvdec', '-i', video_name,
            '-map', '0:v:0', '-vf', 'format=rgb24', '-fps_mode', 'passthrough',
            '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-',
        ])
        print(f"Finished. Consumed {time.time() - T1:.6f} seconds")

        expected_size = bt * h * w * 3
        if len(decoded_bytes) != expected_size:
            probe = _run_command([
                'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,pix_fmt,nb_read_frames',
                '-of', 'default=noprint_wrappers=1', video_name,
            ]).decode().strip().replace('\n', ', ')
            raise RuntimeError(
                f'VVC decoded byte count mismatch: expected {expected_size}, got {len(decoded_bytes)}. '
                f'Stream details: {probe}'
            )

        metadata = _read_vvc_metadata(video_name)
        if metadata.get('RUVC_CODEC', '').lower() != 'vvenc':
            raise RuntimeError('VVC codec metadata is missing or invalid.')
        if metadata.get('RUVC_QP') != str(qp) or metadata.get('RUVC_QP_MAX') != '63':
            raise RuntimeError('VVC QP metadata is missing or invalid.')

        decoded_frames = np.frombuffer(decoded_bytes, dtype=np.uint8).reshape(bt, h, w, 3).copy()
        decoded_video = torch.from_numpy(decoded_frames.astype(np.float32)).permute(0, 3, 1, 2) / 255.0
        return decoded_video, file_size, float(qp)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class CodecPure(nn.Module):
    def __init__(self, cfg: cfg.Configuration):
        super(CodecPure, self).__init__()
        self.codec = cfg.codec
        self.CRF = cfg.x265_CRF
        self.scale_times = cfg.rescaling_times
        self.intermediatedir = cfg.intermediatedir
        self.keyint = cfg.reference_step * cfg.RUVC_GOP + 4
        self.quality_max = 63.0 if self.codec == 'vvenc' else 51.0

        if self.codec == 'vvenc':
            if not self.CRF.is_integer() or not 0 <= self.CRF <= 63:
                raise ValueError('VVC QP must be an integer in [0, 63].')
            print(f'Codec: VVC (QP: {int(self.CRF)}, condition: QP/63)')

    def normalize_quality(self, quality_value):
        return float(quality_value) / self.quality_max

    def forward(self, input):
        if self.codec == 'vvenc':
            return vvencPure.apply(input, int(self.CRF), self.intermediatedir)
        return x265Pure.apply(input, self.CRF, self.intermediatedir, self.keyint)

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


class DenseBlock_SelfC(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True,INN_init = True,is_res = False,seed=None):
        super(DenseBlock_SelfC, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                wi.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1, seed=seed)
            else:
                wi.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1, seed=seed)
            wi.initialize_weights(self.conv5, 0, seed=seed)
        else:
            wi.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4,self.conv5], 1, seed=seed)
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


class FeatureCalapseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, GOP_size=5, scale = 4,init='xavier', gc=32, bias=True,INN_init = True,is_res = False,seed=None):
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
                wi.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1, seed=seed)
            else:
                wi.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1, seed=seed)
            wi.initialize_weights(self.conv5, 0, seed=seed)

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
        

class x265WithSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original_input, DNN_output, q, frame_num):
        
        input      = original_input.clone().detach()
        dev_id     = str(input.device)
        input      = torch.clamp(input, 0, 1)
        output     = (input * 255.).round() 
        bt,c,h,w   = output.size()
        frames     = output.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        video_name = "./tmp/intermedia_video.mkv"
        outptudict = {
            "-s":str(w)+"x"+str(h),
            "-pix_fmt":"yuv444p",
            "-vframes":"100",
            "-c:v":"libx265",
            "-x265-params":"crf="+str(q)+":keyint="+str(CodecWithSurrogate.GOP_size)+":no-info=1"
        }
        
        T1 = time.time()
        writer = skvideo.io.FFmpegWriter(video_name,outputdict = outptudict,verbosity = 0)
        for i in range(bt):
            writer.writeFrame(frames[i, :, :, :])
        writer.close()
        file_size = os.path.getsize(video_name)
        bpp = file_size*8.0/(h*w*CodecWithSurrogate.scale_times*CodecWithSurrogate.scale_times*frame_num)
        outputparameters = {}
        reader = skvideo.io.FFmpegReader(video_name,
                        inputdict={},
                        outputdict={})
        # iterate through the frames
        decoded_frames = []
        for frame in reader.nextFrame():
            decoded_frames += [torch.from_numpy(frame)]
        # print('runing time2 %s ms' % ((T3 - T2)*1000))
        decoded_frames = torch.stack(decoded_frames,dim=0).cuda(input.device)
        decoded_frames = decoded_frames.permute(0,3,1,2)
        decoded_frames = decoded_frames/255.
        # return output/255.
        ctx.save_for_backward(DNN_output,original_input,decoded_frames)
        
        return decoded_frames


class CodecWithSurrogate(nn.Module):
    def __init__(self,cfg:cfg.Configuration):
        super(CodecWithSurrogate, self).__init__()
        CodecWithSurrogate.CRF          = cfg.x265_CRF
        CodecWithSurrogate.GOP_size    = cfg.RUVC_GOP
        CodecWithSurrogate.scale_times = cfg.rescaling_times
        CodecWithSurrogate.RUVC_GOP    = cfg.RUVC_GOP
        CodecWithSurrogate.lambda_corr = cfg.x265_loss_coefficient
        CodecWithSurrogate.file_name   = str(time.time())
        
        intermediate_channel = 24
        self.suggrogate_net  = nn.Sequential(
            DenseBlock_SelfC(4,intermediate_channel,INN_init=False,seed=cfg.random_seed),
            DenseBlock_SelfC(intermediate_channel,intermediate_channel,INN_init=False,is_res=True,seed=cfg.random_seed),
            FeatureCalapseBlock(intermediate_channel,intermediate_channel,CodecWithSurrogate.GOP_size,INN_init=True,is_res=True,seed=cfg.random_seed),
            FeatureCalapseBlock(intermediate_channel,intermediate_channel,CodecWithSurrogate.GOP_size,INN_init=True,is_res=True,seed=cfg.random_seed),
            DenseBlock_SelfC(intermediate_channel,intermediate_channel,INN_init=False,is_res=True,seed=cfg.random_seed),
            DenseBlock_SelfC(intermediate_channel,3,INN_init=False,seed=cfg.random_seed),
        ).cuda()
        
        if isinstance(CodecWithSurrogate.CRF, list):
            self.indicator_fuser = nn.Sequential(
                nn.Linear(2,256),
                nn.ReLU(inplace=True),
                nn.Linear(256,256),
                nn.ReLU(inplace=True),
                nn.Linear(256,1),
            ).cuda()
            
    def forward(self, input):
        bt,c,h,w = input.size()
        t        = self.RUVC_GOP
        b        = bt//t
        input_3d = input.reshape(b,t,c,h,w)
        
        if isinstance(CodecWithSurrogate.CRF, int):
            temporal_indicator = torch.linspace(0,1,t).cuda(input_3d.device)
            indicator          = temporal_indicator.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(b,1,1,h,w)
            current_q          = CodecWithSurrogate.CRF
        elif isinstance(CodecWithSurrogate.CRF, list):
            current_q          = random.randint(CodecWithSurrogate.CRF[0],CodecWithSurrogate.CRF[1]) 
            temporal_indicator = torch.linspace(0,1,t).cuda(input_3d.device)
            q_indicator        = torch.zeros_like(temporal_indicator).cuda(input_3d.device).fill_(current_q/30)
            indicator          = torch.stack([temporal_indicator,q_indicator],dim=1)
            indicator          = self.indicator_fuser(indicator.unsqueeze(0))
            indicator          = indicator.unsqueeze(-1).unsqueeze(-1).repeat(b,1,1,h,w)
            
        input_3d_temporal_ind  = torch.cat([input_3d,indicator],dim=2)

        sug_out                = self.suggrogate_net(input_3d_temporal_ind.reshape(bt,c+1,h,w))+input
        x265_encoder_out       = x265WithSurrogate.apply(input,sug_out,current_q,self.RUVC_GOP)
        x                      = x265_encoder_out.detach()
        y                      = sug_out
        mimick_loss            = torch.mean((x - y)**2.0)

        vx = x - torch.mean(x,dim=0,keepdim=True)
        vy = y - torch.mean(y,dim=0,keepdim=True)

        correlation_param = torch.sum(vx * vy,dim=0,keepdim=True) / (torch.sqrt(torch.sum(vx ** 2,dim=0,keepdim=True)) * torch.sqrt(torch.sum(vy ** 2,dim=0,keepdim=True))+1e-8)
        correlation_param = correlation_param.mean()
        sug_out.data      = x265_encoder_out
        
        return sug_out