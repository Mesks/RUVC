import os, cv2, sys
import torch
import torch.nn as nn
from auxiliary_modules import video_tensor_processor as vtp
from . import configure as cfg

class TestVideoLoading():
    def __init__(self, test_config:cfg.Configuration):
        print(f"\n====================>>>{'Data Loading'.center(25)}<<<====================")
        self.video_width     = 0
        self.video_height    = 0
        self.frame_number    = 0
        self.video           = None
        self.GOP_size        = test_config.RUVC_GOP
        self.reference_step  = test_config.reference_step
        self.random_seed     = test_config.random_seed
        self.rescaling_times = test_config.rescaling_times
        
        image_files                         = sorted([os.path.join(test_config.testdata, f) for f in os.listdir(test_config.testdata) if f.endswith('.jpg') or f.endswith('.png')])
        self.frame_number                   = len(image_files) if test_config.frame_number == -1 else test_config.frame_number
        self.video_height, self.video_width = cv2.imread(image_files[0]).shape[:2]
        
        # batch_size                          = self.frame_number // self.GOP_size if self.frame_number % self.GOP_size == 0 else self.frame_number // self.GOP_size + 1
        # channels                            = 3*self.GOP_size
        # images_tensor                       = torch.zeros((batch_size, channels, self.video_height, self.video_width), dtype=torch.uint8)

        # for i in range(batch_size):
        #     for j in range(self.GOP_size):
        #         if i*self.GOP_size + j < self.frame_number:
        #             img     = cv2.imread(image_files[i * self.GOP_size + j])
        #             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #             images_tensor[i, j*3:(j+1)*3, :, :] = torch.from_numpy(img_rgb).permute(2, 0, 1)
        
        images_tensor                       = torch.zeros((self.frame_number, 3, self.video_height, self.video_width), dtype=torch.uint8, device='cpu')
        for i in range(self.frame_number):
            img     = cv2.imread(image_files[i])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images_tensor[i, :, :, :] = torch.from_numpy(img_rgb).permute(2, 0, 1)

        self.video = images_tensor / 255.0
        
        print(f"Every {self.GOP_size} frames constitutes a video as a training sample.")
        print(f"The resolution is {self.video_width}x{self.video_height}")
        print(f"The frame number is {self.frame_number}")
        
    def get_video(self, device=None, padding=False):
        device = torch.device(device) if device != None else "cpu"            
        return self.video.to(device)
        
    def get_HR_LR_video(self, device=None):
        device                    = torch.device(device) if device != None else "cpu"
        frame_cluster_num         = 1 + self.GOP_size * self.reference_step
        mask                      = torch.ones(self.frame_number, dtype=torch.bool)
        mask[::frame_cluster_num] = False
        to_reference_video        = self.video[::frame_cluster_num]
        to_downsample_video       = self.video[mask]
        return to_reference_video, to_downsample_video
        
    def frequency_detach(self, hybrid_video):
        ''' 
            The results of a frame after downsampling on the channel order is: 
            F1-RL F1-GL F1-BL F2-RL F2-GL F2-BL ... Fn-RL Fn-GL Fn-BL F1-RH F1-GH F1-BH ... F1-RV F1-GV F1-BV ... F1-RD F1-GD F1-BD
        '''
        low_frequency_video  = hybrid_video[:,:hybrid_video.shape[1]//4,:,:]
        high_frequency_video = hybrid_video[:,hybrid_video.shape[1]//4:,:,:]
        return low_frequency_video, high_frequency_video
    
    
class TestVideoProcessing():
    def __init__(self, test_config:cfg.Configuration, test_video:TestVideoLoading):
        self.dim_adjuster = vtp.TensorDimensionAdjuster()
        self.LF_GOP       = test_config.RUVC_GOP
        self.LR_GOP       = test_config.RUVC_GOP * test_config.reference_step
        self.stride       = self.LR_GOP + 4
        self.shuffle      = nn.PixelShuffle(2)
        self.unshuffle    = nn.PixelUnshuffle(2)
        self.video_height = test_video.video_height
        self.video_width  = test_video.video_width
        self.frame_number = test_video.frame_number
        self.pad_factors  = 0
        self.ref_num      = 0
        
    def pad_before_downscale(self, to_reference_video:torch.tensor, to_downsample_video:torch.tensor):
        to_reference_video = self.unshuffle(to_reference_video).reshape(-1,3,4,to_reference_video.shape[-2]//2,to_reference_video.shape[-1]//2)
        to_reference_video = torch.transpose(to_reference_video, 1, 2).reshape(-1,12,to_reference_video.shape[-2], to_reference_video.shape[-1])
        to_reference_video = to_reference_video.reshape(-1,3,to_reference_video.shape[-2],to_reference_video.shape[-1])
        
        LF_mod                  = to_downsample_video.shape[0]%self.LF_GOP
        if LF_mod > 0:
            padding             = torch.zeros(self.LR_GOP-LF_mod, 3, self.video_height, self.video_width)
            to_downsample_video = torch.cat([to_downsample_video, padding],dim=0).reshape(-1,3*self.LF_GOP,self.video_height,self.video_width)
        else:
            to_downsample_video = to_downsample_video.reshape(-1,3*self.LF_GOP,self.video_height,self.video_width)
                        
        return to_reference_video, to_downsample_video
                
    def unpad_after_downscale(self, to_reference_video:torch.tensor, to_downsample_video:torch.tensor):
        encode_video = []
        self.ref_num = to_reference_video.shape[0]//4
        for i in range(self.ref_num):
            encode_video.append(torch.cat([to_reference_video[i*4:(i+1)*4], to_downsample_video[i*self.LR_GOP:(i+1)*self.LR_GOP]], dim=0))
        return torch.cat(encode_video, dim=0)[:self.frame_number+self.ref_num*3,:,:,:]
    
    def pad_after_decode(self, decode_video:torch.tensor):
        LR_padding_frame_num = self.stride - decode_video.shape[0]%self.stride
        decode_video_paded   = self.dim_adjuster.pad_tensor_temporal(decode_video, decode_video.shape[0]+LR_padding_frame_num)
        reference_num        = (decode_video.shape[0] + self.stride - 1) // self.stride
        reference_video      = torch.zeros((reference_num, 3, decode_video.shape[-2]*2, decode_video.shape[-1]*2), device=decode_video.device)
        LF_video             = []
        for i in range(reference_num):
            start = i * self.stride
            end   = start + 4
            if end>decode_video_paded.shape[0]:
                end = decode_video_paded.shape[0]
                
            current = decode_video_paded[start:end]
            if current.shape[0]==0:
                continue
            
            current = torch.transpose(current, 0, 1).reshape(-1,12,current.shape[-2],current.shape[-1])
            reference_video[i:i+1] = self.shuffle(current)
            
            LF_video.append(decode_video_paded[end:(i+1)*self.stride])
        
        LF_video = torch.cat(LF_video, dim=0)
        
        padded_LF_video, self.pad_factors = self.dim_adjuster.pad_tensor_spatial(LF_video, spatial_need_detail=True)
        padded_LF_video                   = padded_LF_video.reshape(-1, 3*self.LF_GOP, padded_LF_video.shape[-2], padded_LF_video.shape[-1])
        self.pad_factors                  = tuple(map(lambda x: x * 2, self.pad_factors))
        padded_reference                  = self.dim_adjuster.pad_tensor_spatial(reference_video, spatial_factors=self.pad_factors)
        
        return padded_reference, padded_LF_video
        
    def unpad_after_upscale(self, reference_video, HR_reconstruction):
        HR_reconstruction    = HR_reconstruction.reshape(-1, 3*self.LR_GOP, HR_reconstruction.shape[-2], HR_reconstruction.shape[-1])
        reconstruction_video = torch.cat([reference_video, HR_reconstruction], dim=1)
        reconstruction_video = self.dim_adjuster.unpad_tensor_spatial(reconstruction_video, spatial_factors=self.pad_factors)
        reconstruction_video = reconstruction_video.reshape(-1,3,self.video_height,self.video_width)
        reconstruction_video = torch.clamp(reconstruction_video, min=0.0, max=1.0)
        reconstruction_video = self.dim_adjuster.unpad_tensor_temporal(reconstruction_video.reshape(-1,3,self.video_height,self.video_width), self.frame_number)
        
        return reconstruction_video
    
    
    
    
class TestVideoProcessing_rawResolution():
    def __init__(self, test_config:cfg.Configuration, test_video:TestVideoLoading):
        self.dim_adjuster = vtp.TensorDimensionAdjuster()
        self.LF_GOP       = test_config.RUVC_GOP
        self.LR_GOP       = test_config.RUVC_GOP * test_config.reference_step
        self.stride       = self.LR_GOP + 1
        self.shuffle      = nn.PixelShuffle(2)
        self.unshuffle    = nn.PixelUnshuffle(2)
        self.video_height = test_video.video_height
        self.video_width  = test_video.video_width
        self.frame_number = test_video.frame_number
        self.pad_factors  = 0
        self.ref_num      = 0
        
    def pad_before_downscale(self, to_reference_video:torch.tensor, to_downsample_video:torch.tensor):
        LF_mod                  = to_downsample_video.shape[0]%self.LF_GOP
        if LF_mod > 0:
            padding             = torch.zeros(self.LR_GOP-LF_mod, 3, self.video_height, self.video_width)
            to_downsample_video = torch.cat([to_downsample_video, padding],dim=0).reshape(-1,3*self.LF_GOP,self.video_height,self.video_width)
        else:
            to_downsample_video = to_downsample_video.reshape(-1,3*self.LF_GOP,self.video_height,self.video_width)
                        
        return to_reference_video, to_downsample_video
                
    def unpad_after_downscale(self, to_reference_video:torch.tensor, to_downsample_video:torch.tensor):
        encode_video = []
        self.ref_num = to_reference_video.shape[0]
        for i in range(self.ref_num):
            LR_video = to_downsample_video[i*self.LR_GOP:(i+1)*self.LR_GOP]
            pad_lr, pad_tb = (to_reference_video.shape[-1]-LR_video.shape[-1])//2, (to_reference_video.shape[-2]-LR_video.shape[-2])//2
            LR_video = self.dim_adjuster.pad_tensor_spatial(LR_video, spatial_factors=[pad_lr, pad_lr, pad_tb, pad_tb])
            encode_video.append(torch.cat([to_reference_video[i:i+1], LR_video], dim=0))
        return torch.cat(encode_video, dim=0)[:self.frame_number,:,:,:]
    
    def pad_after_decode(self, decode_video:torch.tensor):
        LR_padding_frame_num = self.stride - decode_video.shape[0]%self.stride
        decode_video_paded   = self.dim_adjuster.pad_tensor_temporal(decode_video, decode_video.shape[0]+LR_padding_frame_num)
        reference_num        = (decode_video.shape[0] + self.stride - 1) // self.stride
        reference_video      = torch.zeros((reference_num, 3, decode_video.shape[-2], decode_video.shape[-1]), device=decode_video.device)
        LF_video             = []
        for i in range(reference_num):
            start = i * self.stride
            end   = start + 1
            if end>decode_video_paded.shape[0]:
                end = decode_video_paded.shape[0]
                
            current = decode_video_paded[start:end]
            if current.shape[0]==0:
                continue
            
            reference_video[i:i+1] = current
            
            LF_video.append(decode_video_paded[end:(i+1)*self.stride])
        
        LF_video = torch.cat(LF_video, dim=0)
        
        pad_lr, pad_tb = reference_video.shape[-1]//4, reference_video.shape[-2]//4
        LF_video = self.dim_adjuster.unpad_tensor_spatial(LF_video, spatial_factors=[pad_lr, pad_lr, pad_tb, pad_tb])
        
        padded_LF_video, self.pad_factors = self.dim_adjuster.pad_tensor_spatial(LF_video, spatial_need_detail=True)
        padded_LF_video                   = padded_LF_video.reshape(-1, 3*self.LF_GOP, padded_LF_video.shape[-2], padded_LF_video.shape[-1])
        self.pad_factors                  = tuple(map(lambda x: x * 2, self.pad_factors))
        padded_reference                  = self.dim_adjuster.pad_tensor_spatial(reference_video, spatial_factors=self.pad_factors)
        
        return padded_reference, padded_LF_video
        
    def unpad_after_upscale(self, reference_video, HR_reconstruction):
        HR_reconstruction    = HR_reconstruction.reshape(-1, 3*self.LR_GOP, HR_reconstruction.shape[-2], HR_reconstruction.shape[-1])
        reconstruction_video = torch.cat([reference_video, HR_reconstruction], dim=1)
        reconstruction_video = self.dim_adjuster.unpad_tensor_spatial(reconstruction_video, spatial_factors=self.pad_factors)
        reconstruction_video = reconstruction_video.reshape(-1,3,self.video_height,self.video_width)
        reconstruction_video = torch.clamp(reconstruction_video, min=0.0, max=1.0)
        reconstruction_video = self.dim_adjuster.unpad_tensor_temporal(reconstruction_video.reshape(-1,3,self.video_height,self.video_width), self.frame_number)
        
        return reconstruction_video