import os, sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from test_modules import configure as cfg
from auxiliary_modules import network_modules as nm
from auxiliary_modules import weights_initialization
# from auxiliary_modules import optical_flow_estimation as ofe
from auxiliary_modules import optical_flow_estimation2 as ofe
from auxiliary_modules import video_tensor_processor as vtp

class Quantization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, 0, 1)
        output = (input * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class QuantizationModel(nn.Module):
    def __init__(self):
        super(QuantizationModel, self).__init__()

    def forward(self, input):
        return Quantization.apply(input)


# class ResidualBlock(nn.Module):
#     def __init__(self, channel_in=3, inter_channel=16, seed=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channel_in, inter_channel, 3, 1, 1, bias=True)
#         self.conv2 = nn.Conv2d(inter_channel, channel_in, 3, 1, 1, bias=True)
  
#         self.relu  = nn.LeakyReLU(negative_slope=0.2, inplace=False)

#         weights_initialization.initialize_weights_xavier([self.conv1, self.conv2], 0.1, seed=seed)

#     def forward(self, x):
#         out = self.relu(self.conv1(x))
#         out = self.relu(self.conv2(out))
        
#         return x + out


# class HighFrequencyRefineNet(nn.Module):
#     def __init__(self, channel=27, block_num=3, seed=None):
#         super(HighFrequencyRefineNet, self).__init__()

#         self.channel = channel
#         # self.lrelu   = nn.LeakyReLU(negative_slope=0.2, inplace=False)

#         self.Hnet = nn.ModuleList([nm.ResidualinResidualDenseBlock(channel, channel, seed=seed) for _ in range(block_num)])
#         self.Vnet = nn.ModuleList([nm.ResidualinResidualDenseBlock(channel, channel, seed=seed) for _ in range(block_num)])
#         self.Dnet = nn.ModuleList([nm.ResidualinResidualDenseBlock(channel, channel, seed=seed) for _ in range(block_num)])

#     def forward(self, roughHF):
#         rough_H = roughHF[:, :self.channel, :, :]  
#         rough_V = roughHF[:, self.channel:2*self.channel, :, :]
#         rough_D = roughHF[:, 2*self.channel:, :, :]
#         for block in self.Hnet:
#             rough_H = block(rough_H)
#         for block in self.Vnet:
#             rough_V = block(rough_V)
#         for block in self.Dnet:
#             rough_D = block(rough_D)
        
#         refined = torch.cat([rough_H,rough_V,rough_D],1)
#         return refined

    
# class AdjustNet(nn.Module):
#     def __init__(self, channel_in, block_num=4, seed=None):
#         super(AdjustNet, self).__init__()
#         inter_channel = 13
#         self.lrelu    = nn.LeakyReLU(negative_slope=0.2)

#         self.conv_in1 = nm.ResidualinResidualDenseBlock(channel_in, channel_in, gc=inter_channel, seed=seed)
#         self.conv_in2 = nm.ResidualinResidualDenseBlock(channel_in, channel_in, gc=inter_channel, seed=seed)

#         self.conv_up1 = nm.ResidualinResidualDenseBlock(channel_in, channel_in, gc=inter_channel, seed=seed)
#         self.conv_up2 = nm.ResidualinResidualDenseBlock(channel_in, channel_in, gc=inter_channel, seed=seed)
            
#         residual_block = []
#         for i in range(block_num):
#             residual_block.append(ResidualBlock(channel_in=2*channel_in, inter_channel=inter_channel))
#         self.residual_block = nn.Sequential(*residual_block)
            
#         self.catconv   = nm.ResidualinResidualDenseBlock(2*channel_in, channel_in, gc=inter_channel, seed=seed)
#         self.conv_last = nm.ResidualinResidualDenseBlock(channel_in, channel_in, gc=inter_channel, seed=seed)
        
#     def forward(self, frames, referenced_frame):
#         compensated_frames = []
#         frames             = torch.split(frames, 3, dim=1)
#         for frame in frames:
#             x1 = self.lrelu(self.conv_in1(frame))
#             x2 = self.lrelu(self.conv_in2(referenced_frame))
            
#             x1 = self.lrelu(self.conv_up1(x1))
#             x2 = self.lrelu(self.conv_up2(x2))
            
#             x = torch.cat([x1,x2],1)
#             x = self.residual_block(x)
#             x = self.lrelu(self.catconv(x))
#             x = self.lrelu(self.conv_last(x))
                
#             x = x + frame
#             compensated_frames.append(x)
            
#         out = torch.cat(compensated_frames, dim=1)
#         return out
    

class RUVC(nn.Module):
    def __init__(self, cfg:cfg.Configuration):
        super(RUVC,self).__init__()
        print(f"\n====================>>>{'Model Test'.center(25)}<<<====================")
        self.device                    = str(cfg.device)
        self.model                     = cfg.model
        self.optical_flow_model        = cfg.optical_flow_model
        self.GOPbyGOP                  = cfg.GOPbyGOP
        self.reference_step            = cfg.reference_step
        self.useLiteRUVC               = cfg.useLiteRUVC
        self.LR_channel                = 3*cfg.RUVC_GOP
        self.upscaling_reference_frame = cfg.upscaling_reference_frame
        self.rescaling_coefficient     = int(math.log2(cfg.rescaling_times))
        # self.optical_flow_estimation   = ofe.RAFT(use_small=False, use_dehaze=cfg.use_dehaze)
        self.optical_flow_estimation   = ofe.FastFlowNet()

        ## rescaling modules
        # # self.haar_wavelet_transform    = nm.HaarWaveletTransform(self.LR_channel)
        # self.haar_wavelet_transform    = nm.ModulatedHaarWaveletTransform(channel_in=self.LR_channel, modulation_factor=1)
        # # self.frequency_fusion_network  = nn.ModuleList([nm.FeatureFusionBlock(4*self.LR_channel, self.LR_channel, seed=cfg.random_seed) for _ in range(4)])
        # self.quantization_adapter      = nm.QuantizationAdaptionNet(in_channel=self.LR_channel, intermediate_channel=64, seed=cfg.random_seed)
        # # self.HR_frequency_caption      = nm.HaarWaveletTransform(3)
        # self.HR_frequency_caption      = nm.ModulatedHaarWaveletTransform(channel_in=3, modulation_factor=1)
        # # self.adjuster                  = nm.InterframeAttentionFusionNet(seed=cfg.random_seed)
        # self.HF_fusion                 = nm.InterframeAttentionFusionNet(channel_in=3*3, intermediate_channel=64, seed=cfg.random_seed)
        
        self.haar_wavelet_transform    = nm.ModulatedHaarWaveletTransform(channel_in=self.LR_channel, modulation_factor=1)
        self.HR_frequency_caption      = nm.ModulatedHaarWaveletTransform(channel_in=3, modulation_factor=1)
        # self.frequency_fusion_network  = nn.ModuleList([nm.FeatureFusionBlock(4*self.LR_channel, self.LR_channel, seed=cfg.random_seed) for _ in range(4)])
        self.frequency_fusion_network  = nm.FeatureFusionNet(4*self.LR_channel, self.LR_channel, block_num=4, seed=cfg.random_seed)
        if cfg.useLiteRUVC:
            self.quantization_adapter      = nm.LiteQuantizationAdaptionNet(in_channel=self.LR_channel, intermediate_channel=64, seed=cfg.random_seed)
            self.HF_fusion                 = nm.LiteInterframeAttentionFusionNet(channel_in=3*3, intermediate_channel=64, seed=cfg.random_seed)
        else:
            self.quantization_adapter      = nm.QuantizationAdaptionNet(in_channel=self.LR_channel, intermediate_channel=64, seed=cfg.random_seed)
            self.HF_fusion                 = nm.InterframeAttentionFusionNet(channel_in=3*3, intermediate_channel=64, seed=cfg.random_seed)
        
        ## Print Network Model
        ''' 
            Output a visual network model if command parameter contains "--is_print_net True", which is default False.
            The save path depend on parameter "--print_net_path", which is default the python run path.
        '''
        if cfg.is_print_net:
            from auxiliary_modules import print_network as axm_print_network
            axm_print_network.print_a_network(self, cfg.print_net_path, [cfg.batch_size,3*cfg.RUVC_GOP,854,480])
            print(f"The network model have been save in '{cfg.print_net_path}'.")
                
        self.load_model()
                
    def load_model(self):
        if self.model != '':
            print(f"The pre-trained model used from: '{self.model}'.")
            try:
                checkpoint = torch.load(self.model, map_location=self.device)
            except Exception as result:
                print("Pre-trained Model Load Error::",result)
                sys.exit()
                
            try:
                self.load_state_dict(checkpoint)
                # model_state_dict = self.state_dict()
                # filtered_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
                # model_state_dict.update(filtered_state_dict)
                # self.load_state_dict(model_state_dict)
            except Exception as result:
                print("Pre-trained Model Unmatched Error::",result)
                sys.exit()
                
            model_name = 'RUVC' if not self.useLiteRUVC else 'LiteRUVC'
            print("The pre-trained model have been loaded.")
            print(f"{model_name} Parameters Number: {self.count_parameters(contain_opticalFlow=False)}")
        else:
            print("Test model is empty.")
            sys.exit()
        
        # optical flow estimation module initialization
        # state_dict = {k.replace("module.", ""): v for k, v in torch.load(self.optical_flow_model, map_location=self.device).items()}
        # self.optical_flow_estimation.load_state_dict(state_dict)
        # self.optical_flow_estimation.eval()
        # for param in self.optical_flow_estimation.parameters():
        #     param.requires_grad = False
            
        self.eval()
        
    def forward(self, HR_video, LR_video=None, reverse=False, quantization_parameter=1.0):
        if not reverse: # reverse mast be True in RUVC
            downsample_video = []
            if self.GOPbyGOP:
                HR_video_slices = torch.split(HR_video, 1, dim=0)
                for HR_video_slice in HR_video_slices:
                    x, haar_HF = self.haar_wavelet_transform(HR_video_slice.to(self.device))
                    downsample_video.append(x.to('cpu'))
                downsample_video = torch.cat(downsample_video, dim=0)
            else:
                x, HF_constraint = self.haar_wavelet_transform(HR_video.to(self.device))
                downsample_video = x
                
            return downsample_video
        
        else:
            video_reconstruction = []
            HF_reconstruction    = []
            LF_reconstruction    = []
            reference_index      = 0
            if self.GOPbyGOP:
                reference_frame                 = HR_video[reference_index:reference_index+1,:,:,:].to(self.device)
                reference_F_slice, reference_HF = self.HR_frequency_caption(reference_frame)
                reference_LF                    = reference_F_slice[:,:3,:,:]
                LR_video_slices                 = torch.split(LR_video, 1, dim=0)
                
                cuurent_step = 0
                for LR_video_slice in LR_video_slices:
                    cuurent_step  += 1
                    LR_video_slice = LR_video_slice.to(self.device)
                    flow_slice     = self.optical_flow_estimation(reference_LF, LR_video_slice)
                    
                    LR_HF_H_slice  = []
                    LR_HF_V_slice  = []
                    LR_HF_D_slice  = []
                    for i in range(self.LR_channel//3):
                        LR_HF_H_slice.append(self.optical_flow_estimation.warp(reference_HF[:,0:3,:,:], flow_slice[i]))
                        LR_HF_V_slice.append(self.optical_flow_estimation.warp(reference_HF[:,3:6,:,:], flow_slice[i]))
                        LR_HF_D_slice.append(self.optical_flow_estimation.warp(reference_HF[:,6:9,:,:], flow_slice[i]))
                        
                    predicted_HF = torch.cat([torch.cat(LR_HF_H_slice, dim=1), torch.cat(LR_HF_V_slice, dim=1), torch.cat(LR_HF_D_slice, dim=1)], dim=1)
                    x            = torch.cat([LR_video_slice, predicted_HF], dim=1)
                    x            = self.frequency_fusion_network(x)
                    b, _, h, w   = x.shape
                    LF_predicted = x[:,:self.LR_channel,:,:]
                    HF_predicted = x[:,self.LR_channel:,:,:]
                    qp_tensor    = torch.tensor(quantization_parameter, device=LF_predicted.device, dtype=LF_predicted.dtype).view(1, 1, 1, 1).expand(b, 1, h, w)
                    LF_refined   = self.quantization_adapter(LF_predicted, reference_LF, qp_tensor)
                    HF_refined   = self.HF_fusion(HF_predicted, reference_HF)
                    re_x         = torch.cat((LF_refined, HF_refined), dim=1)
                    video_reconstruction_slice, HF_reconstruction_slice = self.haar_wavelet_transform(re_x, reverse=reverse)
                    
                    video_reconstruction.append(video_reconstruction_slice.to('cpu'))
                    HF_reconstruction.append(HF_refined.to('cpu'))
                    LF_reconstruction.append(LF_refined.to('cpu'))
                    
                    if cuurent_step % self.reference_step == 0:
                        reference_index                += 1
                        reference_frame                 = HR_video[reference_index:reference_index+1,:,:,:].to(self.device)
                        reference_F_slice, reference_HF = self.HR_frequency_caption(reference_frame)
                        reference_LF                    = reference_F_slice[:,:3,:,:]
                    
                video_reconstruction = torch.cat(video_reconstruction, dim=0)
                HF_reconstruction    = torch.cat(HF_reconstruction, dim=0)
                LF_reconstruction    = torch.cat(LF_reconstruction, dim=0)
            
            else:
                reference_F, reference_HF = self.HR_frequency_caption(HR_video.to(self.device))
                reference_LF              = reference_F[:,:3,:,:]
                
                flow = self.optical_flow_estimation(reference_LF.to(self.device), LR_video.to(self.device))
                
                LR_HF_H = []
                LR_HF_V = []
                LR_HF_D = []
                for i in range(self.LR_channel//3):
                    LR_HF_H.append(self.optical_flow_estimation.warp(reference_HF[:,0:3,:,:], flow[i]))
                    LR_HF_V.append(self.optical_flow_estimation.warp(reference_HF[:,3:6,:,:], flow[i]))
                    LR_HF_D.append(self.optical_flow_estimation.warp(reference_HF[:,6:9,:,:], flow[i]))
                                
                predicted_HF = torch.cat([torch.cat(LR_HF_H, dim=1), torch.cat(LR_HF_V, dim=1), torch.cat(LR_HF_D, dim=1)], dim=1).detach()
                x = torch.cat([LR_video.to(self.device), predicted_HF], dim=1)
                for block in reversed(self.frequency_fusion_network):
                    x = block.forward(x, reverse)
                LF_reconstruction=x[:,:self.LR_channel,:,:]
                video_reconstruction, HF_reconstruction = self.haar_wavelet_transform(x, reverse=reverse)
                # video_reconstruction = self.adjuster(video_reconstruction, HR_video.to(self.device))
                video_reconstruction = video_reconstruction.to('cpu')
                
            return video_reconstruction, LF_reconstruction, HF_reconstruction
            # return video_reconstruction, LF_reconstruction, HF_reconstruction, OF_time
    
    
    def count_parameters(self, contain_opticalFlow=True) -> int:
        if contain_opticalFlow:
            return sum(map(lambda x: x.numel(), self.parameters()))
        else:
            return sum(map(lambda x: x.numel(), self.parameters())) - sum(map(lambda x: x.numel(), self.optical_flow_estimation.parameters()))
