import os, sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training_modules import configure as cfg
from auxiliary_modules import network_modules as nm
from auxiliary_modules import weights_initialization
# from auxiliary_modules import optical_flow_estimation as ofe
from auxiliary_modules import optical_flow_estimation2 as ofe
from auxiliary_modules import video_tensor_processor as vtp

import torch
import torch.nn.functional as F


def bilateral_filter(img, d=5, sigma_color=50, sigma_space=50):
    batch, channels, height, width = img.shape
    
    coords = torch.arange(d).float() - (sigma_space // 2)
    grid   = coords[None, :] ** 2 + coords[:, None] ** 2
    kernel = torch.exp(-grid / (2 * sigma_space ** 2))
    
    space_kernel = (kernel / kernel.sum()).to(img.device)
    filtered_img = torch.zeros_like(img)
    for i in range(height):
        for j in range(width):
            x_min, x_max = max(i - d // 2, 0), min(i + d // 2 + 1, height)
            y_min, y_max = max(j - d // 2, 0), min(j + d // 2 + 1, width)

            local_patch  = img[:, :, x_min:x_max, y_min:y_max]
            color_diff   = (local_patch - img[:, :, i:i+1, j:j+1]) ** 2
            color_weight = torch.exp(-color_diff.sum(dim=1, keepdim=True) / (2 * sigma_color ** 2))
            weight       = space_kernel[:x_max-x_min, :y_max-y_min] * color_weight
            weight       = weight / weight.sum()
            
            filtered_img[:, :, i, j] = (local_patch * weight).sum(dim=(2, 3))

    return filtered_img


class Quantization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        # x = bilateral_filter(input)
        x = torch.clamp(input, 0, 1)
        output = (x * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    

class QuantizationModel(nn.Module):
    def __init__(self):
        super(QuantizationModel, self).__init__()

    def forward(self, input):
        return Quantization.apply(input)
    
   
class RUVC(nn.Module):
    def __init__(self, cfg:cfg.Configuration, rank):
        super(RUVC,self).__init__()
        ## basic configuration
        if rank == 0:
            # Multithreading happens only once
            print(f"\n====================>>>{'Model Training'.center(25)}<<<====================")
        self.LR_channel                = 3*cfg.RUVC_GOP
        self.useLiteRUVC               = cfg.useLiteRUVC
        self.init_model                = cfg.init_model
        self.init_optimizer            = cfg.init_optimizer
        self.optical_flow_model        = cfg.optical_flow_model
        self.upscaling_reference_frame = cfg.upscaling_reference_frame
        self.rescaling_coefficient     = int(math.log2(cfg.rescaling_times))
        self.optical_flow_estimation   = ofe.FastFlowNet()
        # self.optical_flow_estimation   = ofe.RAFT(use_small=False)
        # self.optical_flow_estimation   = ofe.RAFT(use_small=True)
        
        ## rescaling modules
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
        # self.enhance_network           = nm.PostEnhanceNet(channel_in=self.LR_channel, intermediate_channel=32, block_num=4, seed=cfg.random_seed)
        # self.enhance_network           = nm.HighPerfEnhancer(channel_in=self.LR_channel, intermediate_channel=48, seed=cfg.random_seed)
        
        ## Print Network Model
        '''
            Output a visual network model if command parameter contains "--is_print_net True", which is default False.
            The save path depend on parameter "--print_net_path", which is default the python run path.
        '''
        if cfg.is_print_net:
            from auxiliary_modules import print_network as axm_print_network
            axm_print_network.print_a_network(self, cfg.print_net_path, [cfg.batch_size,3*cfg.RUVC_GOP,854,480])
            print(f"The network model have been save in '{cfg.print_net_path}'.")
        
        ## Optimizers setting
        '''
            Modules requiring gradient updates include three components:
            Inversible Rescaling Network, High-Frequency Component Generation Network, and Inter-Frame Compensation Network.
        '''
        optim_params = []
        # for net in [self.quantization_adapter]:
        for net in [self.quantization_adapter, self.HF_fusion, self.frequency_fusion_network]:
        # for net in [self.quantization_adapter, self.HF_fusion, self.enhance_network]:
        # for net in [self.quantization_adapter, self.HF_fusion, self.enhance_network, self.frequency_fusion_network]:
        # for net in [self.frequency_fusion_network, self.HF_fusion]:
        # for net in [self.frequency_fusion_network, self.quantization_adapter, self.HF_fusion]:
            for k, v in net.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                    
        # self.optimizer = torch.optim.Adam(optim_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2))
        self.optimizer         = torch.optim.Adam(optim_params, lr=cfg.learning_rate)
        self.scheduler         = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=(rank==0))
        self.use_gradient_hook = cfg.use_gradient_hook
        
        
    def load_model(self, rank):
        '''
            Load a pre-trained model, the load path depend on parameter "--init_model".
            If the parameter is '' (the default value is it) or the path pointing a error initialization model the training will start with a empty model.
        '''
        if self.init_model != '':
            if rank == 0:
                # Multithreading happens only once
                try:
                    checkpoint = torch.load(self.init_model, map_location=f'cuda:{rank}')
                    self.load_state_dict(checkpoint)
                    print(f"The training start with pre-trained model from '{self.init_model}'.")
                    if self.init_optimizer != '':
                        try:
                            self.optimizer.load_state_dict(torch.load(self.init_optimizer))
                            print(f"The training start with optimizer state from '{self.init_optimizer}'.")
                        except Exception as result:
                            print("Optimizer State Load Error::",result)
                            print("The training start with an basic optimizer.")
                    else:
                        print(f"The training start with an empty optimizer state.")
                        
                except Exception as result:
                    print("Pre-trained Model Load Error::",result)
                    print("The training start with an empty model.")
            dist.barrier()
            for param in self.parameters():
                dist.broadcast(param.data.contiguous(), src=0)
        else:
            if rank == 0: 
                # Multithreading happens only once
                print("The training start with an empty model.")
                if self.init_optimizer != '':
                    try:
                        self.optimizer.load_state_dict(torch.load(self.init_optimizer))
                        print(f"The training start with optimizer state from '{self.init_optimizer}'.")
                    except Exception as result:
                        print("Optimizer State Load Error::",result)
                        print("The training start with an basic optimizer.")
                else:
                    print(f"The training start with an empty optimizer state.")
            
        ## optical flow estimation module initialization
        state_dict = {k.replace("module.", ""): v for k, v in torch.load(self.optical_flow_model, map_location=f'cuda:{rank}').items()}
        self.optical_flow_estimation.load_state_dict(state_dict)
        self.optical_flow_estimation.training = False
        for param in self.optical_flow_estimation.parameters():
            param.requires_grad = False
        
        ## register gradient hook to debug if the gradient is Nan or Inf
        if self.use_gradient_hook:
            self.gradients = {}
            for name, param in self.named_parameters():
                if param.requires_grad:
                    param.register_hook(
                        lambda grad, name=name: self.gradients.update({name: grad.norm().item()})
                    )
        
        return self
        
    def forward(self, HR_video, LR_video=None, reverse=False, quantization_parameter=1.0):
        if not reverse:
            x, haar_HF = self.haar_wavelet_transform(HR_video)
            haar_LF = x[:,:self.LR_channel,:,:]
            # for block in self.frequency_fusion_network:
            #     x = block(x, reverse)
            
            return x, haar_LF.detach(), haar_HF.detach() 
        
        else:
            reference_F, reference_HF = self.HR_frequency_caption(HR_video)
            reference_LF              = reference_F[:,:3,:,:]
            
            flow = self.optical_flow_estimation(reference_LF, LR_video)
            LR_HF_H = []
            LR_HF_V = []
            LR_HF_D = []
            for i in range(self.LR_channel//3):
                LR_HF_H.append(self.optical_flow_estimation.warp(reference_HF[:,0:3,:,:], flow[i]))
                LR_HF_V.append(self.optical_flow_estimation.warp(reference_HF[:,3:6,:,:], flow[i]))
                LR_HF_D.append(self.optical_flow_estimation.warp(reference_HF[:,6:9,:,:], flow[i]))
                
            rough_HF = torch.cat([torch.cat(LR_HF_H, dim=1), torch.cat(LR_HF_V, dim=1), torch.cat(LR_HF_D, dim=1)], dim=1).detach()
            x        = torch.cat([LR_video, rough_HF], dim=1)
            x = self.frequency_fusion_network(x)
            # for block in self.frequency_fusion_network:
                # x = block(x, reverse)
            
            b, _, h, w   = x.shape
            LF_predicted = x[:,:self.LR_channel,:,:]
            HF_predicted = x[:,self.LR_channel:,:,:]
            qp_tensor    = torch.tensor(quantization_parameter, device=LF_predicted.device, dtype=LF_predicted.dtype).view(1, 1, 1, 1).expand(b, 1, h, w)
            LF_refined   = self.quantization_adapter(LF_predicted, reference_LF, qp_tensor)
            HF_refined   = self.HF_fusion(HF_predicted, reference_HF)
            re_x         = torch.cat((LF_refined, HF_refined), dim=1)
            # re_x         = torch.cat((LF_refined, HF_predicted), dim=1)
            # HR_reconstruction_refined, _ = self.haar_wavelet_transform(re_x, reverse=reverse)
            
            HR_reconstruction, _ = self.haar_wavelet_transform(re_x, reverse=reverse)
            # HR_reconstruction_refined = self.enhance_network(HR_reconstruction)
            
            # return HR_reconstruction_refined, LF_refined, HF_refined, HR_reconstruction
            return HR_reconstruction, LF_refined, HF_refined
                
                    
    def gradient_hook_worker(self, epoch_num:int, iteration_num:int, isNan:bool=False, isInf:bool=False):
        if self.use_gradient_hook:
            if isNan:
                print(f"[Epoch {epoch_num:3d}]-[Iteration {iteration_num:4d}]: Training loss is Nan. Detail: ")
            elif isInf:
                print(f"[Epoch {epoch_num:3d}]-[Iteration {iteration_num:4d}]: Training loss is Inf. Detail: ")
                
            for name, grad_norm in self.gradients.items():
                print(f"Parameter: {name}, Gradient Norm: {grad_norm}")
                
            sys.exit()
        else:
            print('Gradient vanishing or gradient explosion has occurred. Please set the training parameter "use_gradient_hook" to 1 for debugging.')
        
    
class ConstraintComputer(nn.Module):
    def __init__(self, GOP_size:int=6):
        super(ConstraintComputer, self).__init__()
        self.GOP_size          = 3*GOP_size
        self.frequency_caption = nm.ModulatedHaarWaveletTransform(self.GOP_size)
        
    def forward(self, input:torch.Tensor):
        with torch.no_grad():
            output, _ = self.frequency_caption(input)
            LF_constraint = output[:,:self.GOP_size,:,:]
            HF_constraint = output[:,self.GOP_size:,:,:]
            
        return LF_constraint.detach(), HF_constraint.detach()