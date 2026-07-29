import sys
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np

import test_modules.configure as test_cfg

def visualize_4d_tensor(
        video:torch.Tensor, 
        save_path:str, 
        pix_range:float=1.0, 
        need_normalize:bool=False, 
        is_gray:bool=False, 
        is_exit:bool=False, 
        HF_visual_enhance=False, 
        anchor_HF_min=None, 
        anchor_HF_max=None
    ):
    '''
        Visualization of a three-dimensional or four-dimensional tensor, the dimension is [b,c,h,w], or [c.h.w], where c must be an integer multiple of 3 to represent an integer RGB video frame, the default is to receive the tensor elements value in the range of 0~1, if the tensor pixel represents the range of 0~255, set pix_ramge=255.0.
        
        need_normalize: Whether to normalize the tensor, when it is True, the tensor will be min-max normalization.
        is_gray: Whether to save the frames as a gray image, when it is True, the number of channels of the tensor may not be an integer multiple of 3.
        is_exit: Whether to exit the program, just for saving run time when debugging.
    '''
    video = video / pix_range
    if video.dim() == 3:
        video = video.unsqueeze(0)
        
    if HF_visual_enhance:
        need_normalize = False
        if anchor_HF_min is None and anchor_HF_max is None:
            min_vals = video.view(video.size(0), video.size(1), -1).min(dim=2)[0].min(dim=0)[0]
            max_vals = video.view(video.size(0), video.size(1), -1).max(dim=2)[0].max(dim=0)[0]
            min_vals = min_vals[None, :, None, None]
            max_vals = max_vals[None, :, None, None]
        elif anchor_HF_min is not None and anchor_HF_max is not None:
            min_vals = anchor_HF_min
            max_vals = anchor_HF_max
        else:
            raise ValueError("anchor_HF_min and anchor_HF_max must be both None or both not None.")
        video = (video - min_vals) / (max_vals - min_vals + 1e-8)
        
        enhanced_channels = []
        for c in range(video.size(1)):
            channel_data = video[:, c, :, :]
            flat_channel = channel_data.flatten()
            hist = torch.histc(flat_channel, bins=256, min=0, max=1)
            mode_val = (torch.argmax(hist).item() + 0.5) / 256.0
            background_color = 138./255. if c == 0 else 141./255. if c == 1 else 142./255.
            channel_data = channel_data + background_color - mode_val
            
            enhanced_channel = torch.where(
                channel_data != background_color,
                channel_data + 3*(channel_data - background_color),
                channel_data
            )
            enhanced_channels.append(enhanced_channel)
        
        video = torch.stack(enhanced_channels, dim=1) + 1e-8
        
    if is_gray:
        vutils.save_image(video.reshape(-1, 1, video.shape[2], video.shape[3]), save_path, nrow=8, normalize=need_normalize, scale_each=need_normalize)
    else:
        vutils.save_image(video.reshape(-1, 3, video.shape[2], video.shape[3]), save_path, nrow=8, normalize=need_normalize, scale_each=need_normalize)
    
    if is_exit:
        sys.exit()
        
def visualize_diff_4d_tensor(
    video1: torch.Tensor,
    video2: torch.Tensor,
    save_path: str,
    scale: float = 5.0,
    is_gray: bool = False,
    is_exit: bool = False,
):
    assert video1.shape == video2.shape, "The shapes of video1 and video2 must be identical."

    diff = (video1 - video2) * scale

    # Ensure a 4D tensor with shape [B, C, H, W].
    if diff.dim() == 3:
        diff = diff.unsqueeze(0)

    B, C, H, W = diff.shape

    if is_gray:
        # Save a single-channel image.
        img = diff.reshape(-1, 1, H, W)
    else:
        assert C % 3 == 0, "When performing color visualization, C must be a multiple of 3."
        img = diff.reshape(-1, 3, H, W)

    vutils.save_image(
        img,
        save_path,
        nrow=8,
        normalize=False,
        scale_each=False,
    )

    if is_exit:
        sys.exit()

def check_value(tensor:torch.Tensor, save_path:str, is_complicate:bool=False, is_exit:bool=False):
    '''
        Used to output all numbers in tensor as text and save them to save_path.
        
        is_complicate: Whether to save the full txt text, if True, there will be no ellipses, but the running speed will be very slow.
        is_exit: Whether to exit the program, just for saving run time when debugging.
    '''
    if is_complicate:
        torch.set_printoptions(threshold=np.inf)
        
    with open(save_path, 'w') as f:
        f.write(str(tensor))
    
    if is_exit:
        sys.exit()
        
        
class TensorDimensionAdjuster:
    def __init__(self):
        super(TensorDimensionAdjuster, self).__init__()
        
    def pad_tensor_temporal(self, tensor:torch.Tensor, expect_length:int=None):
        assert tensor.dim() == 4, "The input tensor must be 4-dimensional."
        if tensor.shape[1] != 3:
            print("Warning: Function 'pad_tensor_temporal' expects 3-dimensional input [t,3,h,w], but received a tensor with [N] in the second dimension. Only the first (temporal) dimension will be padded. This may lead to misaligned dimensions or unexpected behavior.")
            
        if expect_length < tensor.shape[0]:
            self.unpad_tensor_temporal(tensor, expect_length)
        elif expect_length > tensor.shape[0]:
            padding = torch.zeros(expect_length-tensor.shape[0], tensor.shape[-3], tensor.shape[-2],tensor.shape[-1])
            tensor  = torch.cat([tensor,padding],dim=0)
        
        return tensor
    
    def unpad_tensor_temporal(self, tensor:torch.Tensor, expect_length:int=None):
        assert tensor.dim() == 4, "The input tensor must be 4-dimensional."
        if tensor.shape[1] != 3:
            print("Warning: Function 'unpad_tensor_temporal' expects 3-dimensional input [t,3,h,w], but received a tensor with [N] in the second dimension. Only the first (temporal) dimension will be unpadded. This may lead to misaligned dimensions or unexpected behavior.")
            
        if expect_length > tensor.shape[0]:
            self.pad_tensor_temporal(tensor, expect_length)
        elif expect_length < tensor.shape[0]:
            tensor = tensor[:expect_length]
        
        return tensor
        
    def pad_tensor_spatial(self, tensor:torch.Tensor, spatial_divisor:int=8, spatial_factors:tuple=None, spatial_need_detail:bool=False, pad_value:float=0.0):
        assert tensor.dim() == 4, "The input tensor must be 4-dimensional."

        if spatial_factors is None:        
            h, w           = tensor.size(2), tensor.size(3)
            new_h          = (h + spatial_divisor - 1) // spatial_divisor * spatial_divisor
            new_w          = (w + spatial_divisor - 1) // spatial_divisor * spatial_divisor
            padding_left   = (new_w - w) // 2
            padding_right  = new_w - w - padding_left
            padding_top    = (new_h - h) // 2
            padding_bottom = new_h - h - padding_top
        else:
            padding_left, padding_right, padding_top, padding_bottom = spatial_factors
        x_padded = F.pad(tensor, (padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=pad_value)
        # x_padded = F.pad(tensor, (padding_left, padding_right, padding_top, padding_bottom), mode="reflect")
        if spatial_need_detail:
            return x_padded, (padding_left, padding_right, padding_top, padding_bottom)
        else:
            return x_padded
        
    def unpad_tensor_spatial(self, tensor:torch.Tensor, spatial_factors:tuple=None):
        assert spatial_factors is not None, "Please specify the spatial padding factors."
        
        padding_left, padding_right, padding_top, padding_bottom = spatial_factors

        h = tensor.size(2) - padding_top - padding_bottom
        w = tensor.size(3) - padding_left - padding_right

        unpadded_tensor = tensor[:, :, padding_top : padding_top + h, padding_left : padding_left + w]
        return unpadded_tensor

        
def rgb_to_yuv444(rgb:torch.Tensor):
    assert rgb.dim() == 4, "The input tensor must be 4-dimensional."
    assert rgb.shape[1] == 3, "The input tensor must have 3 channels."
    assert rgb.max() <= 1 and rgb.min() >= 0, "The input tensor must have values in the range of 0~1."
        
    R = rgb[:,0:1, :, :]
    G = rgb[:,1:2, :, :]
    B = rgb[:,2:3, :, :]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.169 * R - 0.331 * G + 0.499 * B + 0.50196
    V = 0.499 * R - 0.41869 * G - 0.08131 * B + 0.50196
    
    yuv = torch.cat([Y, U, V], dim=1)
    return torch.clamp(yuv, 0, 1)
        
        
def yuv444_to_rgb(yuv:torch.Tensor):
    assert yuv.dim() == 4, "The input tensor must be 4-dimensional."
    assert yuv.shape[1] == 3, "The input tensor must have 3 channels."
    assert yuv.max() <= 1 and yuv.min() >= 0, "The input tensor must have values in the range of 0~1."
        
    Y = yuv[:, 0:1, :, :]
    U = yuv[:, 1:2, :, :]
    V = yuv[:, 2:3, :, :]
    
    R = Y + 1.402 * (V - 0.50196)
    G = Y - 0.344136 * (U - 0.50196) - 0.714136 * (V - 0.50196)
    B = Y + 1.772 * (U - 0.50196)
    
    R = torch.clamp(R, 0, 1)
    G = torch.clamp(G, 0, 1)
    B = torch.clamp(B, 0, 1)
    
    rgb = torch.cat([R, G, B], dim=1)
    return rgb

def tensor_gpu_size(tensor: torch.Tensor) -> float:
    """
        Returns the amount of GPU memory used by tensor (in MB).
        If tensor is not on CUDA, returns 0.
    """
    if not tensor.is_cuda:
        return 0.0
    bytes_used = tensor.element_size() * tensor.nelement()
    return bytes_used / 1024**2
