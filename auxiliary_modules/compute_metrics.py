import torch
import lpips

from torchmetrics.image import PeakSignalNoiseRatio as tm_PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as tm_SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as tm_MSSSIM

def video_normalization(tensor:torch.Tensor, normal_min=0, normal_max=1):
    tensor = tensor.clamp(min=normal_min, max=normal_max)
    tensor = (tensor - normal_min) / (normal_max - normal_min)
    return tensor

def compute_metric(
        processed_video :torch.Tensor, 
        target_video    :torch.Tensor, 
        color_space     :str='RGB',
        metric          :str='PSNR', 
        data_range      :float=1.0, 
        step_by_step    :bool=False, 
        device          :torch.device=None
    ) -> float:
    '''
        processed_video: [b,c,h,w] or [c,h,w]
        target_video:    [b,c,h,w] or [c,h,w]
    '''
    torch.cuda.empty_cache()
    with torch.no_grad():
        assert data_range == 1.0 or data_range == 255.0, f"Compute metric Error: Image value range should be [0,1] or [0,255]."
        assert target_video.shape == processed_video.shape, f"Compute metric Error: target_video shape {target_video.shape} is not same with processed_video shape {processed_video.shape}"

        if device == None:
            assert target_video.device == processed_video.device, f"Compute metric Error: target_video device {target_video.device} is not same with processed_video device {processed_video.device}"
            device = processed_video.device

        if len(target_video.shape) == 3:
            target_video    = target_video.unsqueeze(0)
            processed_video = processed_video.unsqueeze(0)
        elif len(target_video.shape) == 4:
            target_video    = target_video.reshape(-1,3,target_video.shape[2],target_video.shape[3])
            processed_video = processed_video.reshape(-1,3,processed_video.shape[2],processed_video.shape[3])
        else:
            raise NotImplementedError
        
        metric      = metric.lower()
        color_space = color_space.lower()
        if color_space in ['yuv', 'yuv444', 'yuv444p', 'ycbcr']:
            target_video    = rgb2ycbcr(target_video)
            processed_video = rgb2ycbcr(processed_video)
        elif color_space in ['y', 'onlyy']:
            assert metric != 'lpips', "LPIPS dont support not onlyY color space."
            target_video = rgb2ycbcr(target_video)[0:1]
            processed_video = rgb2ycbcr(processed_video)[0:1]
        
        result   = 0.
        computer = None
        with torch.no_grad():
            if metric == 'psnr':
                computer = tm_PSNR(data_range=data_range).to(device)
            elif metric == 'ssim':
                computer = tm_SSIM(data_range=data_range).to(device)
            elif metric == 'msssim' or metric == 'ms-ssim':
                computer = tm_MSSSIM(data_range=data_range).to(device)
            elif metric == 'lpips':
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    computer = lpips.LPIPS(net='vgg', verbose=False).to(device)
            else:
                raise NotImplementedError

            try:
                processed_video_clamped = torch.clamp(processed_video, min=0.0, max=data_range)
                if step_by_step or processed_video.shape[0] > 10:
                    for i in range(processed_video.shape[0]):
                        if metric == 'lpips':
                            result += computer(target_video[i:i+1].to(device), processed_video_clamped[i:i+1].to(device), normalize=True).item()
                        else:
                            computer.update(processed_video_clamped[i:i+1].to(device), target_video[i:i+1].to(device))
                            result += computer.compute().item()
                            computer.reset()
                    result /= processed_video.shape[0]
                else:
                    if metric == 'lpips':
                        result = computer(target_video.to(device), processed_video_clamped.to(device), normalize=True).mean().item()
                    else:
                        computer.update(processed_video.to(device),target_video.to(device))
                        result = computer.compute().item()
            except Exception as e:
                print(f"""
                [compute_metric ERROR] type: {type(e).__name__}
                info: {str(e)}
                context:
                - Device: {device}
                - Metric: {metric}
                - processed_video shape: {processed_video.shape if processed_video is not None else 'None'}
                - target_video    shape: {target_video.shape if target_video is not None else 'None'}
                """)
                
                if "CUDA" in str(e):
                    print(">>> Check your GPU status may be helpful. (nvidia-smi)")
                elif "shape" in str(e).lower():
                    print(">>> Check the changes in the tensors' shape during the operation process may be helpful.")
                            
    return result


def rgb2ycbcr(rgb:torch.Tensor) -> torch.Tensor:
    assert rgb.ndim     == 4, "Input must be 4D tensor [B, C, H, W]."
    assert rgb.shape[1] == 3, "Input must have 3 color channels."

    original_dtype = rgb.dtype
    device = rgb.device

    rgb = rgb.to(torch.float32)
    if original_dtype != torch.uint8:
        rgb = rgb * 255.0

    rgb = rgb.permute(0, 2, 3, 1)  # [B, H, W, C]

    # YCbCr conversion matrix (ITU-R BT.601)
    transform_matrix = torch.tensor([
        [65.481,  -37.797,  112.0],
        [128.553, -74.203,  -93.786],
        [24.966,  112.0,   -18.214]
    ], device=device)

    ycbcr = torch.matmul(rgb, transform_matrix.T) / 255.0
    ycbcr += torch.tensor([16.0, 128.0, 128.0], device=device)

    ycbcr = ycbcr.permute(0, 3, 1, 2)  # [B, 3, H, W]

    if original_dtype == torch.uint8:
        ycbcr = torch.round(ycbcr)
    else:
        ycbcr = ycbcr / 255.0

    return ycbcr.to(original_dtype)

def ycbcr2rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    assert ycbcr.ndim == 4, "Input must be 4D tensor [B, C, H, W]."
    assert ycbcr.shape[1] == 3, "Input must have 3 color channels."

    original_dtype = ycbcr.dtype
    device = ycbcr.device

    ycbcr = ycbcr.to(torch.float32)
    if original_dtype != torch.uint8:
        ycbcr = ycbcr * 255.0

    ycbcr = ycbcr.permute(0, 2, 3, 1)  # [B, H, W, C]
    ycbcr -= torch.tensor([16.0, 128.0, 128.0], device=device)

    # Inverse transformation matrix from YCbCr to RGB (ITU-R BT.601)
    inverse_matrix = torch.tensor([
        [0.00456621,  0.0,        0.00625893],
        [0.00456621, -0.00153632, -0.00318811],
        [0.00456621,  0.00791071,  0.0]
    ], device=device) * 255.0

    rgb = torch.matmul(ycbcr, inverse_matrix.T) / 255.0

    rgb = rgb.permute(0, 3, 1, 2)  # [B, 3, H, W]

    if original_dtype == torch.uint8:
        rgb = torch.clamp(torch.round(rgb), 0, 255)
    else:
        rgb = torch.clamp(rgb / 255.0, 0.0, 1.0)

    return rgb.to(original_dtype)
