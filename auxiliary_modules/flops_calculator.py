import torch
import torch.nn as nn
from copy import deepcopy

def get_flops(model, imgsz=640):
    """
    Calculate FLOPs (floating point operations) for a model in billions.

    Attempts two calculation methods: first with a stride-based tensor for efficiency,
    then falls back to full image size if needed (e.g., for RTDETR models). Returns 0.0
    if thop library is unavailable or calculation fails.

    Args:
        model (nn.Module): The model to calculate FLOPs for.
        imgsz (int | list, optional): Input image size.

    Returns:
        (float): The model FLOPs in billions.
    """
    try:
        import thop
    except ImportError:
        thop = None  # conda support without 'ultralytics-thop' installed

    if not thop:
        return 0.0  # if not installed return 0.0 GFLOPs

    try:
        model = de_parallel(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]  # expand if int/float
        try:
            # Method 1: Use stride-based input tensor
            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  # max stride
            im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # stride GFLOPs
            return flops * imgsz[0] / stride * imgsz[1] / stride  # imgsz GFLOPs
        except Exception:
            # Method 2: Use actual image size (required for RTDETR models)
            im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # imgsz GFLOPs
    except Exception:
        return 0.0
    
    
def is_parallel(model):
    """
    Return True if model is of type DP or DDP.

    Args:
        model (nn.Module): Model to check.

    Returns:
        (bool): True if model is DataParallel or DistributedDataParallel.
    """
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))

def de_parallel(model):
    """
    De-parallelize a model: return single-GPU model if model is of type DP or DDP.

    Args:
        model (nn.Module): Model to de-parallelize.

    Returns:
        (nn.Module): De-parallelized model.
    """
    return model.module if is_parallel(model) else model