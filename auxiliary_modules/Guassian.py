import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage.filters as fi

    
# def Guassian_downsample(x, scale=2):
#     assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)
#     if scale == 2:
#         h = gkern(13, 0.8)
#     elif scale == 3:
#         h = gkern(13, 1.2)
#     elif scale == 4:
#         h = gkern(13, 1.6)
#     else:
#         print(f'Invalid factor: {scale} (Must be one of 2, 3, 4)')
#         exit(1)

#     pad_h, pad_w    = 6 + scale * 2, 6 + scale * 2
#     r_h, r_w        = 0, 0
#     t, c, h, w      = x.size()
#     x               = x.reshape(-1, 1, h, w)

#     if scale == 3:
#         r_h = 3 - (h % 3)
#         r_w = 3 - (w % 3)
        
#     with torch.no_grad():
#         gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
        
#         x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')
#         x = F.conv2d(x, gaussian_filter, stride=scale)
#         x = x[:, :, 2:-2, 2:-2]
#         x = x.reshape(t, c, x.size(2), x.size(3))
    
#     return x.contiguous()

# def Guassian_upsample(x, scale=2):
#     assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)
#     if scale == 2:
#         nsig = 0.8
#     elif scale == 3:
#         nsig = 1.2
#     elif scale == 4:
#         nsig = 1.6
#     else:
#         print(f'Invalid factor: {scale} (Must be one of 2, 3, 4)')
#         exit(1)

#     t, c, h, w = x.size()
#     x = x.view(-1, 1, h, w)

#     # Step 1: interpolate
#     up_h, up_w = h * scale, w * scale
#     x = F.interpolate(x, size=(up_h, up_w), mode='bilinear', align_corners=False)

#     # Step 2: apply Gaussian blur
#     gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
#     pad_h, pad_w = 6, 6  # same as kernel radius
#     x = F.pad(x, [pad_w, pad_w, pad_h, pad_h], 'reflect')
#     x = F.conv2d(x, gaussian_filter, stride=1)
#     x = x[:, :, 2:-2, 2:-2]  # match your downsampling crop

#     x = x.view(t, c, x.size(2), x.size(3))
#     return x.contiguous()

class GuassianTransform(nn.Module):
    def __init__(self, channel_in, data_type=torch.float32, rec_smooth:bool=False):
        super(GuassianTransform, self).__init__()
        self.channel_in = channel_in
        self.sigma      = 0.8
        self.shuffle    = nn.PixelShuffle(2)
        self.unshuffle  = nn.PixelUnshuffle(2)
        self.kernel1    = torch.from_numpy(self._gkern(13, self.sigma)).to(data_type)
        self.rec_smooth = rec_smooth
        if self.rec_smooth:
            self.kernel2 = torch.from_numpy(self._gkern(13, 0.4)).to(data_type)
            self.register_buffer("smooth_filter", self.kernel2.unsqueeze(0).unsqueeze(0))
        
        self.register_buffer("gaussian_filter", self.kernel1.unsqueeze(0).unsqueeze(0))

    def forward(self, x, reverse=False):
        if not reverse:
            component_low = self.__downsample(x)
            component_res = self.unshuffle(x - self.__upsample(component_low))
            ## format: frame number - shuffle location - color space
            ## sequence: 1αR, 1αG, 1αB, 1βR, 1βG, 1βB, 1γR, 1γG, 1γB, 1ΔR, 1ΔG, 1ΔB, ...
            return torch.cat((component_low, component_res),dim = 1)
        else:
            component_low = x[:,:self.channel_in,:,:]
            component_res = x[:,self.channel_in:,:,:]
            return self.__upsample(component_low) + self.shuffle(component_res)
        
    def _gkern(self, kernlen=13, nsig=1.6):
        inp = np.zeros((kernlen, kernlen))
        inp[kernlen // 2, kernlen // 2] = 1
        return fi.gaussian_filter(inp, nsig)
    
    def __downsample(self, x):
        pad_h, pad_w = 10, 10
        bt, c, h, w  = x.size()
        
        x = x.reshape(-1, 1, h, w)
        x = F.pad(x, [pad_w, pad_w, pad_h, pad_h], 'reflect')
        x = F.conv2d(x, self.gaussian_filter, stride=2)
        x = x[:, :, 2:-2, 2:-2]
        x = x.reshape(bt, c, x.size(2), x.size(3))
        
        return x
        
    def __upsample(self, x):
        bt, c, h, w  = x.size()
        up_h, up_w   = h * 2, w * 2
        pad_h, pad_w = 6, 6
        
        x = x.reshape(-1, 1, h, w)
        x = F.interpolate(x, size=(up_h, up_w), mode='bicubic', align_corners=False)
        if self.rec_smooth:
            x = F.pad(x, [pad_w, pad_w, pad_h, pad_h], 'reflect')
            x = F.conv2d(x, self.smooth_filter, stride=1)

        x = x.reshape(bt, c, x.size(2), x.size(3))
        
        return x