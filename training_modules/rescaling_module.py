import math
import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from auxiliary_modules import weights_initialization

class ResidualinResidualDenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, gc=32, bias=True, seed=None):
        super(ResidualinResidualDenseBlock, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        weights_initialization.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1, seed=seed)
        weights_initialization.initialize_weights_xavier(self.conv5, 0.1, seed=seed)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            
        return x5


class HaarWaveletTransform(nn.Module):
    def __init__(self, channel_in:int):
        super(HaarWaveletTransform, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        # H
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        
        # V
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        # D
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x:torch.Tensor, reverse=False):
        if not reverse:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out, out[:,self.channel_in:,:,:]
        else:
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            out = F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)
   
            return out, x[:,self.channel_in:,:,:]

class ResidualBlock(nn.Module):
    def __init__(self, intermediate_channel=64, seed=None):
        super(ResidualBlock, self).__init__()
        self.relu  = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        
        self.conv1 = nn.Conv2d(intermediate_channel, intermediate_channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(intermediate_channel, intermediate_channel, 3, 1, 1, bias=True)
        # weights_initialization.initialize_weights([self.conv1, self.conv2], 0.1, seed=seed)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        
        return x + out

class InterframeCompensationBlock(nn.Module):
    def __init__(self, intermediate_channel=16, block_num=8, seed=None):
        super(InterframeCompensationBlock, self).__init__()
        self.lrelu            = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        self.conv_in1 = nn.Conv2d(3, intermediate_channel, 3, 1, 1, bias=True)
        self.conv_in2 = nn.Conv2d(3, intermediate_channel, 3, 1, 1, bias=True)

        # self.conv_l1 = nn.Conv2d(intermediate_channel, intermediate_channel, 3, 2, 1, bias=True)
        # self.conv_l2 = nn.Conv2d(intermediate_channel, intermediate_channel, 3, 2, 1, bias=True)
        self.conv_l1 = nn.Conv2d(intermediate_channel, intermediate_channel, 3, 1, 1, bias=True)
        self.conv_l2 = nn.Conv2d(intermediate_channel, intermediate_channel, 3, 1, 1, bias=True)
            
        residual_block = []
        for i in range(block_num):
            residual_block.append(ResidualBlock(intermediate_channel*2))
        self.residual_block = nn.Sequential(*residual_block)

        # self.upconv        = nn.Conv2d(intermediate_channel*2, intermediate_channel * 4, 3, 1, 1, bias=True)
        # self.pixel_shuffle = nn.PixelShuffle(2)
        # self.HRconv        = nn.Conv2d(intermediate_channel, intermediate_channel, 3, 1, 1, bias=True)
            
        # self.conv_last     = nn.Conv2d(intermediate_channel, 3, 3, 1, 1, bias=True)
        # weights_initialization.initialize_weights_xavier([self.conv_in1,self.conv_in2,self.conv_l1,self.conv_l2,self.upconv,self.HRconv,self.conv_last], 0.1)
            
        self.catconv       = nn.Conv2d(intermediate_channel*2, intermediate_channel, 3, 1, 1, bias=True)
        self.conv_last     = nn.Conv2d(intermediate_channel, 3, 3, 1, 1, bias=True)
        
        # weights_initialization.initialize_weights([self.conv_in1, self.conv_in2, self.conv_l1, self.conv_l2, self.catconv, self.conv_last], 0.1, seed=seed)
        # weights_initialization.initialize_weights_xavier([], 0.1, seed=seed)

    def forward(self, referenced_frame, compensated_frame):
        x1 = self.conv_in1(referenced_frame)
        x2 = self.conv_in2(compensated_frame)
        
        x1 = self.lrelu(x1)
        x2 = self.lrelu(x2)
        
        x1 = self.lrelu(self.conv_l1(x1))
        x2 = self.lrelu(self.conv_l2(x2))
        
        x = torch.cat([x1,x2],1)
        x = self.residual_block(x)
        x = self.lrelu(self.catconv(x))
        x = self.lrelu(self.conv_last(x))
                
        return x


class InversableNeuralNetwork(nn.Module):
    def __init__(self, all_channel, low_frequency_channel, clamp=1., seed=None):
        super(InversableNeuralNetwork, self).__init__()

        self.split_len1 = low_frequency_channel
        self.split_len2 = all_channel - low_frequency_channel
        self.clamp      = clamp

        # self.regular_F  = ResidualinResidualDenseBlock(self.split_len2, self.split_len1, seed=seed)
        # self.regular_G  = ResidualinResidualDenseBlock(self.split_len1, self.split_len2, seed=seed)
        # self.regular_H  = ResidualinResidualDenseBlock(self.split_len1, self.split_len2, seed=seed)
        
        self.reverse_H  = ResidualinResidualDenseBlock(self.split_len1, self.split_len2, seed=seed)
        self.reverse_G  = ResidualinResidualDenseBlock(self.split_len1, self.split_len2, seed=seed)
        self.reverse_F  = ResidualinResidualDenseBlock(self.split_len2, self.split_len1, seed=seed)

    def forward(self, x, reverse=False):
        x1 = x.narrow(1, 0, self.split_len1)
        x2 = x.narrow(1, self.split_len1, self.split_len2)

        if not reverse:
            # y1     = x1 + self.regular_F(x2)
            # self.s = self.clamp * (torch.sigmoid(self.regular_H(y1)) * 2 - 1)
            # y2     = x2.mul(torch.exp(self.s)) + self.regular_G(y1)
            y1 = x1
            y2 = x2
        else:
            self.s = self.clamp * (torch.sigmoid(self.reverse_H(x1)) * 2 - 1)
            y2     = (x2 - self.reverse_G(x1)).div(torch.exp(self.s))
            y1     = x1 - self.reverse_F(y2)
            
        return torch.cat((y1, y2), 1)


# class RescalingBlock(nn.Module):
#     def __init__(self, channel_in, channel_out, rescaling_coefficient=1, seed=None):
#         super(RescalingBlock,self).__init__()
#         blocks          = []
#         current_channel = channel_in
        
#         for i in range(rescaling_coefficient):
#             blocks.append(HaarWaveletDownsampling(current_channel))
#             current_channel *= 4
#             for j in range(4):
#                 blocks.append(InversableCompensationBlock(current_channel, channel_out, seed=seed))

#         self.channel_in            = channel_in
#         # self.referenced_frame      = referenced_frame
#         self.blocks                = nn.ModuleList(blocks)
#         # self.C                     = InterframeCompensationBlock(seed=seed)

#     def forward(self, input, reverse=False):
#         x = input
#         if not reverse:
#             high_frequency_constraint = None
#             for block in self.blocks:
#                 if isinstance(block, HaarWaveletDownsampling):
#                     x, high_frequency_constraint = block(x, reverse)
#                 elif isinstance(block, InversableCompensationBlock):
#                     x = block(x, reverse)
#             return x, high_frequency_constraint
#         else:
#             high_frequency_recontribution = None
#             for block in reversed(self.blocks):
#                 if isinstance(block, InversableCompensationBlock):
#                     x = block(x, reverse)
#                 elif isinstance(block, HaarWaveletDownsampling):
#                     x, high_frequency_recontribution = block(x, reverse)
#             return x, high_frequency_recontribution


# class BiorthogonalWaveletTransform(nn.Module):
#     def __init__(self, channel_in: int):
#         super(BiorthogonalWaveletTransform, self).__init__()
#         self.channel_in = channel_in
        
#         self.decomp_weights = torch.zeros(4, 1, 2, 2)
#         self.recon_weights  = torch.zeros(4, 1, 2, 2)

#         # H0 (lowpass filter)
#         self.decomp_weights[0, 0, 0, 0] = 0.5
#         self.decomp_weights[0, 0, 0, 1] = 0.5
#         self.decomp_weights[0, 0, 1, 0] = 0.5
#         self.decomp_weights[0, 0, 1, 1] = 0.5
        
#         # H1 (highpass filter)
#         self.decomp_weights[1, 0, 0, 0] = -0.5
#         self.decomp_weights[1, 0, 0, 1] = 0.5
#         self.decomp_weights[1, 0, 1, 0] = 0.5
#         self.decomp_weights[1, 0, 1, 1] = -0.5

#         # V filters (H0 and H1) same for vertical
#         self.decomp_weights[2, 0, 0, 0] = -0.5
#         self.decomp_weights[2, 0, 0, 1] = 0.5
#         self.decomp_weights[2, 0, 1, 0] = 0.5
#         self.decomp_weights[2, 0, 1, 1] = 0.5

#         # D filters (H0 and H1) same for diagonal
#         self.decomp_weights[3, 0, 0, 1] = -0.5
#         self.decomp_weights[3, 0, 1, 0] = 0.5
#         self.decomp_weights[3, 0, 1, 1] = 0.5
#         self.decomp_weights[3, 0, 0, 0] = -0.5
        
#         # G0 (lowpass filter)
#         self.recon_weights[0, 0, 0, 0] = 0.5
#         self.recon_weights[0, 0, 0, 1] = 0.5
#         self.recon_weights[0, 0, 1, 0] = 0.5
#         self.recon_weights[0, 0, 1, 1] = 0.5
        
#         # G1 (highpass filter)
#         self.recon_weights[1, 0, 0, 0] = 0.5
#         self.recon_weights[1, 0, 0, 1] = -0.5
#         self.recon_weights[1, 0, 1, 0] = -0.5
#         self.recon_weights[1, 0, 1, 1] = 0.5

#         # V filters (G0 and G1) same for vertical
#         self.recon_weights[2, 0, 0, 0] = 0.5
#         self.recon_weights[2, 0, 0, 1] = -0.5
#         self.recon_weights[2, 0, 1, 0] = -0.5
#         self.recon_weights[2, 0, 1, 1] = -0.5

#         # D filters (G0 and G1) same for diagonal
#         self.recon_weights[3, 0, 0, 1] = 0.5
#         self.recon_weights[3, 0, 1, 0] = -0.5
#         self.recon_weights[3, 0, 1, 1] = -0.5
#         self.recon_weights[3, 0, 0, 0] = 0.5
                
#         self.decomp_weights = torch.cat([self.decomp_weights] * self.channel_in, 0)
#         self.recon_weights = torch.cat([self.recon_weights] * self.channel_in, 0)
        
#         self.decomp_weights = nn.Parameter(self.decomp_weights)
#         self.recon_weights = nn.Parameter(self.recon_weights)

#         self.decomp_weights.requires_grad = False
#         self.recon_weights.requires_grad = False

#     def forward(self, x: torch.Tensor, reverse=False):
#         if not reverse:
#             out = F.conv2d(x, self.decomp_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
#             out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
#             out = torch.transpose(out, 1, 2)
#             out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
#             return out, out[:, self.channel_in:, :, :]
#         else:
#             out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
#             out = torch.transpose(out, 1, 2)
#             out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
#             out = F.conv_transpose2d(out, self.recon_weights, bias=None, stride=2, groups=self.channel_in)
#             return out, x[:, self.channel_in:, :, :]

class BiorthogonalWaveletTransform(nn.Module):
    def __init__(self, channel_in: int):
        super(BiorthogonalWaveletTransform, self).__init__()
        self.channel_in = channel_in
        
        self.decomp_weights = torch.zeros(4, 1, 2, 2)
        self.recon_weights  = torch.zeros(4, 1, 2, 2)
        
        # L decomposition filters
        self.decomp_weights[0, 0, 0, 0] = math.sqrt(2)/2
        self.decomp_weights[0, 0, 0, 1] = math.sqrt(2)/2
        self.decomp_weights[0, 0, 1, 0] = math.sqrt(2)/2
        self.decomp_weights[0, 0, 1, 1] = math.sqrt(2)/2
        
        # H decomposition filters
        self.decomp_weights[1, 0, 0, 0] = math.sqrt(2)/2
        self.decomp_weights[1, 0, 0, 1] = -math.sqrt(2)/2
        self.decomp_weights[1, 0, 1, 0] = math.sqrt(2)/2
        self.decomp_weights[1, 0, 1, 1] = -math.sqrt(2)/2

        # V decomposition filters
        self.decomp_weights[2, 0, 0, 0] = math.sqrt(2)/2
        self.decomp_weights[2, 0, 0, 1] = math.sqrt(2)/2
        self.decomp_weights[2, 0, 1, 0] = -math.sqrt(2)/2
        self.decomp_weights[2, 0, 1, 1] = -math.sqrt(2)/2

        # D decomposition filters
        self.decomp_weights[3, 0, 0, 0] = math.sqrt(2)/2
        self.decomp_weights[3, 0, 0, 1] = -math.sqrt(2)/2
        self.decomp_weights[3, 0, 1, 0] = -math.sqrt(2)/2
        self.decomp_weights[3, 0, 1, 1] = math.sqrt(2)/2
        
        # L reconstruction filters
        self.recon_weights[0, 0, 0, 0] = math.sqrt(2)
        self.recon_weights[0, 0, 0, 1] = math.sqrt(2)
        self.recon_weights[0, 0, 1, 0] = math.sqrt(2)
        self.recon_weights[0, 0, 1, 1] = math.sqrt(2)
        
        # H reconstruction filters
        self.recon_weights[1, 0, 0, 0] = math.sqrt(2)
        self.recon_weights[1, 0, 0, 1] = -math.sqrt(2)
        self.recon_weights[1, 0, 1, 0] = math.sqrt(2)
        self.recon_weights[1, 0, 1, 1] = -math.sqrt(2)

        # V reconstruction filters
        self.recon_weights[2, 0, 0, 0] = math.sqrt(2)
        self.recon_weights[2, 0, 0, 1] = math.sqrt(2)
        self.recon_weights[2, 0, 1, 0] = -math.sqrt(2)
        self.recon_weights[2, 0, 1, 1] = -math.sqrt(2)

        # D reconstruction filters
        self.recon_weights[3, 0, 0, 0] = math.sqrt(2)
        self.recon_weights[3, 0, 0, 1] = -math.sqrt(2)
        self.recon_weights[3, 0, 1, 0] = -math.sqrt(2)
        self.recon_weights[3, 0, 1, 1] = math.sqrt(2)
                
        self.decomp_weights = torch.cat([self.decomp_weights] * self.channel_in, 0)
        self.recon_weights = torch.cat([self.recon_weights] * self.channel_in, 0)
        
        self.decomp_weights = nn.Parameter(self.decomp_weights)
        self.recon_weights = nn.Parameter(self.recon_weights)

        self.decomp_weights.requires_grad = False
        self.recon_weights.requires_grad = False

    def forward(self, x: torch.Tensor, reverse=False):
        if not reverse:
            out = F.conv2d(x, self.decomp_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out, out[:, self.channel_in:, :, :]
        else:
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            out = F.conv_transpose2d(out, self.recon_weights, bias=None, stride=2, groups=self.channel_in)
                        
            return out, x[:, self.channel_in:, :, :]