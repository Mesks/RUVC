import math
import numpy as np
import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from auxiliary_modules import weights_initialization as wi
from einops import rearrange 
from einops.layers.torch import Rearrange
from timm.layers import trunc_normal_
        
class ResidualBlock(nn.Module):
    def __init__(self, intermediate_channel=64, seed=None):
        super(ResidualBlock, self).__init__()
        self.relu  = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        
        self.conv1 = nn.Conv2d(intermediate_channel, intermediate_channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(intermediate_channel, intermediate_channel, 3, 1, 1, bias=True)
        wi.initialize_kaiming([self.conv1], 0.1, seed=seed)
        wi.initialize_kaiming([self.conv2], 0.01, seed=seed)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        
        return x + out
    
class ResidualinResidualDenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, gc=32, bias=True, seed=None, need_res=False):
        super(ResidualinResidualDenseBlock, self).__init__()
        self.need_res = need_res
        self.lrelu    = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.conv1    = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2    = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3    = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4    = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5    = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        wi.initialize_kaiming([self.conv1, self.conv2, self.conv3, self.conv4], 0.1, seed=seed)
        wi.initialize_xavier([self.conv5], 0.01, seed=seed)
        # wi.initialize_xavier([self.conv5], 1, seed=seed)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if self.need_res:
            x5 = x5+x
            
        return x5

class FeatureFusionBlock(nn.Module):
    def __init__(self, all_channel, base_channel, clamp=1., seed=None):
        super(FeatureFusionBlock, self).__init__()

        self.split_len1 = base_channel
        self.split_len2 = all_channel - base_channel
        self.clamp      = clamp
        
        self.F  = ResidualinResidualDenseBlock(self.split_len2, self.split_len1, seed=seed)
        self.G  = ResidualinResidualDenseBlock(self.split_len1, self.split_len2, seed=seed)
        self.H  = ResidualinResidualDenseBlock(self.split_len1, self.split_len2, seed=seed)
        # self.reverse_H  = ResidualinResidualDenseBlock(self.split_len1, self.split_len2, seed=seed)
        # self.reverse_G  = ResidualinResidualDenseBlock(self.split_len1, self.split_len2, seed=seed)
        # self.reverse_F  = ResidualinResidualDenseBlock(self.split_len2, self.split_len1, seed=seed)

    def forward(self, x, reverse=False):
    # def forward(self, x):
        x1 = x.narrow(1, 0, self.split_len1)
        x2 = x.narrow(1, self.split_len1, self.split_len2)

        # if not reverse:
        #     y1 = x1 + self.F(x2)
        #     s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        #     y2 = x2.mul(torch.exp(s)) + self.G(y1)
        # else:
        #     s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
        #     y2 = (x2 - self.G(x1)).div(torch.exp(s))
        #     y1 = x1 - self.F(y2)
            
        y1 = x1 + self.F(x2)
        s  = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(s)) + self.G(y1)
        
        # s = self.clamp * (torch.sigmoid(self.reverse_H(x1)) * 2 - 1)
        # y2     = (x2 - self.reverse_G(x1)).div(torch.exp(s))
        # y1     = x1 - self.reverse_F(y2)
            
        return torch.cat((y1, y2), 1)

class FeatureFusionNet(nn.Module):
    def __init__(self, all_channel, base_channel, block_num=4, seed=None):
        super(FeatureFusionNet, self).__init__()
        self.blocks = nn.ModuleList([FeatureFusionBlock(all_channel, base_channel, seed=seed) for _ in range(block_num)])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
                        
        return x
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8, seed=None):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        wi.initialize_kaiming(self.fc[0], 0.1, seed=seed)
        wi.initialize_xavier(self.fc[2], 0.1, seed=seed)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False, seed=None):
        super(MultiScaleBlock, self).__init__()
        branch_channels = out_channels
        self.is_res  = is_res if in_channels == out_channels else False
        self.branch1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0)
        self.branch2 = nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0)
        )
        self.project = nn.Conv2d(out_channels*4, out_channels, kernel_size=1)
        wi.initialize_xavier([self.branch1, self.branch2, self.branch3, self.branch4], 0.1, seed=seed)
        wi.initialize_xavier([self.project], 0.01, seed=seed)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        if self.is_res:
            return self.project(out) + x
        else:
            return self.project(out)
    
class QuantizationAdaptionNet(nn.Module):
    def __init__(self, in_channel, intermediate_channel, seed=None):
        super(QuantizationAdaptionNet, self).__init__()
        self.fuse_conv   = nn.Conv2d(3*2, 3, 1, groups=3)
        self.input_conv  = nn.Conv2d(3, intermediate_channel//8, 1)
        self.qp_conv     = nn.Conv2d(1, intermediate_channel//8, 1)
        self.output_conv = nn.Conv2d(intermediate_channel, 3, 1)
        
        self.query_gen   = nn.Sequential(
            ResidualinResidualDenseBlock(intermediate_channel//8, intermediate_channel//4, seed=seed),
            ResidualinResidualDenseBlock(intermediate_channel//4, intermediate_channel//2, seed=seed),
            ResidualinResidualDenseBlock(intermediate_channel//2, intermediate_channel, seed=seed),
            ResidualinResidualDenseBlock(intermediate_channel, intermediate_channel, seed=seed)
        )
        self.mask_gen    = nn.Sequential(
            ResidualinResidualDenseBlock(intermediate_channel//8, intermediate_channel//4, seed=seed),
            ResidualinResidualDenseBlock(intermediate_channel//4, intermediate_channel//2, seed=seed),
            ResidualinResidualDenseBlock(intermediate_channel//2, intermediate_channel, seed=seed),
            ResidualinResidualDenseBlock(intermediate_channel, intermediate_channel, seed=seed)
        )
        
        wi.initialize_xavier([self.fuse_conv, self.input_conv, self.qp_conv], 0.1, seed=seed)
        wi.initialize_xavier([self.output_conv], 0.01, seed=seed)
        # wi.initialize_xavier([self.output_conv], 1, seed=seed)

    def forward(self, x, ref, qp):
        output  = []
        frames  = torch.split(x, 3, dim=1)
        for frame in frames:
            input       = torch.cat((ref, frame), 1)[:,[0,3,1,4,2,5],:,:]
            fused_frame = self.fuse_conv(input)
            query       = self.query_gen(self.input_conv(fused_frame))
            mask        = self.mask_gen(self.qp_conv(qp))
            out         = self.output_conv(query * torch.sigmoid(mask)) + frame
            output.append(out)
        return torch.cat(output, 1)

class BaseAttentionBlock(nn.Module):
    def __init__(self, input_dim=64, output_dim=64, head_dim=8, window_size=8, window_type='W', attention_type='reference', seed=None):
        super(BaseAttentionBlock, self).__init__()
        assert window_type in ['W', 'SW']
        assert attention_type in ['reference', 'self']
        
        self.input_dim                = input_dim
        self.output_dim               = output_dim
        self.head_dim                 = head_dim 
        self.scale                    = self.head_dim ** - 0.5
        self.n_heads                  = input_dim//head_dim
        self.window_size              = window_size
        self.wtype                    = window_type
        self.atype                    = attention_type
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))
        self.linear                   = nn.Linear(self.input_dim, self.output_dim)
        if self.atype == 'reference':
            self.key_value_layer1      = nn.Linear(self.input_dim, 2*self.input_dim, bias=True)
            self.key_value_layer2      = nn.Linear(2*self.input_dim, 2*self.input_dim, bias=True)
            self.query_layer1          = nn.Linear(self.input_dim, self.input_dim, bias=True)
            self.query_layer2          = nn.Linear(self.input_dim, self.input_dim, bias=True)
            wi.initialize_xavier([self.linear, self.key_value_layer1, self.key_value_layer2, self.query_layer1, self.query_layer2], 1, seed=seed)
        else:
            self.embedding_layer1      = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
            self.embedding_layer2      = nn.Linear(3*self.input_dim, 3*self.input_dim, bias=True)
            wi.initialize_xavier([self.linear, self.embedding_layer1, self.embedding_layer2], 1, seed=seed)
        
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                
        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.wtype == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x, ref):
        if self.wtype!='W': 
            x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x         = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x         = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        
        if self.atype == 'reference':
            kv   = self.key_value_layer1(x)
            kv   = self.key_value_layer2(kv)
            k, v = rearrange(kv, 'b nw np (twoh c) -> twoh b nw np c', c=self.head_dim).chunk(2, dim=0)
            ref  = rearrange(ref, 'b (w1 p1) (w2 p2) c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
            q    = self.query_layer1(ref)
            q    = self.query_layer2(q)
            q    = rearrange(q, 'b nw np (h c) -> h b nw np c', c=self.head_dim)
        else:
            qkv     = self.embedding_layer1(x)
            qkv     = self.embedding_layer2(qkv)
            q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
            
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        
        if self.wtype != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim       = sim.masked_fill_(attn_mask, float("-inf"))

        probs  = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.wtype!='W': 
            output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, window_type='W', attention_type='reference', seed=None):
        super(AttentionBlock, self).__init__()
        assert window_type in ['W', 'SW']
        assert attention_type in ['reference', 'self']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.wtype = window_type
        self.atype = attention_type
        self.ln1 = nn.LayerNorm(input_dim)            
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim)
        )
        
        if self.atype == 'reference':
            self.msa = BaseAttentionBlock(input_dim, input_dim, head_dim, window_size, self.wtype, attention_type='reference', seed=seed)
        else:
            self.msa = BaseAttentionBlock(input_dim, input_dim, head_dim, window_size, self.wtype, attention_type='self', seed=seed)
            
        wi.initialize_xavier([self.ln1, self.ln2], 0.1, seed=seed)
        wi.initialize_kaiming([self.mlp[0]], 0.1, seed=seed)
        wi.initialize_xavier([self.mlp[2]], 0.01, seed=seed)
        # wi.initialize_xavier([self.mlp[2]], 1, seed=seed)

    def forward(self, x, ref=None):
        if self.atype != 'reference': 
            ref = None
        x = x + self.msa(self.ln1(x), ref)
        x = x + self.mlp(self.ln2(x))
        return x
    
class AttentionNet(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, window_type='W', attention_type='self', seed=None):
        super(AttentionNet, self).__init__()
        assert window_type in ['W', 'SW']
        self.conv_dim    = conv_dim
        self.trans_dim   = trans_dim
        self.head_dim    = head_dim
        self.window_size = window_size
        self.wtype       = window_type
        self.trans_block = AttentionBlock(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.wtype, attention_type=attention_type)
        self.conv1_1     = nn.Conv2d(self.conv_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2     = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim, 1, 1, 0, bias=True)
        # self.conv1_1     = DeformConv2d(self.conv_dim, self.conv_dim+self.trans_dim, 1, seed=seed)
        # self.conv1_2     = DeformConv2d(self.conv_dim+self.trans_dim, self.conv_dim, 1, seed=seed)
        self.conv_block  = ResidualBlock(self.conv_dim, self.conv_dim)
        wi.initialize_xavier([self.conv1_1], 0.1, seed=seed)
        wi.initialize_xavier([self.conv1_2], 0.01, seed=seed)
        # wi.initialize_xavier([self.conv1_2], 1, seed=seed)

    def forward(self, x, mask):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x          = self.conv_block(conv_x)
        att             = Rearrange('b c h w -> b h w c')(mask)
        trans_x         = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x         = self.trans_block(trans_x, att)
        trans_x         = Rearrange('b h w c -> b c h w')(trans_x)
        res             = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        out             = res + x
        return out
    
class SelfAttentionNet(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size=8):
        super(SelfAttentionNet, self).__init__()
        self.block_1     = AttentionBlock(input_dim, output_dim, head_dim, window_size, window_type='W', attention_type='self')
        self.block_2     = AttentionBlock(input_dim, output_dim, head_dim, window_size, window_type='SW', attention_type='self')
        self.window_size = window_size

    def forward(self, x, resize = False):
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(x, (padding_col, padding_col+1, padding_row, padding_row+1))
        trans_x = Rearrange('b c h w -> b h w c')(x)
        trans_x = self.block_1(trans_x)
        trans_x =  self.block_2(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        if resize:
            x = F.pad(x, (-padding_col, -padding_col-1, -padding_row, -padding_row-1))
        return trans_x
        
class InterframeAttentionFusionNet(nn.Module):
    def __init__(self, channel_in=3*3, intermediate_channel=64, head_dim=8, window_size=8, seed=None):
        super(InterframeAttentionFusionNet, self).__init__()
        self.channel_in    = channel_in
        self.channel_out   = channel_in
        self.channel_inter = intermediate_channel
        self.head_dim      = head_dim
        self.window_size   = window_size
        
        self.fusion_query1 = AttentionNet(self.channel_inter, self.channel_inter, self.head_dim, self.window_size, 'W', attention_type='reference', seed=seed)
        self.fusion_resi1  = ResidualinResidualDenseBlock(self.channel_inter, self.channel_inter, gc=self.channel_inter//2, seed=seed)
        self.fusion_query2 = AttentionNet(self.channel_inter, self.channel_inter, self.head_dim, self.window_size, 'SW', attention_type='reference', seed=seed)
        self.fusion_resi2  = ResidualinResidualDenseBlock(self.channel_inter, self.channel_inter, gc=self.channel_inter//2, seed=seed)
        
        self.fusion_mask1  = AttentionNet(self.channel_inter, self.channel_inter, self.head_dim, self.window_size, 'W', attention_type='reference', seed=seed)
        self.fusion_resi3  = ResidualinResidualDenseBlock(self.channel_inter, self.channel_inter, gc=self.channel_inter//2, seed=seed)
        self.fusion_mask2  = AttentionNet(self.channel_inter, self.channel_inter, self.head_dim, self.window_size, 'SW', attention_type='reference', seed=seed)
        self.fusion_resi4  = ResidualinResidualDenseBlock(self.channel_inter, self.channel_inter, gc=self.channel_inter//2, seed=seed)
        
        self.non_local     = SelfAttentionNet(self.channel_inter, self.channel_inter, head_dim, window_size)
        self.in_conv       = nn.Conv2d(self.channel_in, self.channel_inter, kernel_size=1, stride=1)
        self.out_conv      = nn.Conv2d(self.channel_inter, self.channel_out, kernel_size=1, stride=1)
        self.ref_conv      = nn.Conv2d(self.channel_in, self.channel_inter, kernel_size=1, stride=1)
        # self.in_conv       = DeformConv2d(self.channel_in, self.channel_inter, kernel_size=1, seed=seed)
        # self.out_conv      = DeformConv2d(self.channel_inter, self.channel_out, kernel_size=1, seed=seed)
        # self.ref_conv      = DeformConv2d(self.channel_in, self.channel_inter, kernel_size=1, seed=seed)
        
        wi.initialize_xavier([self.in_conv, self.ref_conv], 0.1, seed=seed)
        wi.initialize_xavier([self.out_conv], 0.01, seed=seed)
        # wi.initialize_xavier([self.out_conv], 1, seed=seed)
    
    def forward(self, frames: torch.Tensor, referenced_frame: torch.Tensor) -> torch.Tensor:
        compensated_frames    = []
        frames                = frames.reshape([frames.shape[0], 3, -1, 3, frames.shape[-2], frames.shape[-1]])
        frames                = torch.transpose(frames, 1, 2)
        frames                = frames.reshape([frames.shape[0], -1, frames.shape[-2], frames.shape[-1]])
        frames                = torch.split(frames, 3*3, dim=1)
        referenced_frame      = self.ref_conv(referenced_frame)
        for frame in frames:
            input_frame       = self.in_conv(frame)
            query             = self.fusion_query1(input_frame, referenced_frame)
            query             = self.fusion_resi1(query)
            query             = self.fusion_query2(query, referenced_frame)
            query             = self.fusion_resi2(query)
            mask              = self.non_local(input_frame)
            mask              = self.fusion_mask1(mask, referenced_frame)
            mask              = self.fusion_resi3(mask)
            mask              = self.fusion_mask2(mask, referenced_frame)
            mask              = self.fusion_resi4(mask)
            compensated_frame = query * torch.sigmoid(mask) + input_frame
            output_frame      = self.out_conv(compensated_frame)
            compensated_frames.append(output_frame)
            
        out = torch.cat(compensated_frames, dim=1)
        out = out.reshape([out.shape[0], -1, 3, 3, out.shape[-2], out.shape[-1]])
        out = torch.transpose(out, 1, 2)
        out = out.reshape([out.shape[0], -1, out.shape[-2], out.shape[-1]])
        return out
    

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
    
    
class FeatureCalapseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, scale = 4, gc=32, bias=True, is_res=False, GOP_size=6):
        super(FeatureCalapseBlock, self).__init__()
        self.scale    = scale
        self.is_res   = is_res
        self.GOP_size = GOP_size
        if scale>1:
            self.ds = SpaceToDepth(scale)
            self.us = nn.PixelShuffle(scale)
        channel_in  = (scale**2)*channel_in
        channel_out = (scale**2)*channel_out
        gc = (scale)*gc
        self.conv1 = nn.Conv3d(channel_in, gc, (3,3,3), 1, (1,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,3,3), 1, (1,1,1), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        wi.initialize_kaiming([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        wi.initialize_xavier(self.conv5, 0.01)

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
    
    
class PostEnhanceNet(nn.Module):
    def __init__(self, channel_in=18, intermediate_channel=32, block_num=4, seed=None):
        super(PostEnhanceNet, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_in1 = ResidualinResidualDenseBlock(channel_in, intermediate_channel)
        self.conv_in2 = ResidualinResidualDenseBlock(channel_in, intermediate_channel)

        self.conv_first_11 = MultiScaleBlock(intermediate_channel, intermediate_channel, is_res=True, seed=seed)
        self.conv_first_12 = MultiScaleBlock(intermediate_channel, intermediate_channel, is_res=True, seed=seed)
        
        residual_block = []
        for i in range(block_num):
            residual_block.append(ResidualBlock(intermediate_channel*2))
        self.residual_block = nn.Sequential(*residual_block)


        self.upconv1 = nn.Conv2d(intermediate_channel*2, intermediate_channel * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(intermediate_channel, intermediate_channel, 3, 1, 1, bias=True)

        self.conv_last = nn.Conv2d(intermediate_channel, channel_in, 3, 1, 1, bias=True)
        
        wi.initialize_kaiming([self.upconv1, self.HRconv], 0.1, seed=seed)
        wi.initialize_xavier([self.conv_last], 0.01, seed=seed)

    def forward(self, x):
        x1 = self.conv_in1(x)
        x2 = self.conv_in2(x)
        x1 = self.lrelu(self.conv_first_11(x1))
        x2 = self.lrelu(self.conv_first_12(x2))
        
        res = torch.cat([x1, x2],1)
        res = self.residual_block(res)

        res = self.lrelu(self.pixel_shuffle(self.upconv1(res)))
        res = self.lrelu(self.HRconv(res))
        res = self.conv_last(res)

        return x + res

    
class ModulatedHaarWaveletTransform(nn.Module):
    def __init__(self, channel_in: int, modulation_factor:float=1):
        super(ModulatedHaarWaveletTransform, self).__init__()
        self.channel_in = channel_in
        
        self.decomp_weights = torch.zeros(4, 1, 2, 2)
        self.recon_weights  = torch.zeros(4, 1, 2, 2)
        
        # L decomposition filters
        self.decomp_weights[0, 0, 0, 0] = math.sqrt(modulation_factor)
        self.decomp_weights[0, 0, 0, 1] = math.sqrt(modulation_factor)
        self.decomp_weights[0, 0, 1, 0] = math.sqrt(modulation_factor)
        self.decomp_weights[0, 0, 1, 1] = math.sqrt(modulation_factor)
        
        # H decomposition filters
        self.decomp_weights[1, 0, 0, 0] = math.sqrt(modulation_factor)
        self.decomp_weights[1, 0, 0, 1] = -math.sqrt(modulation_factor)
        self.decomp_weights[1, 0, 1, 0] = math.sqrt(modulation_factor)
        self.decomp_weights[1, 0, 1, 1] = -math.sqrt(modulation_factor)

        # V decomposition filters
        self.decomp_weights[2, 0, 0, 0] = math.sqrt(modulation_factor)
        self.decomp_weights[2, 0, 0, 1] = math.sqrt(modulation_factor)
        self.decomp_weights[2, 0, 1, 0] = -math.sqrt(modulation_factor)
        self.decomp_weights[2, 0, 1, 1] = -math.sqrt(modulation_factor)

        # D decomposition filters
        self.decomp_weights[3, 0, 0, 0] = math.sqrt(modulation_factor)
        self.decomp_weights[3, 0, 0, 1] = -math.sqrt(modulation_factor)
        self.decomp_weights[3, 0, 1, 0] = -math.sqrt(modulation_factor)
        self.decomp_weights[3, 0, 1, 1] = math.sqrt(modulation_factor)
        
        # L reconstruction filters
        self.recon_weights[0, 0, 0, 0] = math.sqrt(1/modulation_factor)
        self.recon_weights[0, 0, 0, 1] = math.sqrt(1/modulation_factor)
        self.recon_weights[0, 0, 1, 0] = math.sqrt(1/modulation_factor)
        self.recon_weights[0, 0, 1, 1] = math.sqrt(1/modulation_factor)
        
        # H reconstruction filters
        self.recon_weights[1, 0, 0, 0] = math.sqrt(1/modulation_factor)
        self.recon_weights[1, 0, 0, 1] = -math.sqrt(1/modulation_factor)
        self.recon_weights[1, 0, 1, 0] = math.sqrt(1/modulation_factor)
        self.recon_weights[1, 0, 1, 1] = -math.sqrt(1/modulation_factor)

        # V reconstruction filters
        self.recon_weights[2, 0, 0, 0] = math.sqrt(1/modulation_factor)
        self.recon_weights[2, 0, 0, 1] = math.sqrt(1/modulation_factor)
        self.recon_weights[2, 0, 1, 0] = -math.sqrt(1/modulation_factor)
        self.recon_weights[2, 0, 1, 1] = -math.sqrt(1/modulation_factor)

        # D reconstruction filters
        self.recon_weights[3, 0, 0, 0] = math.sqrt(1/modulation_factor)
        self.recon_weights[3, 0, 0, 1] = -math.sqrt(1/modulation_factor)
        self.recon_weights[3, 0, 1, 0] = -math.sqrt(1/modulation_factor)
        self.recon_weights[3, 0, 1, 1] = math.sqrt(1/modulation_factor)
                
        self.decomp_weights = torch.cat([self.decomp_weights] * self.channel_in, 0)
        self.recon_weights = torch.cat([self.recon_weights] * self.channel_in, 0)
        
        self.decomp_weights = nn.Parameter(self.decomp_weights)
        self.recon_weights = nn.Parameter(self.recon_weights)

        self.decomp_weights.requires_grad = False
        self.recon_weights.requires_grad  = False

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
        
        
        

class LiteQuantizationAdaptionNet(nn.Module):
    def __init__(self, in_channel, intermediate_channel, seed=None):
        super(LiteQuantizationAdaptionNet, self).__init__()
        self.fuse_conv   = nn.Conv2d(3*2, 3, 1, groups=3)
        self.input_conv  = nn.Conv2d(3, intermediate_channel//2, 1)
        self.qp_conv     = nn.Conv2d(1, intermediate_channel//2, 1)
        self.output_conv = nn.Conv2d(intermediate_channel, 3, 1)
        
        self.query_gen   = nn.Sequential(
            ResidualinResidualDenseBlock(intermediate_channel//2, intermediate_channel, seed=seed)
        )
        self.mask_gen    = nn.Sequential(
            ResidualinResidualDenseBlock(intermediate_channel//2, intermediate_channel, seed=seed)
        )
        
        wi.initialize_xavier([self.fuse_conv, self.input_conv, self.qp_conv], 0.1, seed=seed)
        wi.initialize_xavier([self.output_conv], 0.01, seed=seed)

    def forward(self, x, ref, qp):
        output  = []
        frames  = torch.split(x, 3, dim=1)
        for frame in frames:
            input       = torch.cat((ref, frame), 1)[:,[0,3,1,4,2,5],:,:]
            fused_frame = self.fuse_conv(input)
            query       = self.query_gen(self.input_conv(fused_frame))
            mask        = self.mask_gen(self.qp_conv(qp))
            out         = self.output_conv(query * torch.sigmoid(mask)) + frame
            output.append(out)
        return torch.cat(output, 1)
    
    
class LiteInterframeAttentionFusionNet(nn.Module):
    def __init__(self, channel_in=3*3, intermediate_channel=64, head_dim=8, window_size=8, seed=None):
        super(LiteInterframeAttentionFusionNet, self).__init__()
        self.channel_in    = channel_in
        self.channel_out   = channel_in
        self.channel_inter = intermediate_channel
        self.head_dim      = head_dim
        self.window_size   = window_size
        
        self.query_gen   = nn.Sequential(
            ResidualinResidualDenseBlock(intermediate_channel//4, intermediate_channel//2, seed=seed),
            ResidualinResidualDenseBlock(intermediate_channel//2, intermediate_channel, seed=seed)
        )
        self.mask_gen    = nn.Sequential(
            ResidualinResidualDenseBlock(intermediate_channel//4, intermediate_channel//2, seed=seed),
            ResidualinResidualDenseBlock(intermediate_channel//2, intermediate_channel, seed=seed)
        )
        
        self.in_conv       = nn.Conv2d(self.channel_in, self.channel_inter//4, kernel_size=1, stride=1)
        self.out_conv      = nn.Conv2d(self.channel_inter, self.channel_out, kernel_size=1, stride=1)
        self.ref_conv      = nn.Conv2d(self.channel_in, self.channel_inter//4, kernel_size=1, stride=1)
        # self.in_conv       = DeformConv2d(self.channel_in, self.channel_inter, kernel_size=1, seed=seed)
        # self.out_conv      = DeformConv2d(self.channel_inter, self.channel_out, kernel_size=1, seed=seed)
        # self.ref_conv      = DeformConv2d(self.channel_in, self.channel_inter, kernel_size=1, seed=seed)
        
        wi.initialize_xavier([self.in_conv, self.ref_conv], 0.1, seed=seed)
        wi.initialize_xavier([self.out_conv], 0.01, seed=seed)
        # wi.initialize_xavier([self.out_conv], 1, seed=seed)
    
    def forward(self, frames: torch.Tensor, referenced_frame: torch.Tensor) -> torch.Tensor:
        compensated_frames    = []
        frames                = frames.reshape([frames.shape[0], 3, -1, 3, frames.shape[-2], frames.shape[-1]])
        frames                = torch.transpose(frames, 1, 2)
        frames                = frames.reshape([frames.shape[0], -1, frames.shape[-2], frames.shape[-1]])
        frames                = torch.split(frames, 3*3, dim=1)
        referenced_frame      = self.ref_conv(referenced_frame)
        for frame in frames:
            input_frame       = self.in_conv(frame)
            query             = self.query_gen(input_frame)
            mask              = self.mask_gen(referenced_frame)
            compensated_frame = query * torch.sigmoid(mask)
            output_frame      = self.out_conv(compensated_frame) + frame
            compensated_frames.append(output_frame)
            
        out = torch.cat(compensated_frames, dim=1)
        out = out.reshape([out.shape[0], -1, 3, 3, out.shape[-2], out.shape[-1]])
        out = torch.transpose(out, 1, 2)
        out = out.reshape([out.shape[0], -1, out.shape[-2], out.shape[-1]])
        return out