import torch
import torch.nn as nn
import torch.nn.init as init
# import torch.nn.functional as F

def initialize_kaiming(net_l, scale=1, seed=None):
    '''
        For the previous layer of relu and leakyrelu activation.
    '''
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1.)
                init.constant_(m.bias.data, 0.)


def initialize_xavier(net_l, scale=1, seed=None):
    '''
        For the previous layer of sigmoid and tanh activation, or no activations.
    '''
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1.)
                init.constant_(m.bias.data, 0.)