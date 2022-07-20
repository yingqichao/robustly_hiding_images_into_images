import logging

import torch
import torch.nn as nn

logger = logging.getLogger('base')
import torch.nn.init as init

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out=None,use_spectral_norm=False,init='xavier', dilation=1, clock=1):
        super(ResBlock, self).__init__()
        feature = 64
        channel_out = channel_in if channel_out is None else channel_out
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel_in, feature, kernel_size=3, padding=1*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        # self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=3, padding=1*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=3, padding=1*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=3, padding=1*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        self.conv1_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=5, padding=2*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        # self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=5, padding=2*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        self.conv3_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=5, padding=2*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        self.conv4_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=5, padding=2*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        self.conv5 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)
        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)
        if init == 'xavier':
            initialize_weights_xavier([self.conv1_1, self.conv2_1, self.conv3_1, self.conv4_1], 0.1)
        else:
            initialize_weights([self.conv1_1, self.conv2_1, self.conv3_1, self.conv4_1], 0.1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        residual = self.conv4(residual)
        residual = self.conv1_1(residual)
        residual = self.conv2_1(residual)
        residual = self.conv3_1(residual)
        residual = self.conv4_1(residual)

        input = torch.cat((x, residual), dim=1)
        out = self.conv5(input)

        return out
