import torch
import logging
# import .modules.discriminator_vgg_arch as SRGAN_arch
# from .modules.Inv_arch import *
# from .modules.Subnet_constructor import subnet
import math
import torch.nn as nn
import numpy as np
logger = logging.getLogger('base')
from collections import OrderedDict
import torch.nn.init as init
from .res_block import ResBlock

def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out ,attention
        else:
            return out

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='kaiming', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


def symm_pad(im, padding):
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)

    x_pad = reflect(x_idx, -0.5, w - 0.5)
    y_pad = reflect(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]

def reflect(x, minx, maxx):
    """ Reflects an array around two points making a triangular waveform that ramps up
    and down,  allowing for pad lengths greater than the input length """
    rng = maxx - minx
    double_rng = 2 * rng
    mod = np.fmod(x - minx, double_rng)
    normed_mod = np.where(mod < 0, mod + double_rng, mod)
    out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)


def rgb2gray(rgb):
    b, g, r = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gray = torch.unsqueeze(gray, 1)
    return gray


class UNetDiscriminator(BaseNetwork):
    def __init__(self, in_channels=3, out_channels=1, residual_blocks=4, init_weights=True, use_spectral_norm=True,
                 use_SRM=True, with_attn=False, dim=16, use_sigmoid=False):
        super(UNetDiscriminator, self).__init__()
        # dim = 32
        self.use_SRM = use_SRM
        self.use_sigmoid = use_sigmoid
        self.with_attn = with_attn
        self.clock = 1
        # if self.use_SRM:

        # self.SRMConv2D = nn.Conv2d(in_channels, 9, 5, 1, padding=2, bias=False)
        # self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']
        # ##SRM filters (fixed)
        # for param in self.SRMConv2D.parameters():
        #     param.requires_grad = False

        self.BayarConv2D = nn.Conv2d(in_channels, 3, 5, 1, padding=2, bias=False)
        self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
        self.bayar_mask[2, 2] = 0
        self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
        self.bayar_final[2, 2] = -1
        self.activation = nn.ELU(inplace=True)

        self.init_conv = nn.Sequential(

            spectral_norm(nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )


        self.encoder_1 = nn.Sequential(

            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),

            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2,  kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )

        self.encoder_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),

            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4,  kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True)
        )

        self.encoder_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 8, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 8, out_channels=dim * 8, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 8, out_channels=dim * 8, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 8, out_channels=dim * 8, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True)
        )

        blocks = []
        for _ in range(residual_blocks):  # residual_blocks
            block = ResBlock(dim * 8, dilation=1, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder_3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 8 * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )

        self.decoder_2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 4 * 2, out_channels=dim * 2, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),

            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 2, out_channels=dim * 2,  kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )

        self.decoder_1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 2 * 2, out_channels=dim, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),

            spectral_norm(nn.ConvTranspose2d(in_channels=dim, out_channels=dim,  kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )

        self.decoder_0 = nn.Sequential(
            # spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1),
            #               use_spectral_norm),
            # nn.ELU(),
            # nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=1, padding=0),
        )

        if init_weights:
            self.init_weights()

    def update_clock(self):
        self.clock = min(1.0, self.clock + 1e-4)

    def forward(self, x, qf=None):
        # x = x.contiguous()
        ## **Bayar constraints**
        # if self.use_SRM:
        self.BayarConv2D.weight.data *= self.bayar_mask
        self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
        self.BayarConv2D.weight.data += self.bayar_final

        # Symmetric padding
        # x = symm_pad(x, (2, 2, 2, 2))

        # conv_init = self.vanillaConv2D(x)
        conv_bayar = self.BayarConv2D(x)
        # conv_srm = self.SRMConv2D(x)

        first_block = conv_bayar #torch.cat([conv_init, conv_srm, conv_bayar], axis=1)
        e0 = self.activation(first_block)
        e0 = self.init_conv(e0)


        # for layer in self.middle_and_last_block:
        #
        #     if isinstance(layer, nn.Conv2d):
        #         last_block = symm_pad(last_block, (1, 1, 1, 1))
        #
        #     last_block = layer(last_block)
        #
        # return (torch.tanh(last_block) + 1) / 2

        # e0 = self.encoder_0(x)
        e1 = self.encoder_1(e0)
        # e1_add = self.encoder_1_add(e1)
        # e1 = e1_add #self.clock*e1_add+(1-self.clock)*e1
        e2 = self.encoder_2(e1)
        # e2_add = self.encoder_2_add(e2)
        # e2 = e2_add #self.clock * e2_add + (1 - self.clock) * e2
        e3 = self.encoder_3(e2)
        # e3_add = self.encoder_3_add(e3)
        # e3 = e3_add  # self.clock * e2_add + (1 - self.clock) * e2

        m = self.middle(e3)

        d3 = self.decoder_3(torch.cat((e3, m), dim=1))
        # d3_add = self.decoder_3_add(d3)
        # d3 = d3_add #self.clock * d2_add + (1 - self.clock) * d2
        d2 = self.decoder_2(torch.cat((e2, d3), dim=1))
        # d2_add = self.decoder_2_add(d2)
        # d2 = d2_add #self.clock * d2_add + (1 - self.clock) * d2
        d1 = self.decoder_1(torch.cat((e1, d2), dim=1))
        # d1_add = self.decoder_1_add(d1)
        # d1 = d1_add #self.clock * d1_add + (1 - self.clock) * d1
        x = self.decoder_0(torch.cat((e0, d1), dim=1))
        if self.use_sigmoid:
            x = torch.sigmoid(x)

        return x

class RHI3Net(BaseNetwork):
    def __init__(self, images_in=2, images_out=1, residual_blocks=3, dim=32, init_weights=True, use_spectral_norm=True, use_SRM=True, with_attn=False, additional_conv=False):
        super(RHI3Net, self).__init__()
        self.images_in = images_in
        self.images_out = images_out

        self.encoding_block = []
        for i in range(images_in):
            blocks = []
            blocks.append(
                nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=3, stride=1, padding=1).cuda()
            )
            for _ in range(residual_blocks):  # residual_blocks
                block = ResBlock(dim, dilation=1, use_spectral_norm=use_spectral_norm).cuda()
                blocks.append(block)
            sub_block = nn.Sequential(*blocks)
            self.encoding_block.append(sub_block)

        self.concat_block = nn.Conv2d(in_channels=dim*images_in, out_channels=dim*images_out, kernel_size=3, stride=1, padding=1).cuda()

        self.decoding_block = []
        for i in range(images_out):
            blocks = []
            for _ in range(residual_blocks):  # residual_blocks
                block = ResBlock(dim, dilation=1, use_spectral_norm=use_spectral_norm).cuda()
                blocks.append(block)
            blocks.append(
                nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=3, stride=1, padding=1).cuda()
            )
            sub_block = nn.Sequential(*blocks)
            self.decoding_block.append(sub_block)


        if init_weights:
            self.init_weights()



    def forward(self, x):
        batch_size, channels, width, height = x.shape
        if self.images_in>1:
            assert channels//3==self.images_in, "You need to make sure batch_size is dividable by input_num!"

        enc_feats = []
        for idx, enc_block in enumerate(self.encoding_block):
            enc_feats.append(enc_block(x[:,idx:idx+3]))
        concat_feat = torch.cat(enc_feats,dim=1)

        mid_feat = self.concat_block(concat_feat)

        dec_feats = torch.chunk(mid_feat,chunks=self.images_out,dim=1) # returns a tuple, b[0] size: 1,3,512,512

        y = []
        for idx,dec_block in enumerate(self.decoding_block):
            y.append(dec_block(dec_feats[idx]))

        return y



def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


if __name__ == '__main__':
    input = torch.ones((3,3,64,64)).cuda()
    # model = JPEGGenerator()
    model = RHI3Net(images_in=3,images_out=1).cuda()
    # output = model(input,qf=torch.tensor([[0.2]]))
    output = model(input)
    for i in range(len(output)):
        print(output[i].shape)



