import torch.nn as nn
import functools
import torch
import torch.nn.functional as F

import os
import torchvision.models.vgg as models
from torch.nn.parameter import Parameter

from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
import functools

from models.CASD_module.ContentEncoder import ContentEncoder
from models.CASD_module.VggStyleEncoder import VggStyleEncoder
from models.tools.BasicBlocks import Conv2dBlock_my, LinearBlock, Conv2dBlock, ResBlock
from models.tools.EncoderBlock import EncoderBlock
from models.tools.FreqFusion import FreqFusion
from models.tools.functions import get_norm_layer, get_nonlinearity_layer


# Moddfied with AdINGen
class ADGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, dim, style_dim, n_downsample, n_res, mlp_dim, activ='relu', pad_type='reflect'):
        super(ADGen, self).__init__()

        # style encoder
        input_dim = 3
        self.SP_input_nc = 8
        self.enc_style = VggStyleEncoder(3, input_dim, dim, int(style_dim / self.SP_input_nc), norm='none', activ=activ,
                                         pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(layers=2, ngf=64, img_f=512)

        input_dim = 3
        self.dec = Decoder(style_dim, mlp_dim, n_downsample, n_res, 256, input_dim,
                           self.SP_input_nc, res_norm='adain', activ=activ, pad_type=pad_type)

    def forward(self, img_A, img_B, sem_B):
        content = self.enc_content(img_A)
        style = self.enc_style(img_B, sem_B)

        images_recon = self.dec(content, style,self.enc_style.styleFeature)
        return images_recon


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = torch.reshape(x, (b, c, h, w))
        return x

class MyDecoder(nn.Module):
    def __init__(self, n_upsample, dim, activ='relu', pad_type='zero'):
        super(MyDecoder,self).__init__()
        self.model1=[]
        for i in range(n_upsample):
            self.model1 += [nn.Upsample(scale_factor=2),
                            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        self.model1 = nn.Sequential(*self.model1)
    def forward(self, x):
        return self.model1(x)

class Decoder(nn.Module):
    def __init__(self, style_dim, mlp_dim, n_upsample, n_res, dim, output_dim, SP_input_nc, res_norm='adain',
                 activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.softmax_style = nn.Softmax(dim=2)
        self.SP_input_nc = SP_input_nc
        self.model0 = []
        self.model1 = []
        self.model2 = []
        self.n_res = n_res

        self.mlp = MLP(style_dim, n_res * dim * 4, mlp_dim, 3, norm='none', activ=activ)
        self.fc = LinearBlock(style_dim, style_dim, norm='none', activation=activ)

        # AdaIN residual blocks
        self.model0_0 = [ResBlock_my(dim, res_norm, activ, pad_type=pad_type)]
        self.model0_0 = nn.Sequential(*self.model0_0)
        self.model0_1 = [ResBlock_my(dim, res_norm, activ, pad_type=pad_type)]
        self.model0_1 = nn.Sequential(*self.model0_1)
        self.model0_2 = [ResBlock_my(dim, res_norm, activ, pad_type=pad_type)]
        self.model0_2 = nn.Sequential(*self.model0_2)
        self.model0_3 = [ResBlock_my(dim, res_norm, activ, pad_type=pad_type)]
        self.model0_3 = nn.Sequential(*self.model0_3)
        self.model0_4 = [ResBlock_myDFNM(dim, 'spade', activ, pad_type=pad_type)]
        self.model0_4 = nn.Sequential(*self.model0_4)
        self.model0_5 = [ResBlock_myDFNM(dim, 'spade', activ, pad_type=pad_type)]
        self.model0_5 = nn.Sequential(*self.model0_5)
        self.model0_6 = [ResBlock_myDFNM(dim, 'spade', activ, pad_type=pad_type)]
        self.model0_6 = nn.Sequential(*self.model0_6)
        self.model0_7 = [ResBlock_myDFNM(dim, 'spade', activ, pad_type=pad_type)]
        self.model0_7 = nn.Sequential(*self.model0_7)
        # upsampling blocks
        # self.model1 = MyDecoder(n_upsample, dim, activ, pad_type=pad_type)
        for i in range(n_upsample):
            # self.model1 += [nn.Upsample(scale_factor=2),
            #                 Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # self.model1 = nn.Sequential(*self.model1)


        # use reflection padding in the last conv layer
        self.model2 += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model2 = nn.Sequential(*self.model2)
        # attention parameter

        self.gamma3_1 = nn.Parameter(torch.zeros(1))
        self.gamma3_2 = nn.Parameter(torch.zeros(1))
        self.gamma3_3 = nn.Parameter(torch.zeros(1))
        self.gamma3_style_sa = nn.Parameter(torch.zeros(1))
        in_dim = int(style_dim / self.SP_input_nc)
        self.value3_conv_sa = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.LN_3_style = ILNKVT(256)
        self.LN_3_pose = ILNQT(256)
        self.LN_3_pose_0 = ILNQT(256)
        self.query3_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key3_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value3_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query3_conv_0 = nn.Conv2d(in_channels=in_dim, out_channels=self.SP_input_nc, kernel_size=1)

        self.gamma4_1 = nn.Parameter(torch.zeros(1))
        self.gamma4_2 = nn.Parameter(torch.zeros(1))
        self.gamma4_3 = nn.Parameter(torch.zeros(1))
        self.gamma4_style_sa = nn.Parameter(torch.zeros(1))
        in_dim = int(style_dim / self.SP_input_nc)
        self.value4_conv_sa = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.LN_4_style = ILNKVT(256)
        self.LN_4_pose = ILNQT(256)
        self.LN_4_pose_0 = ILNQT(256)
        self.query4_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key4_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value4_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query4_conv_0 = nn.Conv2d(in_channels=in_dim, out_channels=self.SP_input_nc, kernel_size=1)

        self.FFN3_1 = FFN(256)
        self.FFN4_1 = FFN(256)
        self.up = nn.Upsample(scale_factor=2)

        norm_layer = get_norm_layer(norm_type='instance')
        nonlinearity = get_nonlinearity_layer(activation_type='LeakyReLU')
        self.encoder = EncoderBlock(256, 256, norm_layer,nonlinearity, False, False)

        self.ff1=FreqFusion(hr_channels=256,lr_channels=256)
        self.conv1=Conv2dBlock(256, 128, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)
        self.ff2=FreqFusion(hr_channels=128,lr_channels=128)
        # self.conv2 = Conv2dBlock(128, 64, 3,1, 1, norm='ln', activation=activ, pad_type=pad_type)
        # self.ff3=FreqFusion(hr_channels=64,lr_channels=64)
        self.upSampling = [nn.Upsample(scale_factor=2), Conv2dBlock(128, 64, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        self.upSampling = nn.Sequential(*self.upSampling)
        self.beta_1 = nn.Parameter(torch.zeros(1))
        self.beta_2 = nn.Parameter(torch.zeros(1))


    def forward(self, x, style,styleFeature):
        # fusion module
        style_fusion = self.fc(style.view(style.size(0), -1))
        adain_params = self.mlp(style_fusion)
        # 在通道维度切分指定块数
        adain_params = torch.split(adain_params, int(adain_params.shape[1] / self.n_res), 1)

        x_0 = x
        x = self.model0_0([x, adain_params[0]])
        x = self.model0_1([x, adain_params[1]])
        x = self.model0_2([x, adain_params[2]])
        x = self.model0_3([x, adain_params[3]])

        # 1*256*64*64   1*8*64*64
        x3, enerrgy_sum3 = self.styleatt(x, x_0, style, self.gamma3_1, self.gamma3_2, self.gamma3_3,
                                         self.gamma3_style_sa, self.value3_conv_sa,
                                         self.LN_3_style, self.LN_3_pose, self.LN_3_pose_0,
                                         self.query3_conv, self.key3_conv, self.value3_conv, self.query3_conv_0,
                                         self.FFN3_1)

        x_, enerrgy_sum4 = self.styleatt(x3, x_0, style, self.gamma4_1, self.gamma4_2, self.gamma4_3,
                                         self.gamma4_style_sa, self.value4_conv_sa,
                                         self.LN_4_style, self.LN_4_pose, self.LN_4_pose_0,
                                         self.query4_conv, self.key4_conv, self.value4_conv, self.query4_conv_0,
                                         self.FFN4_1)

        x = self.model0_4([x_0, x_])
        x = self.model0_5([x, x_])
        x = self.model0_6([x, x_])
        # 1*256*64*64
        x = self.model0_7([x, x_])
        # #1*256*64*64 -> 1 * 64 * 256 * 256
        # x = self.model1(x)

        #1*256*32*32
        y3=self.encoder(x)
        _,x3,y3_up=self.ff1(hr_feat=styleFeature['x3'],lr_feat=y3)
        y2=self.beta_1*x3+y3_up
        y2=self.conv1(y2)
        _,x2,y2_up=self.ff2(hr_feat=styleFeature['x2'],lr_feat=y2)
        y1=self.beta_2*x2+y2_up
        y1=self.upSampling(y1)
        # _,x1,y1_up=self.ff3(hr_feat=styleFeature['x1'],lr_feat=y1)
        # y0=x1+y1_up

        return self.model2(y1), [enerrgy_sum3, enerrgy_sum4]

    def styleatt(self, x, x_0, style, gamma1, gamma2, gamma3, gamma_style_sa, value_conv_sa, ln_style, ln_pose,
                 ln_pose_0, query_conv, key_conv, value_conv, query_conv_0, ffn1):
        B, C, H, W = x.size()
        B, Cs, _, _ = style.size()
        K = self.SP_input_nc
        style = style.view((B, K, int(Cs / K)))  # [B,K,C]

        x = ln_pose(x)  # [B,C,H,W]
        style = ln_style(style.permute(0, 2, 1))  # [B,C,K]
        x_0 = ln_pose_0(x_0)

        style = style.permute(0, 2, 1)  # [B,K,C]
        style_sa_value = torch.squeeze(value_conv_sa(torch.unsqueeze(style.permute(0, 2, 1), 3)), 3)  # [B,C,K]
        self_att = self.softmax(torch.bmm(style, style.permute(0, 2, 1))) + 1e-8  # [B,K,K]
        self_att = self_att / torch.sum(self_att, dim=2, keepdim=True)
        style_ = torch.bmm(self_att, style_sa_value.permute(0, 2, 1))
        style = style + gamma_style_sa * style_  # [B,K,C]

        style = style.permute(0, 2, 1)  # [B,C,K]
        x_query = query_conv(x)
        style_key = torch.squeeze(key_conv(torch.unsqueeze(style, 3)).permute(0, 2, 1, 3), 3)
        style_value = torch.squeeze(value_conv(torch.unsqueeze(style, 3)), 3)

        energy_0 = query_conv_0(x_0).view((B, K, H * W))
        energy = torch.bmm(style_key.detach(), x_query.view(B, C, -1))
        enerrgy_sum = energy_0 + energy
        attention = self.softmax_style(enerrgy_sum) + 1e-8
        attention = attention / torch.sum(attention, dim=1, keepdim=True)

        out = torch.bmm(style_value, attention)
        out = out.view(B, C, H, W)
        out = gamma1 * out + x
        out = out + gamma3 * ffn1(out)

        return out, torch.reshape(enerrgy_sum, (B, K, H, W))


class ILNKVT(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2], keepdim=True), torch.var(input, dim=[2], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1], keepdim=True), torch.var(input, dim=[1], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1) * out_in + (1 - self.rho.expand(input.shape[0], -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1) + self.beta.expand(input.shape[0], -1, -1)

        return out


class ILNQT(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1], keepdim=True), torch.var(input, dim=[1], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock_myDFNM(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock_myDFNM, self).__init__()

        model1 = []
        model2 = []
        model1 += [Conv2dBlock_my(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model2 += [Conv2dBlock_my(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        models1 = []
        models1 += [Conv2dBlock(dim, dim, 3, 1, 1, norm='in', activation='relu', pad_type=pad_type)]
        models1 += [Conv2dBlock(dim, 2 * dim, 3, 1, 1, norm='none', activation='none', pad_type=pad_type)]
        models2 = []
        models2 += [Conv2dBlock(dim, dim, 3, 1, 1, norm='in', activation='relu', pad_type=pad_type)]
        models2 += [Conv2dBlock(dim, 2 * dim, 3, 1, 1, norm='none', activation='none', pad_type=pad_type)]
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.models1 = nn.Sequential(*models1)
        self.models2 = nn.Sequential(*models2)

    def forward(self, x):
        style = x[1]
        style1 = self.models1(style)
        style2 = self.models2(style)
        residual = x[0]
        out = self.model1([x[0], style1])
        out = self.model2([out, style2])
        out += residual

        return out


class ResBlock_my(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock_my, self).__init__()

        model1 = []
        model2 = []
        model1 += [Conv2dBlock_my(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model2 += [Conv2dBlock_my(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)

    def forward(self, x):
        style = x[1]
        style1, style2 = torch.split(style, int(style.shape[1] / 2), 1)
        residual = x[0]
        out = self.model1([x[0], style1])
        out = self.model2([out, style2])
        out += residual
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)



