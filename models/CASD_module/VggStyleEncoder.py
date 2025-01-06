import torchvision.models.vgg as models
import torch.nn as nn
import torch
from models.tools.BasicBlocks import Conv2dBlock_my, LinearBlock, Conv2dBlock, ResBlock
import torch.nn.functional as F

# 人物图像纹理特征编码
# CASD_FreqFusion添加了styleFeature变量，记录不同编码阶段特征
class VggStyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(VggStyleEncoder, self).__init__()
        # self.vgg = models.vgg19(pretrained=True).features
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load('dataset/fashion/vgg19-dcbb9e9d.pth'))
        self.vgg = vgg19.features

        for param in self.vgg.parameters():
            param.requires_grad_(False)

        self.conv1 = Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)  # 3->64
        dim = dim * 2
        self.conv2 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)  # 128->128
        dim = dim * 2
        self.conv3 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)  # 256->256
        dim = dim * 2
        self.conv4 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)  # 512->512
        dim = dim * 2

        self.model0 = []
        self.model0 += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model0 = nn.Sequential(*self.model0)

        self.AP = []
        self.AP += [nn.AdaptiveAvgPool2d(1)]
        self.AP = nn.Sequential(*self.AP)
        self.output_dim = dim

        self.styleFeature={}

    def get_features(self, image, model, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1'}
        features = {}
        x = image
        # model._modules is a dictionary holding each module in the model
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def texture_enc(self, x):
        sty_fea = self.get_features(x, self.vgg)
        x = self.conv1(x)
        # self.styleFeature['x1'] = x

        x = torch.cat([x, sty_fea['conv1_1']], dim=1)
        x = self.conv2(x)
        self.styleFeature['x2'] = x

        x = torch.cat([x, sty_fea['conv2_1']], dim=1)
        x = self.conv3(x)
        self.styleFeature['x3'] = x

        x = torch.cat([x, sty_fea['conv3_1']], dim=1)
        x = self.conv4(x)
        x = torch.cat([x, sty_fea['conv4_1']], dim=1)
        x0 = self.model0(x)
        return x0

    def forward(self, x, sem):

        codes = self.texture_enc(x)
        segmap = F.interpolate(sem, size=codes.size()[2:], mode='nearest')

        bs = codes.shape[0]
        hs = codes.shape[2]
        ws = codes.shape[3]
        cs = codes.shape[1]
        f_size = cs

        s_size = segmap.shape[1]
        codes_vector = torch.zeros((bs, s_size, cs), dtype=codes.dtype, device=codes.device)

        for i in range(bs):
            for j in range(s_size):
                component_mask_area = torch.sum(segmap.bool()[i, j])
                if component_mask_area > 0:
                    codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size,
                                                                                                  component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature
                else:
                    tmpmean, tmpstd = calc_mean_std(
                        codes[i].reshape(1, codes[i].shape[0], codes[i].shape[1], codes[i].shape[2]))
                    codes_vector[i][j] = tmpmean.squeeze()

        return codes_vector.view(bs, -1).unsqueeze(2).unsqueeze(3)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std