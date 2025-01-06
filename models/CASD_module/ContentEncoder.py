import torch.nn as nn

from models.tools.EncoderBlock import EncoderBlock
from models.tools.functions import get_norm_layer, get_nonlinearity_layer


# 目标姿势编码器
class ContentEncoder(nn.Module):
    def __init__(self, layers=2, ngf=64, img_f=512, use_spect=False, use_coord=False):
        super(ContentEncoder, self).__init__()

        self.layers = layers
        norm_layer = get_norm_layer(norm_type='instance')
        nonlinearity = get_nonlinearity_layer(activation_type='LeakyReLU')
        self.ngf = ngf
        self.img_f = img_f
        self.block0 = EncoderBlock(30, ngf, norm_layer,
                                   nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(self.layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), self.img_f // self.ngf)
            block = EncoderBlock(self.ngf * mult_prev, self.ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        self.model0 = []
        self.model0 += [norm_layer(128)]
        self.model0 += [nonlinearity]
        self.model0 += [nn.Conv2d(128, 256, 1, 1, 0)]
        self.model0 = nn.Sequential(*self.model0)

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.model0(out)
        return out
