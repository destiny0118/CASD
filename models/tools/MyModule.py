import torch
from skimage.exposure import match_histograms
import numpy as np
from torch import nn


# class EFDM(nn.Module):
#     def __init__(self):
#         super(EFDM, self).__init__()
#         self.f1=exact_feature_distribution_matching()
#
#
#
#     def forward(self, content,style):



## EFDM
def exact_feature_distribution_matching(content_feat, style_feat):
    assert (content_feat.size() == style_feat.size())
    B, C, W, H = content_feat.size(0), content_feat.size(1), content_feat.size(2), content_feat.size(3)
    value_content, index_content = torch.sort(content_feat.view(B,C,-1))  # sort conduct a deep copy here.
    value_style, _ = torch.sort(style_feat.view(B,C,-1))  # sort conduct a deep copy here.
    inverse_index = index_content.argsort(-1)
    new_content = content_feat.view(B,C,-1) + (value_style.gather(-1, inverse_index) - content_feat.view(B,C,-1).detach())

    return new_content.view(B, C, W, H)

## HM
def histogram_matching(content_feat, style_feat):
    assert (content_feat.size() == style_feat.size())
    B, C, W, H = content_feat.size(0), content_feat.size(1), content_feat.size(2), content_feat.size(3)
    x_view = content_feat.view(-1, W,H)
    image1_temp = match_histograms(np.array(x_view.detach().clone().cpu().float().transpose(0, 2)),
                                   np.array(style_feat.view(-1, W, H).detach().clone().cpu().float().transpose(0, 2)),
                                   multichannel=True)
    image1_temp = torch.from_numpy(image1_temp).float().to(content_feat.device).transpose(0, 2).view(B, C, W, H)
    return content_feat + (image1_temp - content_feat).detach()
