from collections import OrderedDict
from functools import partial
from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import BatchNormAct2d, ConvNormAct, create_conv2d
from utils.utils import load_weights_from_state_dict

urls_dict = {
    'dpn68': 'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn68-66bebafa7.pth',
    'dpn68b': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dpn68b_ra-a31ca160.pth',
    'dpn92': 'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn92_extra-b040e4a9b.pth',
    'dpn98': 'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn98-5b90dec4d.pth',
    'dpn131': 'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn131-71dfe43e0.pth',
    'dpn107': 'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn107_extra-1ac7121e2.pth'
}

__all__ = list(urls_dict.keys())

class CatBnAct(nn.Module):
    def __init__(self, in_chs, norm_layer=BatchNormAct2d):
        super(CatBnAct, self).__init__()
        self.bn = norm_layer(in_chs, eps=0.001)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (Tuple[torch.Tensor, torch.Tensor]) -> (torch.Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (torch.Tensor) -> (torch.Tensor)
        pass

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)
        return self.bn(x)


class BnActConv2d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride, groups=1, norm_layer=BatchNormAct2d):
        super(BnActConv2d, self).__init__()
        self.bn = norm_layer(in_chs, eps=0.001)
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, groups=groups)

    def forward(self, x):
        return self.conv(self.bn(x))

class DualPathBlock(nn.Module):
    def __init__(
            self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type == 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type == 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type == 'normal'
            self.key_stride = 1
            self.has_proj = False

        self.c1x1_w_s1 = None
        self.c1x1_w_s2 = None
        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)

        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(
            in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=self.key_stride, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = create_conv2d(num_3x3_b, num_1x1_c, kernel_size=1)
            self.c1x1_c2 = create_conv2d(num_3x3_b, inc, kernel_size=1)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)
            self.c1x1_c1 = None
            self.c1x1_c2 = None

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        pass

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, tuple):
            x_in = torch.cat(x, dim=1)
        else:
            x_in = x
        if self.c1x1_w_s1 is None and self.c1x1_w_s2 is None:
            # self.has_proj == False, torchscript requires condition on module == None
            x_s1 = x[0]
            x_s2 = x[1]
        else:
            # self.has_proj == True
            if self.c1x1_w_s1 is not None:
                # self.key_stride = 1
                x_s = self.c1x1_w_s1(x_in)
            else:
                # self.key_stride = 2
                x_s = self.c1x1_w_s2(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        x_in = self.c1x1_c(x_in)
        if self.c1x1_c1 is not None:
            # self.b == True, using None check for torchscript compat
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


class DPN(nn.Module):
    def __init__(
            self, small=False, num_init_features=64, k_r=96, groups=32, global_pool='avg',
            b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128), output_stride=32,
            num_classes=1000, in_chans=3, drop_rate=0., fc_act_layer=nn.ELU):
        super(DPN, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.b = b
        assert output_stride == 32  # FIXME look into dilation support
        norm_layer = partial(BatchNormAct2d, eps=.001)
        fc_norm_layer = partial(BatchNormAct2d, eps=.001, act_layer=fc_act_layer, inplace=False)
        bw_factor = 1 if small else 4
        blocks = OrderedDict()

        # conv1
        blocks['conv1_1'] = ConvNormAct(
            in_chans, num_init_features, kernel_size=3 if small else 7, stride=2, norm_layer=norm_layer)
        blocks['conv1_pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.feature_info = [dict(num_chs=num_init_features, reduction=2, module='features.conv1_1')]

        # conv2
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        self.feature_info += [dict(num_chs=in_chs, reduction=4, module=f'features.conv2_{k_sec[0]}')]

        # conv3
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        self.feature_info += [dict(num_chs=in_chs, reduction=8, module=f'features.conv3_{k_sec[1]}')]

        # conv4
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        self.feature_info += [dict(num_chs=in_chs, reduction=16, module=f'features.conv4_{k_sec[2]}')]

        # conv5
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        self.feature_info += [dict(num_chs=in_chs, reduction=32, module=f'features.conv5_{k_sec[3]}')]

        blocks['conv5_bn_ac'] = CatBnAct(in_chs, norm_layer=fc_norm_layer)

        self.num_features = in_chs
        self.features = nn.Sequential(blocks)

        # Using 1x1 conv for the FC layer to allow the extra pooling scheme
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features=self.num_features, out_features=num_classes)

    def forward_features(self, x, need_fea=False):
        if need_fea:
            input_size = x.size(2)
            scale = [4, 8, 16, 32]
            features = [None, None, None, None]
            for idx, layer in enumerate(self.features):
                x = layer(x)
                temp_x = torch.cat(x, dim=1) if type(x) is tuple else x
                if input_size // temp_x.size(2) in scale:
                    features[scale.index(input_size // temp_x.size(2))] = temp_x
            out = self.global_pool(x)
            return features, out.flatten(1)
        else:
            x = self.features(x)
            x = self.global_pool(x)
            return x.flatten(1)

    def forward_head(self, x):
        return self.classifier(x)

    def forward(self, x, need_fea=False):
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea=need_fea)
            x = self.forward_head(features_fc)
            return features, features_fc, x
        else:
            x = self.forward_features(x)
            x = self.forward_head(x)
            return x
    
    def cam_layer(self):
        return self.features[-1]


def _create_dpn(variant, pretrained=False, **kwargs):
    model = DPN(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(urls_dict[variant], progress=True, check_hash=True)
        model = load_weights_from_state_dict(model, state_dict)
    return model


def dpn68(pretrained=False, **kwargs):
    model_kwargs = dict(
        small=True, num_init_features=10, k_r=128, groups=32,
        k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64), **kwargs)
    return _create_dpn('dpn68', pretrained=pretrained, **model_kwargs)


def dpn68b(pretrained=False, **kwargs):
    model_kwargs = dict(
        small=True, num_init_features=10, k_r=128, groups=32,
        b=True, k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64), **kwargs)
    return _create_dpn('dpn68b', pretrained=pretrained, **model_kwargs)


def dpn92(pretrained=False, **kwargs):
    model_kwargs = dict(
        num_init_features=64, k_r=96, groups=32,
        k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128), **kwargs)
    return _create_dpn('dpn92', pretrained=pretrained, **model_kwargs)


def dpn98(pretrained=False, **kwargs):
    model_kwargs = dict(
        num_init_features=96, k_r=160, groups=40,
        k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128), **kwargs)
    return _create_dpn('dpn98', pretrained=pretrained, **model_kwargs)


def dpn131(pretrained=False, **kwargs):
    model_kwargs = dict(
        num_init_features=128, k_r=160, groups=40,
        k_sec=(4, 8, 28, 3), inc_sec=(16, 32, 32, 128), **kwargs)
    return _create_dpn('dpn131', pretrained=pretrained, **model_kwargs)


def dpn107(pretrained=False, **kwargs):
    model_kwargs = dict(
        num_init_features=128, k_r=200, groups=50,
        k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128), **kwargs)
    return _create_dpn('dpn107', pretrained=pretrained, **model_kwargs)

if __name__ == '__main__':
    inputs = torch.rand((1, 3, 224, 224))
    model = dpn68(pretrained=False)
    model.eval()
    out = model(inputs)
    print('out shape:{}'.format(out.size()))
    feas, fea_fc, out = model(inputs, True)
    for idx, fea in enumerate(feas):
        print('feature {} shape:{}'.format(idx + 1, fea.size()))
    print('fc shape:{}'.format(fea_fc.size()))
    print('out shape:{}'.format(out.size()))