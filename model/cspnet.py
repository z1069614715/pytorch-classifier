from dataclasses import dataclass, asdict
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from timm.models.helpers import named_apply
from timm.models.layers import ConvNormAct, ConvNormActAa, DropPath, get_attn, create_act_layer, make_divisible
from timm.models.registry import register_model
from utils.utils import load_weights_from_state_dict, fuse_conv_bn

urls_dict = {
    'cspresnet50': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/cspresnet50_ra-d3e8d487.pth',
    'cspresnext50': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/cspresnext50_ra_224-648b4713.pth',
    'cspdarknet53': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/cspdarknet53_ra_256-d05c7c21.pth',
    'darknet53': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/darknet53_256_c2ns-3aeff817.pth',
    'darknetaa53': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/darknetaa53_c2ns-5c28ec8a.pth',
    'cs3darknet_m': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/cs3darknet_m_c2ns-43f06604.pth',
    'cs3darknet_l': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/cs3darknet_l_c2ns-16220c5d.pth',
    'cs3darknet_x': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/cs3darknet_x_c2ns-4e4490aa.pth',
    'cs3darknet_focus_m': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/cs3darknet_focus_m_c2ns-e23bed41.pth',
    'cs3darknet_focus_l': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/cs3darknet_focus_l_c2ns-65ef8888.pth',
    'cs3sedarknet_l': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/cs3sedarknet_l_c2ns-e8d1dc13.pth',
    'cs3sedarknet_x': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/cs3sedarknet_x_c2ns-b4d0abc0.pth',
    'cs3edgenet_x': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/cs3edgenet_x_c2-2e1610a9.pth',
    'cs3se_edgenet_x': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/cs3se_edgenet_x_c2ns-76f8e3ac.pth'
}

__all__ = list(urls_dict.keys())

@dataclass
class CspStemCfg:
    out_chs: Union[int, Tuple[int, ...]] = 32
    stride: Union[int, Tuple[int, ...]] = 2
    kernel_size: int = 3
    padding: Union[int, str] = ''
    pool: Optional[str] = ''


def _pad_arg(x, n):
    # pads an argument tuple to specified n by padding with last value
    if not isinstance(x, (tuple, list)):
        x = (x,)
    curr_n = len(x)
    pad_n = n - curr_n
    if pad_n <= 0:
        return x[:n]
    return tuple(x + (x[-1],) * pad_n)


@dataclass
class CspStagesCfg:
    depth: Tuple[int, ...] = (3, 3, 5, 2)  # block depth (number of block repeats in stages)
    out_chs: Tuple[int, ...] = (128, 256, 512, 1024)  # number of output channels for blocks in stage
    stride: Union[int, Tuple[int, ...]] = 2  # stride of stage
    groups: Union[int, Tuple[int, ...]] = 1  # num kxk conv groups
    block_ratio: Union[float, Tuple[float, ...]] = 1.0
    bottle_ratio: Union[float, Tuple[float, ...]] = 1.  # bottleneck-ratio of blocks in stage
    avg_down: Union[bool, Tuple[bool, ...]] = False
    attn_layer: Optional[Union[str, Tuple[str, ...]]] = None
    attn_kwargs: Optional[Union[Dict, Tuple[Dict]]] = None
    stage_type: Union[str, Tuple[str]] = 'csp'  # stage type ('csp', 'cs2', 'dark')
    block_type: Union[str, Tuple[str]] = 'bottle'  # blocks type for stages ('bottle', 'dark')

    # cross-stage only
    expand_ratio: Union[float, Tuple[float, ...]] = 1.0
    cross_linear: Union[bool, Tuple[bool, ...]] = False
    down_growth: Union[bool, Tuple[bool, ...]] = False

    def __post_init__(self):
        n = len(self.depth)
        assert len(self.out_chs) == n
        self.stride = _pad_arg(self.stride, n)
        self.groups = _pad_arg(self.groups, n)
        self.block_ratio = _pad_arg(self.block_ratio, n)
        self.bottle_ratio = _pad_arg(self.bottle_ratio, n)
        self.avg_down = _pad_arg(self.avg_down, n)
        self.attn_layer = _pad_arg(self.attn_layer, n)
        self.attn_kwargs = _pad_arg(self.attn_kwargs, n)
        self.stage_type = _pad_arg(self.stage_type, n)
        self.block_type = _pad_arg(self.block_type, n)

        self.expand_ratio = _pad_arg(self.expand_ratio, n)
        self.cross_linear = _pad_arg(self.cross_linear, n)
        self.down_growth = _pad_arg(self.down_growth, n)


@dataclass
class CspModelCfg:
    stem: CspStemCfg
    stages: CspStagesCfg
    zero_init_last: bool = True  # zero init last weight (usually bn) in residual path
    act_layer: str = 'leaky_relu'
    norm_layer: str = 'batchnorm'
    aa_layer: Optional[str] = None  # FIXME support string factory for this


def _cs3_cfg(
        width_multiplier=1.0,
        depth_multiplier=1.0,
        avg_down=False,
        act_layer='silu',
        focus=False,
        attn_layer=None,
        attn_kwargs=None,
        bottle_ratio=1.0,
        block_type='dark',
):
    if focus:
        stem_cfg = CspStemCfg(
            out_chs=make_divisible(64 * width_multiplier),
            kernel_size=6, stride=2, padding=2, pool='')
    else:
        stem_cfg = CspStemCfg(
            out_chs=tuple([make_divisible(c * width_multiplier) for c in (32, 64)]),
            kernel_size=3, stride=2, pool='')
    return CspModelCfg(
        stem=stem_cfg,
        stages=CspStagesCfg(
            out_chs=tuple([make_divisible(c * width_multiplier) for c in (128, 256, 512, 1024)]),
            depth=tuple([int(d * depth_multiplier) for d in (3, 6, 9, 3)]),
            stride=2,
            bottle_ratio=bottle_ratio,
            block_ratio=0.5,
            avg_down=avg_down,
            attn_layer=attn_layer,
            attn_kwargs=attn_kwargs,
            stage_type='cs3',
            block_type=block_type,
        ),
        act_layer=act_layer,
    )


model_cfgs = dict(
    cspresnet50=CspModelCfg(
        stem=CspStemCfg(out_chs=64, kernel_size=7, stride=4, pool='max'),
        stages=CspStagesCfg(
            depth=(3, 3, 5, 2),
            out_chs=(128, 256, 512, 1024),
            stride=(1, 2),
            expand_ratio=2.,
            bottle_ratio=0.5,
            cross_linear=True,
        ),
    ),
    cspresnet50d=CspModelCfg(
        stem=CspStemCfg(out_chs=(32, 32, 64), kernel_size=3, stride=4, pool='max'),
        stages=CspStagesCfg(
            depth=(3, 3, 5, 2),
            out_chs=(128, 256, 512, 1024),
            stride=(1,) + (2,),
            expand_ratio=2.,
            bottle_ratio=0.5,
            block_ratio=1.,
            cross_linear=True,
        ),
    ),
    cspresnet50w=CspModelCfg(
        stem=CspStemCfg(out_chs=(32, 32, 64), kernel_size=3, stride=4, pool='max'),
        stages=CspStagesCfg(
            depth=(3, 3, 5, 2),
            out_chs=(256, 512, 1024, 2048),
            stride=(1,) + (2,),
            expand_ratio=1.,
            bottle_ratio=0.25,
            block_ratio=0.5,
            cross_linear=True,
        ),
    ),
    cspresnext50=CspModelCfg(
        stem=CspStemCfg(out_chs=64, kernel_size=7, stride=4, pool='max'),
        stages=CspStagesCfg(
            depth=(3, 3, 5, 2),
            out_chs=(256, 512, 1024, 2048),
            stride=(1,) + (2,),
            groups=32,
            expand_ratio=1.,
            bottle_ratio=1.,
            block_ratio=0.5,
            cross_linear=True,
        ),
    ),
    cspdarknet53=CspModelCfg(
        stem=CspStemCfg(out_chs=32, kernel_size=3, stride=1, pool=''),
        stages=CspStagesCfg(
            depth=(1, 2, 8, 8, 4),
            out_chs=(64, 128, 256, 512, 1024),
            stride=2,
            expand_ratio=(2.,) + (1.,),
            bottle_ratio=(0.5,) + (1.,),
            block_ratio=(1.,) + (0.5,),
            down_growth=True,
            block_type='dark',
        ),
    ),
    darknet17=CspModelCfg(
        stem=CspStemCfg(out_chs=32, kernel_size=3, stride=1, pool=''),
        stages=CspStagesCfg(
            depth=(1,) * 5,
            out_chs=(64, 128, 256, 512, 1024),
            stride=(2,),
            bottle_ratio=(0.5,),
            block_ratio=(1.,),
            stage_type='dark',
            block_type='dark',
        ),
    ),
    darknet21=CspModelCfg(
        stem=CspStemCfg(out_chs=32, kernel_size=3, stride=1, pool=''),
        stages=CspStagesCfg(
            depth=(1, 1, 1, 2, 2),
            out_chs=(64, 128, 256, 512, 1024),
            stride=(2,),
            bottle_ratio=(0.5,),
            block_ratio=(1.,),
            stage_type='dark',
            block_type='dark',

        ),
    ),
    sedarknet21=CspModelCfg(
        stem=CspStemCfg(out_chs=32, kernel_size=3, stride=1, pool=''),
        stages=CspStagesCfg(
            depth=(1, 1, 1, 2, 2),
            out_chs=(64, 128, 256, 512, 1024),
            stride=2,
            bottle_ratio=0.5,
            block_ratio=1.,
            attn_layer='se',
            stage_type='dark',
            block_type='dark',

        ),
    ),
    darknet53=CspModelCfg(
        stem=CspStemCfg(out_chs=32, kernel_size=3, stride=1, pool=''),
        stages=CspStagesCfg(
            depth=(1, 2, 8, 8, 4),
            out_chs=(64, 128, 256, 512, 1024),
            stride=2,
            bottle_ratio=0.5,
            block_ratio=1.,
            stage_type='dark',
            block_type='dark',
        ),
    ),
    darknetaa53=CspModelCfg(
        stem=CspStemCfg(out_chs=32, kernel_size=3, stride=1, pool=''),
        stages=CspStagesCfg(
            depth=(1, 2, 8, 8, 4),
            out_chs=(64, 128, 256, 512, 1024),
            stride=2,
            bottle_ratio=0.5,
            block_ratio=1.,
            avg_down=True,
            stage_type='dark',
            block_type='dark',
        ),
    ),

    cs3darknet_s=_cs3_cfg(width_multiplier=0.5, depth_multiplier=0.5),
    cs3darknet_m=_cs3_cfg(width_multiplier=0.75, depth_multiplier=0.67),
    cs3darknet_l=_cs3_cfg(),
    cs3darknet_x=_cs3_cfg(width_multiplier=1.25, depth_multiplier=1.33),

    cs3darknet_focus_s=_cs3_cfg(width_multiplier=0.5, depth_multiplier=0.5, focus=True),
    cs3darknet_focus_m=_cs3_cfg(width_multiplier=0.75, depth_multiplier=0.67, focus=True),
    cs3darknet_focus_l=_cs3_cfg(focus=True),
    cs3darknet_focus_x=_cs3_cfg(width_multiplier=1.25, depth_multiplier=1.33, focus=True),

    cs3sedarknet_l=_cs3_cfg(attn_layer='se', attn_kwargs=dict(rd_ratio=.25)),
    cs3sedarknet_x=_cs3_cfg(attn_layer='se', width_multiplier=1.25, depth_multiplier=1.33),

    cs3sedarknet_xdw=CspModelCfg(
        stem=CspStemCfg(out_chs=(32, 64), kernel_size=3, stride=2, pool=''),
        stages=CspStagesCfg(
            depth=(3, 6, 12, 4),
            out_chs=(256, 512, 1024, 2048),
            stride=2,
            groups=(1, 1, 256, 512),
            bottle_ratio=0.5,
            block_ratio=0.5,
            attn_layer='se',
        ),
        act_layer='silu',
    ),

    cs3edgenet_x=_cs3_cfg(width_multiplier=1.25, depth_multiplier=1.33, bottle_ratio=1.5, block_type='edge'),
    cs3se_edgenet_x=_cs3_cfg(
        width_multiplier=1.25, depth_multiplier=1.33, bottle_ratio=1.5, block_type='edge',
        attn_layer='se', attn_kwargs=dict(rd_ratio=.25)),
)


class BottleneckBlock(nn.Module):
    """ ResNe(X)t Bottleneck Block
    """

    def __init__(
            self,
            in_chs,
            out_chs,
            dilation=1,
            bottle_ratio=0.25,
            groups=1,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            attn_last=False,
            attn_layer=None,
            drop_block=None,
            drop_path=0.
    ):
        super(BottleneckBlock, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        attn_last = attn_layer is not None and attn_last
        attn_first = attn_layer is not None and not attn_last

        self.conv1 = ConvNormAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.conv2 = ConvNormAct(
            mid_chs, mid_chs, kernel_size=3, dilation=dilation, groups=groups,
            drop_layer=drop_block, **ckwargs)
        self.attn2 = attn_layer(mid_chs, act_layer=act_layer) if attn_first else nn.Identity()
        self.conv3 = ConvNormAct(mid_chs, out_chs, kernel_size=1, apply_act=False, **ckwargs)
        self.attn3 = attn_layer(out_chs, act_layer=act_layer) if attn_last else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.act3 = create_act_layer(act_layer)

    def zero_init_last(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attn2(x)
        x = self.conv3(x)
        x = self.attn3(x)
        x = self.drop_path(x) + shortcut
        # FIXME partial shortcut needed if first block handled as per original, not used for my current impl
        #x[:, :shortcut.size(1)] += shortcut
        x = self.act3(x)
        return x

    def switch_to_deploy(self):
        self.conv1 = nn.Sequential(
            fuse_conv_bn(self.conv1.conv, self.conv1.bn),
            self.conv1.bn.act,
        )
        self.conv2 = nn.Sequential(
            fuse_conv_bn(self.conv2.conv, self.conv2.bn),
            self.conv2.bn.act,
        )
        self.conv3 = nn.Sequential(
            fuse_conv_bn(self.conv3.conv, self.conv3.bn),
        )

class DarkBlock(nn.Module):
    """ DarkNet Block
    """

    def __init__(
            self,
            in_chs,
            out_chs,
            dilation=1,
            bottle_ratio=0.5,
            groups=1,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            attn_layer=None,
            drop_block=None,
            drop_path=0.
    ):
        super(DarkBlock, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer)

        self.conv1 = ConvNormAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.attn = attn_layer(mid_chs, act_layer=act_layer) if attn_layer is not None else nn.Identity()
        self.conv2 = ConvNormAct(
            mid_chs, out_chs, kernel_size=3, dilation=dilation, groups=groups,
            drop_layer=drop_block, **ckwargs)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def zero_init_last(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.attn(x)
        x = self.conv2(x)
        x = self.drop_path(x) + shortcut
        return x

    def switch_to_deploy(self):
        self.conv1 = nn.Sequential(
            fuse_conv_bn(self.conv1.conv, self.conv1.bn),
            self.conv1.bn.act,
        )
        self.conv2 = nn.Sequential(
            fuse_conv_bn(self.conv2.conv, self.conv2.bn),
            self.conv2.bn.act,
        )

class EdgeBlock(nn.Module):
    """ EdgeResidual / Fused-MBConv / MobileNetV1-like 3x3 + 1x1 block (w/ activated output)
    """

    def __init__(
            self,
            in_chs,
            out_chs,
            dilation=1,
            bottle_ratio=0.5,
            groups=1,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            attn_layer=None,
            drop_block=None,
            drop_path=0.
    ):
        super(EdgeBlock, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer)

        self.conv1 = ConvNormAct(
            in_chs, mid_chs, kernel_size=3, dilation=dilation, groups=groups,
            drop_layer=drop_block, **ckwargs)
        self.attn = attn_layer(mid_chs, act_layer=act_layer) if attn_layer is not None else nn.Identity()
        self.conv2 = ConvNormAct(mid_chs, out_chs, kernel_size=1, **ckwargs)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def zero_init_last(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.attn(x)
        x = self.conv2(x)
        x = self.drop_path(x) + shortcut
        return x
    
    def switch_to_deploy(self):
        self.conv1 = nn.Sequential(
            fuse_conv_bn(self.conv1.conv, self.conv1.bn),
            self.conv1.bn.act,
        )
        self.conv2 = nn.Sequential(
            fuse_conv_bn(self.conv2.conv, self.conv2.bn),
            self.conv2.bn.act,
        )


class CrossStage(nn.Module):
    """Cross Stage."""
    def __init__(
            self,
            in_chs,
            out_chs,
            stride,
            dilation,
            depth,
            block_ratio=1.,
            bottle_ratio=1.,
            expand_ratio=1.,
            groups=1,
            first_dilation=None,
            avg_down=False,
            down_growth=False,
            cross_linear=False,
            block_dpr=None,
            block_fn=BottleneckBlock,
            **block_kwargs
    ):
        super(CrossStage, self).__init__()
        first_dilation = first_dilation or dilation
        down_chs = out_chs if down_growth else in_chs  # grow downsample channels to output channels
        self.expand_chs = exp_chs = int(round(out_chs * expand_ratio))
        block_out_chs = int(round(out_chs * block_ratio))
        conv_kwargs = dict(act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'))
        aa_layer = block_kwargs.pop('aa_layer', None)

        if stride != 1 or first_dilation != dilation:
            if avg_down:
                self.conv_down = nn.Sequential(
                    nn.AvgPool2d(2) if stride == 2 else nn.Identity(),  # FIXME dilation handling
                    ConvNormActAa(in_chs, out_chs, kernel_size=1, stride=1, groups=groups, **conv_kwargs)
                )
            else:
                self.conv_down = ConvNormActAa(
                    in_chs, down_chs, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups,
                    aa_layer=aa_layer, **conv_kwargs)
            prev_chs = down_chs
        else:
            self.conv_down = nn.Identity()
            prev_chs = in_chs

        # FIXME this 1x1 expansion is pushed down into the cross and block paths in the darknet cfgs. Also,
        # there is also special case for the first stage for some of the model that results in uneven split
        # across the two paths. I did it this way for simplicity for now.
        self.conv_exp = ConvNormAct(prev_chs, exp_chs, kernel_size=1, apply_act=not cross_linear, **conv_kwargs)
        prev_chs = exp_chs // 2  # output of conv_exp is always split in two

        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_module(str(i), block_fn(
                in_chs=prev_chs,
                out_chs=block_out_chs,
                dilation=dilation,
                bottle_ratio=bottle_ratio,
                groups=groups,
                drop_path=block_dpr[i] if block_dpr is not None else 0.,
                **block_kwargs
            ))
            prev_chs = block_out_chs

        # transition convs
        self.conv_transition_b = ConvNormAct(prev_chs, exp_chs // 2, kernel_size=1, **conv_kwargs)
        self.conv_transition = ConvNormAct(exp_chs, out_chs, kernel_size=1, **conv_kwargs)

    def forward(self, x):
        x = self.conv_down(x)
        x = self.conv_exp(x)
        xs, xb = x.split(self.expand_chs // 2, dim=1)
        xb = self.blocks(xb)
        xb = self.conv_transition_b(xb).contiguous()
        out = self.conv_transition(torch.cat([xs, xb], dim=1))
        return out

    def switch_to_deploy(self):
        if type(self.conv_down) is nn.Sequential:
            self.conv_down = nn.Sequential(
                self.conv_down[0],
                fuse_conv_bn(self.conv_down[1].conv, self.conv_down[1].bn),
                self.conv_down[1].bn.act,
                self.conv_down[1].aa
            )
        elif type(self.conv_down) is nn.Identity:
            pass
        else:
            self.conv_down = nn.Sequential(
                fuse_conv_bn(self.conv_down.conv, self.conv_down.bn),
                self.conv_down.bn.act,
                self.conv_down.aa
            )
        self.conv_exp = nn.Sequential(
            fuse_conv_bn(self.conv_exp.conv, self.conv_exp.bn),
            self.conv_exp.bn.act,
        )
        self.conv_transition_b = nn.Sequential(
            fuse_conv_bn(self.conv_transition_b.conv, self.conv_transition_b.bn),
            self.conv_transition_b.bn.act,
        )
        self.conv_transition = nn.Sequential(
            fuse_conv_bn(self.conv_transition.conv, self.conv_transition.bn),
            self.conv_transition.bn.act,
        )

class CrossStage3(nn.Module):
    """Cross Stage 3.
    Similar to CrossStage, but with only one transition conv for the output.
    """
    def __init__(
            self,
            in_chs,
            out_chs,
            stride,
            dilation,
            depth,
            block_ratio=1.,
            bottle_ratio=1.,
            expand_ratio=1.,
            groups=1,
            first_dilation=None,
            avg_down=False,
            down_growth=False,
            cross_linear=False,
            block_dpr=None,
            block_fn=BottleneckBlock,
            **block_kwargs
    ):
        super(CrossStage3, self).__init__()
        first_dilation = first_dilation or dilation
        down_chs = out_chs if down_growth else in_chs  # grow downsample channels to output channels
        self.expand_chs = exp_chs = int(round(out_chs * expand_ratio))
        block_out_chs = int(round(out_chs * block_ratio))
        conv_kwargs = dict(act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'))
        aa_layer = block_kwargs.pop('aa_layer', None)

        if stride != 1 or first_dilation != dilation:
            if avg_down:
                self.conv_down = nn.Sequential(
                    nn.AvgPool2d(2) if stride == 2 else nn.Identity(),  # FIXME dilation handling
                    ConvNormActAa(in_chs, out_chs, kernel_size=1, stride=1, groups=groups, **conv_kwargs)
                )
            else:
                self.conv_down = ConvNormActAa(
                    in_chs, down_chs, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups,
                    aa_layer=aa_layer, **conv_kwargs)
            prev_chs = down_chs
        else:
            self.conv_down = None
            prev_chs = in_chs

        # expansion conv
        self.conv_exp = ConvNormAct(prev_chs, exp_chs, kernel_size=1, apply_act=not cross_linear, **conv_kwargs)
        prev_chs = exp_chs // 2  # expanded output is split in 2 for blocks and cross stage

        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_module(str(i), block_fn(
                in_chs=prev_chs,
                out_chs=block_out_chs,
                dilation=dilation,
                bottle_ratio=bottle_ratio,
                groups=groups,
                drop_path=block_dpr[i] if block_dpr is not None else 0.,
                **block_kwargs
            ))
            prev_chs = block_out_chs

        # transition convs
        self.conv_transition = ConvNormAct(exp_chs, out_chs, kernel_size=1, **conv_kwargs)

    def forward(self, x):
        x = self.conv_down(x)
        x = self.conv_exp(x)
        x1, x2 = x.split(self.expand_chs // 2, dim=1)
        x1 = self.blocks(x1)
        out = self.conv_transition(torch.cat([x1, x2], dim=1))
        return out

    def switch_to_deploy(self):
        if self.conv_down is not None:
            if type(self.conv_down) is nn.Sequential:
                self.conv_down = nn.Sequential(
                    self.conv_down[0],
                    fuse_conv_bn(self.conv_down[1].conv, self.conv_down[1].bn),
                    self.conv_down[1].bn.act,
                    self.conv_down[1].aa
                )
            else:
                self.conv_down = nn.Sequential(
                    fuse_conv_bn(self.conv_down.conv, self.conv_down.bn),
                    self.conv_down.bn.act,
                    self.conv_down.aa
                )
        self.conv_exp = nn.Sequential(
            fuse_conv_bn(self.conv_exp.conv, self.conv_exp.bn),
            self.conv_exp.bn.act,
        )
        self.conv_transition = nn.Sequential(
            fuse_conv_bn(self.conv_transition.conv, self.conv_transition.bn),
            self.conv_transition.bn.act,
        )

class DarkStage(nn.Module):
    """DarkNet stage."""

    def __init__(
            self,
            in_chs,
            out_chs,
            stride,
            dilation,
            depth,
            block_ratio=1.,
            bottle_ratio=1.,
            groups=1,
            first_dilation=None,
            avg_down=False,
            block_fn=BottleneckBlock,
            block_dpr=None,
            **block_kwargs
    ):
        super(DarkStage, self).__init__()
        first_dilation = first_dilation or dilation
        conv_kwargs = dict(act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'))
        aa_layer = block_kwargs.pop('aa_layer', None)

        if avg_down:
            self.conv_down = nn.Sequential(
                nn.AvgPool2d(2) if stride == 2 else nn.Identity(),   # FIXME dilation handling
                ConvNormActAa(in_chs, out_chs, kernel_size=1, stride=1, groups=groups, **conv_kwargs)
            )
        else:
            self.conv_down = ConvNormActAa(
                in_chs, out_chs, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups,
                aa_layer=aa_layer, **conv_kwargs)

        prev_chs = out_chs
        block_out_chs = int(round(out_chs * block_ratio))
        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_module(str(i), block_fn(
                in_chs=prev_chs,
                out_chs=block_out_chs,
                dilation=dilation,
                bottle_ratio=bottle_ratio,
                groups=groups,
                drop_path=block_dpr[i] if block_dpr is not None else 0.,
                **block_kwargs
            ))
            prev_chs = block_out_chs

    def forward(self, x):
        x = self.conv_down(x)
        x = self.blocks(x)
        return x

    def switch_to_deploy(self):
        if type(self.conv_down) is nn.Sequential:
            self.conv_down = nn.Sequential(
                self.conv_down[0],
                fuse_conv_bn(self.conv_down[1].conv, self.conv_down[1].bn),
                self.conv_down[1].bn.act,
                self.conv_down[1].aa
            )
        else:
            self.conv_down = nn.Sequential(
                fuse_conv_bn(self.conv_down.conv, self.conv_down.bn),
                self.conv_down.bn.act,
                self.conv_down.aa
            )

def create_csp_stem(
        in_chans=3,
        out_chs=32,
        kernel_size=3,
        stride=2,
        pool='',
        padding='',
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        aa_layer=None
):
    stem = nn.Sequential()
    feature_info = []
    if not isinstance(out_chs, (tuple, list)):
        out_chs = [out_chs]
    stem_depth = len(out_chs)
    assert stem_depth
    assert stride in (1, 2, 4)
    prev_feat = None
    prev_chs = in_chans
    last_idx = stem_depth - 1
    stem_stride = 1
    for i, chs in enumerate(out_chs):
        conv_name = f'conv{i + 1}'
        conv_stride = 2 if (i == 0 and stride > 1) or (i == last_idx and stride > 2 and not pool) else 1
        if conv_stride > 1 and prev_feat is not None:
            feature_info.append(prev_feat)
        stem.add_module(conv_name, ConvNormAct(
            prev_chs, chs, kernel_size,
            stride=conv_stride,
            padding=padding if i == 0 else '',
            act_layer=act_layer,
            norm_layer=norm_layer
        ))
        stem_stride *= conv_stride
        prev_chs = chs
        prev_feat = dict(num_chs=prev_chs, reduction=stem_stride, module='.'.join(['stem', conv_name]))
    if pool:
        assert stride > 2
        if prev_feat is not None:
            feature_info.append(prev_feat)
        if aa_layer is not None:
            stem.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            stem.add_module('aa', aa_layer(channels=prev_chs, stride=2))
            pool_name = 'aa'
        else:
            stem.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            pool_name = 'pool'
        stem_stride *= 2
        prev_feat = dict(num_chs=prev_chs, reduction=stem_stride, module='.'.join(['stem', pool_name]))
    feature_info.append(prev_feat)
    return stem, feature_info


def _get_stage_fn(stage_args):
    stage_type = stage_args.pop('stage_type')
    assert stage_type in ('dark', 'csp', 'cs3')
    if stage_type == 'dark':
        stage_args.pop('expand_ratio', None)
        stage_args.pop('cross_linear', None)
        stage_args.pop('down_growth', None)
        stage_fn = DarkStage
    elif stage_type == 'csp':
        stage_fn = CrossStage
    else:
        stage_fn = CrossStage3
    return stage_fn, stage_args


def _get_block_fn(stage_args):
    block_type = stage_args.pop('block_type')
    assert block_type in ('dark', 'edge', 'bottle')
    if block_type == 'dark':
        return DarkBlock, stage_args
    elif block_type == 'edge':
        return EdgeBlock, stage_args
    else:
        return BottleneckBlock, stage_args


def _get_attn_fn(stage_args):
    attn_layer = stage_args.pop('attn_layer')
    attn_kwargs = stage_args.pop('attn_kwargs', None) or {}
    if attn_layer is not None:
        attn_layer = get_attn(attn_layer)
        if attn_kwargs:
            attn_layer = partial(attn_layer, **attn_kwargs)
    return attn_layer, stage_args


def create_csp_stages(
        cfg: CspModelCfg,
        drop_path_rate: float,
        output_stride: int,
        stem_feat: Dict[str, Any]
):
    cfg_dict = asdict(cfg.stages)
    num_stages = len(cfg.stages.depth)
    cfg_dict['block_dpr'] = [None] * num_stages if not drop_path_rate else \
        [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.stages.depth)).split(cfg.stages.depth)]
    stage_args = [dict(zip(cfg_dict.keys(), values)) for values in zip(*cfg_dict.values())]
    block_kwargs = dict(
        act_layer=cfg.act_layer,
        norm_layer=cfg.norm_layer,
    )

    dilation = 1
    net_stride = stem_feat['reduction']
    prev_chs = stem_feat['num_chs']
    prev_feat = stem_feat
    feature_info = []
    stages = []
    for stage_idx, stage_args in enumerate(stage_args):
        stage_fn, stage_args = _get_stage_fn(stage_args)
        block_fn, stage_args = _get_block_fn(stage_args)
        attn_fn, stage_args = _get_attn_fn(stage_args)
        stride = stage_args.pop('stride')
        if stride != 1 and prev_feat:
            feature_info.append(prev_feat)
        if net_stride >= output_stride and stride > 1:
            dilation *= stride
            stride = 1
        net_stride *= stride
        first_dilation = 1 if dilation in (1, 2) else 2

        stages += [stage_fn(
            prev_chs,
            **stage_args,
            stride=stride,
            first_dilation=first_dilation,
            dilation=dilation,
            block_fn=block_fn,
            aa_layer=cfg.aa_layer,
            attn_layer=attn_fn,  # will be passed through stage as block_kwargs
            **block_kwargs,
        )]
        prev_chs = stage_args['out_chs']
        prev_feat = dict(num_chs=prev_chs, reduction=net_stride, module=f'stages.{stage_idx}')

    feature_info.append(prev_feat)
    return nn.Sequential(*stages), feature_info


class CspNet(nn.Module):
    """Cross Stage Partial base model.
    Paper: `CSPNet: A New Backbone that can Enhance Learning Capability of CNN` - https://arxiv.org/abs/1911.11929
    Ref Impl: https://github.com/WongKinYiu/CrossStagePartialNetworks
    NOTE: There are differences in the way I handle the 1x1 'expansion' conv in this impl vs the
    darknet impl. I did it this way for simplicity and less special cases.
    """

    def __init__(
            self,
            cfg: CspModelCfg,
            in_chans=3,
            num_classes=1000,
            output_stride=32,
            global_pool='avg',
            drop_rate=0.,
            drop_path_rate=0.,
            zero_init_last=True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)
        layer_args = dict(
            act_layer=cfg.act_layer,
            norm_layer=cfg.norm_layer,
            aa_layer=cfg.aa_layer
        )
        self.feature_info = []

        # Construct the stem
        self.stem, stem_feat_info = create_csp_stem(in_chans, **asdict(cfg.stem), **layer_args)
        self.feature_info.extend(stem_feat_info[:-1])

        # Construct the stages
        self.stages, stage_feat_info = create_csp_stages(
            cfg,
            drop_path_rate=drop_path_rate,
            output_stride=output_stride,
            stem_feat=stem_feat_info[-1],
        )
        prev_chs = stage_feat_info[-1]['num_chs']
        self.feature_info.extend(stage_feat_info)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Construct the head
        self.num_features = prev_chs
        self.head = nn.Linear(in_features=self.num_features, out_features=num_classes)

        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    def forward_features(self, x, need_fea=False):
        x = self.stem(x)
        if need_fea:
            features = []
            for layer in self.stages:
                x = layer(x)
                features.append(x)
            x = self.avgpool(x)
            return features, torch.flatten(x, start_dim=1, end_dim=3)
        else:
            x = self.stages(x)
            x = self.avgpool(x)
            return torch.flatten(x, start_dim=1, end_dim=3)

    def forward_head(self, x):
        return self.head(x)

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
        return self.stages[-1]


def _init_weights(module, name, zero_init_last=False):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif zero_init_last and hasattr(module, 'zero_init_last'):
        module.zero_init_last()


def _create_cspnet(variant, pretrained=False, **kwargs):
    model = CspNet(model_cfgs[variant])
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(urls_dict[variant], progress=True, check_hash=True)
        model = load_weights_from_state_dict(model, state_dict)
    return model

@register_model
def cspresnet50(pretrained=False, **kwargs):
    return _create_cspnet('cspresnet50', pretrained=pretrained, **kwargs)


@register_model
def cspresnet50d(pretrained=False, **kwargs):
    return _create_cspnet('cspresnet50d', pretrained=pretrained, **kwargs)


@register_model
def cspresnet50w(pretrained=False, **kwargs):
    return _create_cspnet('cspresnet50w', pretrained=pretrained, **kwargs)


@register_model
def cspresnext50(pretrained=False, **kwargs):
    return _create_cspnet('cspresnext50', pretrained=pretrained, **kwargs)


@register_model
def cspdarknet53(pretrained=False, **kwargs):
    return _create_cspnet('cspdarknet53', pretrained=pretrained, **kwargs)


@register_model
def darknet17(pretrained=False, **kwargs):
    return _create_cspnet('darknet17', pretrained=pretrained, **kwargs)


@register_model
def darknet21(pretrained=False, **kwargs):
    return _create_cspnet('darknet21', pretrained=pretrained, **kwargs)


@register_model
def sedarknet21(pretrained=False, **kwargs):
    return _create_cspnet('sedarknet21', pretrained=pretrained, **kwargs)


@register_model
def darknet53(pretrained=False, **kwargs):
    return _create_cspnet('darknet53', pretrained=pretrained, **kwargs)


@register_model
def darknetaa53(pretrained=False, **kwargs):
    return _create_cspnet('darknetaa53', pretrained=pretrained, **kwargs)


@register_model
def cs3darknet_s(pretrained=False, **kwargs):
    return _create_cspnet('cs3darknet_s', pretrained=pretrained, **kwargs)


@register_model
def cs3darknet_m(pretrained=False, **kwargs):
    return _create_cspnet('cs3darknet_m', pretrained=pretrained, **kwargs)


@register_model
def cs3darknet_l(pretrained=False, **kwargs):
    return _create_cspnet('cs3darknet_l', pretrained=pretrained, **kwargs)


@register_model
def cs3darknet_x(pretrained=False, **kwargs):
    return _create_cspnet('cs3darknet_x', pretrained=pretrained, **kwargs)


@register_model
def cs3darknet_focus_s(pretrained=False, **kwargs):
    return _create_cspnet('cs3darknet_focus_s', pretrained=pretrained, **kwargs)


@register_model
def cs3darknet_focus_m(pretrained=False, **kwargs):
    return _create_cspnet('cs3darknet_focus_m', pretrained=pretrained, **kwargs)


@register_model
def cs3darknet_focus_l(pretrained=False, **kwargs):
    return _create_cspnet('cs3darknet_focus_l', pretrained=pretrained, **kwargs)


@register_model
def cs3darknet_focus_x(pretrained=False, **kwargs):
    return _create_cspnet('cs3darknet_focus_x', pretrained=pretrained, **kwargs)


@register_model
def cs3sedarknet_l(pretrained=False, **kwargs):
    return _create_cspnet('cs3sedarknet_l', pretrained=pretrained, **kwargs)


@register_model
def cs3sedarknet_x(pretrained=False, **kwargs):
    return _create_cspnet('cs3sedarknet_x', pretrained=pretrained, **kwargs)


@register_model
def cs3sedarknet_xdw(pretrained=False, **kwargs):
    return _create_cspnet('cs3sedarknet_xdw', pretrained=pretrained, **kwargs)


@register_model
def cs3edgenet_x(pretrained=False, **kwargs):
    return _create_cspnet('cs3edgenet_x', pretrained=pretrained, **kwargs)


@register_model
def cs3se_edgenet_x(pretrained=False, **kwargs):
    return _create_cspnet('cs3se_edgenet_x', pretrained=pretrained, **kwargs)

if __name__ == '__main__':
    inputs = torch.rand((2, 3, 224, 224))
    model = cspresnet50(pretrained=False)
    model.head = nn.Linear(model.head.in_features, 1)
    model.eval()
    out = model(inputs)
    print('out shape:{}'.format(out.size()))
    feas, fea_fc, out = model(inputs, True)
    for idx, fea in enumerate(feas):
        print('feature {} shape:{}'.format(idx + 1, fea.size()))
    print('fc shape:{}'.format(fea_fc.size()))
    print('out shape:{}'.format(out.size()))