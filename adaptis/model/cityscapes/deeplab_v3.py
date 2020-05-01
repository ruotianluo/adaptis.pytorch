import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptis.model.basic_blocks import SeparableConv2D
from .resnet import ResNetBackbone
from torchvision.models.segmentation.deeplabv3 import ASPP

class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone='resnet50', backbone_lr_mult=0.1, **kwargs):
        super(DeepLabV3Plus, self).__init__()

        self._c1_shape = None
        self.backbone_name = backbone
        self.backbone_lr_mult = backbone_lr_mult
        self._kwargs = kwargs

        self.backbone = ResNetBackbone(backbone=self.backbone_name, pretrained_base=False, **kwargs)

        self.head = _DeepLabHead(in_channels=256 + 32, out_channels=256, **kwargs)
        self.skip_project = _SkipProject(256, 32, **kwargs)
        # this ASPP is almost the same as torchvision.models.segmentation.deeplabv3.ASPP
        # except it takes norm_layer as a argument
        self.aspp = _ASPP(2048, 256, [12, 24, 36], **kwargs)

    def load_pretrained_weights(self):
        # load from gluon
        # Unfortunately that's the only way

        # import mxnet as mx
        # from gluoncv.model_zoo.model_store import get_model_file
        # state_dict = mx.ndarray.load(
        #     get_model_file(self.backbone_name+'_v1s', tag=True, root='~/.mxnet/models'))
        # state_dict = {k.replace('gamma', 'weight').replace('beta', 'bias'):\
        #               torch.from_numpy(v.asnumpy()) for k,v in state_dict.items()}
        
        # The above works for single process.
        # For some reason, mx.load doesn't work with mulitprocessing
        # We preprocess the gluon weights and save it in the root
        state_dict = torch.load('resnet50_v1s.pth')
        self.backbone.load_state_dict(state_dict, strict=False)
        for p in self.backbone.parameters():
            p.lr_mult = self.backbone_lr_mult

    def forward(self, x):
        c1, _, c3, c4 = self.backbone(x)
        c1 = self.skip_project(c1)

        if hasattr(c1, 'shape'):
            self._c1_shape = c1.shape

        x = self.aspp(c4)
        x = F.interpolate(x, (self._c1_shape[2], self._c1_shape[3]), mode='bilinear')
        x = torch.cat([x, c1], 1)
        x = self.head(x)

        return x


class _SkipProject(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(_SkipProject, self).__init__()

        self.skip_project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))

    def forward(self, x):
        return self.skip_project(x)


class _DeepLabHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()
        block = []
        block.append(SeparableConv2D(in_channels, 256, dw_kernel=3, dw_padding=1,
                                           activation='relu', norm_layer=norm_layer))
        block.append(SeparableConv2D(256, 256, dw_kernel=3, dw_padding=1,
                                           activation='relu', norm_layer=norm_layer))

        block.append(nn.Conv2d(256, out_channels,
                                     kernel_size=1))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class _ASPP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, atrous_rates=[12, 24, 36], norm_layer=nn.BatchNorm2d):
        super(_ASPP, self).__init__()
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.conv1.add_module('bn', norm_layer(out_channels))
        self.conv1.add_module('relu', nn.ReLU(inplace=True))

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.conv2 = _ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.conv3 = _ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.conv4 = _ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.pool = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer)

        self.project = nn.Sequential()
        self.project.add_module('conv', nn.Conv2d(5*out_channels, out_channels, kernel_size=1, bias=False))
        self.project.add_module('norm', norm_layer(out_channels))
        self.project.add_module('relu', nn.ReLU(inplace=True))
        self.project.add_module('dropout', nn.Dropout(0.5))

    def forward(self, x):
        x = torch.cat([
            self.conv1(x),
            self.conv2(x),
            self.conv3(x),
            self.conv4(x),
            self.pool(x),
        ], 1)
        return self.project(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, **kwargs):
        super(_AsppPooling, self).__init__()
        self._out_h = None
        self._out_w = None
        self.gap = nn.Sequential()
        
        self.gap.add_module('pool', nn.AdaptiveAvgPool2d((1, 1)))
        self.gap.add_module('fc', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.gap.add_module('bn', norm_layer(out_channels))
        self.gap.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        if hasattr(x, 'shape'):
            _, _, h, w = x.shape
            self._out_h = h
            self._out_w = w
        else:
            h, w = self._out_h, self._out_w
            assert h is not None
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode='bilinear')


def _ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential()
    block.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                        kernel_size=3, padding=atrous_rate,
                                        dilation=atrous_rate, bias=False))
    block.add_module('bn', norm_layer(out_channels))
    block.add_module('relu', nn.ReLU(inplace=True))
    return block

