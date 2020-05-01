import torch
import torch.nn as nn

from adaptis.model.cityscapes.deeplab_v3 import DeepLabV3Plus
from adaptis.model.adaptis import AdaptIS
from adaptis.model.ops import AdaIN, ExtractQueryFeatures, AppendCoordFeatures
from adaptis.model.basic_blocks import SepConvHead, FCController, SeparableConv2D
# from .resnet_fpn import SemanticFPNHead, ResNetFPN


def get_cityscapes_model(num_classes, norm_layer, backbone='resnet50',
                         with_proposals=False):
    model = AdaptIS(
        backbone=DeepLabV3Plus(backbone=backbone, norm_layer=norm_layer),
        adaptis_head=CityscapesAdaptISHead(
            FCController(256, 3 * [128], norm_layer=norm_layer),
            in_channels=256,
            out_channels=128, norm_radius=280,
            spatial_scale=1.0/4.0,
            norm_layer=norm_layer
        ),
        segmentation_head=SepConvHead(num_classes, in_channels=256, out_channels=192, num_layers=2, norm_layer=norm_layer),
        proposal_head=SepConvHead(1, in_channels=256, out_channels=128, num_layers=2,
                                  dropout_ratio=0.5, dropout_indx=0, norm_layer=norm_layer),
        with_proposals=with_proposals,
        spatial_scale=1.0/4.0
    )

    return model


# def get_fpn_model(num_classes, norm_layer, backbone='resnet50',
#                          with_proposals=False):
#     model = AdaptIS(
#         feature_extractor=ResNetFPN(backbone=backbone, norm_layer=norm_layer),
#         adaptis_head=CityscapesAdaptISHead(
#             FCController(256, 3 * [128], norm_layer=norm_layer),
#             in_channels=512, out_channels=128, norm_radius=280,
#             spatial_scale=1.0/4.0,
#             norm_layer=norm_layer
#         ),
#         segmentation_head=SemanticFPNHead(num_classes, output_channels=256, norm_layer=norm_layer),
#         proposal_head=SepConvHead(1, channels=128, in_channels=256, num_layers=2,
#                                   dropout_ratio=0.5, dropout_indx=0, norm_layer=norm_layer),
#         with_proposals=with_proposals,
#         spatial_scale=1.0/4.0
#     )
#     return model


class CityscapesAdaptISHead(nn.Module):
    def __init__(self, controller_net, in_channels, out_channels=128, norm_radius=190, spatial_scale=0.25,
                 norm_layer=nn.BatchNorm2d):
        super(CityscapesAdaptISHead, self).__init__()

        self.num_points = None

        self.eqf = ExtractQueryFeatures(extraction_method='ROIAlign', spatial_scale=spatial_scale)
        self.controller_net = controller_net

        self.add_coord_features = AppendCoordFeatures(norm_radius=norm_radius, spatial_scale=spatial_scale)
        in_channels += 2
        if self.add_coord_features.append_dist:
            in_channels += 1

        block0 = []
        block0.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            norm_layer(out_channels),
            SeparableConv2D(out_channels, out_channels, dw_kernel=3, dw_padding=1,
                            norm_layer=norm_layer, activation='relu'),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.LeakyReLU(0.2)
        ])
        self.block0 = nn.Sequential(*block0)

        self.adain = AdaIN(out_channels, out_channels)

        block1 = []
        for i in range(3):
            block1.append(
                SeparableConv2D(in_channels=min(out_channels, 2 * out_channels // (2 ** i)),
                                out_channels=out_channels // (2 ** i),
                                dw_kernel=3, dw_padding=1,
                                norm_layer=norm_layer, activation='relu'),
            )
        block1.append(nn.Conv2d(out_channels // (2 ** i), 1, kernel_size=1))
        self.block1 = nn.Sequential(*block1)

    def forward(self, p1_features, points):
        adaptive_input, controller_input = self._get_point_invariant_features(p1_features)
        return self._get_instance_maps(points, adaptive_input, controller_input)

    def _get_point_invariant_features(self, backbone_features):
        adaptive_input = backbone_features

        if getattr(self.controller_net, 'return_map', False):
            controller_input = self.controller_net(backbone_features)
        else:
            controller_input = backbone_features

        return adaptive_input, controller_input

    def _get_instance_maps(self, points, adaptive_input, controller_input):
        if torch.is_tensor(points):
            self.num_points = points.shape[1]

        if getattr(self.controller_net, 'return_map', False):
            w = self.eqf(controller_input, points)
        else:
            w = self.eqf(controller_input, points)
            w = self.controller_net(w)

        points = points.reshape(-1, 2)
        x = torch.stack([adaptive_input] * self.num_points, 1).reshape(-1, *adaptive_input.shape[1:])
        x = self.add_coord_features(x, points)

        x = self.block0(x)
        x = self.adain(x, w)
        x = self.block1(x)

        return x
