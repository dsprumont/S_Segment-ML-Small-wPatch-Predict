import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResNet
from .utils import Conv2dWithNorm, UpSample2d


class FPN(nn.Module):

    def __init__(
        self,
        bottom_up,
        input_size,
        out_channels,
        num_classes=0,
        in_features=[],
        out_features=[]
    ):
        super().__init__()
        self.bottom_up = bottom_up
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_features = in_features
        self.out_features = out_features

        _stage_outputs = bottom_up.stage_outputs()
        # print(_stage_outputs)
        in_strides = [_stage_outputs[s]['stride'] for s in in_features]
        in_channels = [_stage_outputs[s]['channels'] for s in in_features]

        lateral_convs = []
        output_convs = []
        for n, in_channels in enumerate(in_channels):
            lateral_conv = Conv2dWithNorm(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                norm=nn.BatchNorm2d(out_channels)
            )
            output_conv = Conv2dWithNorm(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(out_channels),
            )
            nn.init.kaiming_uniform_(lateral_conv.weight, a=1)
            nn.init.kaiming_uniform_(output_conv.weight, a=1)
            # if module.bias is not None:
            #     nn.init.constant_(module.bias, 0)

            stage = int(math.log2(in_strides[n]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # reverse order to match bottom_up order
        self.laterals = lateral_convs[::-1]
        self.outputs = output_convs[::-1]

        if 'logits' in self.out_features and num_classes > 0:
            self.upsample = UpSample2d(size=input_size)
            self.logits = Conv2dWithNorm(
                out_channels,
                num_classes,
                kernel_size=1,
                padding=0
            )
            nn.init.kaiming_uniform_(self.logits.weight, a=1)

        self.in_channels = in_channels
        # dicts {stage_name: properties}
        self.out_feature_strides = {
            "p{}".format(int(math.log2(s))): s for s in in_strides}
        self._out_features = list(self.out_feature_strides.keys())
        if 'logits' in self.out_features:
            self._out_features.insert(0, 'logits')
        self.out_feature_channels = {
            k: out_channels for k in self._out_features}
        # print(self._out_features)

    def forward(self, inputs):
        # forward through the bottom_up cnn
        h = self.bottom_up(inputs)
        # list out_feature maps in a top_down fashion
        features = [h[name] for name in self.in_features[::-1]]
        # top layer goes through lateral and output (no fusion)
        curr_features = self.laterals[0](features[0])
        out_features = []
        out_features.append(self.outputs[0](curr_features))
        # we loop through remaining layers: curr = prev + lateral
        for feature, lateral, output in zip(
                features[1:], self.laterals[1:], self.outputs[1:]):
            prev_feature = F.interpolate(
                curr_features, scale_factor=2, mode='nearest')
            curr_features = output(prev_feature + lateral(feature))
            out_features.append(curr_features)
        if 'logits' in self.out_features and self.num_classes > 0:
            feature = self.upsample(curr_features)
            feature_logits = self.logits(feature)
            out_features.append(feature_logits)
        # name are given bottom_up but features are stored top_down
        # so we reverse order of features (back to bottom_up)
        return dict(zip(self._out_features, out_features[::-1]))

    @staticmethod
    def build_resnet_fpn(
        name,
        input_size,
        input_channels,
        output_channels,
        num_classes=0,
        in_features=[],
        out_features=[]
    ):
        resnet = ResNet.build(
            name=name,
            input_channels=input_channels,
            num_classes=0,
            out_features=in_features
        )
        return FPN(
            resnet,
            input_size,
            output_channels,
            num_classes,
            in_features,
            out_features
        )


if __name__ == '__main__':
    inputs = torch.rand((1, 3, 224, 224))
    print(inputs.shape)

    model = FPN.build_resnet_fpn(
        name='resnet50',
        input_size=(224, 224),
        input_channels=3,
        output_channels=256,
        num_classes=1,
        in_features=['stem', 'res2', 'res3', 'res4'],
        out_features=['p5', 'p4', 'p3', 'p2', 'logits']
    )

    outputs = model(inputs)

    print(outputs['p5'].shape)
    print(outputs['p4'].shape)
    print(outputs['p3'].shape)
    print(outputs['p2'].shape)
    # print(outputs['p1'].shape)
    print(outputs['logits'].shape)
