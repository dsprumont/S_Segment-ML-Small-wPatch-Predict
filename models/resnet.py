# common dependencies
import sys

# deep learning dependencies
import torch
import torch.nn as nn
from .utils import Conv2dWithNorm


__all__ = ['ResNet']


class StemBlock(nn.Module):
    """
    First block in a Resnet architecture.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=7,
        stride=2,
        pooling=nn.MaxPool2d,
        norm=nn.BatchNorm2d
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # args stride is for the conv layer,
        # self.stride is for the whole module (stride + maxpool)
        self.stride = stride*2
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.conv1 = Conv2dWithNorm(
            in_channels,
            out_channels,
            kernel_size=kernel_size,  # Original is 7 /!\
            stride=stride,
            padding=int((kernel_size-1)/2),  # Original is 3 /!\
            norm=norm(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        if pooling is not None:
            self.maxpool = pooling(kernel_size=3, stride=2, padding=1)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        h = self.conv1(x)
        h = self.relu(h)
        if self.pooling is not None:
            h = self.maxpool(h)
        return h


class Bottleneck(nn.Module):
    """
    Bottleneck block in ResNet models.
    Composed of 3 Conv2d (1x1, 3x3, 1x1) with element-wise addition shortcut.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        stride=1,
        dilation=1,
        norm=nn.BatchNorm2d
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = 1

        self.conv1 = Conv2dWithNorm(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm=norm(bottleneck_channels)
        )

        self.conv2 = Conv2dWithNorm(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1*dilation,
            dilation=dilation,
            bias=False,
            norm=norm(bottleneck_channels)
        )

        self.conv3 = Conv2dWithNorm(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm=norm(out_channels)
        )

        if in_channels != out_channels or stride > 1:
            self.shortcut = Conv2dWithNorm(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
                norm=norm(out_channels),
            )
        else:
            self.shortcut = None

        self.relu = nn.ReLU(inplace=False)

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

    def forward(self, x):
        h = self.conv1(x)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.relu(h)
        h = self.conv3(h)
        if self.shortcut is not None:
            h += self.shortcut(x)       # shortcut
        else:
            h += x
        h = self.relu(h)
        return h


CFG_SIMPLE0 = {
    'stem_out_channels': 64,
    'stem_stride': 1,
    'stem_kernel': 3,
    'stem_pooling': None,
    'res2_out_channels': 64,
    'res2_bottleneck_channels': 32,
    'resnet_layers': [1, 1, 1, 1],
    'resnet_dilations': [1, 1, 1, 1]
}

CFG_SIMPLE1 = {
    'stem_out_channels': 16,
    'stem_stride': 1,
    'stem_kernel': 7,
    'stem_pooling': nn.MaxPool2d,
    'res2_out_channels': 64,
    'res2_bottleneck_channels': 16,
    'resnet_layers': [1, 1, 1, 1],
    'resnet_dilations': [1, 1, 1, 1]
}

CFG_SIMPLE2 = {
    'stem_out_channels': 32,
    'stem_stride': 1,
    'stem_kernel': 7,
    'stem_pooling': nn.MaxPool2d,
    'res2_out_channels': 128,
    'res2_bottleneck_channels': 32,
    'resnet_layers': [1, 1, 1, 1],
    'resnet_dilations': [1, 1, 1, 1]
}

CFG_SIMPLE3 = {
    'stem_out_channels': 64,
    'stem_stride': 1,
    'stem_kernel': 7,
    'stem_pooling': nn.MaxPool2d,
    'res2_out_channels': 256,
    'res2_bottleneck_channels': 64,  # 16 for Simple3, 64 for Simple3n
    'resnet_layers': [1, 1, 1, 1],
    'resnet_dilations': [1, 1, 1, 1]
}

CFG_SIMPLE4 = {
    'stem_out_channels': 16,
    'stem_stride': 1,
    'stem_kernel': 7,
    'stem_pooling': nn.MaxPool2d,
    'res2_out_channels': 64,
    'res2_bottleneck_channels': 16,
    'resnet_layers': [2, 2, 2, 2],
    'resnet_dilations': [1, 1, 1, 1]
}

CFG_SIMPLE5 = {
    'stem_out_channels': 32,
    'stem_stride': 1,
    'stem_kernel': 7,
    'stem_pooling': nn.MaxPool2d,
    'res2_out_channels': 128,
    'res2_bottleneck_channels': 32,
    'resnet_layers': [2, 2, 2, 2],
    'resnet_dilations': [1, 1, 1, 1]
}

CFG_SIMPLE6 = {
    'stem_out_channels': 64,
    'stem_stride': 1,
    'stem_kernel': 7,
    'stem_pooling': nn.MaxPool2d,
    'res2_out_channels': 256,
    'res2_bottleneck_channels': 64,
    'resnet_layers': [2, 2, 2, 2],
    'resnet_dilations': [1, 1, 1, 1]
}

CFG_MEDIUM1 = {
    'stem_out_channels': 32,
    'stem_stride': 1,
    'stem_kernel': 7,
    'stem_pooling': nn.MaxPool2d,
    'res2_out_channels': 128,
    'res2_bottleneck_channels': 32,
    'resnet_layers': [3, 3, 3, 3],
    'resnet_dilations': [1, 1, 1, 1]
}

CFG_MEDIUM2 = {
    'stem_out_channels': 64,
    'stem_stride': 1,
    'stem_kernel': 7,
    'stem_pooling': nn.MaxPool2d,
    'res2_out_channels': 256,
    'res2_bottleneck_channels': 64,
    'resnet_layers': [3, 3, 3, 3],
    'resnet_dilations': [1, 1, 1, 1]
}

CFG_RESNET50 = {
    'stem_out_channels': 64,
    'stem_stride': 2,
    'stem_kernel': 7,
    'stem_pooling': nn.MaxPool2d,
    'res2_out_channels': 256,
    'res2_bottleneck_channels': 64,
    'resnet_layers': [3, 4, 6, 3],
    'resnet_dilations': [1, 1, 1, 1]
}

CFG_RESNET50b = {
    'stem_out_channels': 64,
    'stem_stride': 1,
    'stem_kernel': 5,
    'stem_pooling': nn.MaxPool2d,
    'res2_out_channels': 256,
    'res2_bottleneck_channels': 64,
    'resnet_layers': [3, 4, 6, 3],
    'resnet_dilations': [1, 1, 1, 1]
}

CFG_RESNET101 = {
    'stem_out_channels': 64,
    'stem_stride': 2,
    'stem_kernel': 7,
    'stem_pooling': nn.MaxPool2d,
    'res2_out_channels': 256,
    'res2_bottleneck_channels': 64,
    'resnet_layers': [3, 4, 23, 3],
    'resnet_dilations': [1, 1, 1, 1]
}

CONFIGURATIONS = {
    'simple0': CFG_SIMPLE0,
    'simple1': CFG_SIMPLE1,
    'simple2': CFG_SIMPLE2,
    'simple3': CFG_SIMPLE3,
    'simple4': CFG_SIMPLE4,
    'simple5': CFG_SIMPLE5,
    'simple6': CFG_SIMPLE6,
    'medium1': CFG_MEDIUM1,
    'medium2': CFG_MEDIUM2,
    'resnet50': CFG_RESNET50,
    'resnet50b': CFG_RESNET50b,
    'resnet101': CFG_RESNET101
}


class ResNet(nn.Module):

    def __init__(self, stem, blocks, num_classes, out_features=['logits']):
        super().__init__()
        self.stem = stem
        self.out_features = out_features
        self.out_feature_channels = {'stem': stem.out_channels}
        self.out_feature_stride = {'stem': stem.stride}

        last_channels = 0
        curr_stride = stem.stride
        self.res = []
        for n, block in enumerate(blocks):
            layers = []
            name = 'res'+str(n+2)
            in_channels = block['in_channels']
            for idx in range(block['count']):
                stride = 1 if (idx > 0 or block['dilation'] > 1) else 2
                out_channels = block['out_channels']
                layers.append(
                    Bottleneck(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bottleneck_channels=block['bottleneck_channels'],
                        stride=stride,
                        dilation=block['dilation']
                    )
                )
                in_channels = out_channels
                last_channels = out_channels
            curr_stride = curr_stride * 2
            self.out_feature_channels[name] = last_channels
            self.out_feature_stride[name] = curr_stride

            module = nn.Sequential(*layers)
            self.res.append([name, module])
            self.add_module(name, module)

        if 'logits' in out_features:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(last_channels, num_classes)
            nn.init.normal_(self.linear.weight, std=0.01)

    def forward(self, x):
        out = {}
        h = self.stem(x)
        if 'stem' in self.out_features:
            out['stem'] = h
        for name, res in self.res:
            h = res(h)
            if name in self.out_features:
                out[name] = h
        if 'logits' in self.out_features:
            h = self.avgpool(h)
            h = torch.flatten(h, 1)  # flatten all dimensions, except batch dim
            h = self.linear(h)
            out['logits'] = h
        return out

    def stage_outputs(self):
        # print(self.out_features)
        return {name: {
            'stride': self.out_feature_stride[name],
            'channels': self.out_feature_channels[name]}
            for name in self.out_features
            }

    @staticmethod
    def build(name, input_channels, num_classes, out_features=['logits']):
        cfg = CONFIGURATIONS.get(name, None)
        if cfg is None:
            raise AttributeError(
                "The given resnet configuration is not available.")
            sys.exit()

        in_channels = input_channels
        bottleneck_channels: int = cfg['stem_out_channels']
        out_channels: int = cfg['res2_out_channels']

        stem = StemBlock(
            in_channels=in_channels,
            out_channels=cfg['stem_out_channels'],
            stride=cfg['stem_stride'],
            kernel_size=cfg['stem_kernel'],
            pooling=cfg['stem_pooling']
        )

        blocks = []
        in_channels = cfg['stem_out_channels']
        bottleneck_channels = cfg['res2_bottleneck_channels']
        out_channels = cfg['res2_out_channels']
        blocks_per_layer = cfg['resnet_layers']
        dilations = cfg['resnet_dilations']

        for idx in range(len(blocks_per_layer)):
            block = {
                'count': blocks_per_layer[idx],
                'in_channels': in_channels,
                'bottleneck_channels': bottleneck_channels,
                'out_channels': out_channels,
                'dilation': dilations[idx]
            }
            in_channels = out_channels
            bottleneck_channels *= 2
            out_channels *= 2
            blocks.append(block)

        return ResNet(stem, blocks, num_classes, out_features)
