import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from .utils import load_state_dict_from_url
from torch import Tensor
from typing import Any, List, Tuple

from .utils import load_state_dict_from_url



__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201']

# Rretrained model
model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
}


class _BottleNeckLayer(nn.Module):
    """
        Architecture: BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3) of H_l layer
        Let each 1x1 convolution produce 4*growth_rate feature-maps.
    """
    def __init__(
        self,
        num_intput_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_BottleNeckLayer, self).__init__()
        self.norm1: nn.BatchNorm2d
        self.add_module('norm1', nn.BatchNorm2d(num_intput_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace = True))
        self.conv1: nn.Conv2d
        self.add_module('conv1', nn.Conv2d(num_intput_features,
                                           growth_rate * bn_size,
                                           kernel_size = 1,
                                           stride = 1,
                                           bias = False))

        self.norm2: nn.BatchNorm2d
        self.add_module('norm2', nn.BatchNorm2d(growth_rate * bn_size))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace = True))
        self.conv2: nn.Conv2d
        self.add_module('conv2', nn.Conv2d(growth_rate * bn_size,
                                           growth_rate,
                                           kernel_size = 1,
                                           stride = 1,
                                           bias = False))

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input: Tensor) -> Tensor:
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p = self.drop_rate,
                                     training = self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    """
        k is imply growth rate
        The l_th layer in the denseblock have k0 + k * (l-1) feature maps
    """
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _BottleNeckLayer(
                num_input_features + i * growth_rate,
                growth_rate = growth_rate,
                bn_size = bn_size,
                drop_rate = drop_rate,
                memory_efficient = memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    """
        This part is a downsampling layer that change the size of feature_maps.
        Implement the transition layers, which do convolution and pooling.
        Consists of a bn_layer and 1x1_conv_layer followed by 2x2_avg_pool.
        The transition layers divide the network into multiple DenseBlock.
    """
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace = True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size = 1, stride = 1, bias = False))
        self.add_module('pool', nn.AvgPool2d(kernel_size = (2, 2), stride = (2, 2)))


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int  = 4,
        drop_rate: float = 0.0,
        num_classes: int = 1000,
        memory_efficient: bool = False
    ) -> None:
        super(DenseNet, self).__init__()

        # Pre convolution and pooling
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                                kernel_size = 7,
                                stride = 2,
                                padding = 3,
                                bias = False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace = True)),
            ('pool0', nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers = num_layers,
                num_input_features = num_features,
                bn_size = bn_size,
                growth_rate = growth_rate,
                drop_rate = drop_rate,
                memory_efficient = memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features = num_features,
                                    num_output_features = num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final BatchNorm layer
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Classification Layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialize weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def _densenet(
        arch: str,
        growth_rate: int,
        block_config: Tuple[int, int, int, int],
        num_init_features: int,
        pretrained: bool,
        progress: bool,
        **kwargs: Any
    ) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress = progress)
        model.load_state_dict(state_dict)
    return model

def densenet121(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)


def densenet169(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     **kwargs)


def densenet201(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)