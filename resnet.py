import torch
from torch import Tensor
import torch.nn as nn
from utils import *

# from modules import acol
# from modules import spg
# from modules import adl
# from modules import se_module
# from modules import nl_module
# from modules import cbam_module
# from modules import eca_module

from models.obj_simi import obj_simi
import torchvision.models as models

import pdb

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

from modules import outersum_module


import torch.nn.functional as F
__all__ = ['ResNet', 'resnet50', 'pretrainnet50']
_ADL_POSITION = [[], [], [], [0], [0, 2]]

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        module_name=None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]


    def _make_layer(self, block: Type[Union[Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, labels=None, return_cam=False) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pre_logit = self.avgpool(x)
        pre_logit = torch.flatten(pre_logit, 1)
        logits = self.fc(pre_logit)

        if return_cam:
            feature_map = x.detach().clone()
            cam_weights = self.fc.weight[labels]
            cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
            return cams

        return {'logits': logits}

    def forward(self, x: Tensor, labels=None, return_cam=False) -> Tensor:
        return self._forward_impl(x, labels, return_cam)

class MoANA(nn.Module):

    def __init__(
        self,
        block: Type[Union[Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(MoANA, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
        #                                dilate=replace_stride_with_dilation[2], propose=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2], propose=True)
        # self.moana = outersum_module.osblock(512*block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.classifier = nn.Sequential(
            # nn.Conv2d(512 * block.expansion, 1000, 3, 1, padding=0),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(1000, num_classes, 1, 1, padding=0)
            nn.Conv2d(512 * block.expansion, num_classes, 1, 1, padding=0)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Conv1d):
            #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]


    def _make_layer(self, block: Type[Union[Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, propose=False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        if propose:
            layers.append(outersum_module.osblock(self.inplanes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, labels=None, return_cam=False) -> Tensor:
        # See note [TorchScript super()]
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x, map = self.layer4(x)
        x = self.layer4(x)
        # x = self.moana(x)

        # --- fc ----
        # pre_logit = self.avgpool(x)
        # pre_logit = torch.flatten(pre_logit, 1)
        # logits = self.fc(pre_logit)
        #
        # if return_cam:
        #     feature_map = x.detach().clone()
        #     cam_weights = self.fc.weight[labels]
        #     cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
        #             feature_map).mean(1, keepdim=False)
        #     return cams

        #  --- classifier ---
        pre_logit = self.classifier(x)
        logits = self.avgpool(pre_logit).squeeze()

        if return_cam:
           # normalized = self.normalize_tensor(pre_logit.detach().clone())
           normalized = pre_logit.detach().clone()
           cams = normalized[range(batch_size), labels]
           # return cams, map
           return cams

        return {'logits': logits}

        # --- visualization attmap ---
        # if return_cam:
        #     normalized = self.normalize_tensor(pre_logit.detach().clone())
        #     cam_weights = self.fc.weight[labels]
        #     cams = (cam_weights.view(*normalized.shape[:2], 1, 1) *
        #             normalized).mean(1, keepdim=False)
        #     return cams

        # return {'logits': logits, 'attmap': map}


    def forward(self, x: Tensor, labels=None, return_cam=False) -> Tensor:
        return self._forward_impl(x, labels, return_cam)

    def normalize_tensor(self, x):
        channel_vector = x.view(x.size()[0], x.size()[1], -1)
        minimum, _ = torch.min(channel_vector, dim=-1, keepdim=True)
        maximum, _ = torch.max(channel_vector, dim=-1, keepdim=True)
        normalized_vector = torch.div(channel_vector - minimum, maximum - minimum)
        normalized_tensor = normalized_vector.view(x.size())
        return normalized_tensor

    # def normalize_tensor(self, x):
    #     channel_vector = x.view(x.size()[0], x.size()[1], -1)
    #     maximum, _ = torch.max(channel_vector, dim=-1, keepdim=True)
    #     normalized_vector = torch.div(channel_vector, maximum)
    #     normalized_tensor = normalized_vector.view(x.size())
    #     return normalized_tensor



def _resnet(
    module_name: str,
    arch: str,
    block: Type[Union[Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
):
    model = {'resnet': ResNet,
             'moana': MoANA,
              }[module_name](block, layers, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)

    return model

def resnet50(module_name, pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(module_name, 'resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def _pretrainnet(
    module_name: str,
    block: Type[Union[Bottleneck]],
    layers: List[int],
    pretrained: bool,
    parallel: bool,
    **kwargs: Any
):
    model = {'resnet': ResNet,
             'moana': MoANA,
             }[module_name](block, layers, **kwargs)

    if pretrained:
        if module_name == "resnet":
            checkpoint = torch.load("./checkpoint/CUB_CAM_final_45.pth")
            # checkpoint = torch.load("./checkpoint/CAM_original_1.pth")
            # checkpoint = torch.load("./checkpoint/CAM_original_3.pth")
            # checkpoint = torch.load("./checkpoint/CAM_original_8.pth")
            # checkpoint = torch.load("./checkpoint/CAM_original_2_6.pth")
            # checkpoint = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        elif module_name == "moana":
            checkpoint = torch.load("./checkpoint/moana_cub.pth")
        #     checkpoint = torch.load("./checkpoint/moana_cub_nonmix.pth")
        #     checkpoint = torch.load("./checkpoint/MoANA_ILSVRC.pth")

        #     checkpoint = torch.load("./checkpoint/MoANA_only_notmix_1000_7.pth")
        #     checkpoint = torch.load("./checkpoint/MoANA_ILSVRC_001_22.pth")
        elif module_name == "moana_i2c":
            checkpoint = torch.load("./checkpoint/AIM_last_cub_46.pth")

        if parallel:
            model = torch.nn.DataParallel(model)

        model.load_state_dict(checkpoint, strict=True)
        model.eval()

    return model


def pretrainnet50(module_name, pretrained: bool = False, parallel: bool = True, **kwargs: Any):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _pretrainnet(module_name, Bottleneck, [3, 4, 6, 3], pretrained, parallel, **kwargs)
