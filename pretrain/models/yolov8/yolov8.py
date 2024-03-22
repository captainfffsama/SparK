# -*- coding: utf-8 -*-
"""
@Author: captainfffsama
@Date: 2024-03-21 19:23:02
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2024-03-22 13:40:51
@FilePath: /SparK/pretrain/models/yolov8/yolov8.py
@Description:
"""

from typing import List
from pathlib import Path

import torch
import torch.nn as nn
from timm.models.registry import register_model

from .module import parse_model


class CSPNetYolov8(nn.Module):
    """
    This is a template for your custom ConvNet.
    It is required to implement the following three functions: `get_downsample_ratio`, `get_feature_map_channels`, `forward`.
    You can refer to the implementations in `pretrain\models\resnet.py` for an example.
    """

    def __init__(self, yaml_path: str, ch=3, verbose=True) -> None:
        super().__init__()
        self.model, self.save, self.depth, self.width, self.max_channels = parse_model(
            yaml_path, ch, verbose
        )
        self.out_idx = (15, 18, 21)

    def get_downsample_ratio(self) -> int:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).

        :return: the TOTAL downsample ratio of the ConvNet.
        E.g., for a ResNet-50, this should return 32.
        """
        return 32

    def get_feature_map_channels(self) -> List[int]:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).

        :return: a list of the number of channels of each feature map.
        E.g., for a ResNet-50, this should return [256, 512, 1024, 2048].
        """
        return [
            int(256 * self.width),
            int(512 * self.width),
            int(self.max_channels * self.width),
        ]

    def forward(self, inp_bchw: torch.Tensor, hierarchical=False):
        """
        The forward with `hierarchical=True` would ONLY be used in `SparseEncoder.forward` (see `pretrain/encoder.py`).

        :param inp_bchw: input image tensor, shape: (batch_size, channels, height, width).
        :param hierarchical: return the logits (not hierarchical), or the feature maps (hierarchical).
        :return:
            - hierarchical == False: return the logits of the classification task, shape: (batch_size, num_classes).
            - hierarchical == True: return a list of all feature maps, which should have the same length as the return value of `get_feature_map_channels`.
              E.g., for a ResNet-50, it should return a list [1st_feat_map, 2nd_feat_map, 3rd_feat_map, 4th_feat_map].
                    for an input size of 224, the shapes are [(B, 256, 56, 56), (B, 512, 28, 28), (B, 1024, 14, 14), (B, 2048, 7, 7)]
        """
        y = []  # outputs
        out = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                inp_bchw = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [inp_bchw if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            inp_bchw = m(inp_bchw)  # run
            y.append(inp_bchw if m.i in self.save else None)  # save output
            if m.i in self.out_idx:
                out.append(inp_bchw)
        if hierarchical:
            return out
        else:
            raise NotImplementedError

    def __repr__(self):
        return self.model.__repr__()


@register_model
def yolov8n(pretrained=False, **kwargs):
    yaml_path = Path(__file__).parent / "yolov8n.yaml"
    return CSPNetYolov8(yaml_path)


@register_model
def yolov8s(pretrained=False, **kwargs):
    yaml_path = Path(__file__).parent / "yolov8s.yaml"
    return CSPNetYolov8(yaml_path)


@register_model
def yolov8m(pretrained=False, **kwargs):
    yaml_path = Path(__file__).parent / "yolov8m.yaml"
    return CSPNetYolov8(yaml_path)


@register_model
def yolov8l(pretrained=False, **kwargs):
    yaml_path = Path(__file__).parent / "yolov8l.yaml"
    return CSPNetYolov8(yaml_path)


@register_model
def yolov8x(pretrained=False, **kwargs):
    yaml_path = Path(__file__).parent / "yolov8x.yaml"
    return CSPNetYolov8(yaml_path)


@torch.no_grad()
def convnet_test():
    from timm.models import create_model

    cnn = create_model("yolov8_n")
    print("get_downsample_ratio:", cnn.get_downsample_ratio())
    print("get_feature_map_channels:", cnn.get_feature_map_channels())

    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()

    # check the forward function
    B, C, H, W = 4, 3, 224, 224
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])

    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio

    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch


if __name__ == "__main__":
    convnet_test()
