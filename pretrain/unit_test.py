# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2024-03-22 10:41:02
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2024-03-22 18:12:26
@FilePath: /SparK/pretrain/unit_test.py
@Description:
'''
import unittest
import torch
import torch.nn as nn

from models import yolov8



class Yolov8Test(unittest.TestCase):

    def test_parse_model(self):
        from timm.models import create_model
        cnn = create_model('yolov8x')
        print('get_downsample_ratio:', cnn.get_downsample_ratio())
        print('get_feature_map_channels:', cnn.get_feature_map_channels())

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


    def test_yolov8(self):
        pass

if __name__ == '__main__':
    # unittest.main()
    a=Yolov8Test()
    a.test_parse_model()