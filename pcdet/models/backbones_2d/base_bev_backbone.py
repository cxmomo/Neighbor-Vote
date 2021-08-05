import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        self.downsample = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 224, 3, stride=2, padding=0),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 224, 1, stride=2, padding=0),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
        )

        self.saconv = SelfAttentionBlock_(224, 128, 224)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(224, 224, 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(224, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(224, 224, 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(224, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.merge = nn.Sequential(
            nn.Conv2d(608, 384, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(384, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(384, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """


        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features

        x_att = self.downsample(x)
        x_att, _ = self.saconv(x_att)

        x_att = self.up(x_att)
        
        data_dict['nonlocal_features'] = x_att

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        
        data_dict['spatial_features_2d_reg'] = x
        x_cat = torch.cat([x,x_att], dim=1)
        x = self.merge(x_cat)
        data_dict['spatial_features_2d'] = x        
    
        return data_dict

class PixelAttentionBlock_(nn.Module):
    def __init__(self, in_channels, key_channels):
        super(PixelAttentionBlock_, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.f_key = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0)),
            ('bn',   nn.BatchNorm2d(key_channels)),
            ('relu', nn.ReLU(True))]))
        self.parameter_initialization()
        self.f_query = self.f_key

    def parameter_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_att(self, x):
        batch_size = x.size(0)
        query = self.f_query(x).view(batch_size, self.key_channels, -1).permute(0, 2, 1) #(b, 175*200, key_channels)
        key = self.f_key(x).view(batch_size, self.key_channels, -1) #(b,key_channels, 175*200)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        return sim_map

    def forward(self, x):
        raise NotImplementedError

class SelfAttentionBlock_(PixelAttentionBlock_):
    def __init__(self, in_channels, key_channels, value_channels, scale=1):
        super(SelfAttentionBlock_, self).__init__(in_channels, key_channels)
        self.scale = scale
        self.value_channels = value_channels
        if scale > 1:
            self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        kernel_size = 3
        self.f_value = nn.Sequential(OrderedDict([
            ('conv1',  nn.Conv2d(in_channels, value_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)),
            ('relu1',  nn.ReLU(inplace=True)),
            ('conv2',  nn.Conv2d(value_channels, value_channels, kernel_size=1, stride=1)),
            ('relu2',  nn.ReLU(inplace=True))]))
        self.parameter_initialization()

    def parameter_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size, c, h, w = x.size()
        if self.scale > 1:
            x = self.pool(x)
        sim_map = self.forward_att(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1).permute(0, 2, 1)
        context = torch.matmul(sim_map, value).permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, h, w)
        if self.scale > 1:
            context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=True)
        return [context, sim_map]