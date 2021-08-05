import numpy as np
import torch.nn as nn
import torch
from .anchor_head_template import AnchorHeadTemplate

import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import torch.nn.functional as F
import pickle as pkl

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.before_vote_head = nn.Sequential(
            nn.Conv2d(384, 224, 3, stride=1, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 224, 3, stride=1, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 224, 3, stride=1, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 224, 3, stride=1, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True)
            ) 

        self.vote_head = nn.Sequential(
            nn.Conv2d(448, 224, 3, stride=1, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 196, 3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),            
            nn.Conv2d(196, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            )

        self.vote_dist_head = nn.Conv2d(96, 2, 3, stride=1, padding=1)
        
        self.vote_angle_head = nn.Conv2d(96, 4, 3, stride=1, padding=1)

        self.neighbor_vote1 = nn.Sequential(
            nn.Conv2d(6, 64, 1, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 224, 3, stride=1, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            )  
        self.vote_att = SelfAttentionBlock_(224, 128, 224)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(224, 224, 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(224, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.neighbor_vote2 = nn.Sequential(
            nn.Conv2d(224, 224, 3, stride=1, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 224, 3, stride=1, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 224, 3, stride=1, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 224, 3, stride=1, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True)
        )

        self.vote_clss_head = nn.Sequential(
            nn.Conv2d(224, 224, 3, stride=1, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, self.num_anchors_per_location * self.num_class, kernel_size=1)
        )

        self.weight = nn.Sequential(
            nn.Conv2d(448, 224, 3, stride=1, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 224, 3, stride=1, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 2, 3, stride=1, padding=1),
      
        ) 
        self.sm = nn.Softmax(dim=1)  

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):

        spatial_features_2d = data_dict['spatial_features_2d']
        spatial_features_2d_reg = data_dict['spatial_features_2d_reg']
        nonlocal_features = data_dict['nonlocal_features']
        image_id = data_dict['frame_id']
        

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d_reg)
        x_vote1 = self.before_vote_head(spatial_features_2d)
        x_vote2 = torch.cat((nonlocal_features, x_vote1), dim=1)
        vote_preds = self.vote_head(x_vote2)

        vote_dist_preds = self.vote_dist_head(vote_preds)
        vote_angle_preds =  self.vote_angle_head(vote_preds)
        vote_preds_cat = torch.cat((vote_dist_preds[:,0,:,:].unsqueeze(1), vote_angle_preds[:,:2,:,:], vote_dist_preds[:,1,:,:].unsqueeze(1), vote_angle_preds[:,2:,:,:]), dim=1)

        
        nv_fea1 = self.neighbor_vote1(vote_preds_cat)       
        x_vote_att, _ = self.vote_att(nv_fea1)
        x_vote_att = self.up4(x_vote_att)

        x_vote3 = self.neighbor_vote2(x_vote_att)
        vote_cls_preds = self.vote_clss_head(x_vote3)

        x_w = torch.cat((x_vote1, x_vote3), dim=1)
        
        vote_weight = self.weight(x_w)
        weight = self.sm(vote_weight)
       
        final_cls_preds = torch.unsqueeze(weight[:,0,:,:], dim = 1) * cls_preds + torch.unsqueeze(weight[:,1,:,:], dim =1) * vote_cls_preds

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        vote_cls_preds = vote_cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        final_cls_preds = final_cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        
        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['vote_cls_preds'] = vote_cls_preds
        self.forward_ret_dict['final_cls_preds'] = final_cls_preds
        self.forward_ret_dict['vote_dist_preds'] = vote_dist_preds
        self.forward_ret_dict['vote_angle_preds'] = vote_angle_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
 
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=final_cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

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