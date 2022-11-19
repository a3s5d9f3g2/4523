#!/usr/bin/python
# -*- coding:utf8 -*-

import torch.nn as nn
import torch.nn.functional as F

import copy
from engine.defaults.constant import MODEL_REGISTRY
from posetimation.layers.basic_layer import conv_bn_relu
from posetimation.layers.basic_model import ChainOfBasicBlocks, ChainOfBasicBlocksFix
from posetimation.layers.conv_rnn import UpdateBlock
from posetimation.backbones.hrnet import HRNetPlus
from torchvision.ops.deform_conv import DeformConv2d
from posetimation.layers.non_local_net import NLBlockND
import kornia
from engine.defaults import TRAIN_PHASE


__all__ = ["DTDMNRV15_2"]
BN_MOMENTUM = 0.1

import logging
import os.path as osp
import torch
from torch.nn.functional import kl_div
from torch.nn import init


@MODEL_REGISTRY.register()
class DTDMNRV15_2(nn.Module):


    @classmethod
    def get_model_hyper_parameters(cls, cfg):
        bbox_enlarge_factor = cfg.DATASET.BBOX_ENLARGE_FACTOR
        rot_factor = cfg.TRAIN.ROT_FACTOR
        SCALE_FACTOR = cfg.TRAIN.SCALE_FACTOR

        if not isinstance(SCALE_FACTOR, list):
            temp = SCALE_FACTOR
            SCALE_FACTOR = [SCALE_FACTOR, SCALE_FACTOR]
        scale_bottom = 1 - SCALE_FACTOR[0]
        scale_top = 1 + SCALE_FACTOR[1]

        paramer = "bbox_{}_rot_{}_scale_{}-{}".format(bbox_enlarge_factor, rot_factor, scale_bottom,
                                                      scale_top)

        if cfg.LOSS.HEATMAP_MSE.USE:
            paramer += f"_MseLoss_{cfg.LOSS.HEATMAP_MSE.WEIGHT}"

        return paramer

    def __init__(self, cfg, is_train, **kwargs):
        super(DTDMNRV15_2, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.pretrained = cfg.MODEL.PRETRAINED
        self.is_train = is_train
        if self.is_train == TRAIN_PHASE:
            self.is_train = True
        else:
            self.is_train = False
        self.pretrained_layers = ['*']
        self.hrnet = HRNetPlus(cfg, self.is_train)
        self.freeze_hrnet_weight = cfg['MODEL']["FREEZE_HRNET_WEIGHTS"]

        #  Difference Modeling
        self.motion_fusion_s1 = ChainOfBasicBlocks(48 * 4, 48, num_blocks=2)

        self.motion_fusion_s2 = ChainOfBasicBlocks(48 * 4, 48, num_blocks=2)
        self.motion_smooth_s2 = ChainOfBasicBlocks(48, 48, num_blocks=1)

        self.motion_fusion_s3 = ChainOfBasicBlocks(48 * 4, 48, num_blocks=2)
        self.motion_smooth_s3 = ChainOfBasicBlocks(48, 48, num_blocks=1)

        self.motion_fusion_s4 = ChainOfBasicBlocks(48 * 4, 48, num_blocks=2)
        self.motion_smooth_s4 = ChainOfBasicBlocks(48, 48, num_blocks=1)

        # Spatial  Alignment & Aggregation

        self.sup_agg = ChainOfBasicBlocks(48 * 4, 48, num_blocks=2)

        self.motion_gap = nn.AdaptiveAvgPool2d((1, 1))

        self.use_motion_structure = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(True),
            nn.Linear(128, 48),
            nn.Sigmoid()
        )

        self.noisy_motion_structure = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(True),
            nn.Linear(128, 48),
            nn.Sigmoid()
        )

        n_kernel_group = 12

        n_offset_channel = 2 * 3 * 3 * n_kernel_group
        n_mask_channel = 3 * 3 * n_kernel_group

        # motion s1 dcn
        self.m_a_agg_s1 = ChainOfBasicBlocks(48 * 2, 48, (3, 3), (1, 1), (1, 1), num_blocks=1)
        self.m_a_dcn_offset_s1 = conv_bn_relu(48, n_offset_channel, 3, 1, padding=3, dilation=3, has_bn=False,
                                              has_relu=False)
        self.m_a_dcn_mask_s1 = conv_bn_relu(48, n_mask_channel, 3, 1, padding=3, dilation=3, has_bn=False,
                                            has_relu=False)
        self.m_a_dcn_s1 = DeformConv2d(48, 48, 3, padding=1, dilation=1)

        # motion s2 dcn
        self.m_a_agg_s2 = ChainOfBasicBlocks(48 * 2, 48, (3, 3), (1, 1), (1, 1), num_blocks=1)
        self.m_a_dcn_offset_s2 = conv_bn_relu(48, n_offset_channel, 3, 1, padding=3, dilation=3, has_bn=False,
                                              has_relu=False)
        self.m_a_dcn_mask_s2 = conv_bn_relu(48, n_mask_channel, 3, 1, padding=3, dilation=3, has_bn=False,
                                            has_relu=False)
        self.m_a_dcn_s2 = DeformConv2d(48, 48, 3, padding=1, dilation=1)

        # motion s3 dcn
        self.m_a_agg_s3 = ChainOfBasicBlocks(48 * 2, 48, (3, 3), (1, 1), (1, 1), num_blocks=1)
        self.m_a_dcn_offset_s3 = conv_bn_relu(48, n_offset_channel, 3, 1, padding=3, dilation=3, has_bn=False,
                                              has_relu=False)
        self.m_a_dcn_mask_s3 = conv_bn_relu(48, n_mask_channel, 3, 1, padding=3, dilation=3, has_bn=False,
                                            has_relu=False)
        self.m_a_dcn_s3 = DeformConv2d(48, 48, 3, padding=1, dilation=1)

        # motion s4 dcn
        self.m_a_agg_s4 = ChainOfBasicBlocks(48 * 2, 48, (3, 3), (1, 1), (1, 1), num_blocks=1)
        self.m_a_dcn_offset_s4 = conv_bn_relu(48, n_offset_channel, 3, 1, padding=3, dilation=3, has_bn=False,
                                              has_relu=False)
        self.m_a_dcn_mask_s4 = conv_bn_relu(48, n_mask_channel, 3, 1, padding=3, dilation=3, has_bn=False,
                                            has_relu=False)
        self.m_a_dcn_s4 = DeformConv2d(48, 48, 3, padding=1, dilation=1)

        # Appearance DCN
        self.combined_feat_layers = ChainOfBasicBlocks(48 * 2, 48, (3, 3), (1, 1), (1, 1), num_blocks=1)

        self.dcn_offset_1 = conv_bn_relu(48, n_offset_channel, 3, 1, padding=3, dilation=3, has_bn=False,
                                         has_relu=False)
        self.dcn_mask_1 = conv_bn_relu(48, n_mask_channel, 3, 1, padding=3, dilation=3, has_bn=False, has_relu=False)
        self.dcn_1 = DeformConv2d(48, 48, 3, padding=3, dilation=3)

        self.dcn_offset_2 = conv_bn_relu(48, n_offset_channel, 3, 1, padding=3, dilation=3, has_bn=False,
                                         has_relu=False)
        self.dcn_mask_2 = conv_bn_relu(48, n_mask_channel, 3, 1, padding=3, dilation=3, has_bn=False, has_relu=False)
        self.dcn_2 = DeformConv2d(48, 48, 3, padding=3, dilation=3)

        self.dcn_offset_3 = conv_bn_relu(48, n_offset_channel, 3, 1, padding=3, dilation=3, has_bn=False,
                                         has_relu=False)
        self.dcn_mask_3 = conv_bn_relu(48, n_mask_channel, 3, 1, padding=3, dilation=3, has_bn=False, has_relu=False)
        self.dcn_3 = DeformConv2d(48, 48, 3, padding=3, dilation=3)

        self.dcn_offset_4 = conv_bn_relu(48, n_offset_channel, 3, 1, padding=3, dilation=3, has_bn=False,
                                         has_relu=False)
        self.dcn_mask_4 = conv_bn_relu(48, n_mask_channel, 3, 1, padding=3, dilation=3, has_bn=False, has_relu=False)
        self.dcn_4 = DeformConv2d(48, 48, 3, padding=3, dilation=3)

        self.all_agg_block = ChainOfBasicBlocks(input_channel=48 * 2, ouput_channel=48, num_blocks=3)

        self.multimodal_fusion = nn.Sequential(
            ChainOfBasicBlocks(input_channel=48 * 2, ouput_channel=48, num_blocks=3)
        )

        self.heatmap_head = nn.Conv2d(48, 17, 3, 1, 1)
        self.motion_heatmap_head = nn.Conv2d(48, 17, 3, 1, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.init_weights()
        self.motion_heatmap_head.apply(self.weights_init_kaiming)

        if self.freeze_hrnet_weight:
            self.hrnet.freeze_weight()

    def forward(self, kf_x, sup_x, **kwargs):
        """
        kf_x:  [batch, 3, 384, 288]
        sup_x: [batch, 3 * num, 384, 288]]
        """
        batch_size, num_sup = kf_x.shape[0], sup_x.shape[1] // 3
        sup_x = torch.cat(torch.chunk(sup_x, num_sup, dim=1), dim=0)

        x = torch.cat([kf_x, sup_x], dim=0)
        x_bb_hm, x_bb_feat = self.hrnet(x, multi_scale=True)

        x_bb_hm_list = torch.chunk(x_bb_hm, num_sup + 1, dim=0)

        x_bb_feat_list = torch.chunk(x_bb_feat[-1], num_sup + 1, dim=0)
        kf_bb_hm, kf_bb_feat = x_bb_hm_list[0], x_bb_feat_list[0]
        sup_bb_hm_list, sup_bb_feat_list = x_bb_hm_list[1:], x_bb_feat_list[1:]

        x_feat_list_s1 = torch.chunk(x_bb_feat[0], num_sup + 1, dim=0)
        x_feat_list_s2 = torch.chunk(x_bb_feat[1], num_sup + 1, dim=0)
        x_feat_list_s3 = torch.chunk(x_bb_feat[2], num_sup + 1, dim=0)

        # Spatial-Aggregation Features (Alignment & Aggregation)
        supp_agg_f = torch.cat(sup_bb_feat_list, dim=1)
        supp_agg_f = self.sup_agg(supp_agg_f)

        combined_feat = self.combined_feat_layers(
            torch.cat([supp_agg_f, kf_bb_feat], dim=1))  # 48

        dcn_offset = self.dcn_offset_1(combined_feat)
        dcn_mask = self.dcn_mask_1(combined_feat)
        combined_feat = self.dcn_1(combined_feat, dcn_offset, dcn_mask)

        dcn_offset = self.dcn_offset_2(combined_feat)
        dcn_mask = self.dcn_mask_2(combined_feat)
        combined_feat = self.dcn_2(combined_feat, dcn_offset, dcn_mask)

        dcn_offset = self.dcn_offset_3(combined_feat)
        dcn_mask = self.dcn_mask_3(combined_feat)
        aligned_sup_feat = self.dcn_3(supp_agg_f, dcn_offset, dcn_mask)

        dcn_offset = self.dcn_offset_4(aligned_sup_feat)
        dcn_mask = self.dcn_mask_4(aligned_sup_feat)
        aligned_sup_feat = self.dcn_4(aligned_sup_feat, dcn_offset, dcn_mask)

        appearance_feat = self.all_agg_block(torch.cat([kf_bb_feat, aligned_sup_feat], dim=1))

        # Motion Encoding

        # Motion Fusion Stage1
        s1_feature_list = x_feat_list_s1[1:]
        kf_s1_feature = x_feat_list_s1[0]
        x1_f, x2_f, x4_f, x5_f = s1_feature_list
        temporal_diff_vectors_s1 = torch.cat(
            [x2_f - x1_f, kf_s1_feature - x2_f, x4_f - kf_s1_feature, x5_f - x4_f], dim=1)
        fuse_motion_feature_s1 = self.motion_fusion_s1(temporal_diff_vectors_s1)
        combined_feat_s1 = self.m_a_agg_s1(torch.cat([kf_bb_feat, fuse_motion_feature_s1], dim=1))
        offset_1 = self.m_a_dcn_offset_s1(combined_feat_s1)
        mask_1 = self.m_a_dcn_mask_s1(combined_feat_s1)
        modulated_motion_feature_s1 = self.m_a_dcn_s1(fuse_motion_feature_s1, offset_1, mask_1)

        motion_features_v1 = modulated_motion_feature_s1

        # # Motion Fusion Stage2
        s2_feature_list = x_feat_list_s2[1:]
        kf_s2_feature = x_feat_list_s2[0]
        x1_f, x2_f, x4_f, x5_f = s2_feature_list
        temporal_diff_vectors_s2 = torch.cat(
            [x2_f - x1_f, kf_s2_feature - x2_f, x4_f - kf_s2_feature, x5_f - x4_f], dim=1)
        fuse_motion_feature_s2 = self.motion_fusion_s2(temporal_diff_vectors_s2)
        combined_feat_s2 = self.m_a_agg_s2(torch.cat([kf_bb_feat, fuse_motion_feature_s2], dim=1))
        offset_2 = self.m_a_dcn_offset_s2(combined_feat_s2)
        mask_2 = self.m_a_dcn_mask_s2(combined_feat_s2)
        modulated_motion_feature_s2 = self.m_a_dcn_s2(fuse_motion_feature_s2, offset_2, mask_2)

        motion_features_v2 = self.motion_smooth_s2(motion_features_v1 + modulated_motion_feature_s2)

        # # Motion Fusion Stage3
        kf_s3_feature = x_feat_list_s3[0]
        s3_feature_list = x_feat_list_s3[1:]
        x1_f, x2_f, x4_f, x5_f = s3_feature_list
        temporal_diff_vectors_s3 = torch.cat(
            [x2_f - x1_f, kf_s3_feature - x2_f, x4_f - kf_s3_feature, x5_f - x4_f], dim=1)
        fuse_motion_feature_s3 = self.motion_fusion_s3(temporal_diff_vectors_s3)
        combined_feat_s3 = self.m_a_agg_s3(torch.cat([kf_bb_feat, fuse_motion_feature_s3], dim=1))
        offset_3 = self.m_a_dcn_offset_s3(combined_feat_s3)
        mask_3 = self.m_a_dcn_mask_s3(combined_feat_s3)
        modulated_motion_feature_s3 = self.m_a_dcn_s3(fuse_motion_feature_s3, offset_3, mask_3)

        motion_features_v3 = self.motion_smooth_s3(motion_features_v2 + modulated_motion_feature_s3)

        x1_f, x2_f, x4_f, x5_f = sup_bb_feat_list
        temporal_diff_vectors_s4 = torch.cat(
            [x2_f - x1_f, kf_bb_feat - x2_f, x4_f - kf_bb_feat, x5_f - x4_f], dim=1)
        fuse_motion_feature_s4 = self.motion_fusion_s4(temporal_diff_vectors_s4)
        combined_feat_s4 = self.m_a_agg_s4(torch.cat([kf_bb_feat, fuse_motion_feature_s4], dim=1))
        offset_4 = self.m_a_dcn_offset_s4(combined_feat_s4)
        mask_4 = self.m_a_dcn_mask_s4(combined_feat_s4)
        modulated_motion_feature_s4 = self.m_a_dcn_s4(fuse_motion_feature_s4, offset_4, mask_4)

        motion_features = self.motion_smooth_s4(motion_features_v3 + modulated_motion_feature_s4)

        # Motion Distillation
        # Global Average Pooling -> reduce
        low_motion_features = self.motion_gap(motion_features)
        low_motion_features = low_motion_features.view(batch_size, -1)
        # useful motion info
        use_motion_feat_mask = self.use_motion_structure(low_motion_features)
        use_motion_feat_mask = use_motion_feat_mask.view(use_motion_feat_mask.shape[0],
                                                         use_motion_feat_mask.shape[1], 1, 1)
        use_motion_feat = use_motion_feat_mask * motion_features
        # noisy motion info
        noisy_motion_feat_mask = self.noisy_motion_structure(low_motion_features)
        noisy_motion_feat_mask = noisy_motion_feat_mask.view(noisy_motion_feat_mask.shape[0],
                                                             noisy_motion_feat_mask.shape[1], 1, 1)
        noisy_motion_feat = noisy_motion_feat_mask * motion_features

        # Final Representation
        final_feature = self.multimodal_fusion(torch.cat([use_motion_feat, appearance_feat], dim=1))
        final_heatmap = self.heatmap_head(final_feature)

        if self.is_train:
            # basic MI term scalar
            # motion distillation terms
            mi_fm_fn = self.feat_feat_mi_estimation(use_motion_feat, noisy_motion_feat, freeze=False)
            mi_fm_f = self.feat_feat_mi_estimation(motion_features, use_motion_feat)

            # multi modal terms
            mi_fm_fa = self.feat_feat_mi_estimation(use_motion_feat, appearance_feat)
            mi_fm_ff = self.feat_feat_mi_estimation(use_motion_feat, final_feature)
            mi_fa_ff = self.feat_feat_mi_estimation(appearance_feat, final_feature)

            # regularization terms
            mi_fm_y = self.feat_label_mi_estimation(use_motion_feat, final_heatmap, motion=False)
            mi_fa_y = self.feat_label_mi_estimation(appearance_feat, final_heatmap, motion=False)

            return final_heatmap, kf_bb_hm, [mi_fm_fn, mi_fm_f, mi_fm_fa, mi_fm_ff, mi_fa_ff, mi_fm_y, mi_fa_y]
        else:
            return final_heatmap, kf_bb_hm

        # if self.is_train:
        #     # {I}( {y}_{t} ;  \boldsymbol{\widetilde{z}}_{t+\delta})         =>  final_hm     &  all_agg_features / aligned_sup_feat
        #     mi_loss_1 = self.feat_label_mi_estimation(all_agg_features, final_hm)
        #     # {I}( {z}_{t} ;  \boldsymbol{\widetilde{z}}_{t+\delta})         =>  kf_bb_feat   &  all_agg_features / aligned_sup_feat
        #     mi_loss_2 = self.feat_feat_mi_estimation(kf_bb_feat, all_agg_features)
        #     # {I}( {y}_{t} ;  {z}_{t+\delta})                                =>  final_hm     &  agg_sup_feat
        #     mi_loss_3 = self.feat_label_mi_estimation(agg_sup_feat, final_hm)
        #     # {I}( {z}_{t+\delta} ; \boldsymbol{\widetilde{z}}_{t+\delta})   =>  agg_sup_feat &  all_agg_features / aligned_sup_feat
        #     mi_loss_4 = self.feat_feat_mi_estimation(agg_sup_feat, all_agg_features)
        #     # {I}( {y}_{t}   ; {z}_{t})                                      =>  final_hm    &  kf_bb_feat
        #     mi_loss_5 = self.feat_label_mi_estimation(kf_bb_feat, final_hm)
        #     # {I}( {z}_{t}   ; \boldsymbol{\widetilde{z}}_{t+\delta})        =>  kf_bb_feat   &  all_agg_features / aligned_sup_feat
        #     mi_loss_6 = self.feat_feat_mi_estimation(kf_bb_feat, all_agg_features)
        #
        #     mi_loss_list = [mi_loss_1, mi_loss_2, mi_loss_3, mi_loss_4, mi_loss_5, mi_loss_6]
        #
        #     return final_hm, [], kf_bb_hm, mi_loss_list
        # else:
        #     return final_hm, kf_bb_hm

    def init_weights(self, *args, **kwargs):
        logger = logging.getLogger(__name__)
        hrnet_name_set = set()
        for module_name, module in self.named_modules():
            if module_name.split('.')[0] == "hrnet":
                hrnet_name_set.add(module_name)

            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)

                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
            else:
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
                    if name in ['weights']:
                        nn.init.normal_(module.weight, std=0.001)

        if osp.isfile(self.pretrained):
            pretrained_state_dict = torch.load(self.pretrained)
            if 'state_dict' in pretrained_state_dict.keys():
                pretrained_state_dict = pretrained_state_dict['state_dict']
            logger.info('{} => loading pretrained model {}'.format(self.__class__.__name__,
                                                                   self.pretrained))

            if list(pretrained_state_dict.keys())[0].startswith('module.'):
                model_state_dict = {k[7:]: v for k, v in pretrained_state_dict.items()}
            else:
                model_state_dict = pretrained_state_dict

            need_init_state_dict = {}
            for name, m in model_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers or self.pretrained_layers[0] is '*':
                    layer_name = name.split('.')[0]
                    if layer_name in hrnet_name_set:
                        need_init_state_dict[name] = m
                    else:
                        new_layer_name = "hrnet.{}".format(layer_name)
                        if new_layer_name in hrnet_name_set:
                            parameter_name = "hrnet.{}".format(name)
                            need_init_state_dict[parameter_name] = m

            self.load_state_dict(need_init_state_dict, strict=False)
            # self.load_state_dict(need_init_state_dict, strict=False)
        elif self.pretrained:
            # raise NotImplementedError
            logger.error('=> please download pre-trained models first!')

        # self.freeze_weight()

        self.logger.info("Finish init_weights")

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    def feat_label_mi_estimation(self, Feat, Y, motion=False):
        """
            F: [B,48,96,72]
            Y: [B,17,96,72]
        """
        batch_size = Feat.shape[0]
        temperature = 0.05
        pred_Y = self.hrnet.final_layer(Feat)  # B,48,96,72 -> B,17,96,72
        pred_Y = pred_Y.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        Y = Y.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        if motion:
            motion_Y = self.motion_heatmap_head(Feat)
            mi = kl_div(input=self.softmax(motion_Y.detach() / temperature),
                        target=self.softmax(Y / temperature),
                        reduction='mean')  # pixel-level
        else:
            mi = kl_div(input=self.softmax(pred_Y.detach() / temperature),
                        target=self.softmax(Y / temperature),
                        reduction='mean')  # pixel-level

        return mi

    def feat_feat_mi_estimation(self, F1, F2, freeze=True):
        """
            F1: [B,48,96,72]
            F2: [B,48,96,72]
            F1 -> F2
        """
        batch_size = F1.shape[0]
        temperature = 0.05
        F1 = F1.reshape(batch_size, 48, -1).reshape(batch_size * 48, -1)
        F2 = F2.reshape(batch_size, 48, -1).reshape(batch_size * 48, -1)
        if freeze:
            mi = kl_div(input=self.softmax(F1.detach() / temperature),
                        target=self.softmax(F2 / temperature))
        else:
            mi = kl_div(input=self.softmax(F1 / temperature),
                        target=self.softmax(F2 / temperature))

        return mi

    def heatmaps_affine_transformation(self, heatmaps: torch.Tensor, offsets):
        """
            heatmaps : (batch, num_joints, map_height, map_width)
            offsets : (batch, num_joints*2)
            theta:[
                [1, 0, c_1],[0, 1, c_2]
            ]
            Note: c_1
        """
        batch_size, num_joints, map_height, map_width = heatmaps.size()
        # (batch, num_joints, 2, 3)
        batch_theta = torch.zeros((batch_size, num_joints, 2, 3), device=heatmaps.device)
        batch_theta[:, :, [0, 1], [0, 1]] = 1
        
        batch_theta[:, :, 0, 2] = offsets[:, ::2]  # x offset
        batch_theta[:, :, 1, 2] = offsets[:, 1::2]  # y offset

        heatmaps = heatmaps.reshape(batch_size * num_joints, 1, map_height, map_width)
        batch_theta = batch_theta.reshape(batch_size * num_joints, 2, 3)
        output = kornia.geometry.warp_affine(heatmaps, batch_theta, dsize=(map_height, map_width))
        output = output.reshape(batch_size, num_joints, map_height, map_width)
        return output

    def temporal_channel_conv(self, features, times, conv):
        """
        features: [Batch, Channel, H, W]s
        """
        batch_size, channel, height, width = features.shape[0], features.shape[1] // times, \
                                             features.shape[2], \
                                             features.shape[3]
        # Channel Concat -> [B,T,C, H,W] -> [B,H,W,C,T] -> [BHW, C, T]
        features = features.contiguous().view(batch_size, -1, channel, height, width).permute(0, 3,
                                                                                              4, 2,
                                                                                              1) \
            .contiguous().view(-1, channel, times)
        agg_sup_feat = conv(features)
        agg_sup_feat = agg_sup_feat.view(batch_size, height, width, channel, times).permute(0, 3, 4,
                                                                                            1, 2) \
            .view(batch_size, -1, height, width)
        return agg_sup_feat

    def motion_agg_module(self):
        pass
