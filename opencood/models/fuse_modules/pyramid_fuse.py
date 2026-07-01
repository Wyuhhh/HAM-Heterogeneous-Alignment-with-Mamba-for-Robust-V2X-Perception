# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.common_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.resblock import ResNetModified, Bottleneck, BasicBlock
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.models.fuse_modules.mamba_blocks import LocalMamba2D, GlobalMamba2D


def weighted_fuse(x, score, record_len, affine_matrix, align_corners):
    """
    Parameters
    ----------
    x : torch.Tensor
            scores_in_ego = torch.squeeze(
                warp_affine_simple(scores, t_matrix, (H, W), align_corners=align_corners), dim=1
            )

            # Stable normalized weights: avoid (-inf)->softmax that can NaN when all
            # scores are invalid. Clamp to >=0 and normalize by sum + eps.
            scores_in_ego = torch.clamp(scores_in_ego, min=0.0)
            denom = scores_in_ego.sum(dim=0).clamp_min(1e-6)  # [H, W]

            # Memory-friendly weighted sum: avoid materializing (L,C,H,W) intermediate
            # from `feature_in_ego * norm_scores`.
            fused = torch.zeros_like(feature_in_ego[0])
            for li in range(feature_in_ego.shape[0]):
                fused = fused + feature_in_ego[li] * (scores_in_ego[li] / denom).unsqueeze(0)
            out.append(fused)
        score, (sum(n_cav), 1, H, W)
        
    record_len : list
        shape: (B)
        
    affine_matrix : torch.Tensor
        normalized affine matrix from 'normalize_pairwise_tfm'
        shape: (B, L, L, 2, 3) 
    """

    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    # score = torch.sum(score, dim=1, keepdim=True)
    split_score = regroup(score, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        score = split_score[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0 # ego
        feature_in_ego = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W), align_corners=align_corners)
        scores_in_ego = warp_affine_simple(split_score[b],
                           t_matrix[i, :, :, :],
                           (H, W), align_corners=align_corners)
        # Stable normalization across agents without using -inf softmax
        # 1) Ensure non-negative scores (after sigmoid + warp)
        scores_in_ego = torch.clamp(scores_in_ego, min=0.0)
        # 2) Normalize along agent dimension with epsilon to avoid divide-by-zero
        scores_sum = torch.sum(scores_in_ego, dim=0, keepdim=True) + 1e-6
        norm_scores = scores_in_ego / scores_sum

        out.append(torch.sum(feature_in_ego * norm_scores, dim=0))
    out = torch.stack(out)
    
    return out

class PyramidFusion(ResNetBEVBackbone):
    def __init__(self, model_cfg, input_channels=64):
        """
        Do not downsample in the first layer.
        """
        super().__init__(model_cfg, input_channels)
        if model_cfg["resnext"]:
            Bottleneck.expansion = 1
            self.resnet = ResNetModified(Bottleneck, 
                                        self.model_cfg['layer_nums'],
                                        self.model_cfg['layer_strides'],
                                        self.model_cfg['num_filters'],
                                        inplanes = model_cfg.get('inplanes', 64),
                                        groups=32,
                                        width_per_group=4)
        self.align_corners = model_cfg.get('align_corners', False)
        print('Align corners: ', self.align_corners)
        
        # add single supervision head
        for i in range(self.num_levels):
            setattr(
                self,
                f"single_head_{i}",
                nn.Conv2d(self.model_cfg["num_filters"][i], 1, kernel_size=1),
            )

        # Mixed Mamba blocks: local window-wise + global Hilbert-serialized
        self.window_size = int(model_cfg.get('mamba_window', 8))
        self.mamba_kernel = int(model_cfg.get('mamba_kernel', 7))
        # Default to False for training stability/memory safety; enable explicitly
        # through YAML with `use_mamba: true`.
        self.use_mamba = bool(model_cfg.get('use_mamba', False))
        if self.use_mamba:
            self.local_mambas = nn.ModuleList([
                LocalMamba2D(self.model_cfg['num_filters'][i], window_size=self.window_size, kernel_size=self.mamba_kernel)
                for i in range(self.num_levels)
            ])
            self.global_mambas = nn.ModuleList([
                GlobalMamba2D(self.model_cfg['num_filters'][i], kernel_size=self.mamba_kernel)
                for i in range(self.num_levels)
            ])

    def forward_single(self, spatial_features):
        """
        This is used for single agent pass.
        """
        feature_list = self.get_multiscale_feature(spatial_features)
        if self.use_mamba:
            feature_list = [
                self.global_mambas[i](self.local_mambas[i](feature_list[i]))
                for i in range(self.num_levels)
            ]
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)
        final_feature = self.decode_multiscale_feature(feature_list)

        return final_feature, occ_map_list
    
    def forward_collab(self, spatial_features, record_len, affine_matrix, agent_modality_list = None, cam_crop_info = None):
        """
        spatial_features : torch.tensor
            [sum(record_len), C, H, W]

        record_len : list
            cav num in each sample

        affine_matrix : torch.tensor
            [B, L, L, 2, 3]

        agent_modality_list : list
            len = sum(record_len), modality of each cav

        cam_crop_info : dict
            {'m2':
                {
                    'crop_ratio_W_m2': 0.5,
                    'crop_ratio_H_m2': 0.5,
                }
            }
        """
        crop_mask_flag = False
        cam_modality_set = None
        cam_agent_mask_dict = None
        if cam_crop_info is not None and len(cam_crop_info) > 0 and agent_modality_list is not None:
            crop_mask_flag = True
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {}
            for cam_modality in cam_modality_set:
                mask_list = [1 if x == cam_modality else 0 for x in agent_modality_list] 
                mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
                cam_agent_mask_dict[cam_modality] = mask_tensor

                # e.g. {m2: [0,0,0,1], m4: [0,1,0,0]}


        feature_list = self.get_multiscale_feature(spatial_features)
        if self.use_mamba:
            feature_list = [
                self.global_mambas[i](self.local_mambas[i](feature_list[i]))
                for i in range(self.num_levels)
            ]
        
        fused_feature_list = []
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])  # [N, 1, H, W]
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and (cam_modality_set is not None) and (cam_agent_mask_dict is not None) and (cam_crop_info is not None) and (not self.training):
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, H, W = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4 # There may be unstable response values at the edges.
                    crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4 # There may be unstable response values at the edges.

                    start_h = int(H//2-crop_H//2)
                    end_h = int(H//2+crop_H//2)
                    start_w = int(W//2-crop_W//2)
                    end_w = int(W//2+crop_W//2)

                    cam_crop_mask[cam_agent_mask_dict[cam_modality],:,start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[cam_agent_mask_dict[cam_modality]] = 1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]

                score = score * cam_crop_mask

            fused_feature_list.append(weighted_fuse(feature_list[i], score, record_len, affine_matrix, self.align_corners))
        
        # Apply deblocks to adjust channels and upsample fused features to same spatial size
        target_h, target_w = fused_feature_list[0].shape[-2:]
        processed_features = []
        for i in range(self.num_levels):
            # Apply deblock to adjust channels
            if len(self.deblocks) > 0:
                deblocked = self.deblocks[i](fused_feature_list[i])
            else:
                deblocked = fused_feature_list[i]
            
            # Upsample to match the spatial size of the first level
            if i > 0:
                deblocked = F.interpolate(deblocked, size=(target_h, target_w), 
                                        mode='bilinear', align_corners=self.align_corners)
            
            processed_features.append(deblocked)
        
        # For PyramidFusion, we fuse features at multiple scales and then concatenate them
        if len(processed_features) > 1:
            fused_feature = torch.cat(processed_features, dim=1)
        else:
            fused_feature = processed_features[0]

        
        return fused_feature, occ_map_list 