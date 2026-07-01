""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from icecream import ic
import torchvision
from collections import OrderedDict, Counter
from opencood.models.common_modules.airv2x_base_model import Airv2xBase
from opencood.models.common_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.task_heads.segmentation_head import BevSegHead
from opencood.models.common_modules.naive_compress import NaiveCompressor
import importlib
# from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn

class Airv2xHEAL(Airv2xBase):
    def __init__(self, args):
        super(Airv2xHEAL, self).__init__(args)
        
        self.args = args

        # here we use image encoder LSS instead of lidar
        self.collaborators = args["collaborators"]
        self.active_sensors = args["active_sensors"]
        
        self.init_encoders(args)
        modality_args = args["modality_fusion"]
        bev_feature_dim = self._infer_backbone_in_channels(args)
        self.backbone = ResNetBEVBackbone(
            modality_args["base_bev_backbone"], bev_feature_dim
        )
        
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if "shrink_header" in modality_args and modality_args["shrink_header"]["use"]:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(modality_args["shrink_header"])
        self.compression = False

        if modality_args["compression"] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args["compression"])
            
        # PyramidFusion will read inplanes from YAML (set to sum(num_upsample_filter))
        self.pyramid_backbone = PyramidFusion(args["fusion_backbone"])

        """
        Shared Heads, Would load from pretrain base.
        """
        if args["task"] == "det":
            self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'] * args["num_class"],
                                    kernel_size=1)
            # Initialize the bias of cls_head for focal loss
            pi = 0.01
            nn.init.constant_(self.cls_head.bias, -np.log((1 - pi) / pi))
            
            self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                    kernel_size=1)
            if args["obj_head"]:
                self.obj_head = nn.Conv2d(
                    args['in_head'], args["anchor_number"], kernel_size=1
                )
        elif args["task"] == "seg":
            self.seg_head = BevSegHead(
                args["seg_branch"], args["seg_hw"], args["seg_hw"], args['in_head'], args["dynamic_class"], args["static_class"],
                seg_res=args["seg_res"], cav_range=args["cav_range"]
            )
        # self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
        #                           kernel_size=1) # BIN_NUM = 2
        
        if args["backbone_fix"]:
            self.backbone_fix(args["backbone_fix"])
            
    def backbone_fix(self, args):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        if type(args) == bool:
        
            if "vehicle" in self.collaborators:
                for p in self.veh_models.parameters():
                    p.requires_grad = False
            if "rsu" in self.collaborators:
                for p in self.rsu_models.parameters():
                    p.requires_grad = False
            if "drone" in self.collaborators:
                for p in self.drone_models.parameters():
                    p.requires_grad = False
        
        elif type(args) == list:
            for i in range(len(args)):
                if args[i] == "vehicle":
                    print("fix vehicle")
                    for p in self.veh_models.parameters():
                        p.requires_grad = False
                elif args[i] == "rsu":
                    print("fix rsu")
                    for p in self.rsu_models.parameters():
                        p.requires_grad = False
                elif args[i] == "drone":
                    print("fix drone")
                    for p in self.drone_models.parameters():
                        p.requires_grad = False
                else:
                    raise ValueError("args should be bool or list")
        
        else:
            raise ValueError("args should be bool or list")

        # 只有当 args 是 bool (True) 时，才冻结 backbone 和检测头
        # 当 args 是 list 时（如 [rsu, drone]），只冻结指定的编码器，保持检测头可训练
        freeze_all = (type(args) == bool and args == True)
        
        if freeze_all:
            for p in self.backbone.parameters():
                p.requires_grad = False

            if self.compression:
                for p in self.naive_compressor.parameters():
                    p.requires_grad = False
            if self.shrink_flag:
                for p in self.shrink_conv.parameters():
                    p.requires_grad = False
                    
            for p in self.pyramid_backbone.parameters():
                p.requires_grad = False

            if self.args["task"] == "det":
                for p in self.cls_head.parameters():
                    p.requires_grad = False
                for p in self.reg_head.parameters():
                    p.requires_grad = False
                if self.args["obj_head"]:
                    for p in self.obj_head.parameters():
                        p.requires_grad = False

            elif self.args["task"] == "seg":
                for p in self.seg_head.parameters():
                    p.requires_grad = False
        else:
            # 当只冻结特定编码器时，打印提示
            print(f"[backbone_fix] Only freezing encoders: {args}, keeping detection heads trainable")
                
                


    def forward(self, data_dict):
        output_dict = {'pyramid': 'single'}

        batch_output_dict, batch_record_len = self.extract_features(data_dict)
        spatial_hw = batch_output_dict["spatial_features"].shape[-2:]
        stride = self.args.get(
            "feature_stride", 2 if self.args.get("task") == "det" else 1
        )
        stride = max(1, int(stride))
        target_hw = tuple(max(1, dim // stride) for dim in spatial_hw)
        batch_output_dict = self.backbone(batch_output_dict)
        comm_rates = batch_output_dict["spatial_features_2d"].count_nonzero().item()
        
        batch_spatial_features_2d = batch_output_dict["spatial_features_2d"]
        # camera features are still in its own coordinate system
        pairwise_t_matrix = data_dict["img_pairwise_t_matrix_collab"]
        
        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                        batch_spatial_features_2d,
                                        batch_record_len, 
                                        pairwise_t_matrix[:, :, :, [0, 1], :][
                                            :, :, :, :, [0, 1, 3]], 
                                    )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        if fused_feature.shape[-2:] != target_hw:
            fused_feature = F.interpolate(
                fused_feature,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        
        # NEW FIX: Clamp the feature values to prevent nan/inf
        # Using a large but safe range for float16
        fused_feature = torch.clamp(fused_feature, min=-10000.0, max=10000.0)

        # Debug: Print statistics of fused_feature
        if self.training and comm_rates > 0: # Print only occasionally or for first batch
             # print(f"Fused Feature Stats - Mean: {fused_feature.mean().item():.4f}, Std: {fused_feature.std().item():.4f}, Min: {fused_feature.min().item():.4f}, Max: {fused_feature.max().item():.4f}")
             
             # Check cls_head output stats
             with torch.no_grad():
                 cls_out = self.cls_head(fused_feature)
                 # print(f"Cls Head Out Stats - Mean: {cls_out.mean().item():.4f}, Min: {cls_out.min().item():.4f}, Max: {cls_out.max().item():.4f}")

        if torch.isnan(fused_feature).any() or torch.isinf(fused_feature).any():
            raise RuntimeError("Fused feature STILL contains nan or inf AFTER clamping!")

        if self.args["task"] == "det":
            psm = self.cls_head(fused_feature)
            rm = self.reg_head(fused_feature)

            if self.args["obj_head"]:
                obj = self.obj_head(fused_feature)
                output_dict.update({"obj": obj})
            output_dict.update(
                {
                    "psm": psm,
                    "rm": rm,
                    "comm_rate": comm_rates,
                }
            )

        elif self.args["task"] == "seg":
            seg_logits = self.seg_head(fused_feature)
            output_dict.update(
                {
                    "comm_rate": comm_rates,
                }
            )
            output_dict.update(seg_logits)
       
        return output_dict
        
    def _infer_backbone_in_channels(self, args: dict) -> int:
        """Infer BEV channel dimension emitted by encoders for backbone input."""
        channel_candidates = []
        for agent_type in ["vehicle", "rsu", "drone"]:
            agent_cfg = args.get(agent_type)
            if not agent_cfg:
                continue
            cam_cfg = agent_cfg.get("cam")
            if cam_cfg and "bevout_feature" in cam_cfg:
                channel_candidates.append(cam_cfg["bevout_feature"])
            lidar_cfg = agent_cfg.get("lidar")
            if lidar_cfg:
                scatter_cfg = lidar_cfg.get("point_pillar_scatter", {})
                if "num_features" in scatter_cfg:
                    channel_candidates.append(scatter_cfg["num_features"])

        channel_candidates = [c for c in channel_candidates if isinstance(c, int)]
        if channel_candidates:
            if len(set(channel_candidates)) > 1:
                raise ValueError(
                    f"Inconsistent BEV feature dims across agents: {channel_candidates}"
                )
            return channel_candidates[0]

        return (
            args.get("modality_fusion", {})
            .get("base_bev_backbone", {})
            .get("inplanes", 64)
        )




