# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# Modifier: Yuheng Wu <yuhengwu@kaist.ac.kr>, Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib


import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """

    def __init__(self, beta: float = 1.0 / 9.0, code_weights: Optional[List[float]] = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n**2 / beta, n - 0.5 * beta)

        return loss

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert (
                weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            )
            loss = loss * weights.unsqueeze(-1)

        return loss


class PointPillarLossMultiClass(nn.Module):
    def __init__(self, args):
        super(PointPillarLossMultiClass, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = args.get("alpha", 0.75)  # Higher weight for positive samples (foreground)
        self.gamma = args.get("gamma", 2.0)

        self.cls_weight = args["cls_weight"]
        self.reg_coe = args["reg"]
        self.obj_weight = args.get("obj_weight", 1.0)  # 添加 objectness 损失权重
        self.loss_dict = {}
        self.cls_num = args["num_class"]

    def forward(self, output_dict, target_dict, prefix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        rm = output_dict["rm{}".format(prefix)]
        psm = output_dict["psm{}".format(prefix)]
        targets = target_dict["targets"]

        # Model outputs
        # psm: [B, A*C, H, W]
        Bm, AC, Hm, Wm = psm.shape
        C = self.cls_num
        assert AC % C == 0, f"psm channels {AC} not divisible by num_class {C}"
        A = AC // C

        cls_preds = psm.permute(0, 2, 3, 1).contiguous()  # [B, H, W, A*C]

        # Labels from dataset are typically [B, H, W, A] but may be bigger due to collate quirks.
        # We always align them to model-output batch and spatial size.
        box_cls_labels = target_dict["pos_equal_one"]
        if box_cls_labels.shape[0] != Bm or box_cls_labels.shape[1] != Hm or box_cls_labels.shape[2] != Wm:
            box_cls_labels = box_cls_labels[:Bm, :Hm, :Wm, :]

        positives = box_cls_labels > 0
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()

        # If no positive anchors exist in this batch, reg loss is not meaningful and can
        # introduce large numerical noise; under DDP all ranks must make the same decision.
        zero_pos_local = (positives.sum() == 0).to(rm.device)
        if dist.is_available() and dist.is_initialized():
            flag = zero_pos_local.to(dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            zero_pos = bool(flag.item())
        else:
            zero_pos = bool(zero_pos_local.item())

        if zero_pos:
            # Return zero loss but mark skip so the trainer can drop this iter.
            # Keep tensors on the right device for AMP/DDP compatibility.
            # CRITICAL: requires_grad=True to avoid "does not require grad" error
            self.loss_dict[prefix + "skip_batch"] = True
            self.loss_dict[prefix + "conf_loss"] = torch.zeros((), device=rm.device, requires_grad=True)
            self.loss_dict[prefix + "reg_loss"] = torch.zeros((), device=rm.device, requires_grad=True)
            self.loss_dict[prefix + "total_loss"] = torch.zeros((), device=rm.device, requires_grad=True)
            return torch.zeros((), device=rm.device, requires_grad=True)

        # Clear skip marker for normal batches
        self.loss_dict[prefix + "skip_batch"] = False

        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        
        # Classification loss: ONLY computed on positive anchors
        # Negative anchors should NOT participate in cls loss, only objectness loss.
        # This avoids the "all-zero target" problem where model can't learn which class to predict.
        num_pos = positives.sum().float().clamp(min=1.0)

        # Classification loss
        cls_labels = target_dict["class_ids"]
        if cls_labels.shape[0] != Bm or cls_labels.shape[1] != Hm or cls_labels.shape[2] != Wm:
            cls_labels = cls_labels[:Bm, :Hm, :Wm, :]
        # Clamp class_ids to be valid
        cls_labels = torch.clamp(cls_labels, 0, self.cls_num - 1).long()
        
        one_hot_targets = torch.zeros(
            *list(cls_labels.shape), self.cls_num,
            dtype=cls_preds.dtype, device=cls_labels.device
        )
        one_hot_targets.scatter_(-1, cls_labels.unsqueeze(dim=-1).long(), 1.0)
        
        # Reshape to [B, H, W, A, C]
        cls_preds = cls_preds.view(Bm, Hm, Wm, A, C)
        one_hot_targets = one_hot_targets.view(Bm, Hm, Wm, A, C)
        
        # FIX: Only positive anchors participate in cls loss
        # Negative anchors only participate in objectness loss (see obj_loss below).
        # This avoids training on all-zero targets which makes the model unable to distinguish classes.
        positives_expanded = positives.view(Bm, Hm, Wm, A, 1)
        cls_weights = positives_expanded.float() / num_pos.clamp(min=1.0)
        
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)
        conf_loss = cls_loss_src.sum() / Bm  # Normalize by batch size
        conf_loss *= self.cls_weight

        # Regression loss
        rm = rm.permute(0, 2, 3, 1).contiguous()
        # Align targets to model batch if needed
        if targets.shape[0] != Bm:
            targets = targets[:Bm]
        # Reshape to [B, -1, 7]
        rm = rm.view(Bm, -1, 7)
        targets = targets.view(Bm, -1, 7)
        reg_weights = reg_weights.view(Bm, -1)

        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm, targets)
        
        loc_loss_src = self.reg_loss_func(
            box_preds_sin, reg_targets_sin, weights=reg_weights
        )
        reg_loss = loc_loss_src.sum() / Bm # Normalize by batch size
        reg_loss *= self.reg_coe

        # Objectness loss (Binary Cross Entropy with positive/negative balance)
        obj_loss = torch.zeros((), device=rm.device, requires_grad=True)
        if "obj{}".format(prefix) in output_dict:
            obj = output_dict["obj{}".format(prefix)]  # [B, A, H, W]
            obj_preds = obj.permute(0, 2, 3, 1).contiguous()  # [B, H, W, A]
            
            # pos_equal_one: [B, H, W, A], values 0 or 1
            pos_mask = target_dict["pos_equal_one"]
            if pos_mask.shape[0] != Bm or pos_mask.shape[1] != Hm or pos_mask.shape[2] != Wm:
                pos_mask = pos_mask[:Bm, :Hm, :Wm, :]
            
            # Balanced BCE: compute positive and negative losses separately
            # then average each by their own count to avoid class imbalance
            obj_preds_sigmoid = torch.sigmoid(obj_preds)
            
            # Positive samples loss (where pos_mask == 1)
            pos_loss = -torch.log(obj_preds_sigmoid + 1e-6)
            # Negative samples loss (where pos_mask == 0)
            neg_loss = -torch.log(1 - obj_preds_sigmoid + 1e-6)
            
            num_pos = pos_mask.sum().clamp(min=1.0)
            num_neg = (1 - pos_mask).sum().clamp(min=1.0)
            
            # Weight positive and negative equally (each normalized by their count)
            pos_obj_loss = (pos_loss * pos_mask).sum() / num_pos
            neg_obj_loss = (neg_loss * (1 - pos_mask)).sum() / num_neg
            
            # Combine: 50% positive + 50% negative (balanced)
            obj_loss = (pos_obj_loss + neg_obj_loss) * 0.5 * self.obj_weight

        total_loss = reg_loss + conf_loss + obj_loss

        self.loss_dict.update({
            f'total_loss{prefix}': total_loss.item(),
            f'reg_loss{prefix}': reg_loss.item(),
            f'conf_loss{prefix}': conf_loss.item(),
            f'obj_loss{prefix}': obj_loss.item() if isinstance(obj_loss, torch.Tensor) else obj_loss,
        })

        return total_loss

    def cls_loss_func(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Focal loss implementation.
        """
        pt = torch.sigmoid(input)
        focal_weight = (target * self.alpha + (1 - target) * (1 - self.alpha)) * torch.pow(torch.abs(target - pt), self.gamma)
        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)
        loss = focal_weight * bce_loss
        
        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = (
            torch.clamp(input, min=0)
            - input * target
            + torch.log1p(torch.exp(-torch.abs(input)))
        )
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim : dim + 1]) * torch.cos(
            boxes2[..., dim : dim + 1]
        )
        rad_tg_encoding = torch.cos(boxes1[..., dim : dim + 1]) * torch.sin(
            boxes2[..., dim : dim + 1]
        )

        boxes1 = torch.cat(
            [boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1 :]], dim=-1
        )
        boxes2 = torch.cat(
            [boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1 :]], dim=-1
        )
        return boxes1, boxes2

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n**2 / beta, n - 0.5 * beta)
        return loss

    def logging(self, epoch, batch_id, batch_len, writer=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        print_msg = "[epoch {}][{}/{}] ".format(epoch, batch_id + 1, batch_len)
        for k, v in self.loss_dict.items():
            print_msg += "|| {}: {:.4f} ".format(k, v)

        if writer:
            for k, v in self.loss_dict.items():
                writer.add_scalar(k, v, epoch * batch_len + batch_id)
                
        return print_msg