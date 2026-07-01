# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# Modifier: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import pickle

from opencood.tools import train_utils
from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.utils.scenario_utils import scenarios_params
from opencood.utils import box_utils


def _unpack_post_process_outputs(post_out):
    """
    Robustly unpack outputs from dataset.post_process variants.

    Supported common formats:
    1) (pred_box_tensor, pred_score, gt_box_tensor)
    2) (pred_box_tensor, pred_score, pred_labels, pred_boxes3d,
        gt_box_tensor, gt_class_label_list, gt_track_list)
    3) tuple/list with >=3 elements (first 3 interpreted as pred_box, pred_score, gt_box fallback)

    Returns
    -------
    pred_box_tensor, pred_score, gt_box_tensor, pred_labels, pred_boxes3d
    """
    if not isinstance(post_out, (list, tuple)):
        raise TypeError(
            f"dataset.post_process must return tuple/list, got {type(post_out)}"
        )

    if len(post_out) < 3:
        raise ValueError(
            f"dataset.post_process returns too few values: len={len(post_out)}"
        )

    # Case A: classic 3-return
    if len(post_out) == 3:
        pred_box_tensor, pred_score, gt_box_tensor = post_out
        return pred_box_tensor, pred_score, gt_box_tensor, None, None

    # Case B: airv2x-style 7-return (or >=5)
    if len(post_out) >= 5:
        pred_box_tensor = post_out[0]
        pred_score = post_out[1]
        pred_labels = post_out[2]
        pred_boxes3d = post_out[3]
        gt_box_tensor = post_out[4]
        return pred_box_tensor, pred_score, gt_box_tensor, pred_labels, pred_boxes3d

    # Case C: uncommon 4-return fallback
    pred_box_tensor = post_out[0]
    pred_score = post_out[1]
    gt_box_tensor = post_out[2]
    pred_labels = post_out[3]
    return pred_box_tensor, pred_score, gt_box_tensor, pred_labels, None


def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    post_out = dataset.post_process(batch_data, output_dict)
    pred_box_tensor, pred_score, gt_box_tensor, _, _ = _unpack_post_process_outputs(post_out)

    return pred_box_tensor, pred_score, gt_box_tensor, output_dict


def inference_no_fusion(batch_data, model, dataset):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict_ego = OrderedDict()

    output_dict_ego["ego"] = model(batch_data["ego"])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process_no_fusion(
        batch_data,  # only for late fusion dataset
        output_dict_ego,
    )

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early/intermediate fusion compatible with multiple dataset post_process signatures.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
    pred_score : torch.Tensor
    gt_box_tensor : torch.Tensor
    pred_boxes3d : torch.Tensor or None
    """
    output_dict = OrderedDict()
    cav_content = batch_data["ego"]
    output_dict["ego"] = model(cav_content)

    post_out = dataset.post_process(batch_data, output_dict)
    pred_box_tensor, pred_score, gt_box_tensor, _, pred_boxes3d = _unpack_post_process_outputs(post_out)

    return pred_box_tensor, pred_score, gt_box_tensor, pred_boxes3d


def inference_intermediate_fusion_withcomm(batch_data, model, dataset):
    """
    Model inference for intermediate fusion with communication statistics.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data["ego"]
    output_dict["ego"] = model(cav_content)

    post_out = dataset.post_process(batch_data, output_dict)
    pred_box_tensor, pred_score, gt_box_tensor, _, _ = _unpack_post_process_outputs(post_out)

    comm_rates = output_dict["ego"]["comm_rate"]
    mask = output_dict["ego"]["mask"]
    each_mask = output_dict["ego"]["each_mask"]
    return pred_box_tensor, pred_score, gt_box_tensor, comm_rates, mask, each_mask
    # return pred_box_tensor, pred_score, gt_box_tensor, comm_rates, mask


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for intermediate fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """

    return inference_early_fusion(batch_data, model, dataset)


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, "%04d_pcd.npy" % timestamp), pcd_np)
    np.save(os.path.join(save_path, "%04d_pred.npy" % timestamp), pred_np)
    np.save(os.path.join(save_path, "%04d_gt.npy" % timestamp), gt_np)


# ====================================================================================================
# airv2x: (1) multiclass (2) segmentation (3) tracking
# ====================================================================================================

def inference_intermediate_fusion_airv2x(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """

    return inference_early_fusion_airv2x(batch_data, model, dataset)


def inference_early_fusion_airv2x(batch_data, model, dataset):
    """
    Model inference for airv2x multiclass/tracking style datasets.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor, pred_score, pred_labels, pred_boxes3d,
    gt_box_tensor, gt_class_label_list, gt_track_list
    """
    output_dict = OrderedDict()
    cav_content = batch_data["ego"]
    output_dict["ego"] = model(cav_content)

    post_out = dataset.post_process(batch_data, output_dict)
    if not isinstance(post_out, (list, tuple)) or len(post_out) < 7:
        raise ValueError(
            f"airv2x expects >=7 outputs from post_process, got {type(post_out)} len={len(post_out) if isinstance(post_out, (list, tuple)) else 'N/A'}"
        )

    pred_box_tensor, pred_score, pred_labels, pred_boxes3d, gt_box_tensor, gt_class_label_list, gt_track_list = post_out[:7]

    return pred_box_tensor, pred_score, pred_labels, pred_boxes3d, gt_box_tensor, gt_class_label_list, gt_track_list


# segmentation
def inference_intermediate_fusion_airv2x_segmentation(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """

    return inference_early_fusion_airv2x_segmentation(batch_data, model, dataset)


def inference_early_fusion_airv2x_segmentation(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data["ego"]
    output_dict["ego"] = model(cav_content)

    pred_dynamic_seg_map, pred_static_seg_map, gt_dynamic_seg_map, gt_static_seg_map = dataset.post_process_seg(
        batch_data, output_dict
    )
    return pred_dynamic_seg_map, pred_static_seg_map, gt_dynamic_seg_map, gt_static_seg_map


def save_preds_airv2x(pred_box_tensor, pred_score, pred_boxes3d, batch_data, save_path, pred_labels=0):
    """
    Save predictions in AirV2X expected per-frame format:
      {
        idx: {
          'location': [...],
          'extent': [...],
          'class': int,
          'confidence': float
        },
        'ego_lidar_pose': [...]
      }

    Robustness:
    - supports tensor / numpy / list inputs
    - handles empty prediction safely
    - avoids shape mismatch between labels/scores and boxes
    """
    # -------- to numpy --------
    if isinstance(pred_score, torch.Tensor):
        pred_score = torch_tensor_to_numpy(pred_score)
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = torch_tensor_to_numpy(pred_labels)
    if isinstance(pred_boxes3d, torch.Tensor):
        pred_boxes3d = torch_tensor_to_numpy(pred_boxes3d)

    if pred_score is not None:
        pred_score = np.asarray(pred_score).reshape(-1)
    if pred_labels is not None and not isinstance(pred_labels, (int, np.integer)):
        pred_labels = np.asarray(pred_labels).reshape(-1)
    if pred_boxes3d is not None:
        pred_boxes3d = np.asarray(pred_boxes3d)

    metadata_path = batch_data["ego"]["metadata_path_list"][0]
    ego_lidar_pose = batch_data["ego"]["ego_lidar_pose_list"][0]

    meta_data_root = os.path.join(*metadata_path.split("/")[-5:-1])
    save_dir = os.path.join(save_path, "preds", meta_data_root)
    os.makedirs(save_dir, exist_ok=True)

    metadata_dict = dict()

    # -------- empty guard --------
    if pred_boxes3d is None or pred_boxes3d.size == 0:
        metadata_dict["ego_lidar_pose"] = ego_lidar_pose
        with open(os.path.join(save_dir, "predictions.pkl"), "wb") as f:
            pickle.dump(metadata_dict, f)
        return

    # convert to format expected by downstream airv2x evaluation
    # NOTE: this utility defines the final layout contract for location/extent.
    pred = box_utils.convert_boxes_to_format(pred_boxes3d)
    pred = np.asarray(pred)

    if pred.ndim != 2:
        # fallback: force 2D view if possible
        pred = pred.reshape(pred.shape[0], -1)

    # Existing evaluator contract in your pipeline:
    # location = first 6 dims, extent = remaining dims
    # (keep this contract stable to avoid downstream parser breakage)
    split_idx = min(6, pred.shape[1])
    location = pred[:, :split_idx]
    extent = pred[:, split_idx:]

    n = pred.shape[0]

    # normalize labels
    if isinstance(pred_labels, (int, np.integer)) or pred_labels is None:
        # default label = provided scalar / fallback 0
        label_scalar = int(pred_labels) if pred_labels is not None else 0
        labels_arr = np.full((n,), label_scalar, dtype=np.int64)
    else:
        labels_arr = pred_labels
        if labels_arr.shape[0] != n:
            m = min(labels_arr.shape[0], n)
            tmp = np.zeros((n,), dtype=np.int64)
            if m > 0:
                tmp[:m] = labels_arr[:m].astype(np.int64)
            labels_arr = tmp
        else:
            labels_arr = labels_arr.astype(np.int64)

    # normalize scores
    if pred_score is None:
        scores_arr = np.ones((n,), dtype=np.float32)
    else:
        scores_arr = pred_score
        if scores_arr.shape[0] != n:
            m = min(scores_arr.shape[0], n)
            tmp = np.ones((n,), dtype=np.float32)
            if m > 0:
                tmp[:m] = scores_arr[:m].astype(np.float32)
            scores_arr = tmp
        else:
            scores_arr = scores_arr.astype(np.float32)

    for idx in range(n):
        metadata = {
            "location": location[idx].tolist(),
            "extent": extent[idx].tolist(),
            "class": int(labels_arr[idx]),
            "confidence": float(scores_arr[idx]),
        }
        metadata_dict[idx] = metadata

    metadata_dict["ego_lidar_pose"] = ego_lidar_pose
    with open(os.path.join(save_dir, "predictions.pkl"), "wb") as f:
        pickle.dump(metadata_dict, f)


def combine_stat(combined_stat, stat):
    """
    Combine statistics from different scenarios.

    Parameters
    ----------
    combined_stat : dict
        Combined statistics.
    stat : dict
        Statistics to be combined.

    Returns
    -------
    dict
        Combined statistics.
    """
    for key, value in stat.items():
        if key in combined_stat:
            combined_stat[key]["tp"].extend(value["tp"])
            combined_stat[key]["fp"].extend(value["fp"])
            combined_stat[key]["gt"] += value["gt"]
            combined_stat[key]["score"].extend(value["score"])
        else:
            combined_stat[key] = value
    return combined_stat


def combine_stat_by_scenarios(result_stat_dict):
    result_stat_init = lambda: {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    combined_stat = defaultdict(result_stat_init)

    for scenario, stat in result_stat_dict.items():
        if scenario not in scenarios_params:
            print(f"Warning: scenario {scenario} not in scenarios_params, skipping...")
            continue
        for key, value in scenarios_params[scenario].items():
            if value:
                combined_stat[key] = combine_stat(combined_stat[key], stat)
        combined_stat["all"] = combine_stat(combined_stat["all"], stat)
    return combined_stat


def combine_stat_by_scenarios_segmentation(result_stat_dict):

    result_stat_init = lambda: {
        "gt_dynamic_seg_map_list": [],
        "pred_dynamic_seg_map_list": [],
        "gt_static_seg_map_list": [],
        "pred_static_seg_map_list": [],
    }
    combined_stat = defaultdict(result_stat_init)

    for scenario, stat in result_stat_dict.items():
        if scenario not in scenarios_params:
            print(f"Warning: scenario {scenario} not in scenarios_params, skipping...")
            continue
        for key, value in scenarios_params[scenario].items():
            if value:
                combined_stat[key]["gt_dynamic_seg_map_list"].extend(stat["gt_dynamic_seg_map_list"])
                combined_stat[key]["pred_dynamic_seg_map_list"].extend(stat["pred_dynamic_seg_map_list"])
                combined_stat[key]["gt_static_seg_map_list"].extend(stat["gt_static_seg_map_list"])
                combined_stat[key]["pred_static_seg_map_list"].extend(stat["pred_static_seg_map_list"])

        combined_stat["all"]["gt_dynamic_seg_map_list"].extend(stat["gt_dynamic_seg_map_list"])
        combined_stat["all"]["pred_dynamic_seg_map_list"].extend(stat["pred_dynamic_seg_map_list"])
        combined_stat["all"]["gt_static_seg_map_list"].extend(stat["gt_static_seg_map_list"])
        combined_stat["all"]["pred_static_seg_map_list"].extend(stat["pred_static_seg_map_list"])

    return combined_stat