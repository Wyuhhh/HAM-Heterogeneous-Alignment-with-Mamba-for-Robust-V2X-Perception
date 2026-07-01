# -*- coding: utf-8 -*-
import argparse
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils


def test_parser():
    parser = argparse.ArgumentParser(description="inference script")
    parser.add_argument("--model_dir", type=str, required=True, help="Continued training path")
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="intermediate",
        help="no, no_w_uncertainty, late, early or intermediate",
    )
    parser.add_argument("--save_vis", default=False, action="store_true", help="whether to save visualization result")
    parser.add_argument("--save_vis_n", type=int, default=0, help="save how many numbers of visualization result")
    parser.add_argument("--save_npy", action="store_true", help="whether to save prediction and gt result in npy file")
    parser.add_argument(
        "--eval_epoch",
        type=int,
        default=-1,
        help="specify one checkpoint epoch to evaluate; -1 means latest",
    )
    parser.add_argument(
        "--eval_best_epoch",
        action="store_true",
        default=False,
        help="evaluate best checkpoint instead of specific epoch/latest",
    )
    parser.add_argument("--comm_thre", type=float, default=None, help="communication threshold override")
    parser.add_argument("--save_pred", action="store_true", help="save per-frame predictions for airv2x style evaluation")
    parser.add_argument("--save_debug_eval", action="store_true", help="save debug sidecar pkl")
    parser.add_argument("--max_samples", type=int, default=0, help="max number of samples to infer, 0 means all")
    parser.add_argument(
        "--score_thre",
        type=float,
        default=0.0,
        help="score threshold before TP/FP evaluation (diagnostic/useful for FP-heavy cases)",
    )
    opt = parser.parse_args()
    return opt


def _safe_get_comm_value(result_stat):
    if not isinstance(result_stat, dict):
        return 0.0
    for k in ["comm_rate", "comm_rates"]:
        if k in result_stat:
            v = result_stat[k]
            try:
                if isinstance(v, list):
                    return float(np.mean(v)) if len(v) > 0 else 0.0
                return float(v)
            except Exception:
                return 0.0
    return 0.0


def _to_numpy_if_tensor(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _len_safe(x):
    if x is None:
        return 0
    try:
        arr = np.asarray(_to_numpy_if_tensor(x))
        if arr.ndim == 0:
            return 1
        return int(arr.shape[0])
    except Exception:
        pass
    try:
        return len(x)
    except Exception:
        return 0


def _is_box_array_like(x):
    """
    可用于 save_pred 的 box 判定:
      [N,8,3] 或 [N,7+] 认为有效
    """
    if x is None:
        return False
    try:
        arr = np.asarray(_to_numpy_if_tensor(x))
        if arr.size == 0:
            return False
        if arr.ndim == 3 and arr.shape[1:] == (8, 3):
            return True
        if arr.ndim == 2 and arr.shape[1] >= 7:
            return True
    except Exception:
        return False
    return False


def _extract_from_nested(obj, keys):
    """
    从 dict/list/tuple 嵌套结构中递归按 key 提取第一个非空值。
    """
    if obj is None:
        return None

    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        for v in obj.values():
            got = _extract_from_nested(v, keys)
            if got is not None:
                return got
        return None

    if isinstance(obj, (list, tuple)):
        for v in obj:
            got = _extract_from_nested(v, keys)
            if got is not None:
                return got
        return None

    return None


def _safe_unpack_postprocess_for_save(pp_out):
    """
    只用于 save_pred，不干扰 eval 主链路。
    返回: pred_labels, pred_boxes3d
    """
    pred_labels = None
    pred_boxes3d = None

    # 1) 常见 tuple/list
    if isinstance(pp_out, (list, tuple)):
        # 常见7项: [pred_box, pred_score, pred_labels, pred_boxes3d, gt_box, ...]
        if len(pp_out) >= 4:
            if pp_out[2] is not None:
                pred_labels = pp_out[2]
            if pp_out[3] is not None:
                pred_boxes3d = pp_out[3]

    # 2) dict 或深层嵌套兜底
    if pred_labels is None:
        pred_labels = _extract_from_nested(pp_out, ["pred_labels", "label", "labels", "pred_cls"])
    if pred_boxes3d is None:
        pred_boxes3d = _extract_from_nested(pp_out, ["pred_boxes3d", "pred_boxes", "pred_box_tensor", "pred_box"])

    return pred_labels, pred_boxes3d


def _normalize_1d(arr_like, n, default_value, dtype):
    """
    将 labels/scores 对齐到长度 n
    """
    if n <= 0:
        return None
    if arr_like is None:
        return np.full((n,), default_value, dtype=dtype)
    arr = np.asarray(_to_numpy_if_tensor(arr_like)).reshape(-1)
    if arr.shape[0] == n:
        return arr.astype(dtype, copy=False)
    m = min(arr.shape[0], n)
    if m <= 0:
        return np.full((n,), default_value, dtype=dtype)
    out = np.full((n,), default_value, dtype=dtype)
    out[:m] = arr[:m].astype(dtype, copy=False)
    return out


def _filter_by_score(pred_box_tensor, pred_score, score_thre):
    """
    根据 score 阈值过滤预测框，用于诊断/抑制FP。
    """
    if pred_box_tensor is None or pred_score is None or score_thre <= 0:
        return pred_box_tensor, pred_score, 0, 0

    before_n = _len_safe(pred_score)

    try:
        if torch.is_tensor(pred_score):
            keep = pred_score >= float(score_thre)
            pred_score = pred_score[keep]
            if torch.is_tensor(pred_box_tensor):
                pred_box_tensor = pred_box_tensor[keep]
            else:
                pred_box_tensor = np.asarray(pred_box_tensor)[keep.detach().cpu().numpy()]
        else:
            ps = np.asarray(pred_score).reshape(-1)
            keep = ps >= float(score_thre)
            pred_score = ps[keep]
            if torch.is_tensor(pred_box_tensor):
                keep_t = torch.from_numpy(keep).to(pred_box_tensor.device)
                pred_box_tensor = pred_box_tensor[keep_t]
            else:
                pred_box_tensor = np.asarray(pred_box_tensor)[keep]
    except Exception:
        # 过滤失败时回退原值
        return pred_box_tensor, pred_score, before_n, before_n

    after_n = _len_safe(pred_score)
    return pred_box_tensor, pred_score, before_n, after_n


def _save_debug_eval_sidecar(saved_path, i, pred_box_tensor, pred_score, gt_box_tensor, pred_label=None, gt_label=None):
    sidecar_root = os.path.join(saved_path, "debug_eval", f"{i:06d}")
    os.makedirs(sidecar_root, exist_ok=True)
    payload = {
        "pred_box_tensor": _to_numpy_if_tensor(pred_box_tensor),
        "pred_score": _to_numpy_if_tensor(pred_score),
        "pred_label": _to_numpy_if_tensor(pred_label),
        "gt_box_tensor": _to_numpy_if_tensor(gt_box_tensor),
        "gt_label_tensor": _to_numpy_if_tensor(gt_label),
        "meta": {"iter": i},
    }
    with open(os.path.join(sidecar_root, "debug_eval.pkl"), "wb") as f:
        pickle.dump(payload, f)


if __name__ == "__main__":
    opt = test_parser()

    assert opt.fusion_method in ["late", "early", "intermediate", "no", "no_w_uncertainty"], "fusion method is not supported"

    hypes = yaml_utils.load_yaml(None, opt)

    if "test_dir" in hypes:
        hypes["validate_dir"] = hypes["test_dir"]

    if opt.comm_thre is not None:
        hypes["model"]["args"]["fusion_args"]["communication"]["thre"] = opt.comm_thre

    print("Dataset Building")
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    print("Creating Model")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.cuda()

    # load model
    if opt.eval_best_epoch:
        print("Loading best checkpoint ...")
        _, model = train_utils.load_saved_model(opt.model_dir, model)
        eval_epoch = "best"
    else:
        if opt.eval_epoch is not None and opt.eval_epoch >= 0:
            print(f"Loading specified epoch checkpoint: {opt.eval_epoch}")
            _, model = train_utils.load_model(opt.model_dir, model, opt.eval_epoch, start_from_best=False)
            eval_epoch = opt.eval_epoch
        else:
            print("Loading latest checkpoint ...")
            _, model = train_utils.load_saved_model(opt.model_dir, model)
            eval_epoch = "latest"

    model.eval()

    result_stat = OrderedDict()
    result_stat[0.3] = {"tp": [], "fp": [], "gt": 0, "score": []}
    result_stat[0.5] = {"tp": [], "fp": [], "gt": 0, "score": []}
    result_stat[0.7] = {"tp": [], "fp": [], "gt": 0, "score": []}
    result_stat["comm_rate"] = []

    infer_info = f"{hypes['name']}_{opt.fusion_method}"
    saved_path = os.path.join(opt.model_dir, "npy")
    os.makedirs(saved_path, exist_ok=True)

    print("Start inference ...")
    print(f"[INFO] eval score_thre = {opt.score_thre:.3f}")

    for i, batch_data in enumerate(tqdm(data_loader)):
        if opt.max_samples > 0 and i >= opt.max_samples:
            break
        if batch_data is None:
            continue

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            # ---------- 1) 先跑 inference ----------
            if opt.fusion_method == "late":
                infer_out = inference_utils.inference_late_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == "early":
                infer_out = inference_utils.inference_early_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == "intermediate":
                infer_out = inference_utils.inference_intermediate_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == "no":
                infer_out = inference_utils.inference_no_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == "no_w_uncertainty":
                infer_out = inference_utils.inference_no_fusion_w_uncertainty(batch_data, model, opencood_dataset)
            else:
                raise NotImplementedError

            # ---------- 2) eval 专用字段 ----------
            pred_box_tensor, pred_score, gt_box_tensor = None, None, None
            output_dict = None
            gt_label_tensor = None

            if isinstance(infer_out, (list, tuple)):
                if len(infer_out) >= 3:
                    pred_box_tensor, pred_score, gt_box_tensor = infer_out[:3]
                if len(infer_out) >= 4:
                    output_dict = infer_out[3]
            elif isinstance(infer_out, dict):
                pred_box_tensor = infer_out.get("pred_box_tensor", None)
                pred_score = infer_out.get("pred_score", infer_out.get("pred_scores", None))
                gt_box_tensor = infer_out.get("gt_box_tensor", None)
                output_dict = infer_out.get("output_dict", None)
                gt_label_tensor = infer_out.get("gt_label", infer_out.get("gt_labels", None))

            # 只在缺失时才尝试 post_process 回填 eval 三元组
            pp_out = None
            if output_dict is not None:
                try:
                    pp_out = opencood_dataset.post_process(batch_data, output_dict)
                    if (pred_box_tensor is None) or (pred_score is None) or (gt_box_tensor is None):
                        if isinstance(pp_out, (list, tuple)) and len(pp_out) >= 3:
                            if pred_box_tensor is None:
                                pred_box_tensor = pp_out[0]
                            if pred_score is None:
                                pred_score = pp_out[1]
                            if gt_box_tensor is None:
                                gt_box_tensor = pp_out[2]
                        elif isinstance(pp_out, dict):
                            if pred_box_tensor is None:
                                pred_box_tensor = pp_out.get("pred_box_tensor", pp_out.get("pred_boxes", None))
                            if pred_score is None:
                                pred_score = pp_out.get("pred_score", pp_out.get("pred_scores", None))
                            if gt_box_tensor is None:
                                gt_box_tensor = pp_out.get("gt_box_tensor", pp_out.get("gt_boxes", None))
                            if gt_label_tensor is None:
                                gt_label_tensor = pp_out.get("gt_label", pp_out.get("gt_labels", None))
                except Exception as e:
                    if i < 5:
                        print(f"[WARN] post_process failed at iter={i}: {e}")

            # ---------- 3) eval ----------
            if pred_box_tensor is not None and pred_score is not None and gt_box_tensor is not None:
                # score threshold filter (diagnostic + practical)
                pred_box_eval, pred_score_eval, before_n, after_n = _filter_by_score(
                    pred_box_tensor, pred_score, opt.score_thre
                )

                if i % 100 == 0:
                    print(
                        f"[EVAL_DEBUG] iter={i} pred_before={before_n} pred_after={after_n} "
                        f"gt={_len_safe(gt_box_tensor)} thre={opt.score_thre}"
                    )

                try:
                    eval_utils.caluclate_tp_fp(pred_box_eval, pred_score_eval, gt_box_tensor, result_stat, 0.3)
                    eval_utils.caluclate_tp_fp(pred_box_eval, pred_score_eval, gt_box_tensor, result_stat, 0.5)
                    eval_utils.caluclate_tp_fp(pred_box_eval, pred_score_eval, gt_box_tensor, result_stat, 0.7)
                except Exception as e:
                    if i < 10:
                        print(f"[WARN] eval skipped at iter={i} due to shape issue: {e}")

            # ---------- 4) comm rate ----------
            try:
                if isinstance(infer_out, dict) and "comm_rates" in infer_out:
                    result_stat["comm_rate"].append(float(infer_out["comm_rates"]))
                else:
                    result_stat["comm_rate"].append(_safe_get_comm_value(infer_out))
            except Exception:
                result_stat["comm_rate"].append(0.0)

            # ---------- 5) save_pred ----------
            if opt.save_pred:
                pred_labels_pp, pred_boxes3d_pp = _safe_unpack_postprocess_for_save(pp_out)

                # boxes 回退链路：pred_boxes3d_pp -> pred_box_tensor -> infer_out深层
                pred_boxes3d_to_save = pred_boxes3d_pp
                if not _is_box_array_like(pred_boxes3d_to_save):
                    pred_boxes3d_to_save = pred_box_tensor
                if not _is_box_array_like(pred_boxes3d_to_save):
                    pred_boxes3d_to_save = _extract_from_nested(
                        infer_out, ["pred_boxes3d", "pred_boxes", "pred_box_tensor", "pred_box"]
                    )

                # labels / scores
                n_boxes = _len_safe(pred_boxes3d_to_save)
                pred_labels_to_save = _normalize_1d(pred_labels_pp, n_boxes, 1, np.int64)
                pred_score_to_save = _normalize_1d(pred_score, n_boxes, 1.0, np.float32)

                if i % 100 == 0:
                    print(
                        f"[SAVE_DEBUG] iter={i} "
                        f"n_boxes={n_boxes} "
                        f"n_labels={_len_safe(pred_labels_to_save)} "
                        f"n_scores={_len_safe(pred_score_to_save)} "
                        f"valid_boxes={_is_box_array_like(pred_boxes3d_to_save)}"
                    )

                try:
                    inference_utils.save_preds_airv2x(
                        pred_box_tensor,          # 保持签名兼容
                        pred_score_to_save,
                        pred_boxes3d_to_save,
                        batch_data,
                        saved_path,
                        pred_labels=pred_labels_to_save,
                    )
                except Exception as e:
                    if i < 10:
                        print(f"[WARN] save_preds_airv2x failed at iter={i}: {e}")

                if opt.save_debug_eval:
                    try:
                        _save_debug_eval_sidecar(
                            saved_path,
                            i,
                            pred_box_tensor,
                            pred_score,
                            gt_box_tensor,
                            pred_label=pred_labels_to_save,
                            gt_label=gt_label_tensor,
                        )
                    except Exception as e:
                        if i < 10:
                            print(f"[WARN] save debug_eval failed at iter={i}: {e}")

            # ---------- 6) optional npy ----------
            if opt.save_npy and pred_box_tensor is not None and gt_box_tensor is not None:
                npy_save_path = os.path.join(saved_path, f"{infer_info}_{i:06d}")
                os.makedirs(npy_save_path, exist_ok=True)
                inference_utils.save_prediction_gt(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data["ego"]["origin_lidar"][0],
                    i,
                    npy_save_path,
                )

    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, opt.model_dir, infer_info)
    comm_rate = float(np.mean(result_stat["comm_rate"])) if len(result_stat["comm_rate"]) > 0 else 0.0
    print(
        f"Epoch: {eval_epoch} | AP @0.3: {ap30:.4f} | AP @0.5: {ap50:.4f} | "
        f"AP @0.7: {ap70:.4f} | comm_rate: {comm_rate:.6f}"
    )