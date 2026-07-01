import os
import glob
import pickle
import numpy as np
from collections import Counter

MODEL_DIR = "/data1/wangyh/heal_framework/opencood/logs/airv2x_HEAL_collab_lidar/stage2_mamba_lossfix_2gpu_nohup_final__2026_03_23_07_58_52"
PRED_ROOT = os.path.join(MODEL_DIR, "preds", "val")


def safe_load_pickle(path: str):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def to_numpy(x):
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def flatten_labels(labels):
    vals = []
    if labels is None:
        return vals
    if isinstance(labels, (list, tuple)):
        for it in labels:
            vals.extend(flatten_labels(it))
        return vals
    arr = to_numpy(labels).reshape(-1)
    out = []
    for v in arr.tolist():
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def flatten_scores(scores):
    vals = []
    if scores is None:
        return vals
    if isinstance(scores, (list, tuple)):
        for it in scores:
            vals.extend(flatten_scores(it))
        return vals
    arr = to_numpy(scores).reshape(-1)
    out = []
    for v in arr.tolist():
        try:
            out.append(float(v))
        except Exception:
            continue
    return out


def flatten_boxes(boxes):
    out = []
    if boxes is None:
        return out
    if isinstance(boxes, (list, tuple)):
        for it in boxes:
            out.extend(flatten_boxes(it))
        return out

    arr = to_numpy(boxes)

    # [N, 8, 3]
    if arr.ndim == 3 and arr.shape[1:] == (8, 3):
        out.extend(arr)
    # [N, 7] / [N, >=6]
    elif arr.ndim == 2:
        out.extend(arr)
    # [8,3] 单框
    elif arr.ndim == 2 and arr.shape == (8, 3):
        out.append(arr)

    return out


def parse_leaf_record(rec: dict):
    labels = None
    scores = None
    boxes = None

    for k in ["pred_labels", "pred_label", "class", "label", "labels"]:
        if k in rec:
            labels = rec[k]
            break

    for k in ["pred_score", "pred_scores", "score", "scores", "confidence"]:
        if k in rec:
            scores = rec[k]
            break

    for k in ["pred_boxes3d", "pred_boxes", "pred_box", "boxes", "bbox_3d"]:
        if k in rec:
            boxes = rec[k]
            break

    return labels, scores, boxes


def iter_leaf_dicts(obj):
    """
    递归遍历:
    - 外层 dict[int -> dict]（你的格式）
    - 或 list/tuple 嵌套
    """
    if isinstance(obj, dict):
        # 如果当前就是叶子记录（包含预测字段）
        keys = set(obj.keys())
        if any(k in keys for k in ["pred_labels", "pred_label", "class", "label", "labels",
                                   "pred_score", "pred_scores", "score", "scores",
                                   "pred_boxes3d", "pred_boxes", "pred_box", "boxes", "bbox_3d"]):
            yield obj
        else:
            for v in obj.values():
                yield from iter_leaf_dicts(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from iter_leaf_dicts(v)
    else:
        return


def summarize_sizes(boxes):
    lwh = []
    for b in boxes:
        b = np.asarray(b)
        if b.ndim == 2 and b.shape == (8, 3):
            mins = b.min(axis=0)
            maxs = b.max(axis=0)
            size = maxs - mins
            lwh.append(size.tolist())
        elif b.ndim == 1 and b.size >= 6:
            lwh.append([b[3], b[4], b[5]])

    if not lwh:
        return None

    arr = np.asarray(lwh, dtype=np.float32)
    return {
        "mean_lwh": arr.mean(axis=0).tolist(),
        "p50_lwh": np.percentile(arr, 50, axis=0).tolist(),
        "p90_lwh": np.percentile(arr, 90, axis=0).tolist(),
    }


def main():
    pkl_files = sorted(glob.glob(os.path.join(PRED_ROOT, "**", "predictions.pkl"), recursive=True))
    if not pkl_files:
        raise FileNotFoundError(f"No predictions.pkl under: {PRED_ROOT}")

    all_labels, all_scores, all_boxes = [], [], []
    per_file_pred_count = []

    print(f"[INFO] found prediction files: {len(pkl_files)}")
    print(f"[INFO] sample file: {pkl_files[0]}")

    for i, p in enumerate(pkl_files):
        data = safe_load_pickle(p)

        file_labels, file_scores, file_boxes = [], [], []
        leaf_count = 0

        for leaf in iter_leaf_dicts(data):
            leaf_count += 1
            labels, scores, boxes = parse_leaf_record(leaf)
            file_labels.extend(flatten_labels(labels))
            file_scores.extend(flatten_scores(scores))
            file_boxes.extend(flatten_boxes(boxes))

        # 聚合
        all_labels.extend(file_labels)
        all_scores.extend(file_scores)
        all_boxes.extend(file_boxes)
        per_file_pred_count.append(len(file_boxes) if len(file_boxes) > 0 else len(file_labels))

        if (i + 1) % 200 == 0:
            print(f"[INFO] parsed {i+1}/{len(pkl_files)} files...")

    print("========== Prediction Diagnostics (All Files) ==========")
    print(f"files: {len(pkl_files)}")
    print(f"total_preds(by boxes/labels): boxes={len(all_boxes)}, labels={len(all_labels)}")

    if per_file_pred_count:
        arr = np.asarray(per_file_pred_count, dtype=np.float32)
        print(f"preds/file: mean={arr.mean():.2f}, p50={np.percentile(arr,50):.2f}, p90={np.percentile(arr,90):.2f}")

    if all_labels:
        cnt = Counter(all_labels)
        print("label_hist:", dict(sorted(cnt.items(), key=lambda x: x[0])))
    else:
        print("label_hist: EMPTY")

    if all_scores:
        s = np.asarray(all_scores, dtype=np.float32)
        print(f"score_stats: min={s.min():.4f}, p25={np.percentile(s,25):.4f}, p50={np.percentile(s,50):.4f}, p75={np.percentile(s,75):.4f}, max={s.max():.4f}")
    else:
        print("score_stats: EMPTY")

    size_stats = summarize_sizes(all_boxes)
    if size_stats is not None:
        print("box_size_stats(l,w,h):", size_stats)
    else:
        print("box_size_stats: unavailable (box format not recognized or boxes missing)")


if __name__ == "__main__":
    main()