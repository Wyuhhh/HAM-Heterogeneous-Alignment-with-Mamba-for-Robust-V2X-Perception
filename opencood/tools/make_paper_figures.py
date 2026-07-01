import os
import glob
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_debug(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    pred = np.asarray(d.get("pred_box_tensor", []))
    score = np.asarray(d.get("pred_score", []), dtype=float)
    gt = np.asarray(d.get("gt_box_tensor", []))
    return pred, score, gt


def draw_box(ax, box8x3, color, lw=1.5, alpha=0.9):
    pts = np.asarray(box8x3)[:4, :2]
    pts = np.vstack([pts, pts[0]])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=alpha)


def draw_scene(ax, pred, score, gt, score_thre=0.97, title=""):
    for g in gt:
        draw_box(ax, g, color="limegreen", lw=1.8, alpha=0.95)

    keep = score >= score_thre
    for b in pred[keep]:
        draw_box(ax, b, color="deepskyblue", lw=1.2, alpha=0.85)

    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.tick_params(labelsize=7)


def pick_frames(debug_dir, score_thre, n_success=6, n_failure=4):
    files = sorted(glob.glob(os.path.join(debug_dir, "*/debug_eval.pkl")))
    rows = []
    for p in files:
        fid = os.path.basename(os.path.dirname(p))
        pred, score, gt = load_debug(p)
        keep = int((score >= score_thre).sum()) if len(score) > 0 else 0
        abs_diff = abs(keep - len(gt))
        rows.append((fid, p, keep, len(gt), abs_diff))

    success = sorted([r for r in rows if r[3] >= 6], key=lambda x: x[4])[:n_success]
    failure = sorted(rows, key=lambda x: -x[4])[:n_failure]
    return success, failure


def make_grid(paths, out_png, score_thre, ncols=3, fig_title=""):
    n = len(paths)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if idx >= n:
            ax.axis("off")
            continue
        fid, p, keep, gt_n, abs_diff = paths[idx]
        pred, score, gt = load_debug(p)
        draw_scene(
            ax,
            pred,
            score,
            gt,
            score_thre=score_thre,
            title=f"frame={fid} | pred={keep} gt={gt_n} | |Δ|={abs_diff}",
        )

    fig.suptitle(fig_title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_dir", type=str, required=True, help=".../npy/debug_eval")
    parser.add_argument("--out_dir", type=str, required=True, help="output figure dir")
    parser.add_argument("--score_thre", type=float, default=0.97)
    parser.add_argument("--n_success", type=int, default=6)
    parser.add_argument("--n_failure", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    success, failure = pick_frames(
        args.debug_dir,
        score_thre=args.score_thre,
        n_success=args.n_success,
        n_failure=args.n_failure,
    )

    fig47 = os.path.join(args.out_dir, "Fig4_7_success.png")
    fig48 = os.path.join(args.out_dir, "Fig4_8_failure.png")

    make_grid(
        success,
        fig47,
        score_thre=args.score_thre,
        ncols=3,
        fig_title="Fig.4-7 Qualitative Success Cases (HAM)",
    )
    make_grid(
        failure,
        fig48,
        score_thre=args.score_thre,
        ncols=2,
        fig_title="Fig.4-8 Failure Cases and Error Types (HAM)",
    )

    print("Saved:")
    print(" ", fig47)
    print(" ", fig48)
    print("\nSelected success frames:", [x[0] for x in success])
    print("Selected failure frames:", [x[0] for x in failure])


if __name__ == "__main__":
    main()
