#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build fine-grained ignore_frames list from spike_cases.txt logs.

It scans one or many run folders under opencood/logs/** and aggregates
(scenario, timestamp) pairs.

Output is a YAML snippet you can copy into hypes yaml:

data_filter:
  train:
    ignore_frames:
      - {scenario: "2025_05_06_11_03_25", timestamp: "395"}

We keep timestamp as string to avoid YAML auto-casting.

Usage examples:
  python opencood/tools/build_ignore_frames_from_spikes.py \
    --logs-root opencood/logs/airv2x_HEAL_collab_lidar \
    --topk 50 \
    --out ignore_frames.yaml

  python opencood/tools/build_ignore_frames_from_spikes.py \
    --spike-files opencood/logs/.../spike_cases.txt opencood/logs/.../spike_cases.txt

"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, Tuple, TypedDict

yaml = None


RE_LOSS = re.compile(r"loss\s*[:=]\s*([0-9]+\.?[0-9]*)", re.I)
RE_SCENARIO = re.compile(r"(20\d\d_\d\d_\d\d_\d\d_\d\d_\d\d)")
RE_TS = re.compile(r"(?:timestamp|ts|frame)\s*[:=]\s*([0-9]+)", re.I)


def _norm_ts(ts) -> str:
    if ts is None:
        return ""
    ts = str(ts).strip()
    return ts.replace(".pkl", "")


def parse_spike_line(line: str):
    scen_m = RE_SCENARIO.search(line)
    if not scen_m:
        return None
    scenario = scen_m.group(1)

    ts = None
    m = RE_TS.search(line)
    if m:
        ts = m.group(1)
    else:
        # fallback: try /000123 or _000123
        m = re.search(r"[\/_](\d{1,})\b", line)
        if m:
            ts = m.group(1)
    ts = _norm_ts(ts)
    if not ts:
        return None

    loss = None
    m = RE_LOSS.search(line)
    if m:
        try:
            loss = float(m.group(1))
        except Exception:
            loss = None

    return scenario, ts, loss


def collect(spike_files):
    class _FrameStat(TypedDict):
        count: int
        max_loss: float
        runs: Set[str]

    stats: DefaultDict[Tuple[str, str], _FrameStat] = defaultdict(
        lambda: {"count": 0, "max_loss": -1.0, "runs": set()}
    )
    bad = 0
    for p in spike_files:
        run = os.path.basename(os.path.dirname(p))
        with open(p, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed = parse_spike_line(line)
                if parsed is None:
                    bad += 1
                    continue
                scenario, ts, loss = parsed
                key = (scenario, ts)
                v = stats[key]
                v["count"] += 1
                v["runs"].add(run)
                if loss is not None:
                    v["max_loss"] = max(v["max_loss"], float(loss))
    return stats, bad


def to_items(stats):
    items = []
    for (scenario, ts), v in stats.items():
        items.append(
            {
                "scenario": scenario,
                "timestamp": str(ts),
                "count": int(v["count"]),
                "max_loss": float(v["max_loss"]),
                "runs": sorted(v["runs"]),
            }
        )
    items.sort(key=lambda x: (-x["max_loss"], -x["count"], x["scenario"], x["timestamp"]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", default=None, help="Root dir to glob **/spike_cases.txt")
    ap.add_argument("--spike-files", nargs="*", default=None, help="Explicit spike_cases.txt paths")
    ap.add_argument("--topk", type=int, default=50, help="Keep top-K by max_loss")
    ap.add_argument("--min-max-loss", type=float, default=None, help="Only keep frames whose max_loss >= this")
    ap.add_argument("--out", default=None, help="Write YAML to this file (default: print)")
    args = ap.parse_args()

    # Lazy import: keep this script independent from any heavy OpenCOOD deps.
    global yaml
    if yaml is None:
        try:
            import yaml as _yaml  # type: ignore

            yaml = _yaml
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required. Install it with: pip install pyyaml"
            ) from e

    spike_files = []
    if args.spike_files:
        spike_files.extend(args.spike_files)
    if args.logs_root:
        spike_files.extend(glob.glob(os.path.join(args.logs_root, "**", "spike_cases.txt"), recursive=True))

    spike_files = sorted(set(spike_files))
    if not spike_files:
        raise SystemExit("No spike_cases.txt found. Provide --logs-root or --spike-files")

    stats, bad = collect(spike_files)
    items = to_items(stats)

    if args.min_max_loss is not None:
        items = [x for x in items if x["max_loss"] >= args.min_max_loss]

    items = items[: max(0, args.topk)]

    payload = {
        "data_filter": {
            "train": {
                "ignore_frames": [{"scenario": x["scenario"], "timestamp": str(x["timestamp"])} for x in items]
            }
        }
    }

    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)

    print(f"spike_files={len(spike_files)} unique_pairs={len(stats)} bad_lines={bad} emitted={len(items)}")


if __name__ == "__main__":
    main()
