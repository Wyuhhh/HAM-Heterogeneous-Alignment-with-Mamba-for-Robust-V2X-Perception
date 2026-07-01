# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yue Hu <18671129361@sjtu.edu.cn>
# Modifier: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics
import sys
import time
import datetime

root_path = os.path.abspath(__file__)
root_path = "/".join(root_path.split("/")[:-3])
sys.path.append(root_path)

import torch
from torch.backends import cudnn, cuda as torch_cuda

# TensorBoard logger is optional; keep training runnable even if neither
# tensorboardX nor tensorboard is installed in the current environment.
try:
    from tensorboardX import SummaryWriter  # preferred if available
except Exception:
    try:
        from torch.utils.tensorboard.writer import SummaryWriter  # fallback
    except Exception:
        SummaryWriter = None
from torch.utils.data import DataLoader, DistributedSampler, Subset

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils, train_utils
from tqdm import tqdm


def _pick_sample_id(ego_dict: dict):
    """Best-effort extract sample identifier fields for debugging spikes.

    Prefer batched "*_list" fields if present (common in AirV2X collate),
    otherwise fall back to single fields.
    """

    if not isinstance(ego_dict, dict):
        return None

    def _as_list(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return list(x)
        # Some datasets may use numpy arrays
        try:
            import numpy as _np  # type: ignore

            if isinstance(x, _np.ndarray):
                return x.tolist()
        except Exception:
            pass
        return None

    # Prefer list form (one id per sample in batch)
    scenario_list = _as_list(ego_dict.get("scenario_index_list"))
    ts_list = _as_list(ego_dict.get("timestamp_key_list"))
    meta_list = _as_list(ego_dict.get("metadata_path_list"))
    if scenario_list is not None:
        n = len(scenario_list)
        pieces = []
        for j in range(n):
            s = scenario_list[j] if j < len(scenario_list) else None
            t = ts_list[j] if isinstance(ts_list, list) and j < len(ts_list) else None
            m = meta_list[j] if isinstance(meta_list, list) and j < len(meta_list) else None
            item = f"scenario={s}"
            if t is not None:
                item += f",ts={t}"
            if m is not None:
                item += f",meta={m}"
            pieces.append(item)
        return " | ".join(pieces)

    # Single sample fields
    for k in ("scenario_index", "timestamp_key", "metadata_path"):
        if k in ego_dict:
            try:
                v = ego_dict.get(k)
                return str(v)
            except Exception:
                pass

    # Fallback: find any useful string-ish field
    for k, v in ego_dict.items():
        if any(
            s in str(k).lower()
            for s in ("scenario", "timestamp", "token", "path", "frame", "meta")
        ):
            try:
                return f"{k}={v}"
            except Exception:
                continue
    return None


def _get_loss_items(criterion):
    """Extract scalar loss components if criterion exposes loss_dict."""
    d = getattr(criterion, "loss_dict", None)
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        try:
            if torch.is_tensor(v):
                out[k] = float(v.detach().item())
            else:
                out[k] = float(v)
        except Exception:
            continue
    return out


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument(
        "--hypes_yaml",
        "-y",
        type=str,
        required=True,
        help="data generation yaml file needed ",
    )
    parser.add_argument("--model_dir", default="", help="Continued training path")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--fusion_method", "-f", default="intermediate", help="passed to inference."
    )
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--worker", default=16, type=int)
    parser.add_argument("--vehicle_dir", default=None, type=str, 
                        help="Model directory of the pretrained vehicle collaborative model.")
    parser.add_argument("--vehicle_epoch", type=int, default=20, 
                        help="Set the checkpoint epoch of the vehicle model.")
    parser.add_argument("--rsu_dir", default=None, type=str, 
                        help="Model directory of the pretrained RSU collaborative model")
    parser.add_argument("--rsu_epoch", type=int, default=20,
                        help="Set the checkpoint epoch of the RSU model.")
    parser.add_argument("--drone_dir", default=None, type=str, 
                        help="Model directory of the pretrained drone collaborative model")
    parser.add_argument("--drone_epoch", type=int, default=20,
                        help="Set the checkpoint epoch of the drone model.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision training (torch.cuda.amp)")
    parser.add_argument(
        "--skip_val",
        action="store_true",
        help="Skip validation phase (useful for DDP smoke test / isolating NCCL timeout).",
    )
    parser.add_argument(
        "--val_max_iters",
        type=int,
        default=-1,
        help="If >0, only run this many validation iterations (per rank).",
    )
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Local rank for distributed training (automatically set by torchrun)")
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)
    hypes["tag"] = opt.tag
    print("Dataset Building")
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes["train_params"]["batch_size"], drop_last=True
        )

        # Choose prefetch_factor only when using multiprocessing workers
        prefetch_train = 1 if opt.worker > 0 else None
        prefetch_val = 1 if opt.worker > 0 else None

        # Dataloader tuning: in DDP, a single slow/hung worker on any rank can
        # stall the whole job and eventually trigger NCCL allreduce timeout.
        # Make these knobs configurable from yaml to stabilize training.
        dl_cfg = hypes.get("train_params", {}).get("dataloader", {})
        pin_memory = bool(dl_cfg.get("pin_memory", False))
        timeout_s = int(dl_cfg.get("timeout", 1800))

        # PyTorch DataLoader requires timeout==0 when num_workers==0.
        if opt.worker <= 0:
            timeout_s = 0

        # persistent_workers is only valid when num_workers > 0.
        persistent_workers = bool(dl_cfg.get("persistent_workers", opt.worker > 0)) and (opt.worker > 0)

        train_loader = DataLoader(
            opencood_train_dataset,
            batch_sampler=batch_sampler_train,
            num_workers=opt.worker,
            timeout=timeout_s,
            collate_fn=opencood_train_dataset.collate_batch_train,
            shuffle=False,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_train,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            opencood_validate_dataset,
            sampler=sampler_val,
            num_workers=opt.worker,
            collate_fn=opencood_validate_dataset.collate_batch_train,
            timeout=timeout_s,
            shuffle=False,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_val,
            drop_last=False,
            persistent_workers=persistent_workers,
        )
    else:
        prefetch_train = 4 if opt.worker > 0 else None
        prefetch_val = 4 if opt.worker > 0 else None
        dl_cfg = hypes.get("train_params", {}).get("dataloader", {})
        pin_memory = bool(dl_cfg.get("pin_memory", False))
        timeout_s = int(dl_cfg.get("timeout", 1800))
        if opt.worker <= 0:
            timeout_s = 0
        persistent_workers = bool(dl_cfg.get("persistent_workers", opt.worker > 0)) and (opt.worker > 0)
        train_loader = DataLoader(
            opencood_train_dataset,
            batch_size=hypes["train_params"]["batch_size"],
            num_workers=opt.worker,
            collate_fn=opencood_train_dataset.collate_batch_train,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=True,
            prefetch_factor=prefetch_train,
            timeout=timeout_s,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            opencood_validate_dataset,
            batch_size=hypes["train_params"]["batch_size"],
            num_workers=opt.worker,
            collate_fn=opencood_validate_dataset.collate_batch_train,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=False,
            prefetch_factor=prefetch_val,
            timeout=timeout_s,
            persistent_workers=persistent_workers,
        )

    print("Creating Model")
    model = train_utils.create_model(hypes)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %d" % (total))
    # print(model)
    # print("Number of parameter: %.2fM" % (total/1e6))

    # device = torch.device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    if opt.distributed:
        # Keep DDP simple and faster: if your model doesn't truly have unused params,
        # setting find_unused_parameters=True avoids extra graph traversal.
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.gpu],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module
    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)  # type: ignore[attr-defined]
    grad_clip = hypes["train_params"].get("grad_clip", None)
    accum_steps = int(hypes["train_params"].get("accum_steps", 1))
    if accum_steps < 1:
        accum_steps = 1

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch  ###
        scheduler = train_utils.setup_lr_schedular(
            hypes, optimizer, init_epoch=init_epoch, n_iter_per_epoch=len(train_loader)
        )
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        print("output result save to: ", saved_path)
        # lr scheduler setup
        scheduler = train_utils.setup_lr_schedular(
            hypes, optimizer, n_iter_per_epoch=len(train_loader)
        )
        
    if opt.drone_dir:
        assert opt.vehicle_dir is not None, "Vehicle model directory should be provided if drone model directory is provided."
        print("Loading pretrained drone model from %s" % opt.drone_dir)
        _, model = train_utils.load_model(opt.drone_dir, model, opt.drone_epoch)
        
    if opt.rsu_dir:
        assert opt.vehicle_dir is not None, "Vehicle model directory should be provided if rsu model directory is provided."
        print("Loading pretrained rsu model from %s" % opt.rsu_dir)
        _, model = train_utils.load_model(opt.rsu_dir, model, opt.rsu_epoch)
        
    # For the current implementation, vehicle model must be loaded last because ego is vehicular by default.
    if opt.vehicle_dir:
        print("Loading pretrained vehicle model from %s" % opt.vehicle_dir)
        _, model = train_utils.load_model(opt.vehicle_dir, model, opt.vehicle_epoch)

    # Freeze backbone parameters during New Agent Training phase
    if hypes['train_params'].get('freeze_backbone', False):
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
        print("Backbone parameters have been frozen.")

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # record training (optional)
    writer = SummaryWriter(saved_path) if SummaryWriter is not None else None

    # Debug/optimization helpers
    spike_thresh = float(hypes["train_params"].get("loss_spike_thresh", 100.0))
    log_interval = int(hypes["train_params"].get("log_interval", 1))
    spike_log_path = os.path.join(saved_path, "spike_cases.txt")
    step_log_path = os.path.join(saved_path, "train_step_metrics.txt")

    # Train stall diagnosis (DDP timeouts are often caused by one slow/hung rank)
    train_slow_log_path = os.path.join(saved_path, "train_slow_batches.txt")

    # global step for logging/debugging (counts optimizer updates)
    global_step = 0

    print("Training start")
    epoches = hypes["train_params"]["epoches"]
    # used to help schedule learning rate
    with_round_loss = False
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print("learning rate %f" % param_group["lr"])

        # Make DistributedSampler shuffle deterministically per-epoch.
        if opt.distributed:
            try:
                sampler_train.set_epoch(epoch)
            except Exception:
                pass

        # shuffle for distributed, 在dataloader中加入打乱顺序（shuffle）的操作
        # if hypes['name'] == 'dair_v2xvit':
        # DistributedSampler(opencood_train_dataset).set_epoch(epoch)
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=(opt.rank != 0))

        # With gradient accumulation we need a single, clean grad reset per update.
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)

        # Lightweight heartbeat logging to localize DDP hangs (data vs compute).
        heartbeat_interval = int(hypes["train_params"].get("heartbeat_interval", 20))
        train_slow_iter_thresh_s = float(
            hypes["train_params"].get("train_slow_iter_thresh_s", 120.0)
        )
        last_iter_end = time.time()

        for i, batch_data in pbar:
            # Time spent waiting for next batch (DataLoader / IO)
            data_time = time.time() - last_iter_end
            iter_start = time.time()
            # print("batch_data: ", batch_data['ego'].keys())
            if batch_data is None:
                continue
            # Proceed without overzealous key filtering; rely on model/dataset to populate required fields.
            # If batch_data is malformed (None), it is already skipped above.
            # the model will be evaluation mode during validation
            model.train()

            with torch.cuda.amp.autocast(enabled=opt.amp):  # type: ignore[attr-defined]
                if "scope" in hypes["name"] or "how2comm" in hypes["name"]:
                    _batch_data = batch_data[0]
                    batch_data = train_utils.to_device(batch_data, device)
                    _batch_data = train_utils.to_device(_batch_data, device)

                    ouput_dict = model(batch_data)
                    final_loss = criterion(ouput_dict, _batch_data["ego"]["label_dict"])
                else:
                    batch_data = train_utils.to_device(batch_data, device)
                    # case1 : late fusion train --> only ego needed,
                    # and ego is (random) selected
                    # case2 : early fusion train --> all data projected to ego
                    # case3 : intermediate fusion --> ['ego']['processed_lidar']
                    # becomes a list, which containing all data from other cavs
                    # as well
                    batch_data["ego"]["epoch"] = epoch
                    output_dict = model(batch_data["ego"])
                    final_loss = criterion(output_dict, batch_data["ego"]["label_dict"])

            # Normalize the loss for gradient accumulation so the effective update
            # magnitude stays comparable to non-accum training.
            loss_for_backward = final_loss / float(accum_steps)

            # Some rare batches may produce a detached loss (no grad_fn), e.g. due to
            # empty targets / invalid labels / internal early-exit in loss.
            # In that case, backward would crash. We skip such iters but keep logging.
            if not getattr(loss_for_backward, "requires_grad", False) or (
                getattr(loss_for_backward, "grad_fn", None) is None
            ):
                if opt.rank == 0:
                    with open(os.path.join(saved_path, "no_grad_batches.txt"), "a+", encoding="utf-8") as f:
                        f.write(
                            f"epoch={epoch} iter={i}/{len(train_loader)} "
                            f"loss={float(final_loss.detach().item()) if torch.is_tensor(final_loss) else final_loss} "
                            f"sample={_pick_sample_id(batch_data.get('ego', {}) if isinstance(batch_data, dict) else {})}\n"
                        )
                continue

            if False:
                # if len(output_dict) > 2:
                single_loss_v = criterion(
                    output_dict,
                    batch_data["ego"]["label_dict_single_v"],
                    prefix="_single_v",
                )
                single_loss_i = criterion(
                    output_dict,
                    batch_data["ego"]["label_dict_single_i"],
                    prefix="_single_i",
                )
                if "fusion_args" in hypes["model"]["args"]:
                    if "communication" in hypes["model"]["args"]["fusion_args"]:
                        comm = hypes["model"]["args"]["fusion_args"]["communication"]
                        if ("round" in comm) and comm["round"] > 1:
                            round_loss_v = 0
                            with_round_loss = True
                            for round_id in range(1, comm["round"]):
                                round_loss_v += criterion(
                                    output_dict,
                                    batch_data["ego"]["label_dict"],
                                    prefix="_v{}".format(round_id),
                                )

            # criterion.logging(epoch, i, len(train_loader), writer)
            print_msg = criterion.logging(epoch, i, len(train_loader), writer) if opt.rank == 0 else ""
            pbar.set_description(print_msg)

            # If criterion indicates this batch should be skipped (e.g., zero-positive anchors),
            # we must skip consistently across all ranks to avoid DDP desync.
            loss_items = _get_loss_items(criterion)
            if loss_items.get("skip_batch") in (True, 1.0):
                # Important: although loss module tries to all_reduce internally,
                # we still do a defensive all_reduce here to guarantee all ranks
                # take the same branch (in case some rank hit non-grad/no-grad path).
                skip_flag = torch.tensor(
                    [1], device=device, dtype=torch.int32
                )
                if opt.distributed and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(skip_flag, op=torch.distributed.ReduceOp.MAX)
                if int(skip_flag.item()) == 1:
                    if opt.rank == 0:
                        try:
                            ego_dict = batch_data.get("ego", {}) if isinstance(batch_data, dict) else {}
                        except Exception:
                            ego_dict = {}
                        with open(
                            os.path.join(saved_path, "zero_pos_batches.txt"),
                            "a+",
                            encoding="utf-8",
                        ) as f:
                            f.write(
                                f"epoch={epoch} iter={i}/{len(train_loader)} "
                                f"sample={_pick_sample_id(ego_dict)}\n"
                            )
                    continue

            # Extra logging for debugging/troubleshooting convergence.
            if opt.rank == 0 and (i % max(log_interval, 1) == 0):
                lr = optimizer.param_groups[0].get("lr", 0.0)
                # Always include total loss + lr in plain text step log
                pieces = [
                    f"time={int(time.time())}",
                    f"epoch={epoch}",
                    f"iter={i}/{len(train_loader)}",
                    f"global_step={global_step}",
                    f"lr={lr}",
                    f"loss={float(final_loss.detach().item())}",
                ]
                for k in sorted(loss_items.keys()):
                    if k in ("total_loss", "loss", "Total_loss"):
                        continue
                    pieces.append(f"{k}={loss_items[k]}")
                with open(step_log_path, "a+", encoding="utf-8") as f:
                    f.write(" ".join(pieces) + "\n")

            # Spike case capture
            if opt.rank == 0:
                try:
                    loss_v = float(final_loss.detach().item())
                except Exception:
                    loss_v = None
                if loss_v is not None and loss_v >= spike_thresh:
                    ego_dict = None
                    try:
                        # When scope/how2comm, our 'batch_data' is already moved to device and has structure.
                        ego_dict = batch_data.get("ego", {}) if isinstance(batch_data, dict) else {}
                    except Exception:
                        ego_dict = {}
                    sample_id = _pick_sample_id(ego_dict)
                    with open(spike_log_path, "a+", encoding="utf-8") as f:
                        f.write(
                            f"Epoch[{epoch}] iter[{i}/{len(train_loader)}] loss={loss_v} lr={optimizer.param_groups[0].get('lr',0.0)} sample={sample_id}\n"
                        )

            if False:
                # if len(output_dict) > 2:
                final_loss += single_loss_v + single_loss_i
                if with_round_loss:
                    final_loss += round_loss_v
            if opt.rank == 0:
                with open(os.path.join(saved_path, "train_loss.txt"), "a+") as f:
                    msg = "Epoch[{}], iter[{}/{}], loss[{}]. \n".format(
                        epoch, i, len(train_loader), final_loss
                    )
                    f.write(msg)

            # print(a)
            # back-propagation (gradient accumulation)
            # Reduce DDP sync frequency by suppressing allreduce on micro-steps via no_sync().
            do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == len(train_loader))
            ddp_no_sync = (
                opt.distributed and hasattr(model, "no_sync") and (not do_step)
            )
            if opt.amp:
                if ddp_no_sync:
                    with model.no_sync():
                        scaler.scale(loss_for_backward).backward()
                else:
                    scaler.scale(loss_for_backward).backward()
                if do_step:
                    if grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    model.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
            else:
                if ddp_no_sync:
                    with model.no_sync():
                        loss_for_backward.backward()
                else:
                    loss_for_backward.backward()
                if do_step:
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    model.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
            # torch.cuda.empty_cache()

            # Time spent on compute for this iteration (forward+loss+backward+step)
            iter_time = time.time() - iter_start
            last_iter_end = time.time()
            if (
                opt.distributed
                and heartbeat_interval > 0
                and (i % heartbeat_interval == 0)
                and (opt.rank == 0)
            ):
                # If we later hit NCCL timeout, these logs help infer whether
                # we're stuck on dataloader (data_time explodes), compute, or sync.
                t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[heartbeat] {t} epoch={epoch} iter={i}/{len(train_loader)} "
                    f"data_time={data_time:.3f}s iter_time={iter_time:.3f}s accum_steps={accum_steps}",
                    flush=True,
                )

            # Slow-batch capture (all ranks). This helps identify the exact sample
            # (scenario/timestamp/meta) that caused one rank to lag and trigger NCCL watchdog.
            if train_slow_iter_thresh_s > 0 and iter_time >= train_slow_iter_thresh_s:
                try:
                    ego_dict = batch_data.get("ego", {}) if isinstance(batch_data, dict) else {}
                except Exception:
                    ego_dict = {}
                sample_id = _pick_sample_id(ego_dict)
                with open(train_slow_log_path, "a+", encoding="utf-8") as f:
                    f.write(
                        f"epoch={epoch} rank={opt.rank} iter={i}/{len(train_loader)} "
                        f"data_time={data_time:.3f}s compute_time={iter_time:.3f}s sample={sample_id}\n"
                    )
        # Step LR scheduler at the end of each epoch
        if hypes["lr_scheduler"]["core_method"] != "cosineannealwarm":
            scheduler.step()

        # Save checkpoint (rank0 only)
        if opt.rank == 0:
            if epoch % hypes["train_params"]["save_freq"] == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(saved_path, "net_epoch%d.pth" % (epoch + 1)),
                )

        # DDP-safe validation: all ranks must enter validation to keep collectives aligned.
        if (not opt.skip_val) and (epoch % hypes["train_params"]["eval_freq"] == 0):
            model.eval()

            # Validation heartbeat / stall diagnosis
            val_heartbeat_interval = int(
                hypes["train_params"].get("val_heartbeat_interval", 50)
            )
            val_slow_iter_thresh_s = float(
                hypes["train_params"].get("val_slow_iter_thresh_s", 120.0)
            )
            val_slow_log_path = os.path.join(saved_path, "val_slow_batches.txt")
            last_val_iter_end = time.time()

            # Local accumulators (per-rank)
            local_loss_sum = 0.0
            local_cnt = 0
            local_items_sum = {}

            with torch.no_grad():
                # Only rank0 shows tqdm to keep logs tidy.
                it = enumerate(val_loader)
                if opt.rank == 0:
                    it = tqdm(it, total=len(val_loader), desc="Validation")

                for i, batch_data in it:
                    # Time spent waiting for the next batch (DataLoader / IO)
                    data_time = time.time() - last_val_iter_end
                    if opt.val_max_iters and opt.val_max_iters > 0 and i >= opt.val_max_iters:
                        break
                    if batch_data is None:
                        continue

                    iter_start = time.time()

                    with torch.cuda.amp.autocast(enabled=opt.amp):  # type: ignore[attr-defined]
                        if "scope" in hypes["name"] or "how2comm" in hypes["name"]:
                            _batch_data = batch_data[0]
                            batch_data = train_utils.to_device(batch_data, device)
                            _batch_data = train_utils.to_device(_batch_data, device)
                            ouput_dict = model(batch_data)
                            final_loss = criterion(ouput_dict, _batch_data["ego"]["label_dict"])
                        else:
                            batch_data = train_utils.to_device(batch_data, device)
                            batch_data["ego"]["epoch"] = epoch
                            ouput_dict = model(batch_data["ego"])
                            final_loss = criterion(ouput_dict, batch_data["ego"]["label_dict"])

                    # Update local stats
                    try:
                        local_loss_sum += float(final_loss.detach().item())
                        local_cnt += 1
                    except Exception:
                        pass
                    for k, v in _get_loss_items(criterion).items():
                        local_items_sum[k] = local_items_sum.get(k, 0.0) + float(v)

                    if opt.rank == 0 and hasattr(it, "set_description"):
                        try:
                            it.set_description(f"Validation Loss: {final_loss.item()}")
                        except Exception:
                            pass

                    # Time spent on compute + device transfer for this batch
                    iter_time = time.time() - iter_start
                    last_val_iter_end = time.time()

                    # Heartbeat: rank0 prints periodically
                    if (
                        val_heartbeat_interval > 0
                        and (i % val_heartbeat_interval == 0)
                        and (opt.rank == 0)
                    ):
                        tnow = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(
                            f"[val-heartbeat] {tnow} epoch={epoch} iter={i}/{len(val_loader)} "
                            f"data_time={data_time:.3f}s iter_time={iter_time:.3f}s",
                            flush=True,
                        )

                    # Slow-batch capture (all ranks, include rank id)
                    if val_slow_iter_thresh_s > 0 and iter_time >= val_slow_iter_thresh_s:
                        try:
                            ego_dict = batch_data.get("ego", {}) if isinstance(batch_data, dict) else {}
                        except Exception:
                            ego_dict = {}
                        sample_id = _pick_sample_id(ego_dict)
                        with open(val_slow_log_path, "a+", encoding="utf-8") as f:
                            f.write(
                                f"epoch={epoch} rank={opt.rank} iter={i}/{len(val_loader)} "
                                f"data_time={data_time:.3f}s compute_time={iter_time:.3f}s sample={sample_id}\n"
                            )

            # Reduce across ranks
            if opt.distributed and torch.distributed.is_initialized():
                t = torch.tensor([local_loss_sum, float(local_cnt)], device=device, dtype=torch.float32)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                global_loss_sum = float(t[0].item())
                global_cnt = int(t[1].item())

                # components: reduce each key
                global_items_sum = {}
                for k, v in local_items_sum.items():
                    tk = torch.tensor(float(v), device=device, dtype=torch.float32)
                    torch.distributed.all_reduce(tk, op=torch.distributed.ReduceOp.SUM)
                    global_items_sum[k] = float(tk.item())
            else:
                global_loss_sum = local_loss_sum
                global_cnt = local_cnt
                global_items_sum = local_items_sum

            valid_ave_loss = (global_loss_sum / max(1, global_cnt))

            # rank0 prints / writes
            if opt.rank == 0:
                print(f"At epoch {epoch}, the validation loss is {valid_ave_loss:.6f}")
                if writer is not None:
                    writer.add_scalar("Validate_Loss", valid_ave_loss, epoch)
                    if global_cnt > 0:
                        for k, v in global_items_sum.items():
                            writer.add_scalar(f"val/{k}", v / global_cnt, epoch)

                with open(os.path.join(saved_path, "validation_loss.txt"), "a+") as f:
                    f.write(f"Epoch[{epoch}], loss[{valid_ave_loss}]. \n")

                if global_cnt > 0 and global_items_sum:
                    with open(
                        os.path.join(saved_path, "validation_step_metrics.txt"),
                        "a+",
                        encoding="utf-8",
                    ) as f:
                        pieces = [f"epoch={epoch}", f"val_loss={valid_ave_loss}"]
                        for k in sorted(global_items_sum.keys()):
                            pieces.append(f"{k}={global_items_sum[k] / global_cnt}")
                        f.write(" ".join(pieces) + "\n")

                # lowest val loss
                # if valid_ave_loss < lowest_val_loss:
                #     lowest_val_loss = valid_ave_loss
                #     best_saved_path = os.path.join(saved_path, 'net_epoch_bestval_at{}.pth'.format(epoch+1))
                #     torch.save(model.state_dict(), best_saved_path)

    # Step schedulers that expect epoch index.
    try:
        scheduler.step(epoches - 1)
    except Exception:
        try:
            scheduler.step()
        except Exception:
            pass

    print("Training Finished, checkpoints saved to %s" % saved_path)
    torch.cuda.empty_cache()
    run_test = True
    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)


if __name__ == "__main__":
    main()
