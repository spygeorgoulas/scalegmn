#!/usr/bin/env python3
"""
python /home/intern/spygeorgoulas/thesis-metanets/scalegmn/scalegmn_inclr_2026_gram.py \
  --conf /home/intern/spygeorgoulas/thesis-metanets/scalegmn/configs/gram/scalegmn.yml
"""

'''
===== Velocity Statistics =====
Files processed : 810
Mean velocity : 37.756039
Std velocity : 19.710636
Min velocity : 0.000000
Max velocity : 194.266403
================================
'''

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import wandb
from tqdm import trange

from src.data import dataset
from src.scalegmn.models import ScaleGMN_equiv
from src.utils.setup_arg_parser import setup_arg_parser
from src.utils.helpers import overwrite_conf, count_parameters, set_seed, mask_input, mask_hidden

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ.setdefault("WANDB_DISABLE_GIT", "true")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")


# ======================================================================================
# Utilities
# ======================================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def residual_param_update(weights, biases, delta_weights, delta_biases):
    new_weights = [weights[j] + delta_weights[j] for j in range(len(weights))]
    new_biases = [biases[j] + delta_biases[j] for j in range(len(biases))]
    return new_weights, new_biases


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.norm(pred - target) / (torch.norm(target) + eps)


def challenge_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Hint-based GRaM metric.

    Original hint:
        metric = (velocity_out - ground_truth).norm(dim=3).mean(dim=(1, 2))

    Here pred and target are per-sample tensors of shape (M, 3), where M = T * N.
    So this becomes mean vector L2 error over all spacetime query points.
    """
    return (pred - target).norm(dim=-1).mean()


def get_output_dir(conf: Dict[str, Any]) -> Path:
    if "output_dir" in conf:
        return Path(conf["output_dir"]).expanduser().resolve()

    if "train_args" in conf and "output_dir" in conf["train_args"]:
        return Path(conf["train_args"]["output_dir"]).expanduser().resolve()

    return Path("./outputs/gram_scalegmn_velocity_operator").resolve()


def get_target_npz_dir(conf: Dict[str, Any]) -> Path:
    data_conf = conf["data"]

    candidate_keys = [
        "target_npz_dir",
        "target_dir",
        "npz_dir",
        "raw_npz_dir",
        "velocity_out_npz_dir",
    ]
    for k in candidate_keys:
        if k in data_conf and data_conf[k] is not None:
            return Path(data_conf[k]).expanduser().resolve()

    raise KeyError(
        "Could not find target npz directory in config. "
        "Please provide one of: data.target_npz_dir / data.target_dir / data.npz_dir / data.raw_npz_dir"
    )


def get_inr_w0(conf: Dict[str, Any]) -> float:
    if "inr_model" in conf and conf["inr_model"] is not None:
        if "w0" in conf["inr_model"]:
            return float(conf["inr_model"]["w0"])
        if "omega_0" in conf["inr_model"]:
            return float(conf["inr_model"]["omega_0"])

    if "data" in conf and "w0" in conf["data"]:
        return float(conf["data"]["w0"])

    return 30.0


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def infer_npz_path_from_inr_path(inr_path: str, target_npz_dir: Path) -> Path:
    """
    Supports:
    - same stem under target root
    - same relative subpath with .npz instead of .pth
    """
    inr_path = Path(inr_path)
    stem = inr_path.stem

    direct = target_npz_dir / f"{stem}.npz"
    if direct.exists():
        return direct

    parts = list(inr_path.parts)
    for idx in range(len(parts)):
        suffix_parts = parts[idx:]
        candidate = target_npz_dir.joinpath(*suffix_parts).with_suffix(".npz")
        if candidate.exists():
            return candidate

    matches = list(target_npz_dir.rglob(f"{stem}.npz"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple .npz files found for stem '{stem}' under {target_npz_dir}")

    raise FileNotFoundError(f"Could not match INR path '{inr_path}' to a target .npz in '{target_npz_dir}'")


def load_velocity_out_npz(npz_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        coords_out: (M, 4) float32, where M = 5 * N
        target_out: (M, 3) float32

    Uses:
        pos:          (N, 3)
        t:            (10,)
        velocity_out: (5, N, 3)

    Assumption:
        velocity_out corresponds to the second half of t, i.e. t[5:10].
        If shapes differ, we fall back safely.
    """
    data = np.load(npz_path)

    pos = data["pos"].astype(np.float32)              # (N, 3)
    t = data["t"].astype(np.float32)                  # (10,)
    vel_out = data["velocity_out"].astype(np.float32) # (5, N, 3)

    n_times = vel_out.shape[0]
    n_points = pos.shape[0]

    if t.shape[0] >= 10 and n_times == 5:
        t_out = t[5:10]
    elif t.shape[0] >= n_times:
        t_out = t[-n_times:]
    else:
        raise ValueError(f"Unexpected t shape {t.shape} for velocity_out shape {vel_out.shape} in {npz_path}")

    pos_rep = np.broadcast_to(pos[None, :, :], (n_times, n_points, 3)).copy()
    t_rep = np.broadcast_to(t_out[:, None, None], (n_times, n_points, 1)).copy()

    coords_out = np.concatenate([pos_rep, t_rep], axis=-1).reshape(n_times * n_points, 4).astype(np.float32)
    target_out = vel_out.reshape(n_times * n_points, 3).astype(np.float32)

    return torch.from_numpy(coords_out), torch.from_numpy(target_out)


def unpack_batch(batch):
    """
    Tries to support common BaseDataset return patterns.

    Expected common cases:
    - (params, w_b, label, path)
    - (params, w_b, path)
    - {"params": ..., "w_b": ..., "path": ...}
    """
    params = None
    w_b = None
    paths = None

    if isinstance(batch, dict):
        params = batch.get("params", batch.get("graph", None))
        w_b = batch.get("w_b", batch.get("weights_biases", None))
        paths = batch.get("path", batch.get("paths", None))

    elif isinstance(batch, (list, tuple)):
        if len(batch) == 4:
            params, w_b, _, paths = batch
        elif len(batch) == 3:
            params, w_b, third = batch
            if isinstance(third, (list, tuple)):
                if len(third) > 0 and isinstance(third[0], str):
                    paths = list(third)
            elif isinstance(third, str):
                paths = [third]
            else:
                paths = getattr(batch, "path", None)
        elif len(batch) == 2:
            params, w_b = batch

    if params is None or w_b is None:
        raise RuntimeError(
            "Could not unpack batch. "
            "Please adapt unpack_batch() to your exact IFWVelocityINRDataset return format."
        )

    if paths is None:
        if hasattr(batch, "path"):
            paths = batch.path
        elif hasattr(batch, "paths"):
            paths = batch.paths
        elif hasattr(params, "path"):
            paths = params.path
        elif hasattr(params, "paths"):
            paths = params.paths

    if isinstance(paths, str):
        paths = [paths]

    return params, w_b, paths


def graph_params_to_siren_params(
    weights: List[torch.Tensor],
    biases: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Convert ScaleGMN / graph-format params to standard batched linear layer params.

    Expected input:
        weights: list of (B, in, out, 1)
        biases:  list of (B, out, 1)

    Returns:
        weights: list of (B, out, in)
        biases:  list of (B, out)
    """
    siren_weights = []
    siren_biases = []

    for w in weights:
        if w.dim() != 4:
            raise ValueError(f"Expected graph-format weight with 4 dims, got shape {tuple(w.shape)}")
        w_lin = w[..., 0].permute(0, 2, 1).contiguous()
        siren_weights.append(w_lin)

    for b in biases:
        if b.dim() != 3:
            raise ValueError(f"Expected graph-format bias with 3 dims, got shape {tuple(b.shape)}")
        b_lin = b[..., 0].contiguous()
        siren_biases.append(b_lin)

    return siren_weights, siren_biases


def batched_siren_forward(
    coords: torch.Tensor,
    weights: List[torch.Tensor],
    biases: List[torch.Tensor],
    w0: float,
) -> torch.Tensor:
    """
    Batched functional SIREN forward.

    Args:
        coords:  (B, M, in_dim)
        weights: list of tensors, each (B, out_dim_l, in_dim_l)
        biases:  list of tensors, each (B, out_dim_l)

    Returns:
        preds:   (B, M, out_dim_last)
    """
    x = coords
    n_layers = len(weights)

    for li in range(n_layers):
        w = weights[li]  # (B, out, in)
        b = biases[li]   # (B, out)

        x = torch.bmm(x, w.transpose(1, 2)) + b.unsqueeze(1)

        if li < n_layers - 1:
            x = torch.sin(w0 * x)

    return x


@torch.no_grad()
def predict_full_field_chunked(
    sample_weights: List[torch.Tensor],
    sample_biases: List[torch.Tensor],
    coords: torch.Tensor,
    w0: float,
    eval_chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Args:
        sample_weights: list of tensors [(1, out, in), ...]
        sample_biases:  list of tensors [(1, out), ...]
        coords:         (M, 4) on CPU or GPU
    Returns:
        pred:           (M, 3)
    """
    preds = []
    m = coords.shape[0]

    for start in range(0, m, eval_chunk_size):
        end = min(start + eval_chunk_size, m)
        coords_chunk = coords[start:end].to(device, non_blocking=True).unsqueeze(0)
        pred_chunk = batched_siren_forward(coords_chunk, sample_weights, sample_biases, w0).squeeze(0)
        preds.append(pred_chunk.detach().cpu())

    return torch.cat(preds, dim=0)


def move_wb_to_device(w_b, device):
    w_b = w_b.to(device)
    weights = w_b.weights
    biases = w_b.biases
    return weights, biases


def get_num_samples_in_batch(paths, weights) -> int:
    if paths is not None:
        return len(paths)
    return weights[0].shape[0]


# ======================================================================================
# Field loss computation
# ======================================================================================

def compute_train_field_loss(
    new_weights: List[torch.Tensor],
    new_biases: List[torch.Tensor],
    paths: List[str],
    target_npz_dir: Path,
    w0: float,
    device: torch.device,
    train_query_batch_size: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Training loss over velocity_out using sampled query points per sample.
    We optimize only MSE, but also compute the challenge metric for logging.
    """
    per_sample_mse_losses = []
    per_sample_hint_metrics = []

    for bi, inr_path in enumerate(paths):
        npz_path = infer_npz_path_from_inr_path(inr_path, target_npz_dir)
        coords_out, target_out = load_velocity_out_npz(npz_path)

        n_total = coords_out.shape[0]
        if train_query_batch_size is not None and train_query_batch_size > 0 and train_query_batch_size < n_total:
            idx = torch.randint(0, n_total, (train_query_batch_size,))
            coords_out = coords_out[idx]
            target_out = target_out[idx]

        coords_out = coords_out.to(device, non_blocking=True).unsqueeze(0)  # (1, M, 4)
        target_out = target_out.to(device, non_blocking=True)               # (M, 3)

        sw = [w[bi:bi+1] for w in new_weights]
        sb = [b[bi:bi+1] for b in new_biases]
        sw, sb = graph_params_to_siren_params(sw, sb)

        pred_out = batched_siren_forward(coords_out, sw, sb, w0).squeeze(0)

        mse_loss = F.mse_loss(pred_out, target_out)
        hint_val = challenge_metric(pred_out, target_out)

        per_sample_mse_losses.append(mse_loss)
        per_sample_hint_metrics.append(hint_val)

    loss = torch.stack(per_sample_mse_losses).mean()
    avg_hint = torch.stack(per_sample_hint_metrics).mean()

    return loss, {
        "train_field_mse": float(loss.detach().item()),
        "train_field_challenge_metric": float(avg_hint.detach().item()),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    target_npz_dir: Path,
    w0: float,
    eval_chunk_size: int,
    num_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full-field evaluation on velocity_out.
    Metrics are averaged over samples.
    """
    model.eval()

    mse_list = []
    rel_l2_list = []
    challenge_metric_list = []

    for i, batch in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break

        params, w_b, paths = unpack_batch(batch)
        params = params.to(device)
        weights, biases = move_wb_to_device(w_b, device)

        delta_weights, delta_biases = model(params, weights, biases)
        new_weights, new_biases = residual_param_update(weights, biases, delta_weights, delta_biases)

        batch_size_actual = get_num_samples_in_batch(paths, weights)

        for bi in range(batch_size_actual):
            if paths is None:
                raise RuntimeError(
                    "Evaluation requires dataset paths. "
                    "Please ensure the dataset returns the original path."
                )

            npz_path = infer_npz_path_from_inr_path(paths[bi], target_npz_dir)
            coords_out, target_out = load_velocity_out_npz(npz_path)

            sw = [w[bi:bi+1] for w in new_weights]
            sb = [b[bi:bi+1] for b in new_biases]
            sw, sb = graph_params_to_siren_params(sw, sb)

            pred_out = predict_full_field_chunked(
                sample_weights=sw,
                sample_biases=sb,
                coords=coords_out,
                w0=w0,
                eval_chunk_size=eval_chunk_size,
                device=device,
            )

            mse_i = F.mse_loss(pred_out, target_out).item()
            rel_l2_i = relative_l2(pred_out, target_out).item()
            challenge_i = challenge_metric(pred_out, target_out).item()

            mse_list.append(mse_i)
            rel_l2_list.append(rel_l2_i)
            challenge_metric_list.append(challenge_i)

    model.train()

    if len(mse_list) == 0:
        return {
            "avg_mse": float("nan"),
            "avg_rel_l2": float("nan"),
            "avg_challenge_metric": float("nan"),
            "num_samples": 0,
        }

    return {
        "avg_mse": float(np.mean(mse_list)),
        "std_mse": float(np.std(mse_list)),
        "avg_rel_l2": float(np.mean(rel_l2_list)),
        "std_rel_l2": float(np.std(rel_l2_list)),
        "avg_challenge_metric": float(np.mean(challenge_metric_list)),
        "std_challenge_metric": float(np.std(challenge_metric_list)),
        "num_samples": int(len(mse_list)),
    }


# ======================================================================================
# Optimization
# ======================================================================================

def build_optimizer_and_scheduler(conf: Dict[str, Any], model_params):
    opt_conf = conf["optimization"]

    optimizer_name = opt_conf.get("optimizer_name", "Adam")
    optimizer_args = opt_conf.get("optimizer_args", {"lr": 1e-4})
    scheduler_args = opt_conf.get("scheduler_args", None)

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model_params, **optimizer_args)
    elif optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model_params, **optimizer_args)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model_params, **optimizer_args)
    else:
        raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")

    scheduler = None
    scheduler_name = None

    if scheduler_args is not None and isinstance(scheduler_args, dict) and "scheduler" in scheduler_args:
        scheduler_name = scheduler_args["scheduler"]
        sched_kwargs = {k: v for k, v in scheduler_args.items() if k != "scheduler"}

        if scheduler_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_kwargs)
        elif scheduler_name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sched_kwargs)
        elif scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_kwargs)
        elif scheduler_name == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **sched_kwargs)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return optimizer, scheduler, scheduler_name


# ======================================================================================
# Main
# ======================================================================================

def main(args=None):
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, vars(args))

    torch.set_float32_matmul_precision("high")

    print(yaml.dump(conf, default_flow_style=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = get_output_dir(conf)
    ensure_dir(output_dir)

    target_npz_dir = get_target_npz_dir(conf)
    inr_w0 = get_inr_w0(conf)

    set_seed(conf["train_args"]["seed"])

    # ------------------------------------------------------------------
    # Dataset / Dataloader
    # ------------------------------------------------------------------
    equiv_on_hidden = mask_hidden(conf)
    get_first_layer_mask = mask_input(conf)

    conf["data"]["return_path"] = False

    train_set = dataset(
        conf["data"],
        split="train",
        debug=conf["debug"],
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=equiv_on_hidden,
        get_first_layer_mask=get_first_layer_mask,
    )
    conf["scalegmn_args"]["layer_layout"] = train_set.get_layer_layout()

    val_set = dataset(
        conf["data"],
        split="val",
        debug=conf["debug"],
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=equiv_on_hidden,
        get_first_layer_mask=get_first_layer_mask,
    )

    test_set = dataset(
        conf["data"],
        split="test",
        debug=conf["debug"],
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=equiv_on_hidden,
        get_first_layer_mask=get_first_layer_mask,
    )

    print(f"Len train set: {len(train_set)}")
    print(f"Len val set:   {len(val_set)}")
    print(f"Len test set:  {len(test_set)}")

    train_loader = torch_geometric.loader.DataLoader(
        dataset=train_set,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=conf["num_workers"],
        pin_memory=True,
        sampler=None,
    )

    val_loader = torch_geometric.loader.DataLoader(
        dataset=val_set,
        batch_size=conf["batch_size"],
        shuffle=False,
        num_workers=conf["num_workers"],
        pin_memory=True,
    )

    test_loader = torch_geometric.loader.DataLoader(
        dataset=test_set,
        batch_size=conf["batch_size"],
        shuffle=False,
        num_workers=conf["num_workers"],
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    net = ScaleGMN_equiv(conf["scalegmn_args"]).to(device)
    print(net)

    cnt_p = count_parameters(net=net)
    print(f"ScaleGMN params: {cnt_p}")

    for p in net.parameters():
        p.requires_grad = True

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    if conf.get("wandb", False):
        wandb_args = conf.get("wandb_args", {})
        wandb.init(
            config=conf,
            dir=str(output_dir),
            save_code=False,
            **wandb_args,
        )
        wandb.log({"number_of_parameters": cnt_p}, step=0)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    model_params = [p for p in net.parameters() if p.requires_grad]
    optimizer, scheduler, scheduler_name = build_optimizer_and_scheduler(conf, model_params)

    # ------------------------------------------------------------------
    # Training settings
    # ------------------------------------------------------------------
    train_query_batch_size = conf["train_args"].get("train_query_batch_size", 16384)
    eval_chunk_size = conf["train_args"].get("eval_chunk_size", 16384)
    eval_every = conf["train_args"]["eval_every"]
    num_epochs = conf["train_args"]["num_epochs"]
    train_eval_num_batches = conf["train_args"].get("train_eval_num_batches", 20)
    clip_grad = conf["optimization"].get("clip_grad", False)
    clip_grad_max_norm = conf["optimization"].get("clip_grad_max_norm", 1.0)

    best_val_challenge_metric = float("inf")
    best_val_results = None
    best_test_results = None
    global_step = 0
    current_test_rel_l2 = float("nan")
    current_test_challenge_metric = float("nan")

    best_ckpt_path = output_dir / "best_model.pt"
    best_metrics_path = output_dir / "best_metrics.json"

    epoch_iter = trange(0, num_epochs)
    net.train()

    for epoch in epoch_iter:
        for i, batch in enumerate(train_loader):
            params, w_b, paths = unpack_batch(batch)
            if paths is None:
                raise RuntimeError(
                    "Training requires dataset paths. "
                    "Please ensure IFWVelocityINRDataset returns the path in __getitem__."
                )

            params = params.to(device)
            weights, biases = move_wb_to_device(w_b, device)

            optimizer.zero_grad(set_to_none=True)

            delta_weights, delta_biases = net(params, weights, biases)
            new_weights, new_biases = residual_param_update(weights, biases, delta_weights, delta_biases)

            loss, loss_dict = compute_train_field_loss(
                new_weights=new_weights,
                new_biases=new_biases,
                paths=paths,
                target_npz_dir=target_npz_dir,
                w0=inr_w0,
                device=device,
                train_query_batch_size=train_query_batch_size,
            )

            loss.backward()

            log = {
                "train/loss": float(loss.item()),
                "train/challenge_metric_sampled": loss_dict["train_field_challenge_metric"],
                "epoch": epoch,
            }

            if clip_grad:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(filter(lambda p: p.requires_grad, net.parameters())),
                    clip_grad_max_norm
                )
                log["train/grad_norm"] = float(grad_norm)

            optimizer.step()

            if scheduler is not None and scheduler_name != "ReduceLROnPlateau":
                scheduler.step()
                log["train/lr"] = float(scheduler.get_last_lr()[0])
            else:
                log["train/lr"] = float(optimizer.param_groups[0]["lr"])

            if conf.get("wandb", False):
                wandb.log(log, step=global_step)

            epoch_iter.set_description(
                f"[{epoch} {i+1}] train_mse={loss.item():.6f} "
                f"best_val_hint={best_val_challenge_metric:.6f} "
                f"test_hint={current_test_challenge_metric:.6f}"
            )

            global_step += 1

            if (global_step % eval_every) == 0:
                train_results = evaluate(
                    model=net,
                    loader=train_loader,
                    device=device,
                    target_npz_dir=target_npz_dir,
                    w0=inr_w0,
                    eval_chunk_size=eval_chunk_size,
                    num_batches=train_eval_num_batches,
                )

                val_results = evaluate(
                    model=net,
                    loader=val_loader,
                    device=device,
                    target_npz_dir=target_npz_dir,
                    w0=inr_w0,
                    eval_chunk_size=eval_chunk_size,
                    num_batches=None,
                )

                test_results = evaluate(
                    model=net,
                    loader=test_loader,
                    device=device,
                    target_npz_dir=target_npz_dir,
                    w0=inr_w0,
                    eval_chunk_size=eval_chunk_size,
                    num_batches=None,
                )

                val_challenge_metric = val_results["avg_challenge_metric"]
                current_test_rel_l2 = test_results["avg_rel_l2"]
                current_test_challenge_metric = test_results["avg_challenge_metric"]

                if scheduler is not None and scheduler_name == "ReduceLROnPlateau":
                    scheduler.step(val_challenge_metric)

                is_best = val_challenge_metric < best_val_challenge_metric

                if is_best:
                    best_val_challenge_metric = val_challenge_metric
                    best_val_results = val_results
                    best_test_results = test_results

                    ckpt = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_challenge_metric": best_val_challenge_metric,
                        "conf": conf,
                    }
                    torch.save(ckpt, best_ckpt_path)

                    best_payload = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_results": best_val_results,
                        "best_test_results": best_test_results,
                        "checkpoint_path": str(best_ckpt_path),
                    }
                    save_json(best_metrics_path, best_payload)

                if conf.get("wandb", False):
                    eval_log = {
                        "train_eval/mse": train_results["avg_mse"],
                        "train_eval/rel_l2": train_results["avg_rel_l2"],
                        "train_eval/challenge_metric": train_results["avg_challenge_metric"],

                        "val/mse": val_results["avg_mse"],
                        "val/rel_l2": val_results["avg_rel_l2"],
                        "val/challenge_metric": val_results["avg_challenge_metric"],

                        "test/mse": test_results["avg_mse"],
                        "test/rel_l2": test_results["avg_rel_l2"],
                        "test/challenge_metric": test_results["avg_challenge_metric"],

                        "overfit_gap/mse": val_results["avg_mse"] - train_results["avg_mse"],
                        "overfit_gap/rel_l2": val_results["avg_rel_l2"] - train_results["avg_rel_l2"],
                        "overfit_gap/challenge_metric": (
                            val_results["avg_challenge_metric"] - train_results["avg_challenge_metric"]
                        ),

                        "best/val_challenge_metric": best_val_challenge_metric,
                        "best/val_rel_l2": best_val_results["avg_rel_l2"] if best_val_results is not None else None,
                        "best/val_mse": best_val_results["avg_mse"] if best_val_results is not None else None,

                        "best/test_challenge_metric": (
                            best_test_results["avg_challenge_metric"] if best_test_results is not None else None
                        ),
                        "best/test_rel_l2": (
                            best_test_results["avg_rel_l2"] if best_test_results is not None else None
                        ),
                        "best/test_mse": (
                            best_test_results["avg_mse"] if best_test_results is not None else None
                        ),

                        "is_best": int(is_best),
                        "epoch": epoch,
                    }
                    wandb.log(eval_log, step=global_step)

    print("\n====================== FINAL SUMMARY ======================")
    print(f"Best checkpoint                : {best_ckpt_path}")
    print(f"Best metrics json              : {best_metrics_path}")
    print(f"Best val Challenge Metric      : {best_val_challenge_metric:.8f}")
    if best_val_results is not None:
        print(f"Best val Relative L2           : {best_val_results['avg_rel_l2']:.8f}")
        print(f"Best val MSE                   : {best_val_results['avg_mse']:.8f}")
    if best_test_results is not None:
        print(f"Best test Challenge Metric     : {best_test_results['avg_challenge_metric']:.8f}")
        print(f"Best test Relative L2          : {best_test_results['avg_rel_l2']:.8f}")
        print(f"Best test MSE                  : {best_test_results['avg_mse']:.8f}")
    print("===========================================================")

    if conf.get("wandb", False):
        wandb.summary["best_val_challenge_metric"] = best_val_challenge_metric
        if best_val_results is not None:
            wandb.summary["best_val_relative_l2"] = best_val_results["avg_rel_l2"]
            wandb.summary["best_val_mse"] = best_val_results["avg_mse"]
        if best_test_results is not None:
            wandb.summary["best_test_challenge_metric"] = best_test_results["avg_challenge_metric"]
            wandb.summary["best_test_relative_l2"] = best_test_results["avg_rel_l2"]
            wandb.summary["best_test_mse"] = best_test_results["avg_mse"]
        wandb.finish()


if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if hasattr(args, "gpu_ids") and isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    main(args=args)