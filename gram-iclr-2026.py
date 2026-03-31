# 

#!/usr/bin/env python3

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

from pathlib import Path
import yaml
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch_geometric
import wandb

from src.data import dataset
from src.utils.setup_arg_parser import setup_arg_parser
from src.scalegmn.models import ScaleGMN_equiv
from src.utils.optim import setup_optimization
from src.utils.helpers import (
    overwrite_conf,
    count_parameters,
    set_seed,
    mask_input,
    mask_hidden,
)


# ============================================================
# INR querying in function space
# ============================================================

class BatchSirenQuery(nn.Module):
    """
    Query a batch of SIREN INRs directly on arbitrary coordinates.

    Expected parameter layout:
        weights[j]: [B, in_dim, out_dim, 1]  or [B, in_dim, out_dim]
        biases[j]:  [B, out_dim, 1]          or [B, out_dim]

    Query shape:
        coords: [B, M, in_dim]

    Output:
        [B, M, out_dim]
    """

    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, weights, biases, coords):
        x = coords  # [B, M, in_dim]
        num_layers = len(weights)

        for j in range(num_layers):
            w = weights[j]
            b = biases[j]

            if w.dim() == 4:
                w = w.squeeze(-1)      # [B, in_dim, out_dim]
            if b.dim() == 3:
                b = b.squeeze(-1)      # [B, out_dim]

            x = torch.einsum("bmi,bio->bmo", x, w) + b.unsqueeze(1)

            if j < num_layers - 1:
                x = torch.sin(self.w0 * x)

        return x


def residual_param_update(weights, biases, delta_weights, delta_biases):
    new_weights = [weights[j] + delta_weights[j] for j in range(len(weights))]
    new_biases = [biases[j] + delta_biases[j] for j in range(len(biases))]
    return new_weights, new_biases


def mse_metric(pred, target):
    return ((pred - target) ** 2).mean()


def relative_l2_metric(pred, target, eps=1e-8):
    return torch.norm(pred - target) / (torch.norm(target) + eps)


# ============================================================
# Ground-truth loading from original .npz files
# ============================================================

def resolve_npz_path_from_inr_path(inr_path, source_npz_dir):
    """
    Accepts either:
      - absolute INR path, e.g. /.../train/1021_1-0.pth
      - relative INR path, e.g. train/1021_1-0.pth

    Tries to resolve the corresponding .npz file under source_npz_dir.
    """
    inr_path = Path(inr_path)
    source_npz_dir = Path(source_npz_dir)

    stem = inr_path.stem

    split = None
    parts = inr_path.parts
    for candidate in ("train", "val", "test"):
        if candidate in parts:
            split = candidate
            break

    candidates = []
    if split is not None:
        candidates.append(source_npz_dir / split / f"{stem}.npz")
    candidates.append(source_npz_dir / f"{stem}.npz")

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not resolve npz for INR path {inr_path}. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def load_velocity_out_queries(npz_path):
    """
    Build full query set for velocity_out.

    .npz contains:
        pos:          (N, 3)
        t:            (10,)
        velocity_out: (5, N, 3)

    We use future times t[5:10].

    Returns:
        queries: (5*N, 4) -> [x, y, z, t]
        target:  (5*N, 3) -> [u, v, w]
    """
    data = np.load(npz_path)

    pos = data["pos"].astype(np.float32)               # (N, 3)
    t = data["t"].astype(np.float32)                   # (10,)
    vel_out = data["velocity_out"].astype(np.float32)  # (5, N, 3)

    t_out = t[5:10]                                    # (5,)
    n_points = pos.shape[0]
    n_times = vel_out.shape[0]

    pos_rep = np.broadcast_to(pos[None, :, :], (n_times, n_points, 3)).copy()
    t_rep = np.broadcast_to(t_out[:, None, None], (n_times, n_points, 1)).copy()

    queries = np.concatenate([pos_rep, t_rep], axis=-1)      # (5, N, 4)
    queries = queries.reshape(n_times * n_points, 4)         # (5N, 4)
    target = vel_out.reshape(n_times * n_points, 3)          # (5N, 3)

    return queries, target


def build_training_query_batch(paths, source_npz_dir, num_query_points, device):
    """
    For each sample in batch:
      - load full future query set from npz
      - random-subsample num_query_points from full set
      - stack into [B, Q, 4], [B, Q, 3]
    """
    q_list = []
    y_list = []

    for p in paths:
        npz_path = resolve_npz_path_from_inr_path(p, source_npz_dir)
        queries_np, target_np = load_velocity_out_queries(npz_path)

        n = queries_np.shape[0]
        if num_query_points is None or num_query_points >= n:
            idx = np.arange(n)
        else:
            idx = np.random.randint(0, n, size=(num_query_points,))

        q = torch.from_numpy(queries_np[idx]).float().to(device, non_blocking=True)
        y = torch.from_numpy(target_np[idx]).float().to(device, non_blocking=True)

        q_list.append(q)
        y_list.append(y)

    queries = torch.stack(q_list, dim=0)   # [B, Q, 4]
    targets = torch.stack(y_list, dim=0)   # [B, Q, 3]
    return queries, targets


@torch.no_grad()
def evaluate_function_space(
    model,
    loader,
    device,
    siren_query,
    source_npz_dir,
    query_eval_chunk_size=16384,
    num_batches=None,
):
    """
    Full function-space evaluation:
      - predict edited INR
      - query predicted INR on the full future space-time set
      - compare to ground-truth velocity_out
    """
    model.eval()

    mse_values = []
    rel_l2_values = []

    for i, batch in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break

        params, w_b, _, paths = unpack_batch_with_paths(batch)

        params = params.to(device)
        w_b = w_b.to(device)
        weights = w_b.weights
        biases = w_b.biases

        delta_weights, delta_biases = model(params, weights, biases)
        pred_weights, pred_biases = residual_param_update(weights, biases, delta_weights, delta_biases)

        batch_size = len(paths)

        for b in range(batch_size):
            npz_path = resolve_npz_path_from_inr_path(paths[b], source_npz_dir)
            queries_np, target_np = load_velocity_out_queries(npz_path)

            queries = torch.from_numpy(queries_np).float().to(device, non_blocking=True)
            target = torch.from_numpy(target_np).float().to(device, non_blocking=True)

            preds = []
            n = queries.shape[0]

            for start in range(0, n, query_eval_chunk_size):
                end = min(start + query_eval_chunk_size, n)
                q_chunk = queries[start:end].unsqueeze(0)  # [1, M, 4]

                pred_chunk = siren_query(
                    [pred_weights[j][b:b+1] for j in range(len(pred_weights))],
                    [pred_biases[j][b:b+1] for j in range(len(pred_biases))],
                    q_chunk,
                )  # [1, M, 3]

                preds.append(pred_chunk.squeeze(0))

            pred_full = torch.cat(preds, dim=0)  # [5N, 3]

            mse_val = mse_metric(pred_full, target)
            rel_l2_val = relative_l2_metric(pred_full, target)

            mse_values.append(mse_val.detach().cpu())
            rel_l2_values.append(rel_l2_val.detach().cpu())

    mse_mean = torch.stack(mse_values).mean() if len(mse_values) > 0 else torch.tensor(float("nan"))
    rel_l2_mean = torch.stack(rel_l2_values).mean() if len(rel_l2_values) > 0 else torch.tensor(float("nan"))

    model.train()
    return {
        "avg_mse": mse_mean,
        "avg_relative_l2": rel_l2_mean,
    }


def unpack_batch_with_paths(batch):
    """
    Expected dataset output when return_path=True:
        (params, w_b, label, path)
    """
    if not isinstance(batch, (list, tuple)):
        raise ValueError("Unexpected batch type. Expected tuple/list.")

    if len(batch) != 4:
        raise ValueError(
            f"Expected batch of length 4: (params, w_b, label, path), got length {len(batch)}"
        )

    params, w_b, label, paths = batch

    if isinstance(paths, tuple):
        paths = list(paths)
    elif not isinstance(paths, list):
        paths = [paths]

    paths = [str(p) for p in paths]
    return params, w_b, label, paths


# ============================================================
# Main
# ============================================================

def main(args=None):
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, vars(args))

    torch.set_float32_matmul_precision("high")
    print(yaml.dump(conf, default_flow_style=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if conf["wandb"]:
        wandb.init(config=conf, **conf["wandb_args"])

    set_seed(conf["train_args"]["seed"])

    # ============================================================
    # Dataset / Dataloader
    # ============================================================
    conf["data"]["return_path"] = True

    equiv_on_hidden = mask_hidden(conf)
    get_first_layer_mask = mask_input(conf)

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

    # ============================================================
    # Models
    # ============================================================
    net = ScaleGMN_equiv(conf["scalegmn_args"]).to(device)
    siren_query = BatchSirenQuery(w0=conf["inr_model"]["w0"]).to(device)

    print(net)
    cnt_p = count_parameters(net=net)
    print(f"ScaleGMN params: {cnt_p}")

    if conf["wandb"]:
        wandb.log({"number of parameters": cnt_p}, step=0)

    for p in net.parameters():
        p.requires_grad = True

    # ============================================================
    # Optimization
    # ============================================================
    criterion = nn.MSELoss()

    conf_opt = conf["optimization"]
    model_params = [p for p in net.parameters() if p.requires_grad]
    optimizer, scheduler = setup_optimization(
        model_params,
        optimizer_name=conf_opt["optimizer_name"],
        optimizer_args=conf_opt["optimizer_args"],
        scheduler_args=conf_opt["scheduler_args"],
    )

    # ============================================================
    # Training loop
    # ============================================================
    best_val_mse = float("inf")
    best_val_results = None
    best_test_results = None
    test_mse_scalar = -1.0
    global_step = 0

    source_npz_dir = conf["data"]["source_npz_dir"]
    num_query_points = conf["train_args"]["num_query_points"]
    query_eval_chunk_size = conf["train_args"].get("query_eval_chunk_size", 16384)

    epoch_iter = trange(conf["train_args"]["num_epochs"])
    net.train()

    for epoch in epoch_iter:
        for i, batch in enumerate(train_loader):
            params, w_b, _, paths = unpack_batch_with_paths(batch)

            params = params.to(device)
            w_b = w_b.to(device)
            weights = w_b.weights
            biases = w_b.biases

            optimizer.zero_grad()

            # ScaleGMN predicts residual edit of the input INR
            delta_weights, delta_biases = net(params, weights, biases)
            pred_weights, pred_biases = residual_param_update(weights, biases, delta_weights, delta_biases)

            # Function-space supervision from raw velocity_out
            queries, targets = build_training_query_batch(
                paths=paths,
                source_npz_dir=source_npz_dir,
                num_query_points=num_query_points,
                device=device,
            )  # queries [B,Q,4], targets [B,Q,3]

            pred_velocity = siren_query(pred_weights, pred_biases, queries)  # [B,Q,3]

            loss = criterion(pred_velocity, targets)
            loss.backward()

            log = {
                "train/loss_mse": loss.item(),
                "global_step": global_step,
            }

            if conf["optimization"]["clip_grad"]:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in net.parameters() if p.requires_grad],
                    conf["optimization"]["clip_grad_max_norm"],
                )
                log["grad_norm"] = float(grad_norm)

            optimizer.step()

            if scheduler[1] is not None and scheduler[1] != "ReduceLROnPlateau":
                log["lr"] = scheduler[0].get_last_lr()[0]
                scheduler[0].step()

            if conf["wandb"]:
                wandb.log(log)

            epoch_iter.set_description(
                f"[epoch {epoch} | batch {i+1}] train_mse={loss.item():.6f}, test_mse={test_mse_scalar:.6f}"
            )

            global_step += 1

            if (global_step + 1) % conf["train_args"]["eval_every"] == 0:
                val_metrics = evaluate_function_space(
                    model=net,
                    loader=val_loader,
                    device=device,
                    siren_query=siren_query,
                    source_npz_dir=source_npz_dir,
                    query_eval_chunk_size=query_eval_chunk_size,
                    num_batches=None,
                )
                test_metrics = evaluate_function_space(
                    model=net,
                    loader=test_loader,
                    device=device,
                    siren_query=siren_query,
                    source_npz_dir=source_npz_dir,
                    query_eval_chunk_size=query_eval_chunk_size,
                    num_batches=None,
                )
                train_metrics = evaluate_function_space(
                    model=net,
                    loader=train_loader,
                    device=device,
                    siren_query=siren_query,
                    source_npz_dir=source_npz_dir,
                    query_eval_chunk_size=query_eval_chunk_size,
                    num_batches=conf["train_args"].get("train_eval_num_batches", 20),
                )

                val_mse = float(val_metrics["avg_mse"])
                test_mse_scalar = float(test_metrics["avg_mse"])

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_val_results = val_metrics
                    best_test_results = test_metrics

                if conf["wandb"]:
                    wandb.log({
                        "epoch": epoch,
                        "global_step": global_step,

                        "train/avg_mse": float(train_metrics["avg_mse"]),
                        "train/avg_relative_l2": float(train_metrics["avg_relative_l2"]),

                        "val/avg_mse": float(val_metrics["avg_mse"]),
                        "val/avg_relative_l2": float(val_metrics["avg_relative_l2"]),

                        "test/avg_mse": float(test_metrics["avg_mse"]),
                        "test/avg_relative_l2": float(test_metrics["avg_relative_l2"]),

                        "val/best_mse": float(best_val_results["avg_mse"]),
                        "val/best_relative_l2": float(best_val_results["avg_relative_l2"]),
                        "test/best_at_best_val_mse": float(best_test_results["avg_mse"]),
                        "test/best_at_best_val_relative_l2": float(best_test_results["avg_relative_l2"]),
                    })

                print(
                    f"\n[Eval @ step {global_step}] "
                    f"train_mse={float(train_metrics['avg_mse']):.6f}, "
                    f"train_relL2={float(train_metrics['avg_relative_l2']):.6f}, "
                    f"val_mse={float(val_metrics['avg_mse']):.6f}, "
                    f"val_relL2={float(val_metrics['avg_relative_l2']):.6f}, "
                    f"test_mse={float(test_metrics['avg_mse']):.6f}, "
                    f"test_relL2={float(test_metrics['avg_relative_l2']):.6f}"
                )

    # ============================================================
    # Final summary
    # ============================================================
    print("\n================ FINAL RESULTS ================")
    if best_val_results is not None:
        print(f"Best val MSE:           {float(best_val_results['avg_mse']):.8f}")
        print(f"Best val Relative L2:   {float(best_val_results['avg_relative_l2']):.8f}")
        print(f"Test MSE @ best val:    {float(best_test_results['avg_mse']):.8f}")
        print(f"Test RelL2 @ best val:  {float(best_test_results['avg_relative_l2']):.8f}")
    print("===============================================")


if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    main(args=args)