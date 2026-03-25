# python /home/intern/spygeorgoulas/thesis-metanets/scalegmn/inference_scalegmn_navier.py --checkpoint /home/intern/spygeorgoulas/thesis-metanets/scalegmn/checkpoints/best_scalegmn_navier_direct_prediction.pt --input_inr_path /home/intern/spygeorgoulas/thesis-metanets/scalegmn/data/navier_pipe_geometry_2x64_2k_rwi_pth_linearonly_splitted_all/test/inr_02183.pth --gt_output_inr_path /home/intern/spygeorgoulas/thesis-metanets/scalegmn/data/navier_h_velocity_2x64_2k_rwi_pth_parallel_linearonly_splitted_all/test/inr_02183.pth --input_inr_dir /home/intern/spygeorgoulas/thesis-metanets/scalegmn/data/navier_pipe_geometry_2x64_2k_rwi_pth_linearonly_splitted_all --output_dir /home/intern/spygeorgoulas/thesis-metanets/scalegmn/inference_scalegmn_navier_imgs

#!/usr/bin/env python3
import os
import yaml
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from src.data.base_datasets import BaseDataset, Batch
from src.scalegmn.models import ScaleGMN_equiv
from src.utils.helpers import mask_input, mask_hidden

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


# ============================================================================================
# HELPERS
# ============================================================================================
def make_ref_grid(H: int, W: int) -> torch.Tensor:
    xs = np.linspace(0.0, 1.0, H, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, W, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return torch.from_numpy(np.stack([xx, yy], axis=-1).reshape(-1, 2))  # [P, 2]


def clean_state_dict(ckpt):
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned[nk] = v
    return cleaned


def state_dict_to_weight_bias_tuples(state_dict: dict):
    """
    Converts linear-only INR state_dict to:
        weights[l] : [in_dim, out_dim, 1]
        biases[l]  : [out_dim, 1]
    """
    weight_items = []
    bias_items = []

    for k, v in state_dict.items():
        if "weight" in k:
            layer_idx = int(k.split(".")[1])
            weight_items.append((layer_idx, v.float()))
        elif "bias" in k:
            layer_idx = int(k.split(".")[1])
            bias_items.append((layer_idx, v.float()))

    weight_items = sorted(weight_items, key=lambda x: x[0])
    bias_items = sorted(bias_items, key=lambda x: x[0])

    weights = tuple(v.permute(1, 0).unsqueeze(-1) for _, v in weight_items)  # [in,out,1]
    biases = tuple(v.unsqueeze(-1) for _, v in bias_items)                    # [out,1]
    return weights, biases


def tuples_to_state_dict(weights, biases):
    sd = OrderedDict()
    for i, (w, b) in enumerate(zip(weights, biases)):
        layer_key = 2 * i
        w_single = w[0].squeeze(-1).permute(1, 0).detach().cpu().contiguous()  # [out,in]
        b_single = b[0].squeeze(-1).detach().cpu().contiguous()                 # [out]
        sd[f"seq.{layer_key}.weight"] = w_single
        sd[f"seq.{layer_key}.bias"] = b_single
    return sd


def make_zero_wb_from_layout(batch_size: int, layer_layout, device):
    weights = []
    biases = []
    for in_dim, out_dim in zip(layer_layout[:-1], layer_layout[1:]):
        w = torch.zeros(batch_size, in_dim, out_dim, 1, device=device)
        b = torch.zeros(batch_size, out_dim, 1, device=device)
        weights.append(w)
        biases.append(b)
    return weights, biases


def project_input_layout_prediction_to_output_layout(
    pred_weights,
    pred_biases,
    input_layout,
    output_layout,
):
    if len(input_layout) != len(output_layout):
        raise ValueError(
            f"Different number of layers in input/output layouts: {input_layout} vs {output_layout}"
        )

    out_weights = []
    out_biases = []

    num_layers = len(input_layout) - 1
    for l in range(num_layers):
        in_in = input_layout[l]
        in_out = input_layout[l + 1]
        out_in = output_layout[l]
        out_out = output_layout[l + 1]

        w = pred_weights[l]
        b = pred_biases[l]

        if l < num_layers - 1:
            if in_in != out_in or in_out != out_out:
                raise ValueError(
                    f"Shared layer mismatch at layer {l}: "
                    f"input ({in_in},{in_out}) vs output ({out_in},{out_out})"
                )
            out_weights.append(w)
            out_biases.append(b)
        else:
            if in_in != out_in:
                raise ValueError(f"Final layer input dim mismatch: {in_in} vs {out_in}")
            if out_out > in_out:
                raise ValueError(
                    f"Cannot expand final output dim by slicing: input {in_out}, output {out_out}"
                )

            out_weights.append(w[:, :, :out_out, :])  # [B, in_dim, out_out, 1]
            out_biases.append(b[:, :out_out, :])      # [B, out_out, 1]

    return out_weights, out_biases


def relative_l2(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)
    return float(np.linalg.norm(pred_flat - gt_flat) / (np.linalg.norm(gt_flat) + eps))


# ============================================================================================
# DATASET
# ============================================================================================
class NavierGeometryINRDataset(BaseDataset):
    """
    Used only to convert the input INR into the graph representation expected by ScaleGMN.
    """

    def __init__(
        self,
        dataset,
        dataset_path,
        split_path=None,
        debug=False,
        split="train",
        node_pos_embed=False,
        edge_pos_embed=False,
        equiv_on_hidden=False,
        get_first_layer_mask=False,
        image_size=(129, 129),
        direction="forward",
        layer_layout=None,
        return_path=False,
        data_format="graph",
        switch_to_canon=False,
    ):
        super().__init__(
            dataset=dataset,
            dataset_path=dataset_path,
            split_path=split_path,
            split=split,
            node_pos_embed=node_pos_embed,
            edge_pos_embed=edge_pos_embed,
            equiv_on_hidden=equiv_on_hidden,
            get_first_layer_mask=get_first_layer_mask,
            image_size=image_size,
            layer_layout=layer_layout,
            direction=direction,
            return_path=return_path,
            data_format=data_format,
            switch_to_canon=switch_to_canon,
        )

        if debug:
            self.dataset = self.dataset[:16]

    def get_path(self, index):
        rel_path = self.dataset[index]
        abs_path = Path(self.dataset_path) / rel_path
        return str(abs_path), None

    def get_label(self, index, state_dict, aux):
        rel_path = Path(self.dataset[index])
        parent_name = rel_path.parent.name
        label = int(parent_name) if parent_name.isdigit() else 0
        return torch.tensor(label, dtype=torch.long)

    def load_dataset(self, split_path=None):
        split_dir = Path(self.dataset_path) / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        file_list = []

        direct_files = sorted(split_dir.glob("*.pth"))
        file_list.extend([str(p.relative_to(self.dataset_path)) for p in direct_files])

        for sub_dir in sorted(split_dir.iterdir()):
            if sub_dir.is_dir():
                sub_files = sorted(sub_dir.glob("*.pth"))
                file_list.extend([str(p.relative_to(self.dataset_path)) for p in sub_files])

        if len(file_list) == 0:
            raise RuntimeError(f"No .pth files found under: {split_dir}")

        return file_list


# ============================================================================================
# INR INFERENCE
# ============================================================================================
class BatchSirenLinearOnlyRefGrid(nn.Module):
    """
    Evaluates linear-only SIREN INRs on the fixed normalized reference grid.

    Input:
        weights[l]: [B, in_dim, out_dim, 1]
        biases[l]:  [B, out_dim, 1]

    Output:
        [B, P, out_dim_last]
    """

    def __init__(self, image_size=(129, 129), w0=30.0, w0_first=30.0):
        super().__init__()
        self.w0 = float(w0)
        self.w0_first = float(w0_first)
        self.image_size = tuple(image_size)

        coords = make_ref_grid(self.image_size[0], self.image_size[1])
        self.register_buffer("coords", coords, persistent=False)

    def forward(self, weights, biases):
        batch_size = weights[0].shape[0]
        x = self.coords.unsqueeze(0).expand(batch_size, -1, -1)  # [B,P,2]

        num_layers = len(weights)
        for i in range(num_layers):
            w = weights[i].squeeze(-1)  # [B,in,out]
            b = biases[i].squeeze(-1)   # [B,out]
            x = torch.einsum("bpi,bio->bpo", x, w) + b.unsqueeze(1)

            if i < num_layers - 1:
                if i == 0:
                    x = torch.sin(self.w0_first * x)
                else:
                    x = torch.sin(self.w0 * x)

        return x


# ============================================================================================
# PATH RESOLUTION
# ============================================================================================
def resolve_split_and_rel_path(input_inr_path: str, input_inr_dir: str):
    root = Path(input_inr_dir).resolve()
    fpath = Path(input_inr_path).resolve()

    if not fpath.exists():
        raise FileNotFoundError(f"Input INR path does not exist: {fpath}")

    try:
        rel_path = fpath.relative_to(root)
    except ValueError as e:
        raise ValueError(
            f"input_inr_path must be inside input_inr_dir.\n"
            f"input_inr_dir={root}\n"
            f"input_inr_path={fpath}"
        ) from e

    if len(rel_path.parts) < 2:
        raise ValueError(
            f"Expected input_inr_path under split folder, e.g. train/inr_xxx.pth.\n"
            f"Got relative path: {rel_path}"
        )

    split = rel_path.parts[0]
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Expected split train/val/test, got: {split}")

    return split, str(rel_path)


# ============================================================================================
# PLOTTING
# ============================================================================================
def save_input_geometry_image(input_field: np.ndarray, save_path: str, title_prefix: str):
    """
    input_field: [H, W, 2]
    """
    x_phys = input_field[:, :, 0]
    y_phys = input_field[:, :, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(x_phys, origin="lower", aspect="auto")
    axes[0].set_title(f"{title_prefix}\nInput geometry - X")
    axes[0].set_xlabel("W")
    axes[0].set_ylabel("H")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(y_phys, origin="lower", aspect="auto")
    axes[1].set_title(f"{title_prefix}\nInput geometry - Y")
    axes[1].set_xlabel("W")
    axes[1].set_ylabel("H")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_geometry_velocity_combined_figure(
    X_geom,
    Y_geom,
    X_gt,
    Y_gt,
    u_pred,
    u_gt,
    rel_l2_value,
    save_path,
    title_prefix,
    scatter_stride=1,
):
    """
    Combined figure in physical space.
    4 panels:
      1) input geometry only
      2) predicted velocity on input geometry
      3) GT velocity on GT geometry
      4) |velocity error| on GT geometry
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    Xg_in = X_geom[::scatter_stride, ::scatter_stride].reshape(-1)
    Yg_in = Y_geom[::scatter_stride, ::scatter_stride].reshape(-1)

    Xg_gt = X_gt[::scatter_stride, ::scatter_stride].reshape(-1)
    Yg_gt = Y_gt[::scatter_stride, ::scatter_stride].reshape(-1)

    Up = u_pred[::scatter_stride, ::scatter_stride].reshape(-1)
    Ug = u_gt[::scatter_stride, ::scatter_stride].reshape(-1)
    Ue = np.abs(u_pred - u_gt)[::scatter_stride, ::scatter_stride].reshape(-1)

    vmin = min(float(u_pred.min()), float(u_gt.min()))
    vmax = max(float(u_pred.max()), float(u_gt.max()))

    # Panel 1: geometry only
    axes[0, 0].scatter(Xg_in, Yg_in, s=6)
    axes[0, 0].set_title("Input geometry")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("Y")

    # Panel 2: predicted velocity on input geometry
    sc1 = axes[0, 1].scatter(Xg_in, Yg_in, c=Up, s=8, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"Predicted velocity on input geometry\nRelative L2 = {rel_l2_value:.6e}")
    axes[0, 1].set_aspect("equal")
    axes[0, 1].set_xlabel("X")
    axes[0, 1].set_ylabel("Y")
    plt.colorbar(sc1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Panel 3: GT velocity on GT geometry
    sc2 = axes[1, 0].scatter(Xg_gt, Yg_gt, c=Ug, s=8, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Ground-truth velocity on ground-truth geometry")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")
    plt.colorbar(sc2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Panel 4: error
    sc3 = axes[1, 1].scatter(Xg_gt, Yg_gt, c=Ue, s=8, cmap="magma")
    axes[1, 1].set_title("|Velocity error| on ground-truth geometry")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_xlabel("X")
    axes[1, 1].set_ylabel("Y")
    plt.colorbar(sc3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(f"{title_prefix}\nGeometry + velocity comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================================================
# MAIN
# ============================================================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_inr_path", type=str, required=True)
    parser.add_argument("--gt_output_inr_path", type=str, required=True)
    parser.add_argument("--input_inr_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("--w0", type=float, default=None)
    parser.add_argument("--w0_first", type=float, default=None)
    parser.add_argument("--scatter_stride", type=int, default=1)

    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load checkpoint and config
    # -------------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    if isinstance(ckpt, dict) and "config" in ckpt:
        conf = ckpt["config"]
    elif args.conf is not None:
        with open(args.conf, "r") as f:
            conf = yaml.safe_load(f)
    else:
        raise ValueError("Checkpoint does not contain config. Please pass --conf path_to_yaml")

    input_layer_layout = conf["data"]["input_layer_layout"]
    output_layer_layout = conf["data"]["output_layer_layout"]
    image_size = tuple(conf["data"]["image_size"])
    node_pos_embed = conf["data"].get("node_pos_embed", False)
    edge_pos_embed = conf["data"].get("edge_pos_embed", False)

    expected_input_layout = [2, 64, 64, 2]
    expected_output_layout = [2, 64, 64, 1]

    if list(input_layer_layout) != expected_input_layout:
        raise ValueError(f"Unexpected input layout. Expected {expected_input_layout}, got {input_layer_layout}")
    if list(output_layer_layout) != expected_output_layout:
        raise ValueError(f"Unexpected output layout. Expected {expected_output_layout}, got {output_layer_layout}")

    w0 = args.w0 if args.w0 is not None else conf["inr_model"]["w0"]
    w0_first = args.w0_first if args.w0_first is not None else conf["inr_model"]["w0_first"]

    equiv_on_hidden = mask_hidden(conf)
    get_first_layer_mask = mask_input(conf)

    # -------------------------------------------------------------------------
    # Resolve dataset path info
    # -------------------------------------------------------------------------
    split, rel_path = resolve_split_and_rel_path(
        input_inr_path=args.input_inr_path,
        input_inr_dir=args.input_inr_dir,
    )

    input_inr_path = str(Path(args.input_inr_path).resolve())
    gt_output_inr_path = str(Path(args.gt_output_inr_path).resolve())

    if not Path(input_inr_path).exists():
        raise FileNotFoundError(f"Missing input INR: {input_inr_path}")
    if not Path(gt_output_inr_path).exists():
        raise FileNotFoundError(f"Missing GT output INR: {gt_output_inr_path}")

    sample_name = Path(input_inr_path).stem  # e.g. inr_02161
    print(f"Sample name: {sample_name}")

    # -------------------------------------------------------------------------
    # Build dataset for graph conversion
    # -------------------------------------------------------------------------
    ds = NavierGeometryINRDataset(
        dataset=conf["data"]["dataset"],
        dataset_path=args.input_inr_dir,
        split=split,
        debug=False,
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=equiv_on_hidden,
        get_first_layer_mask=get_first_layer_mask,
        node_pos_embed=node_pos_embed,
        edge_pos_embed=edge_pos_embed,
        image_size=image_size,
        layer_layout=input_layer_layout,
        return_path=False,
        data_format="graph",
        switch_to_canon=False,
    )

    matching_indices = [i for i, p in enumerate(ds.dataset) if p == rel_path]
    if len(matching_indices) == 0:
        raise RuntimeError(f"Could not find {rel_path} inside dataset list for split={split}")
    ds_index = matching_indices[0]

    params = ds[ds_index]
    params = params.to(device)

    # -------------------------------------------------------------------------
    # Load exact INR files
    # -------------------------------------------------------------------------
    input_sd = clean_state_dict(torch.load(input_inr_path, map_location="cpu"))
    gt_output_sd = clean_state_dict(torch.load(gt_output_inr_path, map_location="cpu"))

    input_weights, input_biases = state_dict_to_weight_bias_tuples(input_sd)
    gt_out_weights, gt_out_biases = state_dict_to_weight_bias_tuples(gt_output_sd)

    input_wb = Batch(
        weights=tuple(w.unsqueeze(0).to(device) for w in input_weights),
        biases=tuple(b.unsqueeze(0).to(device) for b in input_biases),
        label=torch.tensor([0], dtype=torch.long, device=device),
    )

    gt_out_wb = Batch(
        weights=tuple(w.unsqueeze(0).to(device) for w in gt_out_weights),
        biases=tuple(b.unsqueeze(0).to(device) for b in gt_out_biases),
        label=torch.tensor([0], dtype=torch.long, device=device),
    )

    # -------------------------------------------------------------------------
    # Build and load ScaleGMN
    # -------------------------------------------------------------------------
    conf["scalegmn_args"]["layer_layout"] = ds.get_layer_layout()

    net = ScaleGMN_equiv(conf["scalegmn_args"]).to(device)
    net.eval()

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        net.load_state_dict(ckpt["model_state_dict"])
    else:
        net.load_state_dict(ckpt)

    inr_model = BatchSirenLinearOnlyRefGrid(
        image_size=image_size,
        w0=w0,
        w0_first=w0_first,
    ).to(device)
    inr_model.eval()

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    with torch.no_grad():
        zero_in_weights, zero_in_biases = make_zero_wb_from_layout(
            batch_size=1,
            layer_layout=input_layer_layout,
            device=device,
        )

        pred_in_weights, pred_in_biases = net(params, zero_in_weights, zero_in_biases)

        pred_out_weights, pred_out_biases = project_input_layout_prediction_to_output_layout(
            pred_in_weights,
            pred_in_biases,
            input_layout=input_layer_layout,
            output_layout=output_layer_layout,
        )

        # input geometry field [1,P,2]
        input_field = inr_model(input_wb.weights, input_wb.biases)

        # predicted and GT velocity fields [1,P,1]
        pred_out_field = inr_model(pred_out_weights, pred_out_biases)
        gt_out_field = inr_model(gt_out_wb.weights, gt_out_wb.biases)

    H, W = image_size

    input_field_np = input_field[0].detach().cpu().numpy().reshape(H, W, 2)
    pred_out_np = pred_out_field[0].detach().cpu().numpy().reshape(H, W)
    gt_out_np = gt_out_field[0].detach().cpu().numpy().reshape(H, W)

    X_geom = input_field_np[:, :, 0]
    Y_geom = input_field_np[:, :, 1]

    # Since GT output INR is only velocity, we use the same geometry for GT comparison if no separate GT geometry is provided.
    # But here, for comparison on the pipe, the best available geometry from this inference pipeline is the input geometry.
    # If you want GT geometry from another source, you would need to load it separately.
    X_gt = X_geom
    Y_gt = Y_geom

    pred_rel_l2 = relative_l2(pred_out_np, gt_out_np)
    print(f"Predicted output relative L2: {pred_rel_l2:.6e}")

    # -------------------------------------------------------------------------
    # Save predicted INR
    # -------------------------------------------------------------------------
    pred_output_sd = tuples_to_state_dict(pred_out_weights, pred_out_biases)
    pred_output_sd_path = output_dir / f"{sample_name}_predicted_output_inr.pth"
    torch.save(pred_output_sd, pred_output_sd_path)

    # -------------------------------------------------------------------------
    # Save images
    # -------------------------------------------------------------------------
    input_img_path = output_dir / f"{sample_name}_input_geometry.png"
    combined_img_path = output_dir / f"{sample_name}_geometry_velocity_combined.png"

    save_input_geometry_image(
        input_field=input_field_np,
        save_path=str(input_img_path),
        title_prefix=sample_name,
    )

    save_geometry_velocity_combined_figure(
        X_geom=X_geom,
        Y_geom=Y_geom,
        X_gt=X_gt,
        Y_gt=Y_gt,
        u_pred=pred_out_np,
        u_gt=gt_out_np,
        rel_l2_value=pred_rel_l2,
        save_path=str(combined_img_path),
        title_prefix=sample_name,
        scatter_stride=args.scatter_stride,
    )

    print("\nSaved files:")
    print(input_img_path)
    print(combined_img_path)
    print(pred_output_sd_path)


if __name__ == "__main__":
    main()