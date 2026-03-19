# python /home/intern/spygeorgoulas/thesis-metanets/scalegmn/physics_neural_operator.py --conf /home/intern/spygeorgoulas/thesis-metanets/scalegmn/configs/physics/scalegmn.yml --wandb true

#!/usr/bin/env python3
import os
import re
import yaml
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch_geometric
import wandb

from tqdm import trange

from src.data.base_datasets import BaseDataset, Batch
from src.utils.setup_arg_parser import setup_arg_parser
from src.scalegmn.models import ScaleGMN_equiv
from src.utils.helpers import overwrite_conf, count_parameters, set_seed, mask_input, mask_hidden

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ.pop("WANDB_MODE", None)


# ============================================================================================
# HELPERS
# ============================================================================================
def parse_sample_idx_from_path(path_str: str) -> int:
    m = re.search(r"inr_(\d+)\.pth$", str(path_str))
    if m is None:
        raise ValueError(f"Could not parse sample index from path: {path_str}")
    return int(m.group(1))


def make_ref_grid(H: int, W: int) -> torch.Tensor:
    xs = np.linspace(0.0, 1.0, H, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, W, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return torch.from_numpy(np.stack([xx, yy], axis=-1).reshape(-1, 2))  # [P, 2]


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


def batch_to_device_namedtuple(wb: Batch, device):
    return Batch(
        weights=tuple(w.to(device) for w in wb.weights),
        biases=tuple(b.to(device) for b in wb.biases),
        label=wb.label.to(device),
    )


def clone_namedtuple_wb(wb: Batch):
    return Batch(
        weights=tuple(w.clone() for w in wb.weights),
        biases=tuple(b.clone() for b in wb.biases),
        label=wb.label.clone(),
    )


def validate_layouts_for_final_projection(input_layout, output_layout):
    """
    Supports only the case where input/output layouts differ
    only in the final output dimension.

    Example supported:
        [2, 64, 64, 2] -> [2, 64, 64, 1]
    """
    if len(input_layout) != len(output_layout):
        raise ValueError(
            f"Different number of layers: {input_layout} vs {output_layout}"
        )

    if input_layout[:-1] != output_layout[:-1]:
        raise ValueError(
            "This script supports only final-output-dimension mismatch. "
            f"Got input_layout={input_layout}, output_layout={output_layout}"
        )


# ============================================================================================
# FINAL-LAYER PROJECTION HEAD
# ============================================================================================
class FinalLayerProjection(nn.Module):
    """
    Projects the final predicted layer from input final output-dim -> target final output-dim.

    Modes:
      - "linear": learned linear projection
      - "mean":   fixed mean grouping
      - "slice":  keep first out_out channels
      - "identity": no projection, requires same dim

    Input:
      final_w: [B, in_dim, in_out, 1]
      final_b: [B, in_out, 1]

    Output:
      proj_w: [B, in_dim, out_out, 1]
      proj_b: [B, out_out, 1]
    """

    def __init__(self, in_out_dim: int, out_out_dim: int, mode: str = "linear"):
        super().__init__()
        self.in_out_dim = int(in_out_dim)
        self.out_out_dim = int(out_out_dim)
        self.mode = str(mode).lower()

        if self.mode not in {"linear", "mean", "slice", "identity"}:
            raise ValueError(
                f"Unsupported projection mode: {self.mode}. "
                f"Choose from ['linear', 'mean', 'slice', 'identity']."
            )

        if self.mode == "identity":
            if self.in_out_dim != self.out_out_dim:
                raise ValueError(
                    f"Identity projection requires matching dims, got "
                    f"{self.in_out_dim} -> {self.out_out_dim}"
                )
            self.proj = None
        elif self.mode == "linear":
            self.proj = nn.Linear(self.in_out_dim, self.out_out_dim, bias=False)
        else:
            self.proj = None

    def forward(self, final_w, final_b):
        if self.mode == "identity":
            return final_w, final_b

        if self.mode == "slice":
            if self.out_out_dim > self.in_out_dim:
                raise ValueError(
                    f"Cannot slice from smaller to larger dim: {self.in_out_dim} -> {self.out_out_dim}"
                )
            return (
                final_w[:, :, :self.out_out_dim, :],
                final_b[:, :self.out_out_dim, :],
            )

        if self.mode == "mean":
            if self.out_out_dim > self.in_out_dim:
                raise ValueError(
                    f"Cannot mean-project from smaller to larger dim: {self.in_out_dim} -> {self.out_out_dim}"
                )
            if self.in_out_dim % self.out_out_dim != 0:
                raise ValueError(
                    f"Mean projection requires divisible dims, got "
                    f"{self.in_out_dim} -> {self.out_out_dim}"
                )

            group_size = self.in_out_dim // self.out_out_dim

            out_w = final_w.view(
                final_w.shape[0],
                final_w.shape[1],
                self.out_out_dim,
                group_size,
                final_w.shape[3],
            ).mean(dim=3)

            out_b = final_b.view(
                final_b.shape[0],
                self.out_out_dim,
                group_size,
                final_b.shape[2],
            ).mean(dim=2)

            return out_w, out_b

        if self.mode == "linear":
            w = final_w.squeeze(-1)  # [B, in_dim, in_out]
            b = final_b.squeeze(-1)  # [B, in_out]

            out_w = self.proj(w)     # [B, in_dim, out_out]
            out_b = self.proj(b)     # [B, out_out]

            return out_w.unsqueeze(-1), out_b.unsqueeze(-1)

        raise RuntimeError("Unexpected projection mode reached.")


def project_predicted_input_layout_to_output_layout(
    pred_weights,
    pred_biases,
    input_layout,
    output_layout,
    final_layer_projector: FinalLayerProjection,
):
    """
    ScaleGMN predicts tensors in INPUT topology.
    We convert them to OUTPUT topology.

    Shared layers are kept unchanged.
    Final layer is converted with the projection head.
    """
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
                raise ValueError(
                    f"Final layer input dim mismatch: input {in_in} vs output {out_in}"
                )

            proj_w, proj_b = final_layer_projector(w, b)

            if proj_w.shape[2] != out_out:
                raise ValueError(
                    f"Projected final weight dim mismatch: got {proj_w.shape[2]}, expected {out_out}"
                )
            if proj_b.shape[1] != out_out:
                raise ValueError(
                    f"Projected final bias dim mismatch: got {proj_b.shape[1]}, expected {out_out}"
                )

            out_weights.append(proj_w)
            out_biases.append(proj_b)

    return out_weights, out_biases


# ============================================================================================
# INPUT GRAPH DATASET
# ============================================================================================
class NavierGeometryINRDataset(BaseDataset):
    """
    Dataset for Navier geometry INRs stored as .pth files.
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
# INR INFERENCE ON FIXED REFERENCE GRID
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

        coords = make_ref_grid(self.image_size[0], self.image_size[1])  # [P,2]
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

        return x  # [B,P,C]


# ============================================================================================
# PAIRED DATASET
# ============================================================================================
class PairedNavierINRPredictionDataset(torch.utils.data.Dataset):
    """
    Uses:
      - input geometry INR directory
      - output velocity INR directory

    Returns:
      - input graph params from geometry INR
      - input INR params (for initialization)
      - ground-truth output INR params
      - sample index
    """

    def __init__(
        self,
        dataset_name,
        input_inr_dir,
        output_inr_dir,
        split="train",
        debug=False,
        direction="forward",
        equiv_on_hidden=False,
        get_first_layer_mask=False,
        node_pos_embed=False,
        edge_pos_embed=False,
        input_layer_layout=None,
        image_size=(129, 129),
    ):
        super().__init__()

        self.input_graph_ds = NavierGeometryINRDataset(
            dataset=dataset_name,
            dataset_path=input_inr_dir,
            split_path=None,
            debug=debug,
            split=split,
            node_pos_embed=node_pos_embed,
            edge_pos_embed=edge_pos_embed,
            equiv_on_hidden=equiv_on_hidden,
            get_first_layer_mask=get_first_layer_mask,
            image_size=image_size,
            direction=direction,
            layer_layout=input_layer_layout,
            return_path=False,
            data_format="graph",
            switch_to_canon=False,
        )

        self.output_inr_dir = output_inr_dir

    def __len__(self):
        return len(self.input_graph_ds)

    def get_layer_layout(self):
        return self.input_graph_ds.get_layer_layout()

    def __getitem__(self, index):
        params = self.input_graph_ds[index]

        rel_path = self.input_graph_ds.dataset[index]
        input_abs_path = Path(self.input_graph_ds.dataset_path) / rel_path
        output_abs_path = Path(self.output_inr_dir) / rel_path

        if not input_abs_path.exists():
            raise FileNotFoundError(f"Missing input INR checkpoint: {input_abs_path}")
        if not output_abs_path.exists():
            raise FileNotFoundError(f"Missing output INR checkpoint: {output_abs_path}")

        sample_idx = parse_sample_idx_from_path(str(input_abs_path))

        in_sd = torch.load(input_abs_path, map_location="cpu")
        input_weights, input_biases = state_dict_to_weight_bias_tuples(in_sd)
        input_wb = Batch(
            weights=input_weights,
            biases=input_biases,
            label=torch.tensor(0, dtype=torch.long),
        )

        out_sd = torch.load(output_abs_path, map_location="cpu")
        gt_weights, gt_biases = state_dict_to_weight_bias_tuples(out_sd)
        gt_out_wb = Batch(
            weights=gt_weights,
            biases=gt_biases,
            label=torch.tensor(0, dtype=torch.long),
        )

        return params, input_wb, gt_out_wb, torch.tensor(sample_idx, dtype=torch.long)


# ============================================================================================
# EVALUATION
# ============================================================================================
@torch.no_grad()
def evaluate(
    model,
    projector,
    loader,
    device,
    inr_model,
    input_layer_layout,
    output_layer_layout,
    num_batches=None,
    eps=1e-12,
):
    model.eval()
    projector.eval()

    mse_losses = []
    rel_l2_losses = []

    for i, batch in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break

        params, input_wb, gt_out_wb, sample_idx = batch

        params = params.to(device)
        input_wb = batch_to_device_namedtuple(input_wb, device)
        gt_out_wb = batch_to_device_namedtuple(gt_out_wb, device)

        init_in_wb = clone_namedtuple_wb(input_wb)

        pred_in_weights, pred_in_biases = model(
            params,
            list(init_in_wb.weights),
            list(init_in_wb.biases),
        )

        pred_out_weights, pred_out_biases = project_predicted_input_layout_to_output_layout(
            pred_in_weights,
            pred_in_biases,
            input_layout=input_layer_layout,
            output_layout=output_layer_layout,
            final_layer_projector=projector,
        )

        pred_out = inr_model(pred_out_weights, pred_out_biases)  # [B,P,1 or C]
        gt_out = inr_model(gt_out_wb.weights, gt_out_wb.biases)

        if not torch.isfinite(pred_out).all():
            raise RuntimeError("Non-finite values found in pred_out during evaluation")
        if not torch.isfinite(gt_out).all():
            raise RuntimeError("Non-finite values found in gt_out during evaluation")

        mse = ((pred_out - gt_out) ** 2).mean(dim=(1, 2))  # [B]
        mse_losses.append(mse.detach().cpu())

        diff = (pred_out - gt_out).reshape(pred_out.shape[0], -1)
        gt = gt_out.reshape(gt_out.shape[0], -1)

        rel_l2 = diff.norm(dim=1) / (gt.norm(dim=1) + eps)
        rel_l2_losses.append(rel_l2.detach().cpu())

    avg_mse = torch.cat(mse_losses).mean()
    avg_rel_l2 = torch.cat(rel_l2_losses).mean()

    if not torch.isfinite(avg_mse):
        raise RuntimeError("avg_mse is non-finite during evaluation")
    if not torch.isfinite(avg_rel_l2):
        raise RuntimeError("avg_rel_l2 is non-finite during evaluation")

    model.train()
    projector.train()

    return {
        "avg_mse": avg_mse,
        "avg_rel_l2": avg_rel_l2,
    }


# ============================================================================================
# TRAINING
# ============================================================================================
def main(args=None):
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, vars(args))

    torch.set_float32_matmul_precision("high")
    print(yaml.dump(conf, default_flow_style=False), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    run = None
    if conf.get("wandb", False):
        run = wandb.init(
            project=conf["wandb_args"]["project"],
            entity=conf["wandb_args"]["entity"],
            name=conf["wandb_args"]["name"],
            config=conf,
            mode="online",
        )
        print("W&B run URL:", run.url, flush=True)
        print("W&B mode:", run.settings.mode, flush=True)
        print("W&B name:", run.name, flush=True)

        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    set_seed(conf["train_args"]["seed"])

    equiv_on_hidden = mask_hidden(conf)
    get_first_layer_mask = mask_input(conf)

    input_layer_layout = conf["data"]["input_layer_layout"]
    output_layer_layout = conf["data"]["output_layer_layout"]
    image_size = tuple(conf["data"]["image_size"])
    node_pos_embed = conf["data"].get("node_pos_embed", False)
    edge_pos_embed = conf["data"].get("edge_pos_embed", False)

    validate_layouts_for_final_projection(input_layer_layout, output_layer_layout)

    projection_conf = conf.get("projection_head", {})
    projection_mode = projection_conf.get("mode", "linear").lower()

    train_set = PairedNavierINRPredictionDataset(
        dataset_name=conf["data"]["dataset"],
        input_inr_dir=conf["data"]["input_inr_dir"],
        output_inr_dir=conf["data"]["output_inr_dir"],
        split="train",
        debug=conf.get("debug", False),
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=equiv_on_hidden,
        get_first_layer_mask=get_first_layer_mask,
        node_pos_embed=node_pos_embed,
        edge_pos_embed=edge_pos_embed,
        input_layer_layout=input_layer_layout,
        image_size=image_size,
    )

    val_set = PairedNavierINRPredictionDataset(
        dataset_name=conf["data"]["dataset"],
        input_inr_dir=conf["data"]["input_inr_dir"],
        output_inr_dir=conf["data"]["output_inr_dir"],
        split="val",
        debug=conf.get("debug", False),
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=equiv_on_hidden,
        get_first_layer_mask=get_first_layer_mask,
        node_pos_embed=node_pos_embed,
        edge_pos_embed=edge_pos_embed,
        input_layer_layout=input_layer_layout,
        image_size=image_size,
    )

    test_set = PairedNavierINRPredictionDataset(
        dataset_name=conf["data"]["dataset"],
        input_inr_dir=conf["data"]["input_inr_dir"],
        output_inr_dir=conf["data"]["output_inr_dir"],
        split="test",
        debug=conf.get("debug", False),
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=equiv_on_hidden,
        get_first_layer_mask=get_first_layer_mask,
        node_pos_embed=node_pos_embed,
        edge_pos_embed=edge_pos_embed,
        input_layer_layout=input_layer_layout,
        image_size=image_size,
    )

    conf["scalegmn_args"]["layer_layout"] = train_set.get_layer_layout()

    print(f"Len train set: {len(train_set)}", flush=True)
    print(f"Len val set:   {len(val_set)}", flush=True)
    print(f"Len test set:  {len(test_set)}", flush=True)

    train_loader = torch_geometric.loader.DataLoader(
        dataset=train_set,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=conf["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch_geometric.loader.DataLoader(
        dataset=val_set,
        batch_size=conf["batch_size"],
        shuffle=False,
        num_workers=conf["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = torch_geometric.loader.DataLoader(
        dataset=test_set,
        batch_size=conf["batch_size"],
        shuffle=False,
        num_workers=conf["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    net = ScaleGMN_equiv(conf["scalegmn_args"]).to(device)

    projector = FinalLayerProjection(
        in_out_dim=input_layer_layout[-1],
        out_out_dim=output_layer_layout[-1],
        mode=projection_mode,
    ).to(device)

    print(net, flush=True)
    print(projector, flush=True)

    cnt_p_net = count_parameters(net=net)
    cnt_p_projector = count_parameters(net=projector)
    cnt_p_total = cnt_p_net + cnt_p_projector

    print(f"ScaleGMN params:   {cnt_p_net:,}", flush=True)
    print(f"Projector params:  {cnt_p_projector:,}", flush=True)
    print(f"Total params:      {cnt_p_total:,}", flush=True)

    if run is not None:
        run.log(
            {
                "number of parameters/scalegmn": cnt_p_net,
                "number of parameters/projection_head": cnt_p_projector,
                "number of parameters/total": cnt_p_total,
                "global_step": 0,
            },
            step=0,
        )

    for p in net.parameters():
        p.requires_grad = True
    for p in projector.parameters():
        p.requires_grad = True

    inr_model = BatchSirenLinearOnlyRefGrid(
        image_size=image_size,
        w0=conf["inr_model"]["w0"],
        w0_first=conf["inr_model"]["w0_first"],
    ).to(device)

    criterion = nn.MSELoss()

    optimizer_cls = getattr(torch.optim, conf["optimization"]["optimizer_name"])
    optimizer = optimizer_cls(
        [p for p in list(net.parameters()) + list(projector.parameters()) if p.requires_grad],
        **conf["optimization"]["optimizer_args"],
    )

    best_val_rel_l2 = float("inf")
    best_val_results = None
    best_test_results = None

    test_mse = -1.0
    test_rel_l2 = -1.0

    global_step = 0
    start_epoch = 0

    save_best_model = conf.get("save_best_model", False)
    best_model_path = conf.get("best_model_path", "best_scalegmn_navier_direct_prediction.pt")

    epoch_iter = trange(start_epoch, conf["train_args"]["num_epochs"], desc="Epochs")
    net.train()
    projector.train()

    for epoch in epoch_iter:
        for i, batch in enumerate(train_loader):
            params, input_wb, gt_out_wb, sample_idx = batch

            params = params.to(device)
            input_wb = batch_to_device_namedtuple(input_wb, device)
            gt_out_wb = batch_to_device_namedtuple(gt_out_wb, device)

            # Initialization = input INR parameters
            init_in_wb = clone_namedtuple_wb(input_wb)

            optimizer.zero_grad(set_to_none=True)

            pred_in_weights, pred_in_biases = net(
                params,
                list(init_in_wb.weights),
                list(init_in_wb.biases),
            )

            pred_out_weights, pred_out_biases = project_predicted_input_layout_to_output_layout(
                pred_in_weights,
                pred_in_biases,
                input_layout=input_layer_layout,
                output_layout=output_layer_layout,
                final_layer_projector=projector,
            )

            pred_out = inr_model(pred_out_weights, pred_out_biases)
            gt_out = inr_model(gt_out_wb.weights, gt_out_wb.biases)

            if not torch.isfinite(pred_out).all():
                raise RuntimeError("Non-finite values found in pred_out during training")
            if not torch.isfinite(gt_out).all():
                raise RuntimeError("Non-finite values found in gt_out during training")

            loss = criterion(pred_out, gt_out)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected during training: {loss.item()}")

            loss.backward()

            log = {
                "train/loss": float(loss.item()),
                "epoch": epoch,
                "global_step": global_step,
            }

            if conf["optimization"].get("clip_grad", False):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in list(net.parameters()) + list(projector.parameters()) if p.requires_grad],
                    conf["optimization"]["clip_grad_max_norm"],
                )
                log["grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

            optimizer.step()

            if run is not None:
                if projection_mode == "linear":
                    with torch.no_grad():
                        proj_weight = projector.proj.weight.detach().cpu()
                        for r in range(proj_weight.shape[0]):
                            for c in range(proj_weight.shape[1]):
                                log[f"projection_head/W_{r}_{c}"] = float(proj_weight[r, c].item())
                run.log(log, step=global_step)

            epoch_iter.set_description(
                f"Epoch {epoch} | batch {i+1}/{len(train_loader)} | "
                f"train_mse={loss.item():.4e} | test_rel_l2={test_rel_l2:.4e}"
            )

            global_step += 1

            if global_step > 0 and (global_step % conf["train_args"]["eval_every"] == 0):
                val_dict = evaluate(
                    net,
                    projector,
                    val_loader,
                    device,
                    inr_model,
                    input_layer_layout=input_layer_layout,
                    output_layer_layout=output_layer_layout,
                )

                test_dict = evaluate(
                    net,
                    projector,
                    test_loader,
                    device,
                    inr_model,
                    input_layer_layout=input_layer_layout,
                    output_layer_layout=output_layer_layout,
                )

                train_dict = evaluate(
                    net,
                    projector,
                    train_loader,
                    device,
                    inr_model,
                    input_layer_layout=input_layer_layout,
                    output_layer_layout=output_layer_layout,
                    num_batches=100,
                )

                val_mse = float(val_dict["avg_mse"])
                val_rel_l2 = float(val_dict["avg_rel_l2"])

                test_mse = float(test_dict["avg_mse"])
                test_rel_l2 = float(test_dict["avg_rel_l2"])

                if val_rel_l2 < best_val_rel_l2:
                    best_val_rel_l2 = val_rel_l2
                    best_val_results = val_dict
                    best_test_results = test_dict

                    if save_best_model:
                        torch.save(
                            {
                                "model_state_dict": net.state_dict(),
                                "projection_head_state_dict": projector.state_dict(),
                                "config": conf,
                                "best_val_rel_l2": best_val_rel_l2,
                                "global_step": global_step,
                                "epoch": epoch,
                            },
                            best_model_path,
                        )

                if run is not None:
                    eval_log = {
                        "train/avg_mse": float(train_dict["avg_mse"]),
                        "train/avg_rel_l2": float(train_dict["avg_rel_l2"]),
                        "val/avg_mse": val_mse,
                        "val/avg_rel_l2": val_rel_l2,
                        "test/avg_mse": test_mse,
                        "test/avg_rel_l2": test_rel_l2,
                        "epoch": epoch,
                        "global_step": global_step,
                    }

                    if best_val_results is not None and best_test_results is not None:
                        eval_log["val/best_rel_l2"] = float(best_val_results["avg_rel_l2"])
                        eval_log["test/best_at_best_val_rel_l2"] = float(best_test_results["avg_rel_l2"])
                        eval_log["test/best_at_best_val_mse"] = float(best_test_results["avg_mse"])

                    if projection_mode == "linear":
                        with torch.no_grad():
                            proj_weight = projector.proj.weight.detach().cpu()
                            for r in range(proj_weight.shape[0]):
                                for c in range(proj_weight.shape[1]):
                                    eval_log[f"projection_head_eval/W_{r}_{c}"] = float(proj_weight[r, c].item())

                    run.log(eval_log, step=global_step)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if hasattr(args, "gpu_ids") and isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    main(args=args)