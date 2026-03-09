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
    return torch.from_numpy(np.stack([xx, yy], axis=-1).reshape(-1, 2))  # [P, 2], P=H*W


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


def make_zero_wb_from_layout(batch_size: int, layer_layout, device):
    """
    Creates zero weights/biases for a given MLP layout.

    Example:
        [2, 64, 64, 2]
    gives:
        weights: [B,2,64,1], [B,64,64,1], [B,64,2,1]
        biases : [B,64,1],   [B,64,1],    [B,2,1]
    """
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
    """
    ScaleGMN predicts tensors in INPUT topology.
    We externally project them to OUTPUT topology.

    Example:
        input_layout  = [2, 64, 64, 2]
        output_layout = [2, 64, 64, 1]

    Strategy:
      - keep shared layers unchanged
      - slice the final layer output dimension from 2 -> 1
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
                raise ValueError(f"Final layer input dim mismatch: {in_in} vs {out_in}")
            if out_out > in_out:
                raise ValueError(
                    f"Cannot expand final output dim by slicing: input {in_out}, output {out_out}"
                )

            out_weights.append(w[:, :, :out_out, :])  # [B, in_dim, out_out, 1]
            out_biases.append(b[:, :out_out, :])      # [B, out_out, 1]

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
      - ground-truth output INR params (weights, biases)
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

        if not output_abs_path.exists():
            raise FileNotFoundError(f"Missing output INR checkpoint: {output_abs_path}")

        sample_idx = parse_sample_idx_from_path(str(input_abs_path))

        out_sd = torch.load(output_abs_path, map_location="cpu")
        gt_weights, gt_biases = state_dict_to_weight_bias_tuples(out_sd)
        gt_out_wb = Batch(
            weights=gt_weights,
            biases=gt_biases,
            label=torch.tensor(0, dtype=torch.long),
        )

        return params, gt_out_wb, torch.tensor(sample_idx, dtype=torch.long)


# ============================================================================================
# EVALUATION
# ============================================================================================
@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    inr_model,
    input_layer_layout,
    output_layer_layout,
    num_batches=None,
):
    """
    Evaluate directly in function space, not as images.

    pred_out: [B, P, 1]
    gt_out:   [B, P, 1]
    """
    model.eval()

    losses = []

    for i, batch in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break

        params, gt_out_wb, sample_idx = batch

        params = params.to(device)
        gt_out_wb = batch_to_device_namedtuple(gt_out_wb, device)

        batch_size = gt_out_wb.weights[0].shape[0]

        zero_in_weights, zero_in_biases = make_zero_wb_from_layout(
            batch_size=batch_size,
            layer_layout=input_layer_layout,
            device=device,
        )

        pred_in_weights, pred_in_biases = model(params, zero_in_weights, zero_in_biases)

        pred_out_weights, pred_out_biases = project_input_layout_prediction_to_output_layout(
            pred_in_weights,
            pred_in_biases,
            input_layout=input_layer_layout,
            output_layout=output_layer_layout,
        )

        pred_out = inr_model(pred_out_weights, pred_out_biases)         # [B,P,1]
        gt_out = inr_model(gt_out_wb.weights, gt_out_wb.biases)         # [B,P,1]

        loss = ((pred_out - gt_out) ** 2).mean(dim=(1, 2))              # [B]
        losses.append(loss.detach().cpu())

    losses = torch.cat(losses).mean()
    model.train()

    return {
        "avg_loss": losses,
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
    print(net, flush=True)

    cnt_p = count_parameters(net=net)
    print(f"net params: {cnt_p:,}", flush=True)

    if run is not None:
        run.log({"number of parameters": cnt_p, "global_step": 0}, step=0)

    for p in net.parameters():
        p.requires_grad = True

    inr_model = BatchSirenLinearOnlyRefGrid(
        image_size=image_size,
        w0=conf["inr_model"]["w0"],
        w0_first=conf["inr_model"]["w0_first"],
    ).to(device)

    criterion = nn.MSELoss()

    optimizer_cls = getattr(torch.optim, conf["optimization"]["optimizer_name"])
    optimizer = optimizer_cls(
        [p for p in net.parameters() if p.requires_grad],
        **conf["optimization"]["optimizer_args"],
    )

    best_val_loss = float("inf")
    best_test_results = None
    best_val_results = None
    test_loss = -1.0
    global_step = 0
    start_epoch = 0

    save_best_model = conf.get("save_best_model", False)
    best_model_path = conf.get("best_model_path", "best_scalegmn_navier_direct_prediction.pt")

    epoch_iter = trange(start_epoch, conf["train_args"]["num_epochs"], desc="Epochs")
    net.train()
    optimizer.zero_grad()

    for epoch in epoch_iter:
        for i, batch in enumerate(train_loader):
            params, gt_out_wb, sample_idx = batch

            params = params.to(device)
            gt_out_wb = batch_to_device_namedtuple(gt_out_wb, device)

            batch_size = gt_out_wb.weights[0].shape[0]

            optimizer.zero_grad(set_to_none=True)

            zero_in_weights, zero_in_biases = make_zero_wb_from_layout(
                batch_size=batch_size,
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

            pred_out = inr_model(pred_out_weights, pred_out_biases)        # [B,P,1]
            gt_out = inr_model(gt_out_wb.weights, gt_out_wb.biases)        # [B,P,1]

            loss = criterion(pred_out, gt_out)
            loss.backward()

            log = {
                "train/loss": float(loss.item()),
                "epoch": epoch,
                "global_step": global_step,
            }

            if conf["optimization"].get("clip_grad", False):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in net.parameters() if p.requires_grad],
                    conf["optimization"]["clip_grad_max_norm"],
                )
                log["grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

            optimizer.step()

            if run is not None:
                run.log(log, step=global_step)

            epoch_iter.set_description(
                f"Epoch {epoch} | batch {i+1}/{len(train_loader)} | train={loss.item():.4e} | test={test_loss:.4e}"
            )

            global_step += 1

            if global_step > 0 and (global_step % conf["train_args"]["eval_every"] == 0):
                val_dict = evaluate(
                    net,
                    val_loader,
                    device,
                    inr_model,
                    input_layer_layout=input_layer_layout,
                    output_layer_layout=output_layer_layout,
                )

                test_dict = evaluate(
                    net,
                    test_loader,
                    device,
                    inr_model,
                    input_layer_layout=input_layer_layout,
                    output_layer_layout=output_layer_layout,
                )

                train_dict = evaluate(
                    net,
                    train_loader,
                    device,
                    inr_model,
                    input_layer_layout=input_layer_layout,
                    output_layer_layout=output_layer_layout,
                    num_batches=100,
                )

                val_loss = float(val_dict["avg_loss"])
                test_loss = float(test_dict["avg_loss"])

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_results = val_dict
                    best_test_results = test_dict

                    if save_best_model:
                        torch.save(
                            {
                                "model_state_dict": net.state_dict(),
                                "config": conf,
                                "best_val_loss": best_val_loss,
                                "global_step": global_step,
                                "epoch": epoch,
                            },
                            best_model_path,
                        )

                if run is not None:
                    eval_log = {
                        "train/avg_loss": float(train_dict["avg_loss"]),
                        "val/avg_loss": val_loss,
                        "test/avg_loss": test_loss,
                        "val/best_loss": float(best_val_results["avg_loss"]),
                        "test/best_at_best_val": float(best_test_results["avg_loss"]),
                        "epoch": epoch,
                        "global_step": global_step,
                    }
                    run.log(eval_log, step=global_step)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if hasattr(args, "gpu_ids") and isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    main(args=args)