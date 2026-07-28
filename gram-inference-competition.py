#!/usr/bin/env python3

"""
Example:

cd /home/intern/spygeorgoulas/thesis-metanets/scalegmn

python /home/intern/spygeorgoulas/thesis-metanets/scalegmn/gram-inference-competition.py \
    --checkpoint /home/intern/spygeorgoulas/thesis-metanets/scalegmn/outputs/gram_velocity_scalegmn_run2/best_model.pt \
    --inr-dir /home/intern/spygeorgoulas/thesis-metanets/scalegmn/data/gram_ifw_4x64_300k_rwi_pth-comp-test \
    --target-npz-dir /home/intern/spygeorgoulas/thesis-metanets/neural-fields-3d/data/ifw-comp-test \
    --split test \
    --batch-size 1 \
    --eval-chunk-size 8192 \
    --warmup-passes 3
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from tqdm import tqdm

from src.data import dataset
from src.scalegmn.models import ScaleGMN_equiv
from src.utils.helpers import mask_input, mask_hidden


# ---------------------------------------------------------------------
# Parameter and path utilities
# ---------------------------------------------------------------------

def residual_param_update(
    weights,
    biases,
    delta_weights,
    delta_biases,
):
    """Apply the residual parameter update predicted by ScaleGMN."""

    new_weights = [
        weight + delta
        for weight, delta in zip(weights, delta_weights)
    ]

    new_biases = [
        bias + delta
        for bias, delta in zip(biases, delta_biases)
    ]

    return new_weights, new_biases


def flatten_paths(obj) -> List[str]:
    """Extract path strings from a nested collated batch object."""

    paths = []

    if obj is None:
        return paths

    if isinstance(obj, (str, Path)):
        return [str(obj)]

    if isinstance(obj, dict):
        for value in obj.values():
            paths.extend(flatten_paths(value))

        return paths

    if isinstance(obj, (list, tuple)):
        for value in obj:
            paths.extend(flatten_paths(value))

        return paths

    return paths


def unpack_batch(batch):
    """
    Expected dataset outputs include one of:

        (params, w_b, path)
        (params, w_b, label, path)

    or a dictionary containing params, w_b and path.
    """

    params = None
    w_b = None
    paths = None

    if isinstance(batch, dict):
        params = batch.get(
            "params",
            batch.get("graph"),
        )

        w_b = batch.get(
            "w_b",
            batch.get("weights_biases"),
        )

        for key in [
            "path",
            "paths",
            "file_path",
            "file_paths",
            "rel_path",
            "rel_paths",
        ]:
            if key in batch:
                candidate_paths = flatten_paths(batch[key])

                if candidate_paths:
                    paths = candidate_paths
                    break

    elif isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            params = batch[0]
            w_b = batch[1]

        for item in batch[2:]:
            candidate_paths = flatten_paths(item)

            if candidate_paths:
                paths = candidate_paths
                break

    if params is None or w_b is None:
        raise RuntimeError(
            "Could not extract params and w_b from the dataset batch. "
            "Check the return format of "
            "IFWVelocityINRDataset.__getitem__()."
        )

    if paths is None:
        raise RuntimeError(
            "The dataset did not return the .pth paths. "
            "Ensure that conf['data']['return_path'] is supported."
        )

    return params, w_b, [str(path) for path in paths]


def move_wb_to_device(
    w_b,
    device: torch.device,
):
    """Move the weight-bias container to the selected device."""

    w_b = w_b.to(device)

    return w_b.weights, w_b.biases


def find_matching_npz(
    inr_path: str,
    target_npz_dir: Path,
) -> Path:
    """
    Match an input .pth file to a ground-truth .npz file.

    Preferred match:

        input_folder/example.pth
        target_folder/example.npz
    """

    inr_path = Path(inr_path)
    target_npz_dir = Path(target_npz_dir)

    filename = f"{inr_path.stem}.npz"

    direct_match = target_npz_dir / filename

    if direct_match.exists():
        return direct_match

    recursive_matches = list(
        target_npz_dir.rglob(filename)
    )

    if len(recursive_matches) == 1:
        return recursive_matches[0]

    if len(recursive_matches) > 1:
        raise RuntimeError(
            f"Multiple ground-truth files named {filename} were found:\n"
            + "\n".join(
                str(path)
                for path in recursive_matches
            )
        )

    raise FileNotFoundError(
        f"No matching ground-truth file found for:\n"
        f"  INR: {inr_path}\n"
        f"  Expected filename: {filename}\n"
        f"  Search directory: {target_npz_dir}"
    )


# ---------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------

def load_velocity_out_npz(
    npz_path: Path,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load the output coordinates and ground-truth velocity.

    Expected NPZ arrays:

        pos:          (N, 3)
        t:            (10,)
        velocity_out: (5, N, 3)

    Returns:

        coords: (5*N, 4), containing (x, y, z, t)
        target: (5*N, 3), containing (vx, vy, vz)
    """

    with np.load(npz_path) as data:
        required_keys = {
            "pos",
            "t",
            "velocity_out",
        }

        missing_keys = required_keys.difference(
            data.files
        )

        if missing_keys:
            raise KeyError(
                f"{npz_path} is missing arrays: "
                f"{sorted(missing_keys)}"
            )

        pos = data["pos"].astype(np.float32)
        times = data["t"].astype(np.float32)

        velocity_out = data[
            "velocity_out"
        ].astype(np.float32)

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(
            f"Expected pos with shape (N, 3), "
            f"got {pos.shape} in {npz_path}"
        )

    if (
        velocity_out.ndim != 3
        or velocity_out.shape[-1] != 3
    ):
        raise ValueError(
            "Expected velocity_out with shape (T, N, 3), "
            f"got {velocity_out.shape} in {npz_path}"
        )

    num_times = velocity_out.shape[0]
    num_points = velocity_out.shape[1]

    if num_points != pos.shape[0]:
        raise ValueError(
            f"pos contains {pos.shape[0]} points but "
            f"velocity_out contains {num_points} points "
            f"in {npz_path}"
        )

    if times.shape[0] >= num_times:
        output_times = times[-num_times:]
    else:
        raise ValueError(
            f"Not enough time values: t has shape "
            f"{times.shape}, while velocity_out has "
            f"{num_times} time steps in {npz_path}"
        )

    repeated_pos = np.broadcast_to(
        pos[None, :, :],
        (num_times, num_points, 3),
    ).copy()

    repeated_times = np.broadcast_to(
        output_times[:, None, None],
        (num_times, num_points, 1),
    ).copy()

    coords = np.concatenate(
        [
            repeated_pos,
            repeated_times,
        ],
        axis=-1,
    ).reshape(-1, 4)

    target = velocity_out.reshape(-1, 3)

    return (
        torch.from_numpy(coords),
        torch.from_numpy(target),
    )


# ---------------------------------------------------------------------
# Functional SIREN inference
# ---------------------------------------------------------------------

def graph_params_to_siren_params(
    weights: List[torch.Tensor],
    biases: List[torch.Tensor],
):
    """
    Convert ScaleGMN graph parameter shapes into linear-layer shapes.

    Input:

        weight: (B, input_features, output_features, 1)
        bias:   (B, output_features, 1)

    Output:

        weight: (B, output_features, input_features)
        bias:   (B, output_features)
    """

    siren_weights = []
    siren_biases = []

    for weight in weights:
        if weight.ndim != 4:
            raise ValueError(
                "Expected a four-dimensional graph weight, "
                f"got {tuple(weight.shape)}"
            )

        linear_weight = (
            weight[..., 0]
            .permute(0, 2, 1)
            .contiguous()
        )

        siren_weights.append(linear_weight)

    for bias in biases:
        if bias.ndim != 3:
            raise ValueError(
                "Expected a three-dimensional graph bias, "
                f"got {tuple(bias.shape)}"
            )

        linear_bias = bias[..., 0].contiguous()

        siren_biases.append(linear_bias)

    return siren_weights, siren_biases


def batched_siren_forward(
    coords: torch.Tensor,
    weights: List[torch.Tensor],
    biases: List[torch.Tensor],
    w0: float,
) -> torch.Tensor:
    """Evaluate a batch of SIRENs using their predicted parameters."""

    x = coords

    for layer_index, (weight, bias) in enumerate(
        zip(weights, biases)
    ):
        x = torch.bmm(
            x,
            weight.transpose(1, 2),
        )

        x = x + bias.unsqueeze(1)

        if layer_index < len(weights) - 1:
            x = torch.sin(w0 * x)

    return x


@torch.inference_mode()
def predict_field_chunked(
    sample_weights: List[torch.Tensor],
    sample_biases: List[torch.Tensor],
    coords: torch.Tensor,
    w0: float,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Evaluate the complete output field in chunks.

    This timing is not included in the reported ScaleGMN
    forward-pass latency.
    """

    predictions = []

    for start in range(
        0,
        coords.shape[0],
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            coords.shape[0],
        )

        coords_chunk = (
            coords[start:end]
            .to(
                device,
                non_blocking=True,
            )
            .unsqueeze(0)
        )

        prediction_chunk = batched_siren_forward(
            coords=coords_chunk,
            weights=sample_weights,
            biases=sample_biases,
            w0=w0,
        ).squeeze(0)

        predictions.append(
            prediction_chunk.cpu()
        )

    return torch.cat(
        predictions,
        dim=0,
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute the relative L2 error."""

    numerator = torch.linalg.vector_norm(
        prediction - target
    )

    denominator = (
        torch.linalg.vector_norm(target)
        + eps
    )

    return (
        numerator / denominator
    ).item()


# ---------------------------------------------------------------------
# ScaleGMN forward-pass timing
# ---------------------------------------------------------------------

@torch.inference_mode()
def timed_scalegmn_forward(
    model,
    params,
    weights,
    biases,
    device: torch.device,
):
    """
    Measure only:

        model(params, weights, biases)

    The ScaleGMN model modifies the graph object in place, so
    params is cloned before every call.

    The clone is performed before the timer starts and is therefore
    excluded from the reported forward-pass latency.

    Excluded from timing:

    - data loading,
    - CPU-to-GPU transfer,
    - graph cloning,
    - residual parameter update,
    - SIREN evaluation,
    - metric computation.
    """

    params_for_forward = params.clone()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    delta_weights, delta_biases = model(
        params_for_forward,
        weights,
        biases,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    return (
        delta_weights,
        delta_biases,
        elapsed_seconds,
    )


# ---------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------

@torch.inference_mode()
def run_inference(args):
    device = torch.device(
        args.device
        if args.device
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain "
            "'model_state_dict'."
        )

    if "conf" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain "
            "the training configuration 'conf'."
        )

    conf = checkpoint["conf"]

    conf["debug"] = False
    conf["data"]["return_path"] = True

    conf["data"]["dataset_path"] = str(
        Path(args.inr_dir)
        .expanduser()
        .resolve()
    )

    equiv_on_hidden = mask_hidden(conf)
    first_layer_mask = mask_input(conf)

    inference_set = dataset(
        conf["data"],
        split=args.split,
        debug=False,
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=equiv_on_hidden,
        get_first_layer_mask=first_layer_mask,
    )

    if len(inference_set) == 0:
        raise RuntimeError(
            f"No samples were found under "
            f"{args.inr_dir} for split "
            f"'{args.split}'."
        )

    inference_layout = (
        inference_set.get_layer_layout()
    )

    checkpoint_layout = conf[
        "scalegmn_args"
    ].get("layer_layout")

    if (
        checkpoint_layout is not None
        and list(inference_layout)
        != list(checkpoint_layout)
    ):
        raise ValueError(
            "The inference INR architecture differs "
            "from the architecture used during training.\n"
            f"Checkpoint layout: {checkpoint_layout}\n"
            f"Inference layout:  {inference_layout}"
        )

    conf["scalegmn_args"][
        "layer_layout"
    ] = inference_layout

    inference_loader = (
        torch_geometric.loader.DataLoader(
            dataset=inference_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    )

    model = ScaleGMN_equiv(
        conf["scalegmn_args"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    inr_conf = conf.get("inr_model") or {}

    w0 = float(
        inr_conf.get(
            "w0",
            inr_conf.get(
                "omega_0",
                conf["data"].get(
                    "w0",
                    30.0,
                ),
            ),
        )
    )

    target_npz_dir = (
        Path(args.target_npz_dir)
        .expanduser()
        .resolve()
    )

    print(
        f"Number of inference samples: "
        f"{len(inference_set)}"
    )

    print(f"SIREN w0: {w0}")

    print(
        f"ScaleGMN warm-up passes: "
        f"{args.warmup_passes}"
    )

    print()

    sample_results = []
    warmup_completed = False

    for batch in tqdm(
        inference_loader,
        desc="Inference",
    ):
        params, w_b, paths = unpack_batch(batch)

        # Device transfers happen before timing.
        params = params.to(
            device,
            non_blocking=True,
        )

        weights, biases = move_wb_to_device(
            w_b,
            device,
        )

        batch_size_actual = (
            weights[0].shape[0]
        )

        if len(paths) != batch_size_actual:
            raise RuntimeError(
                f"The batch contains "
                f"{batch_size_actual} parameter sets "
                f"but {len(paths)} paths were returned."
            )

        # -------------------------------------------------------------
        # Warm-up
        #
        # A new graph clone is required for each call because ScaleGMN
        # modifies the graph node features in place.
        # -------------------------------------------------------------

        if not warmup_completed:
            for _ in range(args.warmup_passes):
                warmup_params = params.clone()

                model(
                    warmup_params,
                    weights,
                    biases,
                )

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            warmup_completed = True

        # -------------------------------------------------------------
        # Timed ScaleGMN forward pass
        # -------------------------------------------------------------

        (
            delta_weights,
            delta_biases,
            batch_forward_time_sec,
        ) = timed_scalegmn_forward(
            model=model,
            params=params,
            weights=weights,
            biases=biases,
            device=device,
        )

        # With batch size 1, this is the true latency for one sample.
        # With batch size > 1, this is throughput-normalized time.
        forward_time_per_sample_sec = (
            batch_forward_time_sec
            / batch_size_actual
        )

        forward_time_per_sample_ms = (
            forward_time_per_sample_sec
            * 1000.0
        )

        updated_weights, updated_biases = (
            residual_param_update(
                weights,
                biases,
                delta_weights,
                delta_biases,
            )
        )

        for batch_index, inr_path in enumerate(
            paths
        ):
            npz_path = find_matching_npz(
                inr_path,
                target_npz_dir,
            )

            coords, target = load_velocity_out_npz(
                npz_path
            )

            sample_weights = [
                weight[
                    batch_index:batch_index + 1
                ]
                for weight in updated_weights
            ]

            sample_biases = [
                bias[
                    batch_index:batch_index + 1
                ]
                for bias in updated_biases
            ]

            (
                sample_weights,
                sample_biases,
            ) = graph_params_to_siren_params(
                sample_weights,
                sample_biases,
            )

            prediction = predict_field_chunked(
                sample_weights=sample_weights,
                sample_biases=sample_biases,
                coords=coords,
                w0=w0,
                chunk_size=args.eval_chunk_size,
                device=device,
            )

            rel_l2 = relative_l2(
                prediction,
                target,
            )

            mse = F.mse_loss(
                prediction,
                target,
            ).item()

            result = {
                "sample": Path(inr_path).stem,
                "pth": str(inr_path),
                "npz": str(npz_path),
                "relative_l2": rel_l2,
                "mse": mse,
                "scalegmn_forward_time_sec": (
                    forward_time_per_sample_sec
                ),
                "scalegmn_forward_time_ms": (
                    forward_time_per_sample_ms
                ),
                "batch_forward_time_sec": (
                    batch_forward_time_sec
                ),
                "batch_size": batch_size_actual,
            }

            sample_results.append(result)

            print(
                f"{result['sample']}: "
                f"ScaleGMN forward = "
                f"{forward_time_per_sample_ms:.3f} ms/sample, "
                f"relative L2 = {rel_l2:.8f}, "
                f"MSE = {mse:.8f}"
            )

    if not sample_results:
        raise RuntimeError(
            "No inference results were generated."
        )

    relative_l2_values = np.asarray(
        [
            result["relative_l2"]
            for result in sample_results
        ],
        dtype=np.float64,
    )

    mse_values = np.asarray(
        [
            result["mse"]
            for result in sample_results
        ],
        dtype=np.float64,
    )

    forward_time_values_ms = np.asarray(
        [
            result[
                "scalegmn_forward_time_ms"
            ]
            for result in sample_results
        ],
        dtype=np.float64,
    )

    print()
    print(
        "================ INFERENCE SUMMARY ================"
    )

    print(
        f"Checkpoint           : "
        f"{args.checkpoint}"
    )

    print(
        f"Number of samples    : "
        f"{len(sample_results)}"
    )

    print(
        f"Batch size           : "
        f"{args.batch_size}"
    )

    print(
        f"Warm-up passes       : "
        f"{args.warmup_passes}"
    )

    print()

    print(
        f"Average rel. L2      : "
        f"{relative_l2_values.mean():.8f}"
    )

    print(
        f"Std. rel. L2         : "
        f"{relative_l2_values.std():.8f}"
    )

    print(
        f"Median rel. L2       : "
        f"{np.median(relative_l2_values):.8f}"
    )

    print(
        f"Minimum rel. L2      : "
        f"{relative_l2_values.min():.8f}"
    )

    print(
        f"Maximum rel. L2      : "
        f"{relative_l2_values.max():.8f}"
    )

    print(
        f"Average MSE          : "
        f"{mse_values.mean():.8f}"
    )

    print()
    print("ScaleGMN forward-pass latency:")

    print(
        f"Average              : "
        f"{forward_time_values_ms.mean():.3f} ms/sample"
    )

    print(
        f"Standard deviation   : "
        f"{forward_time_values_ms.std():.3f} ms"
    )

    print(
        f"Median               : "
        f"{np.median(forward_time_values_ms):.3f} ms/sample"
    )

    print(
        f"Minimum              : "
        f"{forward_time_values_ms.min():.3f} ms/sample"
    )

    print(
        f"Maximum              : "
        f"{forward_time_values_ms.max():.3f} ms/sample"
    )

    print(
        "==================================================="
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run ScaleGMN inference on INR .pth files "
            "and measure ScaleGMN forward-pass latency."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "/home/intern/spygeorgoulas/"
            "thesis-metanets/scalegmn/"
            "outputs/gram_velocity_scalegmn_run2/"
            "best_model.pt"
        ),
    )

    parser.add_argument(
        "--inr-dir",
        type=str,
        required=True,
        help=(
            "Directory containing the input INR "
            ".pth files."
        ),
    )

    parser.add_argument(
        "--target-npz-dir",
        type=str,
        default=(
            "/home/intern/spygeorgoulas/"
            "thesis-metanets/neural-fields-3d/"
            "data/ifw-comp-test"
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help=(
            "Dataset split passed to the "
            "existing dataset factory."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Use 1 to measure the latency of one "
            "sample. With a larger batch, the script "
            "reports batch latency divided by batch size."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--eval-chunk-size",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--warmup-passes",
        type=int,
        default=3,
        help=(
            "Number of unmeasured ScaleGMN forward "
            "passes before timing begins."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device such as cuda, cuda:0, or cpu."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_inference(
        parse_args()
    )