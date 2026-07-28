

"""
cd /home/intern/spygeorgoulas/thesis-metanets/scalegmn

python /home/intern/spygeorgoulas/thesis-metanets/scalegmn/gram-inf-viz.py \
  --checkpoint /home/intern/spygeorgoulas/thesis-metanets/scalegmn/outputs/gram_velocity_scalegmn_run2/best_model.pt \
  --inr /home/intern/spygeorgoulas/thesis-metanets/scalegmn/data/gram_ifw_4x64_300k_rwi_pth-comp-test/test/1024_13-4.pth \
  --npz /home/intern/spygeorgoulas/thesis-metanets/neural-fields-3d/data/ifw-comp-test/1024_13-4.npz \
  --inr-dir /home/intern/spygeorgoulas/thesis-metanets/scalegmn/data/gram_ifw_4x64_300k_rwi_pth-comp-test \
  --out-dir /home/intern/spygeorgoulas/thesis-metanets/neural-fields-3d/imgs/scalegmn-full-viz \
  --split test \
  --w0 80.0 \
  --slice-thickness 0.02 \
  --grid-res 400 \
  --interp linear \
  --gif-fps 1.5

"""
#!/usr/bin/env python3
"""Visualize one GRaM sample before and after ScaleGMN prediction.

Run from the ScaleGMN repository root so that ``src`` can be imported.

The PNG contains five paired time steps in a 5 x 6 grid:
  input GT | input INR | input error | output GT | predicted INR | output error

The GIF contains the same information, one paired time step per frame.
"""

import argparse
import copy
from pathlib import Path
from typing import List

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from scipy.interpolate import griddata

from src.data import dataset
from src.scalegmn.models import ScaleGMN_equiv
from src.utils.helpers import mask_hidden, mask_input


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Trained ScaleGMN checkpoint")
    p.add_argument("--inr", required=True, help="The single input INR .pth file")
    p.add_argument("--npz", required=True, help="Matching raw GRaM .npz file")
    p.add_argument(
        "--inr-dir", default=None,
        help="Dataset directory containing --inr (defaults to its parent directory)",
    )
    p.add_argument("--out-dir", required=True, help="Folder for the PNG and GIF")
    p.add_argument("--split", default="test", help="Split used by the existing dataset factory")
    p.add_argument("--device", default=None, help="For example cuda, cuda:0, or cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=8192)
    p.add_argument("--grid-res", type=int, default=300)
    p.add_argument("--slice-center", type=float, default=None)
    p.add_argument("--slice-thickness", type=float, default=0.02)
    p.add_argument("--interp", choices=("linear", "nearest", "cubic"), default="linear")
    p.add_argument("--gif-fps", type=float, default=1.5)
    p.add_argument(
        "--w0", type=float, default=None,
        help="Override SIREN w0; otherwise read it from the checkpoint configuration",
    )
    return p.parse_args()


def flatten_paths(obj) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, (str, Path)):
        return [str(obj)]
    if isinstance(obj, dict):
        return [path for value in obj.values() for path in flatten_paths(value)]
    if isinstance(obj, (list, tuple)):
        return [path for value in obj for path in flatten_paths(value)]
    return []


def unpack_batch(batch):
    if isinstance(batch, dict):
        params = batch.get("params", batch.get("graph"))
        wb = batch.get("w_b", batch.get("weights_biases"))
        paths = []
        for key in ("path", "paths", "file_path", "file_paths", "rel_path", "rel_paths"):
            if key in batch:
                paths = flatten_paths(batch[key])
                if paths:
                    break
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        params, wb = batch[:2]
        paths = []
        for item in batch[2:]:
            paths = flatten_paths(item)
            if paths:
                break
    else:
        raise RuntimeError("Unexpected dataset batch format")
    if params is None or wb is None or not paths:
        raise RuntimeError("Dataset batch must provide params, w_b, and the INR path")
    return params, wb, paths


def same_file(candidate, selected: Path) -> bool:
    candidate = Path(candidate).expanduser()
    try:
        return candidate.resolve() == selected.resolve()
    except OSError:
        return candidate.name == selected.name


def residual_update(weights, biases, delta_weights, delta_biases):
    return (
        [w + dw for w, dw in zip(weights, delta_weights)],
        [b + db for b, db in zip(biases, delta_biases)],
    )


def graph_to_linear(weights, biases):
    """Convert graph layout (B,in,out,1)/(B,out,1) to PyTorch linear layout."""
    linear_w = [w[..., 0].permute(0, 2, 1).contiguous() for w in weights]
    linear_b = [b[..., 0].contiguous() for b in biases]
    return linear_w, linear_b


def state_dict_to_linear(state_dict, device):
    """Read seq.<index>.weight/bias tensors from the original SIREN checkpoint."""
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    weight_keys = [k for k in state_dict if k.endswith(".weight")]
    weight_keys.sort(key=lambda k: int(k.split(".")[-2]))
    if not weight_keys:
        raise KeyError("No SIREN Linear weights were found in the input .pth file")
    weights, biases = [], []
    for wk in weight_keys:
        bk = wk[:-6] + "bias"
        if bk not in state_dict:
            raise KeyError(f"Missing matching bias tensor: {bk}")
        weights.append(state_dict[wk].to(device).float().unsqueeze(0))
        biases.append(state_dict[bk].to(device).float().unsqueeze(0))
    return weights, biases


def siren_forward(coords, weights, biases, w0):
    x = coords
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        x = torch.bmm(x, weight.transpose(1, 2)) + bias.unsqueeze(1)
        if i < len(weights) - 1:
            x = torch.sin(w0 * x)
    return x


@torch.inference_mode()
def evaluate_siren(pos, times, weights, biases, w0, chunk_size, device):
    """Return an array shaped (T,N,3)."""
    outputs = []
    pos_t = torch.from_numpy(pos).to(device)
    for time_value in times:
        time_col = torch.full(
            (len(pos), 1), float(time_value), dtype=pos_t.dtype, device=device
        )
        coords = torch.cat((pos_t, time_col), dim=1)
        chunks = []
        for start in range(0, len(coords), chunk_size):
            x = coords[start:start + chunk_size].unsqueeze(0)
            chunks.append(siren_forward(x, weights, biases, w0).squeeze(0).cpu())
        outputs.append(torch.cat(chunks).numpy())
    return np.stack(outputs)


def load_npz(path):
    with np.load(path) as data:
        required = {"pos", "t", "velocity_in", "velocity_out", "idcs_airfoil"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} is missing arrays: {sorted(missing)}")
        values = {key: data[key] for key in required}
    pos = values["pos"].astype(np.float32)
    times = values["t"].astype(np.float32)
    velocity_in = values["velocity_in"].astype(np.float32)
    velocity_out = values["velocity_out"].astype(np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected pos (N,3), got {pos.shape}")
    if velocity_in.shape != velocity_out.shape or velocity_in.shape[0] != 5:
        raise ValueError(
            f"Expected velocity_in and velocity_out to both be (5,N,3); got "
            f"{velocity_in.shape} and {velocity_out.shape}"
        )
    if times.shape[0] < 10:
        raise ValueError(f"Expected at least 10 time values, got {times.shape}")
    return pos, times[:5], times[-5:], velocity_in, velocity_out, values["idcs_airfoil"]


def make_slice_geometry(pos, idcs_airfoil, center, thickness, grid_res):
    y = pos[:, 1]
    mask = np.abs(y - center) <= thickness / 2.0
    if mask.sum() < 50:
        raise ValueError(f"Only {mask.sum()} slice points; increase --slice-thickness")
    xs, zs = pos[mask, 0], pos[mask, 2]
    xi = np.linspace(xs.min(), xs.max(), grid_res)
    zi = np.linspace(zs.min(), zs.max(), grid_res)
    Xi, Zi = np.meshgrid(xi, zi)
    airfoil = np.zeros(len(pos), dtype=bool)
    airfoil[np.asarray(idcs_airfoil).reshape(-1)] = True
    air_slice = mask & airfoil
    extent = [xs.min(), xs.max(), zs.min(), zs.max()]
    return mask, Xi, Zi, extent, pos[air_slice, 0], pos[air_slice, 2]


def interpolate_magnitude(pos, velocity, mask, Xi, Zi, method):
    points = np.column_stack((pos[mask, 0], pos[mask, 2]))
    magnitude = np.linalg.norm(velocity, axis=-1)[mask]
    image = griddata(points, magnitude, (Xi, Zi), method=method)
    if np.isnan(image).any():
        nearest = griddata(points, magnitude, (Xi, Zi), method="nearest")
        image = np.where(np.isnan(image), nearest, image)
    return image


def build_images(pos, gt, pred, mask, Xi, Zi, method):
    gt_images, pred_images, error_images = [], [], []
    for gt_t, pred_t in zip(gt, pred):
        gt_images.append(interpolate_magnitude(pos, gt_t, mask, Xi, Zi, method))
        pred_images.append(interpolate_magnitude(pos, pred_t, mask, Xi, Zi, method))
        error_images.append(interpolate_magnitude(pos, gt_t - pred_t, mask, Xi, Zi, method))
    return gt_images, pred_images, error_images


def draw_panel(ax, image, title, extent, x_air, z_air, cmap, vmin, vmax):
    im = ax.imshow(
        image, origin="lower", extent=extent, aspect="auto", cmap=cmap,
        vmin=vmin, vmax=vmax,
    )
    if len(x_air):
        ax.scatter(x_air, z_air, s=0.25, c="black", alpha=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def render_static(all_images, input_times, output_times, geometry, out_path, limits):
    in_gt, in_pred, in_err, out_gt, out_pred, out_err = all_images
    extent, x_air, z_air = geometry
    value_min, value_max, error_min, error_max = limits
    fig, axes = plt.subplots(5, 6, figsize=(21, 15), constrained_layout=True)
    value_im = error_im = None
    for k in range(5):
        panels = (
            (in_gt[k], f"Input NPZ | t={input_times[k]:.4g}", "plasma", value_min, value_max),
            (in_pred[k], "Input INR", "plasma", value_min, value_max),
            (in_err[k], "Input error", "magma", error_min, error_max),
            (out_gt[k], f"Output NPZ | t={output_times[k]:.4g}", "plasma", value_min, value_max),
            (out_pred[k], "ScaleGMN INR", "plasma", value_min, value_max),
            (out_err[k], "Output error", "magma", error_min, error_max),
        )
        for j, panel in enumerate(panels):
            im = draw_panel(axes[k, j], panel[0], panel[1], extent, x_air, z_air, *panel[2:])
            if j == 0:
                value_im = im
            if j == 2:
                error_im = im
    fig.colorbar(value_im, ax=axes[:, [0, 1, 3, 4]], shrink=0.55, label="Velocity magnitude |v|")
    fig.colorbar(error_im, ax=axes[:, [2, 5]], shrink=0.55, label="Vector error |GT - INR|")
    fig.suptitle("Raw flow, input INR reconstruction, and ScaleGMN future prediction", fontsize=15)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_gif_frames(
    all_images,
    input_times,
    output_times,
    geometry,
    limits,
    first_timestep_path=None,
):
    in_gt, in_pred, in_err, out_gt, out_pred, out_err = all_images
    extent, x_air, z_air = geometry
    value_min, value_max, error_min, error_max = limits
    frames = []
    for k in range(5):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        panels = (
            (in_gt[k], f"Input NPZ | t={input_times[k]:.4g}", "plasma", value_min, value_max),
            (in_pred[k], "Input INR", "plasma", value_min, value_max),
            (in_err[k], "Input error", "magma", error_min, error_max),
            (out_gt[k], f"Output NPZ | t={output_times[k]:.4g}", "plasma", value_min, value_max),
            (out_pred[k], "ScaleGMN INR", "plasma", value_min, value_max),
            (out_err[k], "Output error", "magma", error_min, error_max),
        )
        value_im = error_im = None
        for ax, panel in zip(axes.flat, panels):
            im = draw_panel(ax, panel[0], panel[1], extent, x_air, z_air, *panel[2:])
            if panel[2] == "plasma":
                value_im = im
            else:
                error_im = im
        fig.colorbar(value_im, ax=axes[:, :2], shrink=0.8, label="Velocity magnitude |v|")
        fig.colorbar(error_im, ax=axes[:, 2], shrink=0.8, label="Vector error |GT - INR|")
        fig.suptitle(f"Paired timestep {k + 1}/5", fontsize=14)
        if k == 0 and first_timestep_path is not None:
            fig.savefig(first_timestep_path, dpi=300, bbox_inches="tight")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    return frames


@torch.inference_mode()
def main():
    args = parse_args()
    selected_inr = Path(args.inr).expanduser().resolve()
    npz_path = Path(args.npz).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    inr_dir = Path(args.inr_dir).expanduser().resolve() if args.inr_dir else selected_inr.parent
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in (selected_inr, npz_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint or "conf" not in checkpoint:
        raise KeyError("Checkpoint must contain model_state_dict and conf")
    conf = copy.deepcopy(checkpoint["conf"])
    conf["debug"] = False
    conf["data"]["return_path"] = True
    conf["data"]["dataset_path"] = str(inr_dir)

    inference_set = dataset(
        conf["data"], split=args.split, debug=False,
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=mask_hidden(conf),
        get_first_layer_mask=mask_input(conf),
    )
    conf["scalegmn_args"]["layer_layout"] = inference_set.get_layer_layout()
    loader = torch_geometric.loader.DataLoader(
        inference_set, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = ScaleGMN_equiv(conf["scalegmn_args"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    selected = None
    available_names = []
    for batch in loader:
        params, wb, paths = unpack_batch(batch)
        available_names.extend(Path(p).name for p in paths)
        if any(same_file(p, selected_inr) for p in paths):
            selected = (params, wb)
            break
    if selected is None:
        preview = ", ".join(available_names[:10])
        raise FileNotFoundError(
            f"{selected_inr.name} was not returned by split '{args.split}' under {inr_dir}. "
            f"First dataset files: {preview}"
        )

    params, wb = selected
    params = params.to(device)
    wb = wb.to(device)
    graph_weights, graph_biases = wb.weights, wb.biases
    delta_weights, delta_biases = model(params.clone(), graph_weights, graph_biases)
    predicted_graph_w, predicted_graph_b = residual_update(
        graph_weights, graph_biases, delta_weights, delta_biases
    )
    predicted_weights, predicted_biases = graph_to_linear(predicted_graph_w, predicted_graph_b)

    inr_state = torch.load(selected_inr, map_location=device, weights_only=False)
    input_weights, input_biases = state_dict_to_linear(inr_state, device)
    inr_conf = conf.get("inr_model") or {}
    w0 = args.w0 if args.w0 is not None else float(
        inr_conf.get("w0", inr_conf.get("omega_0", conf["data"].get("w0", 30.0)))
    )

    pos, input_times, output_times, velocity_in, velocity_out, idcs_airfoil = load_npz(npz_path)
    input_prediction = evaluate_siren(
        pos, input_times, input_weights, input_biases, w0, args.chunk_size, device
    )
    output_prediction = evaluate_siren(
        pos, output_times, predicted_weights, predicted_biases, w0, args.chunk_size, device
    )

    center = float(pos[:, 1].mean()) if args.slice_center is None else args.slice_center
    mask, Xi, Zi, extent, x_air, z_air = make_slice_geometry(
        pos, idcs_airfoil, center, args.slice_thickness, args.grid_res
    )
    in_images = build_images(pos, velocity_in, input_prediction, mask, Xi, Zi, args.interp)
    out_images = build_images(pos, velocity_out, output_prediction, mask, Xi, Zi, args.interp)
    all_images = (*in_images, *out_images)
    value_images = in_images[:2] + out_images[:2]
    error_images = (in_images[2], out_images[2])
    value_min = min(float(np.nanmin(im)) for group in value_images for im in group)
    value_max = max(float(np.nanmax(im)) for group in value_images for im in group)
    error_min = min(float(np.nanmin(im)) for group in error_images for im in group)
    error_max = max(float(np.nanmax(im)) for group in error_images for im in group)
    limits = value_min, value_max, error_min, error_max
    geometry = extent, x_air, z_air

    stem = npz_path.stem
    png_path = out_dir / f"{stem}_scalegmn_full_comparison.png"
    first_timestep_path = out_dir / f"{stem}_scalegmn_first_timestep.png"
    gif_path = out_dir / f"{stem}_scalegmn_full_comparison.gif"
    render_static(all_images, input_times, output_times, geometry, png_path, limits)
    frames = render_gif_frames(
        all_images,
        input_times,
        output_times,
        geometry,
        limits,
        first_timestep_path=first_timestep_path,
    )
    imageio.mimsave(gif_path, frames, duration=1.0 / args.gif_fps, loop=0)

    input_rel_l2 = np.linalg.norm(input_prediction - velocity_in) / np.linalg.norm(velocity_in)
    output_rel_l2 = np.linalg.norm(output_prediction - velocity_out) / np.linalg.norm(velocity_out)
    print(f"Device: {device}")
    print(f"SIREN w0: {w0}")
    print(f"Input INR relative L2: {input_rel_l2:.8f}")
    print(f"ScaleGMN output relative L2: {output_rel_l2:.8f}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved first-timestep PNG: {first_timestep_path}")
    print(f"Saved GIF: {gif_path}")


if __name__ == "__main__":
    main()