#!/usr/bin/env python3
"""Inspect the time and velocity arrays of one warped-IFW NPZ sample."""

import argparse
from pathlib import Path

import numpy as np


DEFAULT_NPZ = (
    "/home/intern/spygeorgoulas/thesis-metanets/neural-fields-3d/"
    "data/ifw-comp-test/1024_13-4.npz"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=DEFAULT_NPZ, help="Path to the NPZ sample")
    parser.add_argument(
        "--point-index",
        type=int,
        default=None,
        help="Optionally print the complete 10-step velocity sequence at one point",
    )
    return parser.parse_args()


def velocity_statistics(velocity):
    magnitude = np.linalg.norm(velocity, axis=-1)
    return {
        "mag_min": float(magnitude.min()),
        "mag_mean": float(magnitude.mean()),
        "mag_max": float(magnitude.max()),
        "u_mean": float(velocity[:, 0].mean()),
        "v_mean": float(velocity[:, 1].mean()),
        "w_mean": float(velocity[:, 2].mean()),
    }


def main():
    args = parse_args()
    path = Path(args.file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path) as data:
        print(f"File: {path}")
        print("\nArrays:")
        for key in data.files:
            print(f"  {key:16s} shape={data[key].shape}, dtype={data[key].dtype}")

        required = {"t", "pos", "velocity_in", "velocity_out"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Missing required arrays: {sorted(missing)}")

        times = data["t"].astype(np.float64)
        pos = data["pos"]
        velocity_in = data["velocity_in"].astype(np.float64)
        velocity_out = data["velocity_out"].astype(np.float64)

    if times.ndim != 1 or len(times) != 10:
        raise ValueError(f"Expected t with shape (10,), got {times.shape}")
    expected_velocity_shape = (5, len(pos), 3)
    if velocity_in.shape != expected_velocity_shape:
        raise ValueError(
            f"Expected velocity_in {expected_velocity_shape}, got {velocity_in.shape}"
        )
    if velocity_out.shape != expected_velocity_shape:
        raise ValueError(
            f"Expected velocity_out {expected_velocity_shape}, got {velocity_out.shape}"
        )

    print("\nComplete time array:")
    print("  " + np.array2string(times, precision=10, separator=", "))
    print(f"  Strictly increasing: {bool(np.all(np.diff(times) > 0))}")
    print(f"  Time increments: {np.array2string(np.diff(times), precision=10)}")

    print("\nExact dataset mapping:")
    print("  Known interval — velocity_in")
    for i in range(5):
        print(f"    velocity_in[{i}]  -> t[{i}] = {times[i]:.10g}")
    print("  Future interval — velocity_out")
    for i in range(5):
        print(f"    velocity_out[{i}] -> t[{i + 5}] = {times[i + 5]:.10g}")

    print("\nPer-timestep statistics:")
    print(
        "  source           time         |v| min       |v| mean      |v| max"
        "       mean(u)       mean(v)       mean(w)"
    )
    for source, values, time_values in (
        ("velocity_in", velocity_in, times[:5]),
        ("velocity_out", velocity_out, times[5:]),
    ):
        for i, (velocity, time_value) in enumerate(zip(values, time_values)):
            s = velocity_statistics(velocity)
            print(
                f"  {source}[{i}]  {time_value:12.8f}  "
                f"{s['mag_min']:12.6f}  {s['mag_mean']:12.6f}  "
                f"{s['mag_max']:12.6f}  {s['u_mean']:12.6f}  "
                f"{s['v_mean']:12.6f}  {s['w_mean']:12.6f}"
            )

    boundary_difference = velocity_out[0] - velocity_in[-1]
    boundary_relative_l2 = (
        np.linalg.norm(boundary_difference.reshape(-1))
        / (np.linalg.norm(velocity_in[-1].reshape(-1)) + 1e-12)
    )
    boundary_pointwise_change = np.linalg.norm(boundary_difference, axis=-1)
    print("\nInput/output boundary:")
    print(f"  Last known field:   velocity_in[4] at t={times[4]:.10g}")
    print(f"  First future field: velocity_out[0] at t={times[5]:.10g}")
    print(f"  Boundary time step: {times[5] - times[4]:.10g}")
    print(f"  Relative field change: {boundary_relative_l2:.10f}")
    print(
        "  Pointwise |velocity_out[0] - velocity_in[4]|: "
        f"min={boundary_pointwise_change.min():.6f}, "
        f"mean={boundary_pointwise_change.mean():.6f}, "
        f"max={boundary_pointwise_change.max():.6f}"
    )

    if args.point_index is not None:
        index = args.point_index
        if not 0 <= index < len(pos):
            raise IndexError(f"point-index must be between 0 and {len(pos) - 1}")
        complete_velocity = np.concatenate((velocity_in, velocity_out), axis=0)
        print(f"\nPoint {index}:")
        print(f"  position = {pos[index]}")
        for i, (time_value, velocity) in enumerate(
            zip(times, complete_velocity[:, index])
        ):
            interval = "input " if i < 5 else "output"
            print(
                f"  t[{i}]={time_value:.10g} ({interval}) -> "
                f"velocity={velocity}, |v|={np.linalg.norm(velocity):.10f}"
            )


if __name__ == "__main__":
    main()