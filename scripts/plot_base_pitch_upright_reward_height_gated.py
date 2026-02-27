#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot Spot's `base_pitch_upright_reward` (pitch band-pass × height band-pass).

This script is intentionally standalone: it does not require launching Isaac Sim or constructing an env.
It reproduces the scalar math used by:
`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py`

What gets plotted:
1) Pitch band-pass factor vs pitch angle.
2) Height band-pass factor vs base height (world z).
3) Final reward r(pitch, z) = r_pitch(pitch) * r_height(z) as a 2D heatmap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _bandpass_reward(
    x: torch.Tensor,
    *,
    x_min: float,
    x_max: float,
    std: float,
) -> torch.Tensor:
    """Smooth band-pass reward: 1.0 inside [min,max], Gaussian decay outside.

    Args:
        x: Input values. Shape: (N,).
        x_min: Lower band edge (same units as x).
        x_max: Upper band edge (same units as x).
        std: Decay length-scale outside the band (same units as x).

    Returns:
        Reward in (0, 1]. Shape: (N,).
    """
    x_min_t = x.new_tensor(x_min)
    x_max_t = x.new_tensor(x_max)
    x_min_t, x_max_t = torch.min(x_min_t, x_max_t), torch.max(x_min_t, x_max_t)

    # Distance outside band (0 inside): (N,)
    d_below = torch.relu(x_min_t - x)
    d_above = torch.relu(x - x_max_t)
    d_out = d_below + d_above

    std_t = torch.clamp(x.new_tensor(std), min=1.0e-6)
    return torch.exp(-torch.square(d_out / std_t))


def base_pitch_upright_reward_bandpass_curve(
    pitch_deg: torch.Tensor,
    *,
    base_height_m: float,
    min_pitch_deg: float,
    max_pitch_deg: float,
    pitch_std_deg: float,
    min_height_m: float,
    max_height_m: float,
    height_std_m: float,
) -> torch.Tensor:
    """Standalone curve for pitch-band-pass × height-band-pass reward.

    Args:
        pitch_deg: Pitch angles in degrees. Shape: (N,).
        base_height_m: Scalar base height (world z) [m].
        ...: Same meaning as in `base_pitch_upright_reward` params.

    Returns:
        Reward in (0, 1]. Shape: (N,).
    """
    r_pitch = _bandpass_reward(pitch_deg, x_min=min_pitch_deg, x_max=max_pitch_deg, std=pitch_std_deg)  # (N,)
    z = pitch_deg.new_full(pitch_deg.shape, float(base_height_m))  # (N,) [m]
    r_height = _bandpass_reward(z, x_min=min_height_m, x_max=max_height_m, std=height_std_m)  # (N,)
    return r_pitch * r_height


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot band-pass base_pitch_upright_reward factors and product.")
    # Pitch band-pass params (defaults match Spot flat config)
    parser.add_argument("--min_pitch_deg", type=float, default=60.0)
    parser.add_argument("--max_pitch_deg", type=float, default=80.0)
    parser.add_argument("--pitch_std_deg", type=float, default=30.0)
    # Height band-pass params
    parser.add_argument("--min_height_m", type=float, default=0.5)
    parser.add_argument("--max_height_m", type=float, default=0.8)
    parser.add_argument("--height_std_m", type=float, default=0.24)
    # Plot ranges
    parser.add_argument("--pitch_min_deg", type=float, default=-30.0)
    parser.add_argument("--pitch_max_deg", type=float, default=110.0)
    parser.add_argument("--num", type=int, default=2000)
    parser.add_argument("--z_min_m", type=float, default=0.0)
    parser.add_argument("--z_max_m", type=float, default=1.2)
    parser.add_argument("--z_num", type=int, default=2000)
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional output path (e.g. reward_pitch_height_bandpass.png). If empty, shows an interactive window.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    pitch_deg = torch.linspace(args.pitch_min_deg, args.pitch_max_deg, args.num)  # (N,)
    z_w = torch.linspace(args.z_min_m, args.z_max_m, args.z_num)  # (M,)
    r_pitch = _bandpass_reward(
        pitch_deg, x_min=args.min_pitch_deg, x_max=args.max_pitch_deg, std=args.pitch_std_deg
    )  # (N,)
    r_height = _bandpass_reward(z_w, x_min=args.min_height_m, x_max=args.max_height_m, std=args.height_std_m)  # (M,)
    # r_total: (M, N)
    r_total = torch.outer(r_height, r_pitch)

    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, (ax_pitch_bp, ax_height_bp, ax_reward) = plt.subplots(1, 3, figsize=(15.0, 4.0))

    # (1) Pitch band-pass factor
    ax_pitch_bp.plot(pitch_deg.cpu().numpy(), r_pitch.cpu().numpy(), linewidth=2.0)
    ax_pitch_bp.axvline(args.min_pitch_deg, linestyle="--", linewidth=1.5, label="min_pitch_deg")
    ax_pitch_bp.axvline(args.max_pitch_deg, linestyle="--", linewidth=1.5, label="max_pitch_deg")
    ax_pitch_bp.set_title("Pitch band-pass factor")
    ax_pitch_bp.set_xlabel("pitch [deg]")
    ax_pitch_bp.set_ylabel("r_pitch")
    ax_pitch_bp.set_xlim(args.pitch_min_deg, args.pitch_max_deg)
    ax_pitch_bp.set_ylim(-0.05, 1.05)
    ax_pitch_bp.grid(True, alpha=0.35)
    ax_pitch_bp.legend(loc="best")

    # (2) Height band-pass factor
    ax_height_bp.plot(z_w.cpu().numpy(), r_height.cpu().numpy(), linewidth=2.0)
    ax_height_bp.axvline(args.min_height_m, linestyle="--", linewidth=1.5, label="min_height_m")
    ax_height_bp.axvline(args.max_height_m, linestyle="--", linewidth=1.5, label="max_height_m")
    ax_height_bp.set_title("Height band-pass factor")
    ax_height_bp.set_xlabel("base height z (world) [m]")
    ax_height_bp.set_ylabel("r_height")
    ax_height_bp.set_xlim(args.z_min_m, args.z_max_m)
    ax_height_bp.set_ylim(-0.05, 1.05)
    ax_height_bp.grid(True, alpha=0.35)
    ax_height_bp.legend(loc="best")

    # (3) Final reward heatmap r(pitch, z) = r_pitch(pitch) * r_height(z)
    im = ax_reward.imshow(
        r_total.cpu().numpy(),
        origin="lower",
        aspect="auto",
        extent=[args.pitch_min_deg, args.pitch_max_deg, args.z_min_m, args.z_max_m],
        vmin=0.0,
        vmax=1.0,
    )
    ax_reward.set_title("Final reward: r(pitch, z)")
    ax_reward.set_xlabel("pitch [deg]")
    ax_reward.set_ylabel("base height z (world) [m]")
    cbar = fig.colorbar(im, ax=ax_reward, fraction=0.046, pad=0.04)
    cbar.set_label("reward")

    fig.suptitle(
        "base_pitch_upright_reward = r_pitch(pitch) × r_height(z)\n"
        f"pitch_band=[{args.min_pitch_deg}, {args.max_pitch_deg}] deg, pitch_std={args.pitch_std_deg} deg | "
        f"height_band=[{args.min_height_m}, {args.max_height_m}] m, height_std={args.height_std_m} m"
    )
    fig.tight_layout()

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        print(f"[INFO] Saved plot to: {out_path.resolve()}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

