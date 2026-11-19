"""using PyTorchFire on the Bear 2020 fire (California)

it loads the example arrays that ship with the PyTorchFire repo, runs the
cellular-automata fire simulator step by step, and writes to results/

    bear2020_inputs.png    the input maps (fuel, slope, wind, ignition)
    bear2020.gif           simulated spread, observed final perimeter in blue
    bear2020_final.png     where simulation and observation agree / disagree
    bear2020_metrics.json  the numbers for the README

Run:
    uv run python scripts/01_bear2020.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window: only write image files
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pytorchfire import WildfireModel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "external" / "PyTorchFire" / "examples" / "Bear_2020"
OUT = ROOT / "results"

LAYERS = ["p_veg", "p_den", "slope", "wind_velocity",
          "wind_towards_direction", "initial_ignition", "target"]

# colours (RGB in 0..1)
BURNING = (1.00, 0.40, 0.00)
BURNED = (0.30, 0.30, 0.30)
OBSERVED = "#1f6fd1"
BOTH = (0.55, 0.10, 0.10)
SIM_ONLY = (1.00, 0.60, 0.20)
OBS_ONLY = (0.20, 0.45, 0.85)


def load_array(data_dir: Path, name: str) -> np.ndarray:
    """Each layer is one .npz file holding one array."""
    with np.load(data_dir / f"{name}.npz") as f:
        if len(f.files) > 1:
            print(f"  note: {name}.npz holds {f.files}, using the first one")
        arr = f[f.files[0]]
    print(f"  {name:<24} shape={str(arr.shape):<22} dtype={arr.dtype}")
    return arr

def as_frames(arr: np.ndarray) -> np.ndarray:
    """Time-varying layers as [T, H, W]; a single 2-D map becomes T = 1."""
    return arr[None] if arr.ndim == 2 else arr

def ignition_mask(arr: np.ndarray, shape: tuple) -> np.ndarray:
    """Ignition as a boolean map. Also accepts a list of (row, col) cells."""
    if arr.ndim == 3:  # [T, H, W]: the ignition is the first frame
        arr = arr[0]
    if arr.shape == shape:
        return arr > 0
    mask = np.zeros(shape, dtype=bool)
    rc = np.asarray(arr).reshape(-1, 2).astype(int)
    mask[rc[:, 0], rc[:, 1]] = True
    print(f"  initial_ignition given as {len(rc)} (row, col) cells -> mask")
    return mask

def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.nan_to_num(np.asarray(arr, dtype=np.float32)))

def build_model(data: dict, wind_v0, wind_d0, ignition, params: dict) -> WildfireModel:
    env = {
        "p_veg": to_tensor(data["p_veg"]),          # [H, W]  vegetation type factor
        "p_den": to_tensor(data["p_den"]),          # [H, W]  vegetation density factor
        "slope": to_tensor(data["slope"]),          # [H, W, 3, 3]  slope to each neighbour (deg)
        "wind_velocity": to_tensor(wind_v0),        # [H, W]  m/s
        "wind_towards_direction": to_tensor(wind_d0),  # [H, W]  deg, from East, counter-clockwise
        "initial_ignition": torch.as_tensor(ignition),  # [H, W]  bool
    }
    model = WildfireModel(env_data=env,
                          params={k: torch.tensor(float(v)) for k, v in params.items()})
    model.eval()  # forward simulation only: skip the gradient bookkeeping used for calibration
    return model


def state_image(model: WildfireModel) -> np.ndarray:
    """0 = unburned, 1 = burning, 2 = burned out."""
    burning, burned = model.state.cpu().numpy()
    img = np.zeros(burning.shape, dtype=np.uint8)
    img[burned] = 2
    img[burning] = 1
    return img


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union (Jaccard index) of two boolean maps."""
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def wind_index(step: int, n_wind: int, n_steps: int) -> int:
    """Which wind map is active at this step (Bear 2020: 20 maps over 300 steps = every 15 steps)."""
    return min(step * n_wind // n_steps, n_wind - 1)


def simulate(model, wind_v, wind_d, targets, args, device):
    """One model step per observed frame, swapping in the next wind map when it is due.

    The data has one observed burned-area map per simulation step (target[t]),
    so after step t we compare the simulated burned area with target[t].
    """
    n_steps = len(targets)
    snapshots, log = [], []
    current_wind = -1
    for t in range(n_steps):
        w = wind_index(t, len(wind_v), n_steps)
        if w != current_wind:  # new wind map
            model.wind_velocity = torch.as_tensor(wind_v[w], device=device)
            model.wind_towards_direction = torch.as_tensor(wind_d[w], device=device)
            current_wind = w

        model.compute()  # one cellular-automata step

        if (t + 1) % args.save_every == 0:
            snapshots.append((t + 1, w, state_image(model)))
        simulated = (model.state[0] | model.state[1]).cpu().numpy()
        log.append({"step": t + 1, "wind_map": w + 1,
                    "observed_cells": int(targets[t].sum()),
                    "simulated_cells": int(simulated.sum()),
                    "iou": round(iou(simulated, targets[t]), 3)})
        if (t + 1) % args.print_every == 0 or t + 1 == n_steps:
            d = log[-1]
            print(f"  step {d['step']:>3}/{n_steps} (wind map {d['wind_map']:>2}) | observed "
                  f"{d['observed_cells']:>6} cells | simulated {d['simulated_cells']:>6} | IoU {d['iou']:.2f}")
        if not model.state[0].any():
            print("  fire burned out, stopping")
            break

    if not snapshots or snapshots[-1][0] != t + 1:
        snapshots.append((t + 1, w, state_image(model)))  # always keep the final state
    return log, snapshots, simulated


## figures


def background(p_veg: np.ndarray, p_den: np.ndarray) -> np.ndarray:
    """Light beige (little fuel) to green (lots of fuel), as an RGB map."""
    fuel = np.nan_to_num((1 + p_veg) * (1 + p_den))  # the fuel term of the spread probability
    f = (fuel - fuel.min()) / (np.ptp(fuel) + 1e-9)
    low, high = np.array([0.96, 0.95, 0.92]), np.array([0.62, 0.78, 0.55])
    return low + f[..., None] * (high - low)


def to_rgb(img: np.ndarray, base: np.ndarray) -> np.ndarray:
    rgb = base.copy()
    rgb[img == 2] = BURNED
    rgb[img == 1] = BURNING
    return rgb


def mean_wind(speed: np.ndarray, toward_deg: np.ndarray) -> tuple[float, float]:
    """Average wind as (speed m/s, direction it blows towards, deg from East CCW)."""
    th = np.deg2rad(toward_deg)
    u, v = float((speed * np.cos(th)).mean()), float((speed * np.sin(th)).mean())
    return float(np.hypot(u, v)), float(np.rad2deg(np.arctan2(v, u)) % 360)


def draw_wind_arrow(ax, speed, toward_deg) -> list:
    """Arrow in the top-right corner. Row 0 of the grid is north (up)."""
    s, d = mean_wind(speed, toward_deg)
    dx, dy = 0.07 * np.cos(np.deg2rad(d)), 0.07 * np.sin(np.deg2rad(d))
    x0, y0 = 0.88, 0.88
    arrow = ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0 - dx, y0 - dy),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", lw=2, color="black"))
    label = ax.text(x0, y0 - 0.12, f"wind {s:.1f} m/s", transform=ax.transAxes,
                    ha="center", fontsize=8)
    return [arrow, label]


def save_inputs_figure(data, wind_v, wind_d, ignition, path: Path):
    steepness = np.abs(np.asarray(data["slope"])).max(axis=(2, 3))
    panels = [
        ("Vegetation type  (p_veg)", data["p_veg"], "YlGn"),
        ("Vegetation density  (p_den)", data["p_den"], "YlGn"),
        ("Slope to steepest neighbour (deg)", steepness, "copper_r"),
        ("Wind speed, first wind map (m/s)", wind_v[0], "Blues"),
    ]
    rows, cols = np.nonzero(ignition)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), dpi=90)
    for ax, (title, arr, cmap) in zip(axes.flat, panels):
        im = ax.imshow(arr, cmap=cmap)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.scatter(cols, rows, s=6, c="red")
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    draw_wind_arrow(axes[1, 1], wind_v[0], wind_d[0])
    fig.suptitle("Bear 2020: PyTorchFire inputs (red = ignition)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_gif(snapshots, base, final_observed, wind_v, wind_d, path: Path,
             max_frames: int = 50, fps: int = 8):
    stride = max(1, len(snapshots) // max_frames)
    frames = snapshots[::stride]
    if frames[-1] is not snapshots[-1]:
        frames.append(snapshots[-1])

    fig, ax = plt.subplots(figsize=(6, 6.4), dpi=64)  # small enough to show in a README
    im = ax.imshow(to_rgb(frames[0][2], base))
    ax.contour(final_observed.astype(float), levels=[0.5], colors=OBSERVED, linewidths=1.2)
    ax.set_axis_off()
    title = ax.set_title("")
    ax.legend(handles=[Patch(color=BURNING, label="burning"),
                       Patch(color=BURNED, label="burned"),
                       Line2D([], [], color=OBSERVED, label="observed final perimeter")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
              frameon=False, fontsize=8)
    arrow = []

    def update(i):
        nonlocal arrow
        step, w, img = frames[i]
        im.set_data(to_rgb(img, base))
        title.set_text(f"Bear 2020 · step {step} · wind map {w + 1}")
        for artist in arrow:
            artist.remove()
        arrow = draw_wind_arrow(ax, wind_v[w], wind_d[w])
        return [im, title, *arrow]

    FuncAnimation(fig, update, frames=len(frames)).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_comparison(simulated, observed, base, path: Path):
    rgb = base.copy()
    rgb[simulated & observed] = BOTH
    rgb[simulated & ~observed] = SIM_ONLY
    rgb[~simulated & observed] = OBS_ONLY
    fig, ax = plt.subplots(figsize=(6, 6.4), dpi=90)
    ax.imshow(rgb)
    ax.set_axis_off()
    ax.set_title(f"Final step: simulated vs observed (IoU = {iou(simulated, observed):.2f})")
    ax.legend(handles=[Patch(color=BOTH, label="burned in both"),
                       Patch(color=SIM_ONLY, label="simulated only"),
                       Patch(color=OBS_ONLY, label="observed only")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
              frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)



## main


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--device", default="cpu", help="cpu, mps (Apple GPU) or cuda")
    p.add_argument("--seed", type=int, default=0, help="the model is random: fix the seed")
    p.add_argument("--save-every", type=int, default=2, help="GIF snapshot every N steps")
    p.add_argument("--print-every", type=int, default=15, help="progress line every N steps")
    # Uncalibrated starting values, same as PyTorchFire's firedataforge_simulate.py example
    p.add_argument("--a", type=float, default=0.1, help="slope effect")
    p.add_argument("--p_h", type=float, default=0.4, help="base spread probability")
    p.add_argument("--c_1", type=float, default=0.05, help="wind speed effect")
    p.add_argument("--c_2", type=float, default=0.05, help="wind direction effect")
    p.add_argument("--p_continue", type=float, default=0.3, help="chance a cell keeps burning")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"Data not found: {args.data}\n"
                         "Clone PyTorchFire into external/ first (see README).")
    OUT.mkdir(exist_ok=True)

    print(f"Loading {args.data}")
    data = {name: load_array(args.data, name) for name in LAYERS}
    shape = data["p_veg"].shape
    wind_v = as_frames(data["wind_velocity"]).astype(np.float32)
    wind_d = as_frames(data["wind_towards_direction"]).astype(np.float32)
    targets = as_frames(data["target"]) > 0  # observed burned area, one map per step
    ignition = ignition_mask(data["initial_ignition"], shape)
    print(f"Grid {shape[0]} x {shape[1]} cells, {len(targets)} observed frames (= steps), "
          f"{len(wind_v)} wind maps (one every {len(targets) // len(wind_v)} steps), "
          f"{int(ignition.sum())} ignition cell(s)")

    params = {k: getattr(args, k) for k in ["a", "p_h", "c_1", "c_2", "p_continue"]}
    model = build_model(data, wind_v[0], wind_d[0], ignition, params).to(args.device)
    model.reset(seed=args.seed)

    print(f"Simulating on {args.device}, seed {args.seed}")
    t0 = time.perf_counter()
    with torch.inference_mode():
        log, snapshots, simulated = simulate(model, wind_v, wind_d, targets, args, args.device)
    runtime = time.perf_counter() - t0
    total_steps = len(log)
    print(f"Done: {total_steps} steps in {runtime:.1f} s "
          f"({1000 * runtime / max(total_steps, 1):.1f} ms/step)")

    print("Writing figures")
    base = background(np.asarray(data["p_veg"], np.float32), np.asarray(data["p_den"], np.float32))
    save_inputs_figure(data, wind_v, wind_d, ignition, OUT / "bear2020_inputs.png")
    save_gif(snapshots, base, targets[-1], wind_v, wind_d, OUT / "bear2020.gif")
    save_comparison(simulated, targets[total_steps - 1], base, OUT / "bear2020_final.png")

    final = log[-1]
    metrics = {
        "fire": "Bear 2020 (California)",
        "grid": list(shape),
        "device": args.device,
        "seed": args.seed,
        "params": params,
        "total_steps": total_steps,
        "runtime_s": round(runtime, 2),
        "ms_per_step": round(1000 * runtime / max(total_steps, 1), 2),
        "final_iou": final["iou"],
        "final_area_ratio_sim_over_obs": round(final["simulated_cells"] / max(final["observed_cells"], 1), 2),
        "steps": log,
    }
    (OUT / "bear2020_metrics.json").write_text(json.dumps(metrics, indent=2))
    for f in sorted(OUT.glob("bear2020*")):
        print(f"  {f.relative_to(ROOT)}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
