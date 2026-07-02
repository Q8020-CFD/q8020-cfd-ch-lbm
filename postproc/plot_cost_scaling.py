"""Circuit-cost scaling figure: CH vs QLBM across the q-ladder.

Reads a circuit_cost_scaling JSON (series -> points with q / n_qubits /
cx / circuit_depth) and plots CX, depth, and qubit count vs q on three
panels, log-scaled, so QLBM's flat grid-independent cost contrasts with
CH's grid-driven growth.

Usage:  plot_cost_scaling.py <cost.json> [--out fig.png]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="path to circuit_cost_scaling JSON")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    ds_path = Path(args.dataset).resolve()
    ds = json.loads(ds_path.read_text())
    series = ds.get("series", [])
    if not series:
        print("No series in dataset.", file=sys.stderr)
        sys.exit(1)

    metrics = [
        ("cx", "CX gates", True),
        ("circuit_depth", "circuit depth", True),
        ("n_qubits", "qubits", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    cmap = plt.get_cmap("tab10")

    for ax, (key, ylabel, logy) in zip(axes, metrics):
        for i, s in enumerate(series):
            pts = sorted(s["points"], key=lambda p: p["q"])
            xs = [p["q"] for p in pts]
            ys = [p[key] for p in pts]
            ax.plot(
                xs, ys, marker="o", lw=1.8, ms=6,
                color=cmap(i % 10), label=s.get("label", s["method"]),
            )
        ax.set_xlabel("q (qubits dialed)")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(ylabel)
    axes[0].legend(loc="best", fontsize=9)

    fig.suptitle(
        f"{ds.get('name', ds_path.stem)} — circuit cost vs resolution "
        f"(CH grid-driven, QLBM grid-independent)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(args.out) if args.out else ds_path.with_suffix(".png")
    fig.savefig(str(out), dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
