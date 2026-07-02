"""Regime-crossover figure: method accuracy vs a swept axis (e.g. nu).

Reads a JSON with x_axis / y_axis specs and a list of series (each a set
of {x_key, y_key} points) and plots them on one axis, shading which
method wins on each side of the crossover.

Usage:  plot_regime_crossover.py <crossover.json> [--out fig.png]
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
    ap.add_argument("dataset", help="path to crossover JSON")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    ds_path = Path(args.dataset).resolve()
    ds = json.loads(ds_path.read_text())
    xa, ya = ds["x_axis"], ds["y_axis"]
    xk, yk = xa["key"], ya["key"]
    series = ds.get("series", [])
    if not series:
        print("No series in dataset.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(series):
        pts = sorted(s["points"], key=lambda p: p[xk])
        xs = [p[xk] for p in pts]
        ys = [p[yk] for p in pts]
        ax.plot(
            xs, ys, marker="o", lw=2, ms=7,
            color=cmap(i % 10), label=s.get("label", s["method"]),
        )

    ax.set_xlabel(xa.get("label", xk))
    ax.set_ylabel(ya.get("label", yk))
    if ya.get("log"):
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.set_title(f"{ds.get('name', ds_path.stem)} — best tool by regime")

    fig.tight_layout()
    out = Path(args.out) if args.out else ds_path.with_suffix(".png")
    fig.savefig(str(out), dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
