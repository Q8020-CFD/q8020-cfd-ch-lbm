"""Regime-pair figure: final-state relL2 of CH vs QLBM, two panels.

Takes two dataset JSONs (e.g. case2A shock | case2B margin), each with
q{4,6,8} CH/QLBM runs + an ftcs_reference, and draws a grouped bar chart
of final-state relL2 vs q per panel.  Missing runs (final-only, no
solution_steps) are drawn as a hatched "no frames" placeholder; bars
above --cap are clipped and annotated with their true value (a detonated
CH run otherwise crushes the axis).

Usage:
  plot_regime_pair.py A.json B.json [--out fig.png] [--cap 1.2]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _final(path: Path):
    af = path / "q8020_artifacts_0.json"
    if not af.is_file():
        return None
    a = json.loads(af.read_text())
    ss = a.get("solution_steps", {})
    if not ss:
        return None
    last = str(max(int(k) for k in ss))
    return np.array(a["grid"]), np.array(ss[last])


def _rel_l2(path: Path, gref: np.ndarray, uref: np.ndarray):
    got = _final(path)
    if got is None:
        return None
    g, u = got
    ui = np.interp(gref, g, u) if len(g) != len(gref) else u
    return float(np.sqrt(np.mean((ui - uref) ** 2)) / np.sqrt(np.mean(uref ** 2)))


def _panel_data(ds_path: Path):
    """Return (title, {q: {'ch':relL2|None, 'qlbm':relL2|None}})."""
    ds = json.loads(ds_path.read_text())
    base = ds_path.parent
    runs = {r["label"]: r for r in ds["runs"]}
    ref = runs.get("ftcs_reference")
    gref, uref = _final((base / ref["path"]).resolve())
    out = {}
    qs = sorted({r["q"] for r in ds["runs"] if r.get("method") != "ftcs_reference"})
    for q in qs:
        row = {}
        for meth, key in (("ch", f"q{q}_ch"), ("qlbm", f"q{q}_qlbm")):
            r = runs.get(key)
            row[meth] = (
                _rel_l2((base / r["path"]).resolve(), gref, uref)
                if r else None
            )
        out[q] = row
    g = ds.get("global", {})
    title = (
        f"{ds.get('name', ds_path.stem)}\n"
        f"A={g.get('ic_amplitude')}, nu={g.get('nu')}, "
        f"~{g.get('shock_pct_effective', '?')}% shock"
    )
    return title, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs=2, help="two dataset JSONs (A then B)")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--cap", type=float, default=1.2,
                    help="clip bars above this; annotate true value")
    args = ap.parse_args()

    panels = [_panel_data(Path(p).resolve()) for p in args.datasets]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    width = 0.38
    colors = {"ch": "tab:blue", "qlbm": "tab:orange"}

    for ax, (title, data) in zip(axes, panels):
        qs = sorted(data)
        xs = np.arange(len(qs))
        for j, meth in enumerate(("ch", "qlbm")):
            off = (j - 0.5) * width
            label = "CH" if meth == "ch" else "QLBM/QALB"
            for i, q in enumerate(qs):
                v = data[q][meth]
                x = xs[i] + off
                if v is None:
                    ax.bar(x, args.cap, width, color="none",
                           edgecolor="0.6", hatch="//",
                           label=label if i == 0 else None)
                    ax.text(x, args.cap * 0.5, "no\nframes", ha="center",
                            va="center", fontsize=7, color="0.4", rotation=90)
                    continue
                drawn = min(v, args.cap)
                ax.bar(x, drawn, width, color=colors[meth],
                       label=label if i == 0 else None)
                if v > args.cap:
                    ax.text(x, args.cap, f"{v:.0f}", ha="center",
                            va="bottom", fontsize=8, fontweight="bold",
                            color=colors[meth])
        ax.axhline(1.0, color="0.5", ls=":", lw=1)
        ax.text(xs[-1] + 0.5, 1.0, "err = signal", fontsize=7,
                color="0.5", va="bottom", ha="right")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"q{q}" for q in qs])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("qubits (grid resolution)")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("final-state relative L2 vs FTCS")
    axes[0].set_ylim(0, args.cap)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "Best tool by regime: CH-favorable (shock) vs QLBM-favorable (margin)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = (Path(args.out) if args.out
           else Path(args.datasets[0]).resolve().parent / "case2_regime_pair.png")
    fig.savefig(str(out), dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
