"""Segment-size <-> circuit-depth tradeoff figure.

Plots final-state relL2 vs segment-size for CH under two conditions:
  * noiseless  -- only seam error; falls monotonically as segments lengthen
  * noisy sim  -- seam error + depth decoherence; U-shaped, an optimum
with depth-per-circuit (rising with segment-size) on a right axis to
explain the noisy curve's upturn.

Reads a descriptor JSON (see seg_depth_tradeoff_q7.json): a reference
path, a depth_key, and points each with segment_size + noiseless_path +
noisy_path (either may be null until its run exists).

Usage:  plot_seg_depth_tradeoff.py <descriptor.json> [--out fig.png]
"""

import argparse
import json
import sys
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
    return np.array(a["grid"]), np.array(ss[str(max(int(k) for k in ss))])


def _rel_l2(path: Path, gref, uref):
    got = _final(path)
    if got is None:
        return None
    g, u = got
    ui = np.interp(gref, g, u) if len(g) != len(gref) else u
    return float(np.sqrt(np.mean((ui - uref) ** 2)) / np.sqrt(np.mean(uref ** 2)))


def _depth(path: Path, key: str):
    af = path / "q8020_analysis_0.json"
    if not af.is_file():
        return None
    a = json.loads(af.read_text())
    import re
    m = re.search(rf'"{key}":\s*([0-9.]+)', json.dumps(a))
    return float(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="path to tradeoff descriptor JSON")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    ds_path = Path(args.dataset).resolve()
    ds = json.loads(ds_path.read_text())
    base = ds_path.parent
    depth_key = ds.get("depth_key", "circuit_depth")

    gref, uref = _final((base / ds["reference"]).resolve())

    segs, noiseless, noisy, depths = [], [], [], []
    for p in sorted(ds["points"], key=lambda r: r["segment_size"]):
        segs.append(p["segment_size"])
        nl = (base / p["noiseless_path"]).resolve() if p.get("noiseless_path") else None
        nz = (base / p["noisy_path"]).resolve() if p.get("noisy_path") else None
        noiseless.append(_rel_l2(nl, gref, uref) if nl else None)
        noisy.append(_rel_l2(nz, gref, uref) if nz else None)
        # depth: prefer the noisy run, fall back to noiseless (same circuit)
        d = (_depth(nz, depth_key) if nz else None) or (_depth(nl, depth_key) if nl else None)
        depths.append(d)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax2 = ax.twinx()

    def _plot(ys, **kw):
        xs = [s for s, y in zip(segs, ys) if y is not None]
        vv = [y for y in ys if y is not None]
        if vv:
            ax.plot(xs, vv, **kw)

    _plot(noiseless, marker="s", lw=1.8, ls="--", color="0.55",
          label="noiseless (seam error only)")
    _plot(noisy, marker="o", lw=2.2, color="tab:red",
          label="noisy sim (seam + depth)")

    dxs = [s for s, d in zip(segs, depths) if d is not None]
    dvs = [d for d in depths if d is not None]
    if dvs:
        ax2.plot(dxs, dvs, marker="^", lw=1.5, ls=":", color="tab:blue",
                 label="depth / circuit")

    ax.axhline(1.0, color="0.7", ls=":", lw=1)
    ax.text(segs[-1], 1.0, " err = signal", fontsize=7, color="0.6",
            va="bottom", ha="right")
    ax.set_xlabel("segment-size (steps per measure-reprepare)")
    ax.set_ylabel("final-state relative L2 vs FTCS")
    ax2.set_ylabel("circuit depth per segment", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_xticks(segs)
    ax.grid(True, axis="y", alpha=0.3)
    g = ds.get("global", {})
    ax.set_title(
        f"{ds.get('name', ds_path.stem)} — CH seam-vs-depth tradeoff\n"
        f"q{g.get('q')}, A={g.get('ic_amplitude')}, nu={g.get('nu')}; "
        f"{g.get('noise_model', 'noisy sim')}",
        fontsize=10,
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8)
    fig.tight_layout()

    out = Path(args.out) if args.out else ds_path.with_suffix(".png")
    fig.savefig(str(out), dpi=130)
    print(f"wrote {out}")
    miss = [s for s, n in zip(segs, noisy) if n is None]
    if miss:
        print(f"  note: noisy data still missing for segment-size {miss} "
              "(run burgers_ch_segdepth_noisy_q7.toml, fill noisy_path)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
