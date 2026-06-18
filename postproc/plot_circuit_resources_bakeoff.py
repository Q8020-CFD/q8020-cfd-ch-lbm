"""Circuit-resource figure for a CH-vs-QLBM bakeoff dataset.

Reads a dataset JSON in the case1_bakeoff shape (a 'runs' list, each with a
'path' to a sweep case dir) and re-harvests the circuit cost of every run from
its q8020_analysis_*.json.  Because it reads the live analysis files via the
dataset's paths, regenerating any case and re-pointing its path in the JSON
then re-running this script regenerates the figure from the new numbers.

2x2 layout: a qubits panel (top-left), a combined per-frame panel (top-right)
and a combined cumulative panel (bottom-right), with the legend/notes in the
bottom-left.  The two combined panels each overlay three metrics for both
methods on one log-y axis -- colour = method, linestyle+marker = metric:
  per-frame  (per measure-reprepare segment): # circuits, deepest circuit
             depth, total CX in the frame (= # circuits x CX/circuit)
  cumulative (summed over the run's frames): # circuits, depth, CX

The qubits panel is a stacked bar: for CH the register width is split into
data (the 2^q grid), state-prep ancillas (MPS bond qubits) and evolution
ancillas (heat block-encoding / post-selection), reconstructed by rebuilding
the actual segment circuit; QLBM has no prep register and stays a flat bar.

Also writes a sidecar <out>.json of the harvested numbers so the figure's
values are inspectable / citable without re-running.

Usage:
    plot_circuit_resources_bakeoff.py <dataset.json> [--out fig.png]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Match the movie's per-method colours (plot_method_compare.METHOD_STYLE).
METHOD_STYLE = {
    "cole_hopf_circuit": {"color": "#17becf", "label": "Cole-Hopf"},
    "qlbm_circuit": {"color": "#d62728", "label": "QLBM"},
}

# Stacked-qubit segment colours (data / state-prep / evolution).
QUBIT_PARTS = [
    ("data", "data (2^q grid)", "#1f77b4"),
    ("prep", "state-prep ancillas", "#ff7f0e"),
    ("evo", "evolution ancillas", "#2ca02c"),
]


def _ch_qubit_split(q: int, segment_size: int, bond_dim: int | None,
                    propagator: str, total: int) -> dict:
    """Decompose a Cole-Hopf run's register width into data / state-prep /
    evolution qubits by rebuilding its actual segment circuit (the single
    source of truth for n_bond / n_heat_anc).  Falls back to the analytic
    split if the solver source can't be imported."""
    try:
        import numpy as np
        src = Path(__file__).resolve().parent.parent / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from burgers_cole_hopf_circuit import build_segment_circuit
        n = 2 ** q
        x = np.arange(n) / n
        phi = np.exp(0.3 * np.sin(2 * np.pi * x))      # generic positive state
        phi = phi / np.linalg.norm(phi)
        _, total_q, n_bond, n_heat_anc = build_segment_circuit(
            phi, q, nu=0.03, dt=0.1 / n, segment_size=segment_size,
            L_box=1.0, bc="periodic", propagator=propagator,
            bond_dim=bond_dim, use_mps_prep=True,
        )
        return {"data": q, "prep": n_bond, "evo": n_heat_anc}
    except Exception as e:           # pragma: no cover - diagnostic fallback
        print(f"  CH qubit-split rebuild failed ({e}); inferring from total",
              file=sys.stderr)
        n_heat_anc = min(q, segment_size)
        return {"data": q, "prep": max(total - q - n_heat_anc, 0),
                "evo": n_heat_anc}


def _find_analysis(o: dict | list) -> dict | None:
    """Locate the harvested analysis fragment (the one carrying the
    per-segment circuit metrics) anywhere in a results JSON."""
    if isinstance(o, dict):
        if "per_step_metrics" in o:
            return o
        for v in o.values():
            r = _find_analysis(v)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _find_analysis(v)
            if r is not None:
                return r
    return None


def _depth(m: dict) -> int:
    """Post-transpile depth of the frame's circuit."""
    after = (m.get("transpile") or {}).get("after") or {}
    return int(m.get("circuit_depth", after.get("depth")) or 0)


def _cx(m: dict) -> int:
    """Post-transpile CX count of one circuit in the frame."""
    after = (m.get("transpile") or {}).get("after") or {}
    g = m.get("gate_counts", after.get("gate_counts")) or {}
    return int(g.get("cx", 0) or 0)


def _harvest(case_dir: Path) -> dict | None:
    """Pull per-frame and cumulative circuit metrics for one run.  Returns
    None if the case has no circuit analysis (e.g. the FTCS reference)."""
    hits = sorted(case_dir.glob("q8020_analysis_*.json"))
    if not hits:
        return None
    an = _find_analysis(json.loads(hits[0].read_text()))
    if an is None or an.get("n_qubits") is None:
        return None
    psm = sorted(
        (m for m in (an.get("per_step_metrics") or [])
         if m.get("step") is not None),
        key=lambda m: m["step"],
    )
    if not psm:
        return None
    # Per frame: # circuits, deepest depth, total CX (# circuits x CX/circuit).
    frame_ncirc = [int(m.get("n_circuits", 0) or 0) for m in psm]
    frame_depth = [_depth(m) for m in psm]
    frame_cx_each = [_cx(m) for m in psm]
    frame_cx_total = [n * c for n, c in zip(frame_ncirc, frame_cx_each)]
    return {
        "n_qubits": int(an["n_qubits"]),
        "n_frames": len(psm),
        # Per-frame representative values (max over frames; constant for these
        # runs, but max keeps the figure honest if a future run varies).
        "circ_per_frame": max(frame_ncirc, default=0),
        "deepest_depth": max(frame_depth, default=0),
        "cx_per_frame": max(frame_cx_total, default=0),
        "cx_per_circuit": max(frame_cx_each, default=0),
        # Cumulative over the run.
        "cum_circ": sum(frame_ncirc),
        "cum_depth": sum(frame_depth),
        "cum_cx": sum(frame_cx_total),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="path to the bakeoff dataset JSON")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    ds_path = Path(args.dataset).resolve()
    ds = json.loads(ds_path.read_text())
    base = ds_path.parent

    # series[method] = list of harvested points (one per q), sorted by q.
    series: dict[str, list[dict]] = {}
    for run in ds.get("runs", []):
        method = run.get("method", "")
        if method not in METHOD_STYLE:
            continue  # skip FTCS reference / classical
        case_dir = (base / run["path"]).resolve()
        h = _harvest(case_dir)
        if h is None:
            print(f"  {run.get('label')}: no circuit metrics, skipping",
                  file=sys.stderr)
            continue
        h["q"] = int(run["q"])
        h["label"] = run.get("label", "")
        # CH register split (data / prep / evolution).  segment_size from the
        # dataset global; bond_dim / propagator from the run entry.
        if method == "cole_hopf_circuit":
            g = ds.get("global", {})
            h["qubit_split"] = _ch_qubit_split(
                q=h["q"],
                segment_size=int(g.get("segment_size", 5)),
                bond_dim=run.get("bond_dim"),
                propagator=run.get("propagator",
                                   g.get("propagator", "qft-diagonal")),
                total=h["n_qubits"],
            )
        series.setdefault(method, []).append(h)
    for method in series:
        series[method].sort(key=lambda p: p["q"])
    if not series:
        print("No circuit-bearing runs found in dataset.", file=sys.stderr)
        sys.exit(1)

    # Each combined panel overlays 3 metrics x 2 methods on a log y-axis:
    # colour = method, (linestyle, marker) = metric.  (key, label, ls, marker).
    per_frame = [
        ("circ_per_frame", "circuits / frame", ":", "o"),
        ("deepest_depth", "deepest depth", "-", "s"),
        ("cx_per_frame", "CX / frame", "--", "^"),
    ]
    cumulative = [
        ("cum_circ", "circuits", ":", "o"),
        ("cum_depth", "depth", "-", "s"),
        ("cum_cx", "CX", "--", "^"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    all_q = sorted({p["q"] for pts in series.values() for p in pts})

    def _draw_combined(ax, metrics, title):
        """Overlay several metrics for both methods on one log-y panel."""
        for method, pts in series.items():
            mcolor = METHOD_STYLE[method]["color"]
            xs = [p["q"] for p in pts]
            for key, _lbl, ls, mk in metrics:
                ys = [p[key] for p in pts]
                ax.plot(xs, ys, ls=ls, marker=mk, lw=1.8, ms=6,
                        color=mcolor)
        ax.set_xlabel("q")
        ax.set_xticks(all_q)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(title, fontsize=11)

    def _draw_qubits_stacked(ax):
        """Grouped bars per q: CH stacked into data / prep / evolution,
        QLBM a single flat bar (no prep register)."""
        bw = 0.36
        methods = list(series)
        for mi, method in enumerate(methods):
            pts = {p["q"]: p for p in series[method]}
            off = (mi - (len(methods) - 1) / 2) * (bw + 0.04)
            for q in all_q:
                p = pts.get(q)
                if p is None:
                    continue
                xpos = all_q.index(q) + off
                split = p.get("qubit_split")
                if split:
                    bottom = 0
                    for kkey, _lbl, kcol in QUBIT_PARTS:
                        hgt = split.get(kkey, 0)
                        if hgt <= 0:
                            continue
                        ax.bar(xpos, hgt, bw, bottom=bottom, color=kcol,
                               edgecolor="white", linewidth=0.5)
                        ax.text(xpos, bottom + hgt / 2, str(hgt),
                                ha="center", va="center", fontsize=6,
                                color="white", fontweight="bold")
                        bottom += hgt
                    ax.text(xpos, bottom, str(p["n_qubits"]), ha="center",
                            va="bottom", fontsize=7,
                            color=METHOD_STYLE[method]["color"])
                else:
                    ax.bar(xpos, p["n_qubits"], bw,
                           color=METHOD_STYLE[method]["color"])
                    ax.text(xpos, p["n_qubits"], str(p["n_qubits"]),
                            ha="center", va="bottom", fontsize=7,
                            color=METHOD_STYLE[method]["color"])
        from matplotlib.ticker import MaxNLocator
        ax.set_xticks(range(len(all_q)))
        ax.set_xticklabels([str(q) for q in all_q])
        ax.set_xlabel("q")
        ax.set_ylabel("qubits")
        ax.margins(y=0.18)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, axis="y", alpha=0.3)
        # CH = stacked multicolour, QLBM = solid (per the legend); no per-bar
        # method label needed.
        ax.set_title("qubits", fontsize=10)

    # 2x2: qubits (stacked) | per-frame combined ; legend/notes | cumulative.
    _draw_qubits_stacked(axes[0][0])
    _draw_combined(axes[0][1], per_frame, "per frame")
    _draw_combined(axes[1][1], cumulative, "cumulative over run")
    axes[1][0].axis("off")

    # Legend: method = colour; metric = linestyle+marker; CH qubit split.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    method_h = [Line2D([], [], color=METHOD_STYLE[m]["color"], lw=2.5,
                       label=METHOD_STYLE[m]["label"]) for m in series]
    metric_h = [Line2D([], [], color="0.35", ls=ls, marker=mk, lw=1.8,
                       label=lbl) for key, lbl, ls, mk in
                [("cum_circ", "circuits", ":", "o"),
                 ("cum_depth", "depth", "-", "s"),
                 ("cum_cx", "CX", "--", "^")]]
    split_h = [Patch(facecolor=c, label=lbl) for _, lbl, c in QUBIT_PARTS]
    blank = Line2D([], [], color="none", label="")
    handles = (method_h + [blank] + metric_h + [blank] + split_h)
    axes[1][0].legend(
        handles=handles, loc="upper center", fontsize=9, frameon=False,
        labelspacing=0.7,
        title="method (colour)   metric (line)   CH qubits (fill)",
        title_fontsize=10,
    )
    axes[1][0].text(
        0.5, 0.16,
        "CX / frame = (circuits/frame) x (CX/circuit)\n"
        "QLBM: fixed 9q circuit, count scales with grid\n"
        "CH: 1 circuit/frame, circuit grows with grid\n"
        "CH qubits = data (2^q) + prep + evolution ancillas",
        transform=axes[1][0].transAxes, ha="center", va="top",
        fontsize=8, color="0.35",
    )

    fig.suptitle(
        f"{ds.get('name', ds_path.stem)} — circuit resources "
        f"(CH grid-driven, QLBM grid-independent per-circuit)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(args.out) if args.out else ds_path.with_name(
        ds_path.stem + "_circuit_resources.png")
    fig.savefig(str(out), dpi=150)
    plt.close(fig)

    # Sidecar JSON of the harvested numbers.
    sidecar = out.with_suffix(".json")
    dump = {
        "source_dataset": ds_path.name,
        "series": {
            METHOD_STYLE[m]["label"]: series[m] for m in series
        },
    }
    sidecar.write_text(json.dumps(dump, indent=2))
    print(f"wrote {out}")
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
