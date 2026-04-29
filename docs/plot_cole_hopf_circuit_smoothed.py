"""Cole-Hopf circuit evolution with spatially smoothed quantum curve.

Companion to plot_cole_hopf_circuit_evolution.py.  Re-reads the same
sweep output but adds a Savitzky-Golay-smoothed quantum curve to
suppress per-frame shot noise while preserving waveform structure.

Invocation:
  python plot_cole_hopf_circuit_smoothed.py --sweep-dir ~/q8020/<id>
  python plot_cole_hopf_circuit_smoothed.py <group_postproc.json>
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

_pkg_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_pkg_root))
sys.path.insert(0, str(_pkg_root / "src"))

from q8020_cfd_metautil.harvest import harvest_metadata
from q8020_cfd_metautil.metakeys import _walk_case_dirs


# ── helpers (shared with the original script) ───────────────────────


def _find_solver_entry(entries: list[dict], key: str) -> dict:
    for e in entries:
        if key in e:
            return e
    return {}


def _extract_case_params(meta: dict) -> dict | None:
    cases = meta.get("case", [])
    if not cases:
        return None
    for c in cases:
        if c.get("_source") == "solver":
            return c
    for c in cases:
        params = c.get("params", {})
        if params:
            return {
                k.lstrip("-"): v
                for k, v in params.items()
                if not k.startswith("_")
            }
    return cases[0]


def _load_group_params(sweep_dir: Path) -> dict:
    """Read --params from the _group_postproc_*.json in sweep_dir."""
    import glob as _glob
    jsons = _glob.glob(str(sweep_dir / "_group_postproc_*.json"))
    if jsons:
        with open(jsons[0], encoding="utf-8") as f:
            gp = json.load(f)
        return gp.get("params", {})
    return {}


def _load_cases_from_sweep(sweep_dir: Path) -> list[dict]:
    case_dirs, no_meta = _walk_case_dirs(sweep_dir)
    out = []
    for cd in case_dirs + no_meta:
        cd = Path(cd)
        if not cd.is_dir():
            continue
        meta, _, _ = harvest_metadata(cd, read_only=True)
        if meta.get("case"):
            out.append(meta)
    return out


def _smooth(u: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smooth with wrap-around for periodic BC."""
    N = len(u)
    if window >= N:
        window = N - 1 if N % 2 == 0 else N
    if window < polyorder + 2:
        return u.copy()
    # Pad symmetrically to handle periodicity
    pad = window
    u_ext = np.concatenate([u[-pad:], u, u[:pad]])
    s = savgol_filter(u_ext, window, polyorder)
    return s[pad : pad + N]


# ── main ────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("postproc_json", nargs="?", default=None)
    p.add_argument("--sweep-dir", type=Path, default=None)
    p.add_argument("--outfile", default=None)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument(
        "--window", type=int, default=7,
        help="Savitzky-Golay window length (odd integer, default 7)",
    )
    p.add_argument(
        "--polyorder", type=int, default=3,
        help="Savitzky-Golay polynomial order (default 3)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    window = args.window if args.window % 2 == 1 else args.window + 1

    # --- locate data ---
    if args.postproc_json and Path(args.postproc_json).is_file():
        with open(args.postproc_json, encoding="utf-8") as f:
            pp = json.load(f)
        run_dir = Path(pp["run_dir"])
        all_cases = _load_cases_from_sweep(run_dir)
        group_params = _load_group_params(run_dir)
        outfile = args.outfile or str(
            run_dir / "cole_hopf_circuit_smoothed.gif",
        )
    elif args.sweep_dir:
        sd = args.sweep_dir.expanduser().resolve()
        all_cases = _load_cases_from_sweep(sd)
        group_params = _load_group_params(sd)
        outfile = args.outfile or str(
            sd / "cole_hopf_circuit_smoothed.gif",
        )
    else:
        print(
            "Provide either a postproc JSON or --sweep-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- filter to cole_hopf_circuit cases ---
    candidates = []
    for c in all_cases:
        cm = _extract_case_params(c)
        if cm and cm.get("method") == "cole_hopf_circuit":
            candidates.append(c)
    if not candidates:
        print("No cole_hopf_circuit cases found.", file=sys.stderr)
        sys.exit(1)

    def _anchor_key(c: dict) -> tuple[int, int]:
        cm = _extract_case_params(c) or {}
        bd = cm.get("bond_dim") or 0
        if bd is None:
            bd = 999
        art = _find_solver_entry(
            c.get("artifacts", []), "solution_steps",
        )
        n = len(art.get("solution_steps", {}))
        return (int(bd), n)

    anchor = max(candidates, key=_anchor_key)
    anchor_meta = _extract_case_params(anchor) or {}
    art = _find_solver_entry(
        anchor.get("artifacts", []), "solution_steps",
    )
    sol_steps = art.get("solution_steps", {})
    x = np.array(art.get("grid", []))
    if not sol_steps or x.size == 0:
        print("No solution_steps/grid.", file=sys.stderr)
        sys.exit(1)

    res = _find_solver_entry(
        anchor.get("results", []), "u_final_classical",
    )

    step_keys = sorted(sol_steps.keys(), key=int)
    frames_q = [np.array(sol_steps[k]) for k in step_keys]

    # --- classical FTCS reference ---
    from burgers_classical import solve_burgers, source_term_sine
    from burgers_cole_hopf import run_cole_hopf_simulation

    dt = float(anchor_meta.get("dt", 0.0))
    nu = float(anchor_meta.get("nu", 1e-2))
    bc = anchor_meta.get("bc", "periodic")
    source = anchor_meta.get("source", "sine")
    n_steps_total = int(
        anchor_meta.get("n_steps", len(step_keys) - 1),
    )
    u0 = frames_q[0].copy()

    source_fn = source_term_sine if source == "sine" else None
    forced = source_fn is not None
    sols_classical = solve_burgers(
        u0, x, nu, dt, n_steps_total,
        source_fn=source_fn, bc=bc,
    )
    frames_cl = [sols_classical[int(k)] for k in step_keys]

    if not forced:
        print("  Computing Cole-Hopf MPS reference ...", file=sys.stderr)
        sols_mps, _ = run_cole_hopf_simulation(
            u0, x, nu, dt, n_steps_total, bc=bc,
        )
        frames_mps = [sols_mps[int(k)] for k in step_keys]
    else:
        frames_mps = None

    # --- smoothed quantum frames ---
    frames_smooth = [
        _smooth(f, window, args.polyorder) for f in frames_q
    ]

    # --- truncate at divergence ---
    amp0 = max(np.max(np.abs(u0)), 1.0)
    amp_limit = 10.0 * amp0
    valid_end = len(step_keys)
    for i, (fq, fc) in enumerate(zip(frames_q, frames_cl)):
        ok = (
            np.all(np.isfinite(fq))
            and np.all(np.isfinite(fc))
            and np.max(np.abs(fq)) < amp_limit
            and np.max(np.abs(fc)) < amp_limit
        )
        if not ok:
            valid_end = i
            break
    if valid_end == 0:
        print("No valid frames.", file=sys.stderr)
        sys.exit(1)
    step_keys = step_keys[:valid_end]
    frames_q = frames_q[:valid_end]
    frames_smooth = frames_smooth[:valid_end]
    frames_cl = frames_cl[:valid_end]
    if frames_mps is not None:
        frames_mps = frames_mps[:valid_end]

    # --- animation ---
    def _fmax(fs):
        vals = [
            np.abs(f[np.isfinite(f)]).max()
            for f in fs
            if np.any(np.isfinite(f))
        ]
        return max(vals) if vals else 1.0

    all_frames = [frames_q, frames_cl]
    if frames_mps is not None:
        all_frames.append(frames_mps)
    ymax = max(_fmax(fs) for fs in all_frames) * 1.15

    q_val = anchor_meta.get("q", "?")
    propagator = (
        anchor_meta.get("propagator")
        or group_params.get("--propagator", "qft-diagonal")
    )
    shots = anchor_meta.get("shots", 0)
    shots_label = (
        f"{int(shots) // 1000}k shots" if shots else "statevector"
    )

    fig, (ax_u, ax_err) = plt.subplots(
        2, 1, figsize=(9, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(
        f"Cole-Hopf Circuit (smoothed): q={q_val} (N={2**int(q_val)})"
        f", $\\nu$={nu:.0e}, {propagator}, {shots_label}",
        fontsize=12, fontweight="bold",
    )

    ax_u.set_xlim(x[0], x[-1] + x[1] - x[0])
    ax_u.set_ylim(-ymax, ymax)
    ax_u.set_ylabel("u(x, t)")
    ax_u.grid(alpha=0.2)

    ax_u.plot(x, u0, "k--", alpha=0.2, lw=1, label="IC")
    if frames_mps is not None:
        mps_line, = ax_u.plot(
            [], [], "-", color="#2ca02c", lw=2.2,
            label="Exact classical (tensor net)",
        )
    else:
        mps_line = None
    cl_line, = ax_u.plot(
        [], [], "b-", lw=1.2, alpha=0.5,
        label="FTCS (num. diffusive)",
    )
    raw_line, = ax_u.plot(
        [], [], "-", color="#d62728", lw=0.6, alpha=0.35,
        label="CH circuit (raw)",
    )
    smooth_line, = ax_u.plot(
        [], [], "-", color="#d62728", lw=2.0,
        label=f"CH circuit (SG w={window}, p={args.polyorder})",
    )
    ax_u.legend(loc="lower left", fontsize=8)
    time_text = ax_u.text(
        0.02, 0.95, "", transform=ax_u.transAxes,
        fontsize=10, va="top", fontfamily="monospace",
    )

    # --- error panel ---
    ref_label = "vs MPS" if frames_mps is not None else "vs FTCS"
    ref_frames = frames_mps if frames_mps is not None else frames_cl
    err_raw = np.zeros(len(step_keys))
    err_smooth = np.zeros(len(step_keys))
    for i in range(len(step_keys)):
        nrm = np.linalg.norm(ref_frames[i])
        if nrm > 1e-15:
            err_raw[i] = (
                np.linalg.norm(frames_q[i] - ref_frames[i]) / nrm
            )
            err_smooth[i] = (
                np.linalg.norm(frames_smooth[i] - ref_frames[i]) / nrm
            )
    times = np.array([int(k) * dt for k in step_keys])

    line_err_raw, = ax_err.plot(
        [], [], "-", color="#d62728", lw=0.8, alpha=0.5,
        label=f"raw {ref_label}",
    )
    line_err_smooth, = ax_err.plot(
        [], [], "-", color="#d62728", lw=1.5,
        label=f"smoothed {ref_label}",
    )
    ax_err.set_xlim(times[0], times[-1])
    all_err = np.concatenate([err_raw[err_raw > 0],
                              err_smooth[err_smooth > 0]])
    if all_err.size == 0:
        all_err = np.array([1e-6])
    ax_err.set_yscale("log")
    ax_err.set_ylim(all_err.min() * 0.5, all_err.max() * 2.0)
    ax_err.set_xlabel("Time")
    ax_err.set_ylabel("Rel. L2 error")
    ax_err.legend(loc="upper left", fontsize=8)
    ax_err.grid(alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    du0dx = np.gradient(u0, x[1] - x[0])
    max_grad = np.max(np.abs(du0dx))
    t_shock = 1.0 / max_grad if max_grad > 0 else 1.0

    def update(frame_idx):
        step = int(step_keys[frame_idx])
        t = step * dt
        t_pct = 100.0 * t / t_shock
        cl_line.set_data(x, frames_cl[frame_idx])
        raw_line.set_data(x, frames_q[frame_idx])
        smooth_line.set_data(x, frames_smooth[frame_idx])
        artists = [
            cl_line, raw_line, smooth_line, time_text,
            line_err_raw, line_err_smooth,
        ]
        if mps_line is not None:
            mps_line.set_data(x, frames_mps[frame_idx])
            artists.append(mps_line)
        time_text.set_text(
            f"step {step}/{n_steps_total}  "
            f"t={t:.4f} ({t_pct:.0f}% T_shock)  "
            f"err_raw={err_raw[frame_idx]:.2e}  "
            f"err_smooth={err_smooth[frame_idx]:.2e}"
        )
        sl = slice(None, frame_idx + 1)
        line_err_raw.set_data(times[sl], err_raw[sl])
        line_err_smooth.set_data(times[sl], err_smooth[sl])
        return tuple(artists)

    anim = manimation.FuncAnimation(
        fig, update, frames=len(step_keys),
        interval=1000 / args.fps, blit=False,
    )
    writer = manimation.PillowWriter(fps=args.fps)
    anim.save(outfile, writer=writer, dpi=120)
    plt.close(fig)
    print(
        f"Saved smoothed animation to {outfile}  "
        f"({len(step_keys)} frames @ {args.fps} fps)",
    )


if __name__ == "__main__":
    main()
