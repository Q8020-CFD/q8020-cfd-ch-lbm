"""Animated comparison: classical shift-Euler vs F2 TEBD vs F10 Cole-Hopf.

Produces a GIF showing three methods evolving a sine IC through viscous
Burgers, with per-frame error metrics.  Self-contained: runs the
simulations inline, no sweep infrastructure needed.

Usage:
    python animate_tebd_comparison.py [--q 7] [--nu 0.01] [--steps 300]
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from burgers_cole_hopf import run_cole_hopf_simulation
from burgers_tebd import run_tebd_simulation
from burgers_trotter import run_simulation, shift_euler_step


def run_all(
    q: int, nu: float, dt: float, n_steps: int,
    include_circuit: bool = False,
) -> dict:
    """Run all three methods and return trajectory arrays."""
    N = 2**q
    dx = 1.0 / N
    x = np.linspace(0, 1, N, endpoint=False)
    u0 = np.sin(2.0 * np.pi * x)

    results: dict = {"x": x, "u0": u0, "q": q, "nu": nu, "dt": dt}

    # --- Classical shift-Euler (baseline) ---
    t0 = time.time()
    sols_shift = [u0.copy()]
    u_s = u0.copy()
    for _ in range(n_steps):
        u_s, _ = shift_euler_step(u_s, dx, dt, nu)
        sols_shift.append(u_s.copy())
    results["shift"] = sols_shift
    results["t_shift"] = time.time() - t0

    # --- F2 TEBD (classical mirror) ---
    t0 = time.time()
    sols_tebd, mets_tebd = run_tebd_simulation(
        u0, x, nu, dt, n_steps,
    )
    results["tebd"] = sols_tebd
    results["t_tebd"] = time.time() - t0
    results["mets_tebd"] = mets_tebd

    # --- F10 Cole-Hopf (no classical mirror) ---
    t0 = time.time()
    sols_ch, mets_ch = run_cole_hopf_simulation(
        u0, x, nu, dt, n_steps,
    )
    results["cole_hopf"] = sols_ch
    results["t_cole_hopf"] = time.time() - t0
    results["mets_cole_hopf"] = mets_ch

    # --- F2 Phase B.2 tebd_circuit (statevector path; slow, small-q only) ---
    # Phase B.2 currently diverges after ~hundreds of steps due to the
    # Zaletel gate fusion-mismatch floor (see F2 spec §10a).  Run step by
    # step so we can freeze the trajectory on NaN / zero-vector collapse
    # and still visualize the rest of the run.
    if include_circuit:
        from burgers_trotter import tebd_circuit_step
        t0 = time.time()
        sols_tc: list = [u0.copy()]
        mets_tc: list = []
        u_c = u0.copy()
        collapsed_at: int | None = None
        for step in range(n_steps):
            try:
                u_c, met = tebd_circuit_step(
                    u_c, x, dt, nu, g=None, bc="periodic",
                )
                if not np.all(np.isfinite(u_c)):
                    raise ValueError("non-finite state")
            except ValueError as exc:
                collapsed_at = step
                print(
                    f"  tebd_circuit collapsed at step {step}: {exc}"
                )
                u_c = sols_tc[-1].copy()  # freeze
                met = {"collapsed": True}
            sols_tc.append(u_c.copy())
            mets_tc.append(met)
        results["tebd_circuit"] = sols_tc
        results["t_tebd_circuit"] = time.time() - t0
        results["mets_tebd_circuit"] = mets_tc
        results["tebd_circuit_collapsed_at"] = collapsed_at

    return results


def animate(results: dict, out_path: str, fps: int = 20) -> None:
    """Build and save the comparison animation."""
    x = results["x"]
    n_frames = len(results["shift"])
    dt = results["dt"]
    q = results["q"]
    nu = results["nu"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={
        "height_ratios": [3, 1],
    })
    fig.suptitle(
        f"Burgers Evolution: q={q} (N={2**q}), "
        f"$\\nu$={nu:.0e}, dt={dt:.0e}",
        fontsize=13, fontweight="bold",
    )

    ax_u = axes[0]
    ax_err = axes[1]

    # Wave panel
    line_shift, = ax_u.plot(
        x, results["shift"][0], "k-", lw=2, label="Shift-Euler (classical)",
    )
    line_tebd, = ax_u.plot(
        x, results["tebd"][0], "b--", lw=1.5,
        label="F2 TEBD (classical mirror)",
    )
    line_ch, = ax_u.plot(
        x, results["cole_hopf"][0], "r:", lw=2,
        label="F10 Cole-Hopf (no mirror)",
    )
    has_circuit = "tebd_circuit" in results
    line_tc = None
    collapsed_at = results.get("tebd_circuit_collapsed_at")
    if has_circuit:
        circuit_label = "F2 tebd_circuit (shape-normed; amp dies)"
        if collapsed_at is not None:
            circuit_label += f" [collapsed@step {collapsed_at}]"
        # Plot shape-normalized to its own amp so evolution is visible
        # even as amplitude collapses to zero.
        u_tc0 = results["tebd_circuit"][0]
        amp0 = max(np.max(np.abs(u_tc0)), 1e-20)
        line_tc, = ax_u.plot(
            x, u_tc0 / amp0, "g-.", lw=1.5, label=circuit_label,
        )
    ax_u.set_xlim(0, 1)
    ymax = 1.3 * np.max(np.abs(results["shift"][0]))
    ax_u.set_ylim(-ymax, ymax)
    ax_u.set_ylabel("u(x, t)")
    ax_u.legend(loc="upper right", fontsize=9)
    ax_u.grid(True, alpha=0.3)
    time_text = ax_u.text(
        0.02, 0.95, "", transform=ax_u.transAxes,
        fontsize=10, verticalalignment="top",
        fontfamily="monospace",
    )

    # Error panel
    err_tebd_arr = np.zeros(n_frames)
    err_ch_arr = np.zeros(n_frames)
    err_tc_arr = np.zeros(n_frames) if has_circuit else None
    for i in range(n_frames):
        n_sh = min(i, len(results["shift"]) - 1)
        n_tb = min(i, len(results["tebd"]) - 1)
        n_ch = min(i, len(results["cole_hopf"]) - 1)
        err_tebd_arr[i] = np.max(np.abs(
            results["tebd"][n_tb] - results["shift"][n_sh]
        ))
        err_ch_arr[i] = np.max(np.abs(
            results["cole_hopf"][n_ch] - results["shift"][n_sh]
        ))
        if has_circuit:
            n_tc = min(i, len(results["tebd_circuit"]) - 1)
            err_tc_arr[i] = np.max(np.abs(
                results["tebd_circuit"][n_tc] - results["shift"][n_sh]
            ))

    times = np.arange(n_frames) * dt
    line_err_tebd, = ax_err.plot(
        [], [], "b-", lw=1.5, label="F2 vs shift",
    )
    line_err_ch, = ax_err.plot(
        [], [], "r-", lw=1.5, label="F10 vs shift",
    )
    line_err_tc = None
    if has_circuit:
        line_err_tc, = ax_err.plot(
            [], [], "g-", lw=1.5, label="F2-circuit vs shift",
        )
    ax_err.set_xlim(0, (n_frames - 1) * dt)
    err_stack = [np.max(err_tebd_arr), np.max(err_ch_arr)]
    if has_circuit:
        err_stack.append(np.max(err_tc_arr))
    err_max = max(err_stack) * 1.2
    if err_max < 1e-15:
        err_max = 1e-10
    ax_err.set_ylim(0, err_max)
    ax_err.set_xlabel("Time")
    ax_err.set_ylabel("Max |error| vs shift")
    ax_err.legend(loc="upper left", fontsize=9)
    ax_err.grid(True, alpha=0.3)

    # Right-hand log axis: tebd_circuit amplitude (|u_tc|_max) — visualizes
    # the per-step damping factor that kills the field within ~5 steps.
    line_amp_tc = None
    amp_tc_arr = None
    if has_circuit:
        amp_tc_arr = np.array([
            max(np.max(np.abs(u)), 1e-20)
            for u in results["tebd_circuit"]
        ])
        ax_amp = ax_err.twinx()
        line_amp_tc, = ax_amp.plot(
            [], [], "g:", lw=1.2, label="tebd_circuit |u|_max (log)",
        )
        ax_amp.set_yscale("log")
        ax_amp.set_ylim(
            max(amp_tc_arr.min() * 0.5, 1e-20),
            amp_tc_arr.max() * 2,
        )
        ax_amp.set_ylabel("|u_tc|_max (log)", color="green")
        ax_amp.tick_params(axis="y", labelcolor="green")
        ax_amp.legend(loc="upper right", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Frame skip for reasonable GIF size
    total_target_frames = min(n_frames, 200)
    skip = max(1, n_frames // total_target_frames)

    def update(frame_idx):
        i = frame_idx * skip
        if i >= n_frames:
            i = n_frames - 1

        i_sh = min(i, len(results["shift"]) - 1)
        i_tb = min(i, len(results["tebd"]) - 1)
        i_ch = min(i, len(results["cole_hopf"]) - 1)

        line_shift.set_ydata(results["shift"][i_sh])
        line_tebd.set_ydata(results["tebd"][i_tb])
        line_ch.set_ydata(results["cole_hopf"][i_ch])
        tc_amp_str = ""
        if has_circuit:
            i_tc = min(i, len(results["tebd_circuit"]) - 1)
            u_tc_i = results["tebd_circuit"][i_tc]
            amp_i = np.max(np.abs(u_tc_i))
            line_tc.set_ydata(u_tc_i / max(amp_i, 1e-20))
            tc_amp_str = f"\ntebd_circuit |u|_max = {amp_i:.2e}"

        t_now = i * dt
        time_text.set_text(
            f"t = {t_now:.4f}  step {i}/{n_frames-1}{tc_amp_str}"
        )

        line_err_tebd.set_data(times[:i+1], err_tebd_arr[:i+1])
        line_err_ch.set_data(times[:i+1], err_ch_arr[:i+1])
        if has_circuit:
            line_err_tc.set_data(times[:i+1], err_tc_arr[:i+1])
            line_amp_tc.set_data(times[:i+1], amp_tc_arr[:i+1])

        artists = [
            line_shift, line_tebd, line_ch,
            time_text, line_err_tebd, line_err_ch,
        ]
        if has_circuit:
            artists.extend([line_tc, line_err_tc, line_amp_tc])
        return tuple(artists)

    actual_frames = (n_frames + skip - 1) // skip
    anim = manimation.FuncAnimation(
        fig, update, frames=actual_frames,
        interval=1000 // fps, blit=True,
    )

    writer = manimation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"\nSaved animation to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Animate TEBD comparison for Burgers equation",
    )
    parser.add_argument("--q", type=int, default=7)
    parser.add_argument("--nu", type=float, default=1e-2)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--out", type=str,
        default=str(
            Path(__file__).resolve().parent / "tebd_comparison.gif"
        ),
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--include-circuit", action="store_true",
        help="Also run F2 tebd_circuit (Phase B.2). Slow; use small q.",
    )
    args = parser.parse_args()

    print(
        f"Running: q={args.q}, nu={args.nu}, "
        f"dt={args.dt}, steps={args.steps}"
    )
    print()

    results = run_all(
        args.q, args.nu, args.dt, args.steps,
        include_circuit=args.include_circuit,
    )

    print(f"\nTimings:")
    print(f"  Shift-Euler:  {results['t_shift']:.2f}s")
    print(f"  F2 TEBD:      {results['t_tebd']:.2f}s")
    print(f"  F10 Cole-Hopf: {results['t_cole_hopf']:.2f}s")
    if args.include_circuit:
        print(f"  F2 tebd_circuit: {results['t_tebd_circuit']:.2f}s")

    # Final-step errors
    u_shift = results["shift"][-1]
    u_tebd = results["tebd"][-1]
    u_ch = results["cole_hopf"][-1]
    print(f"\nFinal-step max errors vs shift-Euler:")
    print(f"  F2 TEBD:       {np.max(np.abs(u_tebd - u_shift)):.2e}")
    print(f"  F10 Cole-Hopf: {np.max(np.abs(u_ch - u_shift)):.2e}")
    if args.include_circuit:
        u_tc = results["tebd_circuit"][-1]
        print(
            f"  F2 tebd_circuit: {np.max(np.abs(u_tc - u_shift)):.2e}"
        )

    print(f"\nBuilding animation ({len(results['shift'])} frames)...")
    animate(results, args.out, fps=args.fps)


if __name__ == "__main__":
    main()
