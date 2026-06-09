# SPEC — Taking QLBM and CH to q = 8–9 (single-CPU Aer)

Self-contained. Goal: make `qlbm_circuit` (QALB) and `cole_hopf_circuit`
(CH) **both viable at q = 8 (N=256) and q = 9 (N=512)** on a single-CPU
Aer statevector/shots run, and produce a fair head-to-head comparison.

**Out of scope (deliberately):** Qiskit 2.3→3 deprecation (FUTURE-WORK
#11) and real-backend / `--backend-type` wiring (#27.3). Peaked-φ shots
readout (#10) is invoked **only if** the chosen ν forces it — see §2; the
target regime here is chosen so it does **not**.

---

## 1. The headline: q=8–9 is nearly free for CH, real work for QLBM

| | CH | QLBM (QALB) |
|---|---|---|
| circuit width | `q + ~1` (qft-diagonal) → **9–10 qubits** | k=1 per-site: **`3·qc` = 9 qubits**; full-lattice k>1: `3·qc+q+2+anc` |
| statevector RAM | ≤ 64 KB | per-site negligible; full-lattice ≈ 1–2 GB (§4) |
| time bottleneck | conditional-Ry depth `O(2^q)`/step (fine at q≤9) | **`2^q` circuits per step** (256–512) — prohibitive as built |
| new work needed at q=8–9 | **none** (periodic + qft-diagonal) | **LCU/Trotter collision synthesis (#27.1)** — required |

CH already fits in space and time at q=8–9; the open q≥7 CH items (#5
QSVT, #6 QROM) become load-bearing only at **q ≥ 10**. QLBM is the one
that needs an algorithmic change to be viable here.

---

## 2. Regime (chosen so #10 is NOT needed)

`τ = 0.5 + ν·N`. QLBM needs `τ > 1`; CH needs φ = exp(−∫u/2ν) not too
peaked (peakedness ≈ `exp(A/2πν)` for a sine IC of amplitude `A`).

**Target: ν = 0.03, periodic, sine, `--ic-amplitude 0.3`** (same as
`burgers_aligned`, scaled to higher q):
- QLBM: `τ = 8.2` (q=8) / `15.9` (q=9) → `Δt/τ ≤ 0.12`. Very stable;
  QLBM stability *improves* with q at fixed ν.
- CH: φ varies only ~5× → comfortably out of the peaked-φ regime, so
  **#10 is not required.**

The point of q=8–9 here is **spatial resolution / scaling**, not lower
ν. If a run drops below **ν ≈ 0.015** (CH φ ≳ 25×), CH shots accuracy
degrades and #10 *does* become load-bearing — treat ν=0.015 as the joint
floor for this comparison; go lower only with #10 in hand. QLBM's own
floor (`ν > 0.5/N`) is far lower (2e-3 at q=8) and not binding here.

---

## 3. CH at q = 8–9 — what's needed

**Space.** φ on `q` qubits + 1 conditional-Ry ancilla = 9–10 qubits.
Non-issue.

**Time.** Use `--propagator qft-diagonal` (periodic) — `O(2^q)` rotations
per step: 256 (q=8) / 512 (q=9). Statevector handles this directly;
shots add no width. The Möbius `build_conditional_ry` term count is
`O(2^q)` and is the only thing that grows — manageable at q≤9, becomes
the q≥10 bottleneck (#6 QROM) but **not needed here**.

**Required work: none.** Avoid `dense-block` (`O(4^q)` = 65 k gates at
q=8). If a Dirichlet comparison is wanted, `qft-diagonal` falls back to
`dense-block`, so #4 (DST-diagonal) would be needed — but the target
regime is **periodic**, so skip it.

CH measure-reprepare(`k`) already works; it sets the `k` reference the
QLBM comparison must meet.

---

## 4. QLBM at q = 8–9 — what's needed

The collision and encoding are shipped and validated (OVERVIEW §5.3); the
blocker is purely **architecture × cost**.

### 4.1 Minimal path (k=1): LCU/Trotter collision synthesis — REQUIRED

The current per-site k=1 loop runs **`2^q × n_steps_lbm` circuits**
(≈2560 at q=8, 10 steps), each applying a **dense `UnitaryGate`** for the
collision (~10⁴ transpiled depth). That is hours–days and is the reason
q=8 is not viable as built.

Fix = **FUTURE-WORK #27.1**: synthesize `e^{−iΔtĤ′}` as an LCU/Trotter of
the App B Pauli decomposition (Itani Eqs. B5–B8) instead of a dense
unitary. Each per-site circuit drops to shallow depth; the run becomes
`2^q × n_steps` *shallow* 9-qubit (`3·qc`, qc=3) circuits ≈ **tens of
minutes** on Aer. **Space is trivial** (9 qubits). This is the minimum to
make q=8–9 QLBM viable, and it gives a **k=1** comparison.

### 4.2 Fuller path (k>1): full-lattice coherent circuit — for the purity comparison

To match CH's `k>1` coherence (the scientifically interesting dial), the
lattice must live in **one** circuit so `k` collide+stream steps run
before a measurement (FUTURE-WORK #27.2): register
`3·qc + q + 2 (+ ~7 LCU ancilla)`.

| q | qc | data qubits | + LCU anc | total | statevector |
|---|----|------------|-----------|-------|-------------|
| 8 | 3 | 19 | ~7 | ~26 | ~1 GB |
| 9 | 3 | 20 | ~7 | ~27 | ~2 GB |

Both fit a single CPU. Needs **quantum log-depth streaming** (SPEC §3.6)
and the measure-reprepare break at segment boundaries (Itani §VIII.A
makes simultaneous coherent collide+stream at the full lattice
impossible, so `k` is finite). `k` is bounded by Itani App A's logistic
truncation divergence — keep `k` small. This is the larger lift; it is
optional for first viability but required for a `k`-scaling comparison.

### 4.3 qc

Use **qc = 3** (qc=2 overshoots — the documented cartoon failure). qc=3
is convergent at this ν and keeps the data register at 9 qubits.

---

## 5. Comparison protocol

Mirror `burgers_aligned`, scaled to q ∈ {8, 9}:
- **Shared**: q, ν=0.03, `--cfl 0.1`, periodic, sine, amp 0.3, common
  aligned time grid; `--n-steps` a multiple of 10 so the QLBM lattice
  cadence lands on stored snapshots.
- **Reference**: `ftcs_reference` (resolved FTCS, already shipped) as the
  shared classical baseline; optionally also run classical `lbm` (Euler
  D1Q3) — it shares QLBM's *scheme*, so it separates the flow-LBM scheme
  gap from quantum/Fock error.
- **Cases**: `cole_hopf_circuit` (qft-diagonal, measure-reprepare) and
  `qlbm_circuit` (qc=3), first **both at k=1** (CH `--segment-size 1`)
  for an apples-to-apples readout comparison; then k>1 once §4.2 lands.

**Metrics (per method, per step):** L²-error vs `ftcs_reference`; u_max
trajectory; **transpiled depth / CX per step** (honest-cost metric, the
real scaling story); shots + any p_success; and `k` (purity).

**Interpretation caveat (state it in results):** CH is exact-via-
transform; QLBM is a *flow*-LBM with an O(Ω²)/step **scheme gap** vs FTCS
(~0.11 final error) that does **not** shrink with qc. So compare QLBM to
`ftcs_reference` *and* to classical `lbm`; the QLBM-vs-`lbm` residual is
the quantum/truncation error, the `lbm`-vs-FTCS residual is the scheme
gap. Comparing only QLBM-vs-FTCS conflates the two.

---

## 6. Acceptance

1. **CH** runs end-to-end at q=8 and q=9 on single-CPU Aer (qft-diagonal,
   periodic), L²-error vs `ftcs_reference` within its shots tolerance at
   ν=0.03 (no peaked-φ degradation), depth/CX reported.
2. **QLBM** (after #27.1) runs at q=8 within a ≲1 hr Aer budget at qc=3,
   amplitude tracks `ftcs_reference` to within the scheme gap (no qc=2
   blow-up), depth/CX reported and **shallow** (LCU, not dense unitary).
3. A single `method_compare` artifact overlays CH, QLBM, and the FTCS
   reference on the common grid at q=8 (and q=9), with the gate-cost
   table.
4. (If §4.2 done) a `k`-sweep showing both methods' error/purity vs `k`.

## 7. Work summary

| Item | Method | Needed for | Source |
|---|---|---|---|
| LCU/Trotter collision synthesis | QLBM | k=1 viability (the keystone) | #27.1 |
| Full-lattice circuit + quantum streaming | QLBM | k>1 purity comparison | #27.2 |
| (none) | CH | already viable at q≤9 | — |
| DST-diagonal | CH | *only* if Dirichlet wanted | #4 |
| Peaked-φ readout | CH | *only* if ν < ~0.015 | #10 |

Critical path for a q=8–9 comparison: **#27.1**, then everything else is
optional refinement. CH needs nothing.
