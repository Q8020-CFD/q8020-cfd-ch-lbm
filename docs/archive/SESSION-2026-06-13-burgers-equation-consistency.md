# Session log — 2026-06-13: Burgers equation consistency check & paper intro

Full record of a working session on the IEEE QC4PDE 2026 paper (CH vs.
QLBM/QALB) and the `q8020-mps-burgers` codebase. Captured verbatim-in-
substance at the user's request; conversational, not a spec.

---

## 1. Best-Practices loading (housekeeping)

**Q.** Are the user's Best-Practices loaded at startup, and does it happen
automatically every session?

**Findings.**
- The memory chain (`~/.claude/CLAUDE.md` → `@SKILL.md` = `0-kit/SKILL.md`,
  plus `proj/src/CLAUDE.md`) auto-loads, but it only *pointed to*
  `Best-Practices.md` ("Read the guidelines from Best-Practices.md"). The
  pointer loads; the file **contents** did not. So at session start the 29
  rules were not actually in context.
- A `SessionStart` hook would fix the main session but **not** sub-agents
  (hooks don't fire for them). The memory chain *is* inherited by
  sub-agents, so an `@`-import is the robust fix covering every agent.

**Action taken.** Converted both pointers to real `@`-imports so the full
text inlines into every context (main + sub-agents):
- `0-kit/SKILL.md`: `Read the guidelines from Best-Practices.md...` →
  `The guidelines from Best-Practices.md are imported below... @Best-Practices.md`
- `proj/src/CLAUDE.md`: `Read and follow...` →
  `... @0-kit/Best-Practices.md`

Caveat noted: takes effect on the next session/agent spawn.

---

## 2. Inserted the Burgers equation into the paper intro

**Request.** Inject the LaTeX for Burgers' equation at line 21 of the intro
doc (`IEEE-QC4PDE/2-Intro-QC4PDE26-CH-QLBM.tex`).

**Action.** Inserted the 1-D **viscous, non-conservative** Burgers equation
in an `equation` environment with `\label{eq:burgers}`:

```latex
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x}
    = \nu \frac{\partial^2 u}{\partial x^2}
```

with `u(x,t)` the velocity field and `ν` the kinematic viscosity, plus a
one-line lead-in. (User later refined the surrounding prose, incl. a note
that using "a lesser computational model than the full Navier-Stokes" is
itself in-scope, and reworded the lead-in to "the case selected for
comparison is the viscous Burgers equation, a nonlinear PDE that serves as a
model for advection-diffusion".)

---

## 3. The core task — confirm all sources agree on the SAME Burgers equation

**Request.** Check that `itani2024qalb` and `uchida2024burgers` solve the
same Burgers equation, then inspect the `qlbm_circuit`/`qalb` and
`cole_hopf_circuit` code pathways to ensure they all agree. This is the
equation that should go in the paper.

### 3.1 Sources inspected

Code (active tree `q8020/q8020-mps-burgers/src/`):
- `burgers_lbm.py` — shared classical D1Q3 LBM core (equilibrium,
  `tau_from_nu`, collide/stream), consumed by both QLBM and QALB.
- `burgers_qlbm_circuit.py` — hybrid QLBM (amplitude-encoded, classical
  collision mirror each step).
- `burgers_qalb_circuit.py` — pure-quantum QALB (Itani App B/C bosonic
  encoding, Hermitised collision).
- `burgers_cole_hopf_circuit.py` + `burgers_cole_hopf.py` — Cole-Hopf
  pathway (Burgers → heat eq → circuit propagator).
- Spec: `docs/archive/SPEC-qlbm-pure-quantum-qalb.md`.

Papers (read full text — Uchida via ar5iv HTML, Itani via the actual PDF
pulled to disk and `pdftotext`'d; abstracts alone were insufficient and the
arXiv/ar5iv HTML for Itani 404'd / failed conversion).

### 3.2 What each source actually solves

**Uchida (`uchida2024burgers`, arXiv:2412.17206) — EXACT MATCH.**
- Eq. (1): `∂_t u + u ∂_x u = ν ∂_x² u` (non-conservative, coefficient ν).
- Eq. (3): `ψ = exp(−(1/2ν) ∫^x u dy)`.
- Eq. (4): `u = −2ν ∂_x ψ / ψ`.
- Eq. (5): `∂_t ψ = ν ∂_x² ψ` (heat equation).

**`cole_hopf_circuit` pathway — matches Uchida exactly.**
`burgers_cole_hopf.py` header states `u_t + u u_x = nu u_xx`,
`phi = exp(-(1/(2nu))∫u)`, `u = -2nu phi_x/phi`. Same equation, same
transform, same coefficients (1/2ν and −2ν).

**`qlbm_circuit` / `qalb_circuit` pathways — agree with each other and with
Uchida.** Both consume `burgers_lbm.equilibrium` (`burgers_lbm.py:30`), the
density-conserving D1Q3 Burgers equilibrium with rest state `(0,1,0)`:
- `f_0^eq = ρ(1−u²)`, `f_{±1}^eq = ρ(u²±u)/2`  ⇒  `Π_eq = ρu²`  ⇒
  Chapman-Enskog recovers viscous Burgers.
- `qalb_circuit` rederives its collision Ω against this SAME equilibrium
  (`burgers_qalb_circuit.py:65,162-175`: `F_EQ0=(0,1,0)`), validated to
  machine precision. Same `tau_from_nu` (`ν=(τ−½)dx²/dt`).

### 3.3 THE INCONSISTENCY — Itani is Navier-Stokes, not Burgers

`itani2024qalb` (arXiv:2304.05915), read from the PDF directly:
- Title/target: "...Simulation of **Incompressible Fluids**." The word
  **"Burgers" never appears** in the paper.
- Eq. (5)/(10): the standard **Navier-Stokes** equilibrium
  `f_i^eq = w_i ρ(1 + c_i·u/c_s² + Q_i u²/2c_s⁴)`, explicitly "compatible
  with the Navier-Stokes equation of incompressible fluid dynamics."
- Eq. (7), D1Q3: rest weights `(1/6, 2/3, 1/6)` and second moment
  `Π_eq = c_s² + u² = 1/3 + u²` — i.e. it **keeps** the lattice-pressure
  term `c_s²`.
- Numerics (§VIII.A.1): a **single-site, 0-D** repeated-collision demo at
  `u(t=0)=0.1` — not a spatial Burgers PDE solve.

The repo's Burgers equilibrium **drops** exactly that `c_s²` pressure term
(rest `(0,1,0)`, `Π_eq = u²`), which is precisely what converts an NS
lattice scheme into a Burgers (advection-diffusion) one. The spec
(`SPEC-qlbm-pure-quantum-qalb.md`) already documents this ("rederived
against THIS repo's density-conserving equilibrium (rest `(0,1,0)`)").

**Relationship, stated correctly:** Itani is the source of the *algorithmic
machinery* (App B/C bosonic encoding, finite-position embedding Eq. C39,
Hermitisation Eq. 85), which the QALB code adapts to the Burgers
equilibrium. Itani is **not** a paper that "solves the same Burgers
equation."

### 3.4 Verdict / paper guidance

- The equation for the paper is the **non-conservative 1-D viscous Burgers
  equation** `∂_t u + u ∂_x u = ν ∂_x² u` — as inserted.
- ✅ Cite `uchida2024burgers` as solving the identical equation (direct prior
  art for the CH pathway).
- ⚠️ Cite `itani2024qalb` as the **QALB method/algorithm** being adapted
  (original applies it to incompressible NS); do **not** lump it with
  Uchida as "the same Burgers equation." The `refs.bib` note is fine; only
  prose that conflates the two needs rewording.

---

## 4. Discussion — "Is Burgers a stepping stone to Navier-Stokes?"

Fair and standard framing ("poor man's Navier-Stokes"), but precision
matters and the strength is pathway-dependent.

**Where the claim is strong** (Burgers isolates the hard-but-shared parts):
- Nonlinear self-advection `u ∂_x u` ≅ NS convective `(u·∇)u` — the
  same quadratic nonlinearity a *linear* quantum computer must confront.
- Viscous dissipation `ν ∂_xx u` is the NS diffusion term verbatim
  (advection–diffusion / Reynolds-number balance present).
- Underpins "Burgulence" (cf. `alhawwary2018`, already cited).

**Where it is genuinely not NS** (name these gaps honestly):
1. **No pressure / no incompressibility** (`∇p`, `∇·u=0`, the elliptic
   Poisson coupling). For quantum CFD this global non-local coupling is the
   hardest piece to map to a circuit, and Burgers omits it entirely.
2. **1-D / scalar** — no vortex stretching (the engine of 3-D turbulence).
3. **Cole-Hopf linearizability** — Burgers is exactly linearizable to the
   heat equation; NS is not.

**Pathway-dependent nuance (a real differentiator for the comparative
study):**
- **QLBM/QALB → load-bearing stepping stone.** The LBM machinery (BGK
  collision, streaming, Hermitisation of a nonlinear collision) is the same
  algorithm one would use for NS — Itani builds it for incompressible NS and
  reaches D2Q9/D3Q27 by tensor products of D1Q3. The Burgers specialization
  (dropping `c_s²`) is a true subset; "Burgers → NS" is a mechanical
  extension path.
- **Cole-Hopf → methodological stepping stone only.** The linearization
  trick is Burgers-specific and does **not** extend to NS. What transfers is
  the hardware-implementation experience (state prep, readout, propagator
  realization, error behavior), not the handling of the nonlinearity.

Suggested framing: *Burgers is a stepping stone to NS, but the QLBM
pathway's stepping-stone is load-bearing (the algorithm extends) whereas the
Cole-Hopf pathway's is methodological (the engineering lessons extend, the
linearization does not).* This asymmetry speaks directly to the intro's
"pathway to utility in a FTQC era" framing.

---

## 5. Net changes made this session

- `0-kit/SKILL.md`, `proj/src/CLAUDE.md` — `@`-import Best-Practices (full
  text now inlines into every session/sub-agent).
- `IEEE-QC4PDE/2-Intro-QC4PDE26-CH-QLBM.tex` — inserted the viscous Burgers
  equation (`\label{eq:burgers}`) + lead-in (user subsequently refined
  surrounding prose).
- This archive doc.

No code under `src/` was modified; the equation-consistency task was
read-only inspection.
