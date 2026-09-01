
The reader is recommended to first see [Cole-Hopf method](./CH-method.md) for some background on Burgers' equation and the Cole-Hopf transform used in the comparison. Here is a description of the quantum lattice Boltzmann portion of this codebase.


# The Lattice Boltzmann Method

Instead of evolving the velocity field u(x,t) directly, LBM tracks a small set of mesoscopic distribution functions - populations of fictitious particles - and recovers u as a moment (an average) of them. For 1D we use the D1Q3 stencil: at every grid cell there are three populations f₋₁, f₀, f₊₁, riding lattice velocities c = (−1, 0, +1), one sitting still and one moving to each neighbor. The macroscopic quantities are simple sums over the three: density ρ = Σ fᵢ, momentum ρu = −f₋₁ + f₊₁, so u = (−f₋₁ + f₊₁)/ρ.

Each timestep is two operations - collide, then stream. Collision relaxes the populations toward a local equilibrium $f^{\mathrm{eq}}$ using the single-relaxation-time BGK rule $f^* = f - \frac{1}{\tau}(f - f^{\mathrm{eq}})$. The equilibrium is chosen so that its moments reproduce Burgers': $f_0^{\mathrm{eq}} = \rho(1-u^2)$, $f_{+1}^{\mathrm{eq}} = \rho(u^2+u)/2$, $f_{-1}^{\mathrm{eq}} = \rho(u^2-u)/2$, which conserves both density and momentum and, through a Chapman-Enskog expansion, recovers the ν·∂²u/∂x² − u·∂u/∂x dynamics with the nonlinearity carried by the u² term. This form is valid for |u| < 1 (lattice Mach number below one), which is why the LBM family requires the IC amplitude A < 1. 

Streaming then shifts each population by its lattice velocity - f₋₁ one cell left, f₊₁ one cell right, f₀ not at all - with periodic wrap (or bounce-back at a wall for Dirichlet).

The viscosity enters entirely through the relaxation time: ν = (τ − 1/2)·dx²/dt, i.e. τ = ν·dt/dx² + 1/2. A subtlety of LBM is that the streaming step moves exactly one lattice site per step, so the time step is locked to the grid: dt_lbm = dx. The solver honors the caller's dt·n_steps as the end time T_end but runs the lattice at its native cadence, computing n_steps_lbm = round(T_end/dx) genuine steps and mapping them back onto the caller's time grid. Stability requires τ > 1/2; below that the classical scheme is unstable.

This traditional LBM formulation can be executed entirely classically, and our code implements this when triggered by the proper switch (see below).

**References**

- G. R. McNamara and G. Zanetti, "Use of the Boltzmann Equation to Simulate Lattice-Gas Automata," *Phys. Rev. Lett.* **61**, 2332–2335 (1988) — the foundational lattice Boltzmann method.


# From Classical LBM to QALB

A quantum LBM naturally splits along the collide/stream seam. Collision is local - it touches only the three populations at one cell - while streaming is a shift to neighbors. In our implementation the collision is done with quantum code and the streaming is done classically, which lines up neatly with the measure-and-reprepare structure used on the Cole-Hopf side: evolve quantum, measure, reprepare, and let a classical step move the result along. The price of classical streaming is decoherence at each seam, the same tradeoff the CH loop makes.

The hard part is the collision, because it is nonlinear (the $u^2$ in $f^{\mathrm{eq}}$). A conventional quantum LBM that amplitude-encodes the field - putting the population values in the amplitudes of the state - would have to rebuild the nonlinear collision map from the current classical state every single step, since the operator depends on the data it acts on. The variation proposed by Itani et al. sidesteps this. Rather than amplitude encoding, each quasi-particle density value is written into its own small register using a value/Fock (bosonic) encoding: the value is carried by a superposition over the register's Fock (number) basis states |0⟩, |1⟩, |2⟩, …, held in a register of qc qubits. Here qc is the number of qubits in each density register (the "--fock-qubits" argument, qc=3 in this study); it truncates the otherwise-unbounded Fock space to its lowest $2^{\mathrm{qc}}$ number states, and so sets the resolution of the encoding. 

Under this encoding the nonlinearity becomes an operator, and that is what makes the quantum collision cheap to set up. In circuit terms it is one fixed macro-gate: the collision unitary is synthesized once at setup and then applied unchanged to every cell and at every timestep. What differs between invocations is only the input - each per-cell circuit is a short, data-dependent preparation that writes the current population into the register (a displacement rotation on the register), followed by that same collision gate, a fixed rotation into the q̂ readout basis, and measurement. So the expensive operator synthesis is paid once and amortized over every cell and every step of the run, whereas an amplitude/field-encoded scheme would have to recompile the collision each step. Streaming does move population between cells, but those new values re-enter only through the preparation rotations at the start of the next step.

A further payoff is sign handling: negative particle populations are perfectly admissible (the readout estimates a signed expectation value ⟨q̂⟩), so unlike a magnitude-only amplitude readout there is no need for a Hadamard-type test to recover the sign of the solution - the same virtue the Cole-Hopf positivity gives on its side.

Following Itani, we refer to this approach as QALB (the Quantum Algorithm for Lattice Boltzmann). Because our implementation is not the full QALB proposal - we handle streaming classically and stop short of the LCU collision construction, see below - the paper uses the generic term QLBM for it.

**References**

- W. Itani, K. R. Sreenivasan, and S. Succi, "Quantum Algorithm for Lattice Boltzmann (QALB) Simulation of Incompressible Fluids with a Nonlinear Collision Term," *Phys. Fluids* **36**, 017112 (2024); arXiv:2304.05915 — the QALB construction adapted here.

# Discretization

As with Cole-Hopf, the length L=1 is divided into N = 2^q cells, so the argument q sets the spatial grid. But the qubits used by the quantum collision are separate from q. The collision acts on one cell at a time, on a register of 3·qc qubits: three density registers (f₋₁, f₀, f₊₁) this being 1D, each holding qc Fock qubits. This qc is exposed as the "--fock-qubits" argument. It controls the truncation of the bosonic encoding: qc=2 is too coarse for a full run, while qc=3 (9 qubits total for D1Q3) converges and is the value used in this study. This is a very different qubit story from CH - the CH width grows with q as the field lives on the grid register, whereas the QALB collision uses a constant 3·qc qubits regardless of q, because it is applied cell-by-cell. In the paper's circuit-cost figure this shows up as QALB keeping a flat qubit count while CH climbs with q, at the cost of QALB circuits being constant but deep.

Where q enters into the picture in this code is the number of times the fixed circuit must run - this is a function of the space discretization, which is a function of q. For time stepping, the lattice internally locks dt_lbm = dx for one collision/stream step and runs its own native internal step count. This is similar but different to what we see in the Cole-Hopf method where "CFL" is also not the normal interpretation.

We typically drive the runs by "--n-steps" directly, and the solver lines up the QLBM, CH, and classical-reference timelines against each other so the results can be compared frame-for-frame as in a movie.

One important asymmetry from CH: the CH segment can internally evolve k time steps coherently between measurements, so n_steps = S segments · k internal steps, but the QALB loop here runs only one lattice step per segment (k=1), so n_steps = S. After each quantum collision we measure, stream classically, and reload the full measured populations to begin the next step. Allowing k>1 (coherent interim streaming steps) is noted as future work; integrating quantum streaming would deepen circuits that are already at a concerning depth.


# Relation to Prior Work

The approach follows W. Itani to a point. Setting aside any pure statevector approaches (which the code implements but which is not specifically used in the study), the path used in this study is their Appendix B construction: the bosonic-mode Hermitized collision, run on shots. Hermitizing the collision is what lets its exponential be an exactly unitary gate that runs on shots with no post-selection. The one subtlety is that on this encoding the register carries a residual vacuum variance, which is subtracted off for the Hermitized collision to reproduce the classical flow. We need to adapt our approach from their target - that of incompressible flow with a nonlinear collision - to our 1D Burgers on the D1Q3 stencil, rederiving the collision functional Ω against the same equilibrium $f^{\mathrm{eq}}$ our classical LBM uses (the Burgers form whose populations sum back to the density ρ, so mass is conserved).

From there our shots-path implementation differs from Itani's full QALB in two main ways. First, the collision synthesis: Itani constructs the collision as a block encoding via a linear combination of unitaries (LCU), which carries an ancilla and post-selection; we stop short of that and instead hand the fixed Hermitized H′ to Qiskit's UnitaryGate for a dense synthesis. LCU for us is future work. (There are alternatives to the Qiskit UnitaryGate in the code, but these are not exercised here. This drives the point that these solvers have many steps, each subject to configuration and optimization, and this makes the systematic study of these algorithms a challenge requiring a flexible and well-documenting execution harness.)

Second, the streaming: Itani constructs streaming as a coherent log-depth shift on a position register (though they demonstrate only the single-cell collision numerically), whereas we handle only the collision in the quantum circuit and stream classically in a measure-and-reprepare loop with one lattice step per segment (k=1) - collide, read out ⟨q̂⟩, stream classically, reload the full measured populations, repeat. 

Because streaming is classical and the collision is not the LCU block encoding, this is a partial realization of the full QALB proposal - which is why the accompanying paper refers to it with the generic name QLBM.


***************************************************************************************

# Code Arguments

Code arguments and typical [values] used in this study. See [Cole-Hopf method](./CH-method.md) for the common args. These differ: 

- method [qlbm_circuit] - the pure-quantum QALB path (Phase 2)
- fock-qubits (qc): [3] - bosonic Fock qubits per density register; 3 registers → 9 qubits total for D1Q3. qc=2 is too coarse; qc=3 converges


***************************************************************************************
