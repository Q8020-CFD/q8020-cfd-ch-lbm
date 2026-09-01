

A comparison of the Cole-Hopf transformation versus quantum lattice Boltzmann method was performed and implemented on a simulator and on real quantum computers. Here is a description of the Cole-Hopf portion of that code.

# Burgers' Equation 

The Burgers' equation is ∂u/∂t  =  ν·∂²u/∂x²  −  u·∂u/∂x, the rate of change of velocity at a fixed point in space is impacted by the advection of a wave and inhibited by the viscosity. This is a simplification of the general Navier-Stokes equations for flow - the pressure term and incompressibility constraint in Nav-Stokes are dropped. We will consider just 1D, but Burgers' can be applied to 2D / 3D, albeit with the noted,  important, and computationally challenging omissions. However, Burgers' is not a toy and is useful in some real world cases - the 3D system might contain a nonlinear steepening wave which competes with a resisting diffusion term in some dominant direction - here Burgers' applies naturally. Some examples include weak-shock acoustics and modeling automobile traffic flow. Some inappropriate uses would be when the omitted pressure term drives the flow (e.g. a channel with sudden size differences), where the amplitude of the wave is large (strong nonlinearity, e.g. a detonation), or where there are memory effects (e.g. as with certain, perhaps human but illogical, traffic behaviors). 

We intend to solve the 1D viscous Burgers' equation on a quantum computer. In our case, the initial condition is a sine wave representing the velocity along a line of length L. Every location on the line is u(x,t), some velocity at location x at time t, and for t=0 that is the initial sine wave with amplitude A. The boundary condition is periodic, so the line is really a loop. A quarter of the way along the length, at u(L/4,0) = A, and at u(3L/4,0) = -A. For our purposes, we'll set L=1 and discretize a 1D grid within it. The sine wave case will show negative value handling by the algorithm, which a Gaussian signal would not. The periodic BC will align well with the sine IC, there will be no edges to consider, and we'll not use the source / sink term in the Burgers' equation - nothing enters or leaves the system. Other boundary conditions will require handling other than what is described here, which are also implemented by the code, but not exercised in this study. 

This is the viscous Burgers', so kinematic viscosity ν (nu) can be small but not zero - at zero there will be discontinuity and the numerics break down. The sine wave peak will propel itself along the line with a rate (velocity) which is equal to the height of the wave A - the trough will similarly move but in the opposite direction, toward the encounter with the peak at the zero crossing at L/2. A larger A means the time to encounter is less, the potential shock sharper. This advection movement will be inhibited by the viscosity ν - the amplitude of the advection steepens, the viscosity smooths. The ratio A/ν - the Reynolds number, Re = A·L/ν = A/ν for L=1, will determine the outcome - a large Re means a thin sharp shock (the thickness being ~ ν/A), and a small Re=A/ν results in the wave smearing and decaying before a shock front forms. All values are in working units with domain length L=1; the dynamics are governed by the single dimensionless group Re = A/ν, with A=0.3 indicating the peak velocity and ν the viscosity in those units.

Given a velocity at a point on the line in time, u(x,t), the next value at the same point u(x,t+1) =  u(x,t) +  dt·(ν·∂²u/∂x² − u·∂u/∂x). Here ν·∂²u/∂x² is due to the viscosity and it's linear (the u term appears once), and u·∂u/∂x is the advection, which is non-linear (u appears twice in u·∂u/∂x). This nonlinear part is the subject of much research attention, as quantum computers are inherently linear. Some way to handle the nonlinearity is needed. One method is to linearize the nonlinear portion, another is to adopt a transform which includes both parts of the Burgers' equation, changing the problem entirely. 

# The Cole-Hopf Transform 

The Cole-Hopf transform comes from Cole & Hopf separately in the early 1950s. The transformation linearizes the Burgers' equation, including the nonlinear term, perfectly into the linear heat equation. There are many other known linearization techniques, but for this particular case, Cole-Hopf does a good job of it through a change of variables. The new variable phi, φ(x,t) = exp( −(1/2ν) ∫u dx ). Here ∫u dx is the running area under the velocity curve at the fixed instant, and that value is scaled by -(1/2ν). The exp() means the value stays positive everywhere - the Cole-Hopf transform bakes-in the sign handling for the sine wave. This also cleanly enables, without extra phase estimation, a measure-and-reprepare scheme for reducing circuit depth, which we explain below.

The Cole-Hopf transformation is handled classically. After evolving the new system φ in time, we'll have to perform a classical reverse Cole-Hopf transform to return us to the original velocity variable.

# Discretization

Now we discretize. In 1D space, the length L is divided into N = 2^q cells, with its obvious space benefits over classical processing. As q increases, the granularity of the spatial grid increases, and so does the number of probabilistic attempts (shots) needed to resolve the field within a desired error tolerance, so there will be NISQ-era practical limits to how high we can set q. Unlike an explicit classical scheme, however, a finer grid does not force a finer time step here: as we show below, our φ-solver applies the heat evolution exactly, so it carries no stability bound tying dt to dx. The step size is a free choice; the reasons to control the number of steps are circuit depth and the per-step damping cost, both explained below, not stability or grid resolution.

In classical CFD the Courant number (aka CFL number) is C = u·dt/dx, i.e. a velocity times a time step over a cell width. It answers "how many cells does a signal travel in one step," and for an explicit advection scheme it's a stability criterion (roughly C ≤ 1 or the scheme blows up). CFL is about a signal being transported at some speed u.

The Cole-Hopf transform is going to convert the Burgers' equation, and in doing so is going to drop the velocity term. This is going to change the meaning of "CFL" for the solver. Perhaps regrettably, the name of the argument to the code remains "--cfl". We keep the name but treat it purely as the normalized step size dt/dx, so dt = "cfl" · dx. For example, at cfl = 0.1 and q = 8, dx = 1/2^8 = 0.00390625, so the step size is dt = 0.1 · 0.00390625 = 0.000390625.

The diffusive counterpart to the Courant number is the von Neumann number (also called the diffusion number), d = ν·dt/dx², which for our step dt = cfl·dx and dx = 2^(−q) becomes d = ν·cfl·2^q. It measures how strongly heat diffuses in one step relative to a cell. An explicit scheme is stable only for d ≤ 1/2. Our propagator instead evolves each Fourier mode exactly - each is multiplied by exp(−ν k² dt), which has magnitude ≤ 1 for any dt, so it is unconditionally stable and carries no d-bound. In our runs we typically specify the n_steps argument. 

Given the space discretization, we can load the initial Cole-Hopf transformed φ field into the cells by amplitude encoding. We could also compress the information before loading with a more efficient but potentially lossy Matrix Product State (MPS) encoder, which is tunable with the bond dimension parameter. Normally an amplitude encoding circuit would be O(2^q) gates, whereas MPS prep costs only O(q·χ²) gates, where χ is the bond dimension, akin to the amount of compression. The initial condition sine is smooth, and the transformed φ is also smooth, which makes it amenable to compression. For these small problems with small numbers of qubits and a low bond dim, the MPS is effectively lossless.

For the time stepping, we can evolve the IC n_steps in one circuit, but n_steps has direct impact on circuit depth. On real QC at some point depth will become prohibitive. We use a measure-and-reprepare technique to reduce circuit depth. Given n_steps, we can divide the steps up into S segments, each which internally evolves the system k time steps (n_steps = S·k). After k steps, we measure, then reprepare the quantum state to evolve another k step-segment. We do not need to repeat the Cole-Hopf transform as we remain processing the φ field until all n_steps are finished. We do however perform the MPS encoding at the start of each segment. Then after the final segment measurement we perform a reverse Cole-Hopf to return the velocity field. The error of deep circuits must be balanced against the error introduced at the measure-and-reprepare segment seams, as we cross back and forth between the classical and quantum domains. Better understanding these trade-offs is part of this study. The positivity of φ means that when we measure we can obtain the solution in this transformed field without sign / phase ambiguity.


# Quantum Implementation

After state preparation, the time stepping is performed in the next part of the quantum circuit. For each time step, we solve the heat equation on the transformed field by computing ∂φ/∂t = ν·∂²φ/∂x², which by virtue of the derivative couples neighboring points. In its basic form, we first do a QFT into the Fourier space, then apply the heat propagator which is conveniently diagonal in Fourier space, then the inverse QFT (QFT⁻¹) back into the grid position space, completing one time step. For this one ancilla is needed. Given this, we notice that for circuits which try to evolve the system more steps (ancilla) than the physical number of qubits allow, ancilla would get recycled by expensive mid-circuit measurements and ancilla reuse. An improvement implemented is to perform k time steps between the QFT / QFT⁻¹ sandwich - one sandwich per time stepping segment S. This reduces the number of ancilla needed to one per QFT / QFT⁻¹ sandwich, and shrinks circuit size for the segment due to consolidation / cancelling of adjacent QFT⁻¹/QFT steps.

The middle propagator step is the diagonal damping of the heat equation, and in gate terms it's a block-encoded (your matrix embedded in the top-left of a larger unitary) stack of controlled rotation gates on ancilla, which captures the non-unitary change due to dampening / dissipation of the heat equation, which is otherwise not captured by rotation on the data qubits alone. High frequency modes in the signal get more rotation than small ones. 

The shots budget is used to accumulate samples for post-selecting on the ancilla. A larger per-step damping — the price of a larger dt — lowers the success rate and throws away more shots. A small dt means deep circuits, or more segment seams. 

Note these ancilla qubits used during the time evolution are not the same as those used for state preparation, and are also distinct from the base qubit register used to represent the spatial grid. We bound the number of total qubits the code may use as "max-total-qubits", and allow it to grow to O(C·q), under the observation that today's QC are relatively wide, but still inhibited by depth - allowing an algorithm to grow its number of utilized qubits is not of much benefit if the circuits which use them, and which are dependent on q, cannot be usefully deep.

Finally, there is the phi modes filter, a low pass filter on φ which is performed classically and has the effect of smoothing the noisy signal prior to the reverse Cole-Hopf transform, which is u = −2ν·∂/∂x(ln φ). The derivative amplifies high-frequency measurement noise, so a low-pass before the reverse CH helps u from becoming noise dominated. The filter takes an argument which is the number of modes to keep, and can be set below the full modes count, keeping enough to cover the pertinent information in a signal given its anticipated smoothness (e.g. as a function of viscosity). For phi-modes ≥ N/2, where N=2^q, this is equivalent to phi-modes=0, the first an effective and the second a literal no-op where the full signal is retained.

The code also permits a shots budget and the setting of QC PEC parameters. 


# Relation to Prior Work

The quantum Cole-Hopf approach for Burgers' was recently studied by Uchida et al. "Quantum simulation of Burgers turbulence: Nonlinear transformation and direct evaluation of statistical quantities" (arXiv:2412.17206, 2024; authors' code at https://github.com/fu-230/QC_Burgers). Like our implementation, they apply the classical Cole-Hopf transform to linearize Burgers' into the heat equation and then evolve the transformed field on a quantum computer. From there our implementations differ.

The Uchida implementation is aimed at the FTQC era: they evolve the transformed field by block-encoding the diffusion propagator via a linear combination of unitaries (LCU) over a central-difference spatial operator, and they characterize the cost as oracle counts rather than running circuits. The non-unitary decay of the heat equation is absorbed into a block-encoding, and the objective is to measure statistical quantities of the turbulence - n-point / structure functions up to a common normalization - rather than reconstruct the full real-space field. Their results are test calculations validated classically, not runs on real quantum hardware.


***************************************************************************************

# Code Arguments

Code arguments and typical [values] used in this study.

Case: 
- nu: [] kinematic viscosity 
- ic: [sine] - shows handling of pos & neg values, vs. gaussian - the code contains other IC, but sine is the only one exercised in this study
- A:  [0.3] sine amplitude 
- source: [none] - no source or sink in the system
- bc: [periodic] - simplest case, other BC are in the code but unexercised in this study
- q: [8] - grid has N = 2^q points; CH adds ancilla, see real circuit qubit count
- cfl: [0.1] 
- n-steps [512]
- evolution-mode [measure_reprepare] vs. single t=0 to t=end evolution in one circuit
- auto-cadence: takes snapshots at the seams
- segment-size [] - the k, the per-segment interim steps - the number of segments S = n_steps / segment_size 
- bond-dim - for the MPS encoding state preparation of the Cole-Hopf field
- phi-modes - for the post-measurement smoothing function
- max-total-qubits - enforceable as a constraint - suggest O(C·q)
- shots: [] - minimum and sufficient; we do sim runs to get an estimate for real QC
- backend args: typical Qiskit args, backend name, PEC, transpilation optimization


***************************************************************************************
