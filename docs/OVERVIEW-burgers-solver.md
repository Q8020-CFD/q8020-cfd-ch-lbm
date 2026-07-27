# Quantum Methods for the 1-D Viscous Burgers Equation

*Human author*

---

## Overview

We are interested in *cases × codes × backends*, in running the same case on more than one code implementing the same case - in this instance, the 1D Burgers equation. We can run this case x code on many backends - simulators, real QC of various vendors.

In this code we implement three algorithms - a Cole-Hopf linearization followed by a time evolution (aka CH), a variation on the quantum lattice Boltzmann (aka QLBM), and a classical reference.

The code is organized as follows:

- burgers_solver - main entry point, exposes numerous knobs for both the CH and LBM routines. Uses a common q8020-cfd-metautil library to pull in some common QC arguments (e.g. "--shots").

- lib_classical - the reference computation to be used - FTCS - CH and LBM will be compared to this. Includes initial condition generators for sine, etc.

- lib_cole_hopf_circuit - build the circuit from the linearized problem, block encoded, and evolve it in time

- lib_cole_hopf - perform the classical linearization, and the reverse

- lib_fd - spacial derivative functions, used by the classical solver, postproc, computing L2 error, etc.

- lib_fw - q8020-cfd-metautil exposes a "solverfw" - a "solver framework" which this code tries to adapt to. The idea is to create pluggable components for a reusable solver scaffolding. This framework is a work-in-progress. Mostly this means providing classes which implement the solverfw abstract classes - wrappers on existing python methods in this code.

- lib_lbm - The implementation of the classical LBM. This could be used as a reference, but we use FTCS. Our QLBM is actually quantum collision with classical streaming. 

- lib_mps - MPS encoding routines. At this time used by CH only.

- lib_postprocess - Driven by the metautil "sweeper" (a TOML-driven way to invoke this code across a range of args), this code is called at the end to mine results.

- lib_qalb_circuit - The quantum LBM collision step. 


## CH

The linearization step converts Burgers including the non-linear portion into the linear heat equation. It does this, for this case, perfectly. The transformation is classical. At the very end of the CH simulation, a reverse-CH is performed to return us to the original basis. In many ways this is no different than other linearizations know to us - it is chosen here specifically because it does a good job with Burgers', not because it is generalizable. It also handles the sign of the solution without extra Hadamard-type tests.

At this point we could use amplitude encoding to load the initial state, but the code also permits an MPS encoding, with a settable bond dimension parameter that controls the amount of compression. Either can be used.

During the time evolution, if the length of time is long, a single evolution can require very deep circuits. Thus we use an iterative measure-and-reprepare approach. The CH transform is done once up front, the circuit is created and run for the first time slice, then a measurement is taken. From there, the state is reprepared and evolved through the next time slice. Note "slice here" - call it what you like, but its a collection of time steps (see lib_cole_hopf_circuit.heat_qft_full_circuit() for the Trotterization circuit - see also a separate method for shots=0 pure SV). After all of these steps, the reverse CH is performed.

Needless to say there is error introduced at each of these time slice seams - the tradeoff between that error and the error of long circuits can be compared.


## QLBM aka QALB

The LBM method involves two steps - a collision, and a streaming to nearest neighbors. In our implementation, the collision is handled with quantum code, and the streaming with classical. This lines up nicely with the CH code, which uses a similar quantum-classical setup via the measure-and-reprepare loop. Thus for the purposes of results movie-making, we start with a "% to shock" argument and compute where the time seams belong to line up QLBM, CH, and the classical reference in time. Note that unlike CH which can run interim time steps per "slice" or movie frame, QLBM only runs one. Allowing the QLBM to run interim steps (k>1) is noted as a future work item. 

The QLBM algorithm is a variation proposed by Itani et al. Value/Fock (bosonic) encoding is used instead of amplitude encoding. There is no MPS encoding option in the code. In amplitude encoding, the field lives in the amplitudes of the state, so the nonlinear collision map would have to be rebuilt every step from the current classical state. "QALB" (i.e. Itani QLBM) encodes each quasi-particle density value into a Fock register (3 qubits/site), so the nonlinearity becomes an operator, and the collision becomes a single, state-independent Hamiltonian built once and reused. (Note: the human author admits the math of this is above his head - read Itani.) Unlike regular QLBM, QALB handles the sign without extra test - negative particle populations are possible.

Itani offers math but no code. We offer implementation, but stop short of Itani's LCU approach for defining the collision operator. For us this is future work. We use Qiskit's UnitaryGate and let it do the construction and Trotterize. We also support an option where trotter_steps > 1 and a SparsePauliOp is used, although this was not used in the tests for the paper so far.

