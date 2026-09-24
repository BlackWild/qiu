# Qiu

Qiu is a monorepo holding various packages and projects related to quantum computing and quantum programming. The main focus is to provide better tools on top of the Qiskit framework.

## Structure

The repository is structured as follows:

- `shareable-packages/`: Contains reusable libraries and tools for things not necessarily related to scientific computation or quantum computing. Think of them as packages that enhance standard python.
  - `python-encore`: A package for enhancing standard Python functionalities.
  - `python-signals`: A package for uniformly sampled axes and signals on them, for classical and quantum numerics alike.
  - `python-wave-optics`: Paraxial wave optics of Gaussian beams through sliced lenses: optical elements, the lens experiment and its storage, its simulation loop for any backend, and its classical references.

- `dev-packages/`: Contains packages which are not meant to be published for others but just to be used inside this monorepo for development purposes.
  - `python-pytest-helper`: The floating-point comparisons and the `hypothesis` strategies of numbers, axes and signals shared by the unit tests, depending only on `python-signals`.
  - `qiskit-pytest-helper`: The quantum counterparts on top of it: strategies of quantum states and qubit axes, assertions with Qiskit's equality, and typed circuit helpers.

- `packages/`: Contains reusable libraries and tools for quantum computing.
  - `qiskit-encore`: Circuit building blocks on top of Qiskit (robust state preparation, QFT, uniformly controlled rotations), each available as a dense unitary, a Qiskit gate or a decomposed circuit.
  - `qiskit-aer-encore`: Aer simulators configured for the available hardware (CPU or GPU).
  - `qiskit-phase-propagator`: Circuits applying phases `e^(i f(x))` of `python-signals` signals to qubit registers, directly or sample-based.
  - `qiskit-hamiltonian-simulation`: Time evolution under potentials and kinetic energies given as `python-signals` signals.

- `apps/`: Contains standalone applications and projects that utilize the other packages, with section-based (`# %%`) scripts instead of notebooks.
  - `wave_optics_propagation`: The lens experiments of the paper simulated with Qiskit: the simulation run on the cluster and the analyses of the paper.
  - `wave_optics_propagation_qutip`: The same lens experiments simulated with QuTiP.

There are then other directories which are very specific to some task and are not used very frequently:
- `cluster/`: The SLURM jobs running the simulations of `apps/wave_optics_propagation`.
- `project-diary/`
- and the other dot folders `.*/`
