# python-wave-optics

`python-wave-optics` models the paraxial wave optics of a Gaussian beam through a plano-convex lens, sliced along the optical axis into thin transparent plates, and through free space behind it. It holds everything about this lens experiment that does not depend on how the phases are applied: the optical elements as phase signals, the parameters of an experiment and everything derived from them, the simulation loop for any backend, the classical references (exact split-step numerics and thin lens analytics), the storage of runs and their analysis.

The simulations of the monorepo share it: [wave_optics_propagation](../wave_optics_propagation/) applies the phases with Qiskit circuits and [wave_optics_propagation_qutip](../wave_optics_propagation_qutip/) with QuTiP operators, each implementing a `PropagationBackend`. The fields and phases are signals of [python-signals](../python-signals/). It depends on NumPy, Matplotlib and `python-signals`.

## Installation

```sh
pip install python-wave-optics
```

Inside the monorepo, it is installed with all other packages by `uv sync --all-packages`.

## Quick start

```python
from python_wave_optics.classical_numerics import classical_numerics_simulation
from python_wave_optics.cli import parse_parameters
from python_wave_optics.simulation import ExactBackend, simulate

# the experiment of the paper, on a small grid with few lens slices
parameters, _ = parse_parameters(
    "", ".result", ["--max-delta=0.1", "--num-qubits=5", "--lens-slices=20"]
)
result = simulate(parameters, ExactBackend())

behind_lens = parameters.step_size_after_lens * parameters.num_of_steps_after_lens
reference = classical_numerics_simulation(parameters, behind_lens)
assert abs(abs(result.snapshots["final"] @ reference.conj()) - 1) < 1e-9
```

## Where next

- The [User Guide](user-guide.md) explains the physical model, the parameters of an experiment, the simulation loop and its backends, the references, the storage and the analysis.
- The [Examples](examples.md) compare a simulation with the thin lens analytics, implement a backend of the phase protocol, store and load runs and follow a beam through free space.
- The [API Reference](reference/python_wave_optics/index.md) documents every public object, generated from the docstrings.
