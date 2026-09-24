# Wave Optics Propagation

The lens experiments of the paper, simulated with Qiskit: a Gaussian beam through a plano-convex lens, sliced into thin plates whose phases the sample-based phase protocol applies, and through free space behind it. The experiment, its simulation loop and its analysis live in [`python-wave-optics`](../../shareable-packages/python-wave-optics/README.md); this app adds the Qiskit backend and the scripts.

## Qiskit backend

`wave_optics_propagation.backend.QiskitBackend` applies

- the phase of each lens slice, and of each free space step with the sample-based propagator, with the statevector simulation of the circuits of [`qiskit-phase-propagator`](../../packages/qiskit-phase-propagator/README.md), post-selected on the success of each cycle, with `|phi>` prepared by `SynthesisMethod.DENSE` by default;
- the direct propagator with the circuit `MomentumDomainEvolutionQuadratic` of [`qiskit-hamiltonian-simulation`](../../packages/qiskit-hamiltonian-simulation/README.md).

## Scripts

The scripts in `scripts/` are section-based (`# %%`), to run as a whole or cell by cell, e.g. in the interactive window of VS Code:

- `simulate.py`: Simulates the experiment of the paper and stores the run in `.result/<uuid>/`; run on the cluster by `cluster/run.slurm` and `cluster/batch-run.slurm`. See `--help` for the options, e.g. `--max-delta`, `--reverse-order` and `--direct-propagator`.
- `forward_analysis.py`: The propagation figure of the paper of a single run: the beam waist against an ideal thin lens, and the intensity map with the lens, the focal point and the principal plane.
- `batch_analysis.py`: The figures of the paper of the batch over `max_delta` in `.result/batch/`: the fidelity to the classical numerics and the success probability.
- `parameter_validity.py`: An exploration of how well the phases of the lens slices and propagators are sampled.

The analyses typeset with LaTeX and save their figures to `.output/`.

## Results

`.result/` (not committed) holds the runs, one folder per run: the local runs of December 2025 and January 2026 by their timestamps, those of the cluster by their uuids, and the batch in `.result/batch/`. `.result/legacy/` keeps the runs of an early notebook, stored before the lens model was saved with them.

## Changes to the former analyses

- The reverse order passes the slices from the plane side, `N-1, ..., 0`; the former code passed them as `0, N-1, ..., 1`. The difference in the final field of the experiment of the paper is an infidelity of about `2e-6`.
- The batch analysis reads the success probability at the analyzed snapshot; the former one read it one free space step earlier.
- Each run gets its own uuid and time; formerly, all runs of one process shared them.

## Tests

From the repository root:

```sh
uv run pytest apps/wave_optics_propagation
```
