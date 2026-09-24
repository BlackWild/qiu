# Wave Optics Propagation with QuTiP

The lens experiments of [`wave_optics_propagation`](../wave_optics_propagation/README.md), simulated with QuTiP instead of Qiskit. The experiment, its simulation loop and its analysis live in [`python-wave-optics`](../../shareable-packages/python-wave-optics/README.md); this app adds the QuTiP backend and the scripts.

## QuTiP backend

`wave_optics_propagation_qutip.backend.QutipBackend` applies

- the sample-based phase protocol to the ket `|phi> (x) |psi>` of an ancilla prepared in `|phi>` and the field: the partial phase (`partial_phase_operator`) multiplies the states where both registers agree by `e^(i delta)`, and the projection of the ancilla onto `<phi|` keeps the successful outcome of each cycle;
- the direct propagator as the operator `F^dagger e^(i f) F` of the orthonormal DFT `F` (`dft_operator`).

## Scripts

- `scripts/simulate.py`: Simulates the experiment of the paper, with the same options as the Qiskit app, and stores the run in `.result/<uuid>/`.
- `scripts/analysis.py`: A single run against the thin lens and the classical numerics: the beam waist along the propagation and snapshots behind the lens.

## Runs of December 2025

`.result/13-lens-simulation-after-classical-numerics/` (not committed) holds the runs of the former script. It used the Fresnel approximation and phases reduced modulo `2 pi`, which the runs did not store, so `scripts/analysis.py` gives them as defaults:

- `2025-12-10_07-53-57`: normal order
- `2025-12-10_08-16-27`: reverse order of the previous

with the order as a parameter:

- `2025-12-10_14-34-19`: normal order, `max_delta=0.01`, 10000 slices
- `2025-12-10_14-47-52`: reverse
- `2025-12-10_15-02-34`: normal, `max_delta=0.1`, 1000 slices
- `2025-12-10_15-04-13`: reverse, `max_delta=0.1`
- `2025-12-10_15-10-05`: normal, `max_delta=0.1`, 10000 slices
- `2025-12-10_15-12-36`: reverse, `max_delta=0.1`, 10000 slices
- `2025-12-10_15-16-37`: normal order, `max_delta=0.01`, 1000 slices
- `2025-12-10_15-33-18`: normal, `max_delta=0.2`, 10000 slices
- `2025-12-10_15-35-45`: reverse, `max_delta=0.2`, 10000 slices

## Tests

From the repository root:

```sh
uv run pytest apps/wave_optics_propagation_qutip
```
