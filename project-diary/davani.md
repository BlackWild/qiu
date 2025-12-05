# Project Diary

### Log 05.12.2025

- To make CUDA work locally in this project, I had to
  - Add `module load nvidia/x.x.x` to the `.bashrc` file to load the ENVs
  - Also super duper important was that I needed to exactly match the version with the version I get from `nvidia-smi` inside the compute node UP TO THE EXACT DIGITS and not just `11.x.x` vs `12.x.x`.
  - Then I added a local `.env` file to the repo to make sure vscode reads it.
  - But that did not really fix the issue for the interactive sessions and I had to work with ChatGPT to create a python kernel specification that loads the ENV from the system.
  - Now it works.

### Log 02.12.2025

- For some reason, transforming a `QuantumCircuit` to and an `Operator` is super inefficient and takes a lot of memory+time.
- Avoid that!
- But `evolve()` using a quantum circuit seems to work much faster.

### Log 15.10.2025

- make sure to use `norm="ortho"` in all Fourier transforms. This ensures that the Fourier transform is unitary and preserves the norm of the state vector.

### Log 10.10.2025

- `pytest-xdist` enables parallel test execution, allowing for faster test runs by distributing tests across multiple CPU cores.
- I moved to using QuantumCircuit instead of Gate. Gates are more limited in functionality and lead to unintuitive bugs.

### Log 09.10.2025

- It seems the `.inverse()` of the `StatePreparation` does not work as expected. The problem seems to be that qiskit does not know how to transpile the inverse of the `StatePreparation` gate. The workaround manually pass a `basis_gates` argument to the `transpile` function. Passing a `backend` argument does not work.

### Log 08.10.2025

- I think `BlueprintCircuit` object is supposed to be used in cases the circuit initialization is costly and you are just gonna wait until its first application to initialize it. Which in general means that you can change the internal parameters before the first usage and the parameters will be validated before the build.

### Log 07.10.2025 later the same day

- The previous point is not entirely true. The reason is that `Gate` does not support classical bits, while `QuantumCircuit` does. So if you need classical bits, you have to use `QuantumCircuit`.
- Use qiskit's classical feedforward feature to reuse the classical register and not to perform quantum operations for the next iteration if there has been an error in the previous ones.
  - For now I made a classical for loop with qiskit's `if_test` functionality. But would be better I could use `for_loop` and breaks instead.

### Log 07.10.2025

- Most of the circuit-like entities in the packages should be implemented using the `Gate` class, not `QuantumCircuit`. That is because `QuantumCircuit` is supposed to also hold the information about which qubits and classical bits it acts on, while `Gate` is more like a template that can be applied to any underlying register.
- Whenever something can be applied to different qubits, it should be a `Gate`. Use `QuantumCircuit` for applications and complete simulations; that is whenever you actually want to run something on a quantum computer or simulator.

### Log 06.10.2025

- No point fighting over using `jax` or `numpy`. Just use `numpy` for now, since it's more standard.
- Move fast, create big. If you just try being vigilant enough, your can achieve a lot.
