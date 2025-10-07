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
