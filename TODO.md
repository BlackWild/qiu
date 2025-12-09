# TODO

### Added on 15.10.2025

- [ ] shall do a better work on tests for the assertion of correct phase application. The problem is that at the moment they are based on a `FIDELITY_TOLERANCE` set to a number. But the fidelity actually depends on the parameters. So I have to assert the correct bounds on the fidelity depending on the parameters.
- [ ] the classical feedforward `if` block should contain the quantum FT blocks as well in the case of fourier domain propagation. This is not the case at the moment.

- [ ] there should actually also be unit tests which measure the runtime execution time and soft-ensure they are reasonable, namely they should give warnings
