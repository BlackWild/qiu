import numpy as np
import qutip as qt
import scipy.constants as constants
from qiskit.quantum_info import Statevector
from qiskit_hamiltonian_simulation.time_independent.direct import (
    MomentumDomainEvolutionQuadratic,
)
from qiskit_signals.sample_based_signal import ArbitrarySignalForSampleBasedProtocol
from tqdm import tqdm
from wave_optics_propagation.elements import free_space_signal_generator
from wave_optics_propagation.storage import (
    save_initial_parameters,
    save_numpy_results,
)
from wave_optics_propagation_qutip.helpers import apply_phase_protocol

from wave_optics_propagation_enhanced.parameters import ExperimentParameters
from wave_optics_propagation_enhanced.result import ExperimentResult


class Experiment:
    parameters: ExperimentParameters
    result: ExperimentResult | None = None

    def __init__(
        self,
        parameters: ExperimentParameters,
    ):
        self.parameters = parameters

    def is_valid(self) -> bool:
        return self.parameters.is_valid()

    def run(self, save_results: bool = True) -> None:
        initial_beam = self.parameters.initial_beam_profile
        lens_thickness = self.parameters.lens_thickness
        lens_slices = self.parameters.lens_slices
        lens_signals = self.parameters.lens_signals
        lens_reverse_order = self.parameters.lens_reverse_order
        k_axis = self.parameters.k_axis
        vacuum_wavelength = self.parameters.vacuum_wavelength
        max_delta = self.parameters.max_delta
        num_of_steps_after_lens = self.parameters.num_of_steps_after_lens
        step_size_after_lens = self.parameters.step_size_after_lens
        lens_slice_thickness = self.parameters.lens_slice_thickness
        reduced_wavelength = self.parameters.reduced_wavelength
        experiment_datetime = self.parameters.experiment_datetime

        current_state = qt.Qobj(initial_beam.normalized_data)

        snapshots = dict([])

        snapshots[f"step_{0}"] = current_state.full()

        inside_lens_propagation_remained = lens_thickness
        total_lenses_simulated = 0
        total_probability_of_success = 1.0
        ### Lens
        tqdm_loop = tqdm(
            range(lens_slices),
            desc="Lens Slices",
            total=lens_slices,
        )
        for i in tqdm_loop:
            # print(f"Doing lens slice {i + 1}/{lens_slices}")

            lens_signal = lens_signals[-i] if lens_reverse_order else lens_signals[i]

            sample_based_lens_signal = (
                ArbitrarySignalForSampleBasedProtocol.from_generic_signal(lens_signal)
            )

            if not np.isclose(np.std(sample_based_lens_signal.data), 0):
                # print(f"max value in signal: {np.max(np.abs(sample_based_lens_signal.data))}")
                # print(
                #     f"sum of amplitude square: {np.sum(np.abs(sample_based_lens_signal.data) ** 2)}"
                # )

                current_state, local_probability_of_success = apply_phase_protocol(
                    current_state, sample_based_lens_signal, max_delta
                )

                total_lenses_simulated += 1
                total_probability_of_success *= local_probability_of_success

            else:
                pass
                # print("Skipped the lens slice because it is a flat phase.")

            propagator_signal = free_space_signal_generator(
                k_axis=k_axis,
                delta_t=lens_slice_thickness / constants.c,
                wavelength=vacuum_wavelength,
                c=constants.c,
            )
            propagator = MomentumDomainEvolutionQuadratic(
                quadratic_signal=propagator_signal
            )

            # TODO: this is for now the easiest way to do it which is to fall back to qiskit for the evolution in the momentum domain,
            qiskit_statevector = Statevector(current_state.full().flatten())
            evolved_statevector = qiskit_statevector.evolve(propagator)
            current_state = qt.Qobj(evolved_statevector.data)

            inside_lens_propagation_remained -= lens_slice_thickness

            snapshots[f"step_lens_{i}"] = current_state.full()

        # remaining propagation once the lens slice grows larger than the simulation window
        while inside_lens_propagation_remained > 0:
            propagator_signal = free_space_signal_generator(
                k_axis=k_axis,
                delta_t=inside_lens_propagation_remained / constants.c,
                wavelength=reduced_wavelength,
                c=constants.c,
            )
            propagator = MomentumDomainEvolutionQuadratic(
                quadratic_signal=propagator_signal
            )

            qiskit_statevector = Statevector(current_state.full().flatten())
            evolved_statevector = qiskit_statevector.evolve(propagator)
            current_state = qt.Qobj(evolved_statevector.data)

            inside_lens_propagation_remained -= inside_lens_propagation_remained

        snapshots["after_lens"] = current_state.full()

        tqdm_loop = tqdm(
            range(num_of_steps_after_lens),
            desc="Free space steps",
            total=num_of_steps_after_lens,
        )
        for i in tqdm_loop:
            # print(f"Doing free space step {i + 1}/{num_of_steps_after_lens}")

            propagator_signal = free_space_signal_generator(
                k_axis=k_axis,
                delta_t=step_size_after_lens / constants.c,
                wavelength=vacuum_wavelength,
                c=constants.c,
            )
            propagator = MomentumDomainEvolutionQuadratic(
                quadratic_signal=propagator_signal
            )

            qiskit_statevector = Statevector(current_state.full().flatten())
            evolved_statevector = qiskit_statevector.evolve(propagator)
            current_state = qt.Qobj(evolved_statevector.data)

            snapshots[f"step_after_lens_{i + 1}"] = current_state.full()

        snapshots["final"] = current_state.full()

        self.result = ExperimentResult(
            snapshots=snapshots,
            total_lenses_simulated=total_lenses_simulated,
            total_probability_of_success=total_probability_of_success,
        )

        if save_results:
            self.save_result()

        print(
            f"Simulation complete. ID: {experiment_datetime.strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        print(f"Total probability of success: {total_probability_of_success}")

    def save_result(self) -> None:
        if self.result is None:
            raise ValueError("No result to save. Please run the experiment first.")

        # TODO: make it nicer by handling the items directly using the ExperimentParameters class, probably makes more sense to define save methods internal to the Parameters and Result classes, and not mix total_lenses_simulated with others for example

        initial_parameters = {
            "experiment_datetime": self.parameters.experiment_datetime.isoformat(),
            "vacuum_wavelength": self.parameters.vacuum_wavelength,
            "beam_FWHM": self.parameters.beam_FWHM,
            "focal_length": self.parameters.focal_length,
            "lens_diameter": self.parameters.lens_diameter,
            "refractive_index": self.parameters.refractive_index,
            "lens_thickness": self.parameters.lens_thickness,
            "propagation_after_lens": self.parameters.propagation_after_lens,
            "transverse_length": self.parameters.transverse_length,
            "num_of_steps_after_lens": self.parameters.num_of_steps_after_lens,
            "lens_slices": self.parameters.lens_slices,
            "num_qubits": self.parameters.num_qubits,
            "max_delta": self.parameters.max_delta,
            "total_lenses_simulated": self.result.total_lenses_simulated,
            "lens_reverse_order": self.parameters.lens_reverse_order,
            "fresnel_approximation": self.parameters.fresnel_approximation,
            "scale_down_phases": self.parameters.scale_down_phases,
            "total_probability_of_success": self.result.total_probability_of_success,
        }

        save_initial_parameters(
            initial_parameters, self.parameters.experiment_datetime, wrapper_folder=None
        )
        save_numpy_results(
            self.result.snapshots,
            self.parameters.experiment_datetime,
            wrapper_folder=None,
        )
