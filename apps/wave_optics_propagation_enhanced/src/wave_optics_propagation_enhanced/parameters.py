from datetime import datetime
from functools import cached_property

import numpy as np
from qiskit_signals.helper_types import EncodingType
from qiskit_signals.quantum_axis import (
    AngularWavenumberAxis,
    PositionAxis,
)
from wave_optics_propagation.analytics import GenericQuantumSignal, gaussian_signal
from wave_optics_propagation.elements import (
    radius_of_convex_planar_lens_as_a_func_of_z,
    thin_transparent_plate_signal_generator,
)


class ExperimentParameters:
    def __init__(
        self,
        vacuum_wavelength: float,
        beam_FWHM: float,
        focal_length: float,
        refractive_index: float,
        propagation_after_lens: float,
        transverse_length: float,
        num_of_steps_after_lens: int,
        lens_slices: int,
        num_qubits: int,
        max_delta: float,
        lens_reverse_order: bool,
        fresnel_approximation: bool,
        scale_down_phases: bool = True,
        experiment_datetime: datetime | None = None,
    ):
        # Input parameters
        self.vacuum_wavelength = vacuum_wavelength
        self.beam_FWHM = beam_FWHM
        self.focal_length = focal_length
        self.refractive_index = refractive_index
        self.propagation_after_lens = propagation_after_lens
        self.transverse_length = transverse_length
        self.num_of_steps_after_lens = num_of_steps_after_lens
        self.lens_slices = lens_slices
        self.num_qubits = num_qubits
        self.max_delta = max_delta
        self.lens_reverse_order = lens_reverse_order
        self.fresnel_approximation = fresnel_approximation
        self.scale_down_phases = scale_down_phases

        # Experiment metadata
        self.experiment_datetime = experiment_datetime or datetime.now()

    def is_valid(self) -> bool:
        # TODO: Add any necessary validation logic here
        if self.delta_x > self.vacuum_wavelength:
            print(
                f"Invalid parameters: delta_x > vacuum_wavelength, {self.delta_x} > {self.vacuum_wavelength}"
            )
            return False  # ensure sampling condition

        if self.gaussian_beam_waist < 10 * self.vacuum_wavelength:
            print(
                f"Invalid parameters: gaussian_beam_waist < 10 * vacuum_wavelength, {self.gaussian_beam_waist} < {10 * self.vacuum_wavelength}"
            )
            return False  # ensure paraxial approximation validity

        return True

    # Derived parameters
    @cached_property
    def k0(self) -> float:
        return 2 * np.pi / self.vacuum_wavelength

    @cached_property
    def reduced_wavelength(self) -> float:
        return self.vacuum_wavelength / self.refractive_index

    @cached_property
    def radius_of_curvature(self) -> float:
        return self.focal_length * (self.refractive_index - 1)

    @cached_property
    def lens_diameter(self) -> float:
        return self.transverse_length

    @cached_property
    def lens_radius(self) -> float:
        return self.lens_diameter / 2

    @cached_property
    def lens_thickness(self) -> float:
        return self.radius_of_curvature - np.sqrt(
            self.radius_of_curvature**2 - self.lens_radius**2
        )

    @cached_property
    def dimension(self) -> int:
        return 2**self.num_qubits

    @cached_property
    def delta_x(self) -> float:
        return self.transverse_length / self.dimension

    @cached_property
    def gaussian_mean(self) -> float:
        return self.transverse_length / 2

    @cached_property
    def gaussian_beam_waist(self) -> float:
        return self.beam_FWHM / np.sqrt(2 * np.log(2))

    @cached_property
    def lens_slice_thickness(self) -> float:
        return self.lens_thickness / self.lens_slices

    @cached_property
    def step_size_after_lens(self) -> float:
        return self.propagation_after_lens / self.num_of_steps_after_lens

    @cached_property
    def lens_slice_positions(self) -> np.ndarray:
        return (
            np.linspace(0, self.lens_thickness, self.lens_slices, endpoint=False)
            + self.lens_slice_thickness / 2
        )  # choosing the midpoint in each transverse slice

    @cached_property
    def lens_transverse_radii(self) -> list[float]:
        return [
            radius_of_convex_planar_lens_as_a_func_of_z(
                self.radius_of_curvature,
                z,
                self.lens_thickness,
                fresnel_approximation=self.fresnel_approximation,
            )
            for z in self.lens_slice_positions
        ]

    @cached_property
    def x_axis(self) -> PositionAxis:
        return PositionAxis(
            num_qubits=self.num_qubits,
            delta_x=self.delta_x,
            encoding=EncodingType.UNSIGNED,
        )

    @cached_property
    def k_axis(self) -> AngularWavenumberAxis:
        return AngularWavenumberAxis.from_position_axis(self.x_axis)

    @cached_property
    def initial_beam_profile(self) -> GenericQuantumSignal:
        return gaussian_signal(
            self.x_axis, self.gaussian_beam_waist, self.gaussian_mean
        )

    @cached_property
    def lens_signals(self) -> list[GenericQuantumSignal]:
        return [
            thin_transparent_plate_signal_generator(
                x_axis=self.x_axis,
                refractive_index=self.refractive_index,
                thickness=self.lens_slice_thickness,
                radius=lens_radius,
                wavelength=self.vacuum_wavelength,
                scale_down=self.scale_down_phases,
            )
            for lens_radius in self.lens_transverse_radii
        ]

    @cached_property
    def sum_lens_signal_data(self) -> np.ndarray:
        sum_signal = np.zeros_like(self.lens_signals[0].data)
        for signal in self.lens_signals:
            sum_signal += signal.data
        return sum_signal
