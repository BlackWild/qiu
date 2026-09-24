"""The parameters of a lens experiment: a Gaussian beam through a plano-convex lens.

A Gaussian beam of a waist given by its FWHM enters a plano-convex lens, sliced along
the optical axis into thin transparent plates, and then propagates freely behind it.
The transverse field is sampled on `2**num_qubits` points of a window of
`transverse_length`, which the lens fills.
"""

import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from functools import cached_property
from typing import Any

import numpy as np
import numpy.typing as npt
from python_signals.algebraic_signal import AlgebraicSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import AngularWavenumberAxis, PositionAxis

from python_wave_optics.elements import (
    convex_planar_lens_radius,
    transparent_plate_phase,
    vacuum_wavenumber,
)
from python_wave_optics.gaussian_beam import gaussian_signal

LEGACY_KEYS = {
    "timestamp": "experiment_datetime",
    "reverse_order": "lens_reverse_order",
}
"""Former names of stored parameters, mapped to the current ones."""

LEGACY_DEFAULTS: dict[str, Any] = {"direct_propagator": True, "uuid": ""}
"""Values of parameters missing in older results, which always used them."""


@dataclass(frozen=True)
class ExperimentParameters:
    """The parameters of a lens experiment, from which all others are derived."""

    vacuum_wavelength: float
    beam_FWHM: float
    focal_length: float
    refractive_index: float
    propagation_after_lens: float
    transverse_length: float
    num_of_steps_after_lens: int
    lens_slices: int
    num_qubits: int
    """The number of qubits `n` of the `2**n` transverse samples."""
    max_delta: float
    """The maximum phase per cycle of the sample-based phase protocol."""
    lens_reverse_order: bool
    """If True, the beam enters through the plane side of the lens."""
    fresnel_approximation: bool
    """If True, the lens surface is approximated by a paraboloid."""
    scale_down_phases: bool
    """If True, the phases of the lens slices are reduced modulo `2 pi`."""
    direct_propagator: bool
    """If True, free propagation is applied directly, else with the phase protocol."""
    experiment_datetime: datetime = field(default_factory=datetime.now)
    uuid: str = field(default_factory=lambda: uuid.uuid4().hex)

    def validity_problems(self) -> list[str]:
        """Return the violated conditions of the sampling and the paraxial regime."""
        problems = []
        if self.delta_x > self.vacuum_wavelength:
            problems.append(
                f"delta_x > vacuum_wavelength: {self.delta_x} > {self.vacuum_wavelength}"
            )
        if self.gaussian_beam_waist < 10 * self.vacuum_wavelength:
            problems.append(
                "gaussian_beam_waist < 10 * vacuum_wavelength (paraxial regime): "
                f"{self.gaussian_beam_waist} < {10 * self.vacuum_wavelength}"
            )
        return problems

    def is_valid(self) -> bool:
        """Whether the parameters satisfy the conditions of `validity_problems`."""
        return not self.validity_problems()

    # derived parameters

    @cached_property
    def k0(self) -> float:
        """The vacuum wavenumber."""
        return vacuum_wavenumber(self.vacuum_wavelength)

    @cached_property
    def reduced_wavelength(self) -> float:
        """The wavelength inside the lens."""
        return self.vacuum_wavelength / self.refractive_index

    @cached_property
    def radius_of_curvature(self) -> float:
        """The radius of curvature of the convex surface, from the lensmaker's equation."""
        return self.focal_length * (self.refractive_index - 1)

    @cached_property
    def lens_diameter(self) -> float:
        """The diameter of the lens, filling the transverse window."""
        return self.transverse_length

    @cached_property
    def lens_radius(self) -> float:
        """The transverse radius of the lens."""
        return self.lens_diameter / 2

    @cached_property
    def lens_thickness(self) -> float:
        """The thickness of the lens at its center."""
        return self.radius_of_curvature - np.sqrt(
            self.radius_of_curvature**2 - self.lens_radius**2
        )

    @cached_property
    def dimension(self) -> int:
        """The number of transverse samples."""
        return 2**self.num_qubits

    @cached_property
    def delta_x(self) -> float:
        """The transverse sampling period."""
        return self.transverse_length / self.dimension

    @cached_property
    def gaussian_mean(self) -> float:
        """The transverse position of the beam's center, the center of the window."""
        return self.transverse_length / 2

    @cached_property
    def gaussian_beam_waist(self) -> float:
        """The waist radius of the beam, from its FWHM."""
        return self.beam_FWHM / np.sqrt(2 * np.log(2))

    @cached_property
    def lens_slice_thickness(self) -> float:
        """The thickness of each lens slice."""
        return self.lens_thickness / self.lens_slices

    @cached_property
    def step_size_after_lens(self) -> float:
        """The length of each free propagation step behind the lens."""
        return self.propagation_after_lens / self.num_of_steps_after_lens

    @cached_property
    def lens_slice_positions(self) -> npt.NDArray[np.float64]:
        """The depths of the midpoints of the lens slices, from the vertex."""
        return (
            np.linspace(0, self.lens_thickness, self.lens_slices, endpoint=False)
            + self.lens_slice_thickness / 2
        )

    @cached_property
    def lens_transverse_radii(self) -> list[float]:
        """The transverse radii of the lens slices, from the vertex."""
        return [
            convex_planar_lens_radius(
                self.radius_of_curvature,
                depth,
                self.lens_thickness,
                fresnel_approximation=self.fresnel_approximation,
            )
            for depth in self.lens_slice_positions
        ]

    @cached_property
    def x_axis(self) -> PositionAxis:
        """The transverse position axis, from 0 to `transverse_length`."""
        return PositionAxis(self.dimension, self.delta_x, IndexOrdering.NATURAL)

    @cached_property
    def k_axis(self) -> AngularWavenumberAxis:
        """The angular wavenumber axis of the angular spectrum, in the FFT ordering."""
        return AngularWavenumberAxis.from_position_axis(self.x_axis)

    @cached_property
    def initial_beam_profile(self) -> AlgebraicSignal:
        """The field of the Gaussian beam entering the lens."""
        return gaussian_signal(
            self.x_axis, self.gaussian_beam_waist, self.gaussian_mean
        )

    @cached_property
    def initial_state(self) -> npt.NDArray[np.complex128]:
        """The normalized amplitudes of the beam entering the lens."""
        return self.initial_beam_profile.to_signal().normalized_data.astype(
            np.complex128
        )

    @cached_property
    def lens_signals(self) -> list[AlgebraicSignal]:
        """The phase signals of the lens slices, from the vertex."""
        return [
            transparent_plate_phase(
                x_axis=self.x_axis,
                refractive_index=self.refractive_index,
                thickness=self.lens_slice_thickness,
                radius=radius,
                wavelength=self.vacuum_wavelength,
                scale_down=self.scale_down_phases,
            )
            for radius in self.lens_transverse_radii
        ]

    @cached_property
    def ordered_lens_signals(self) -> list[AlgebraicSignal]:
        """The phase signals of the lens slices, in the order the beam passes them."""
        if self.lens_reverse_order:
            return self.lens_signals[::-1]
        return self.lens_signals

    # storage

    def to_dict(self) -> dict[str, Any]:
        """Return the parameters, with the derived lens geometry, as JSON values."""
        values = asdict(self)
        values["experiment_datetime"] = self.experiment_datetime.isoformat()
        values["lens_diameter"] = self.lens_diameter
        values["lens_thickness"] = self.lens_thickness
        return values

    @classmethod
    def from_dict(
        cls, values: dict[str, Any], defaults: dict[str, Any] | None = None
    ) -> "ExperimentParameters":
        """Create the parameters from stored values, e.g. of `to_dict`.

        Values of results stored under former names (`LEGACY_KEYS`) are renamed, and
        `LEGACY_DEFAULTS` fill in parameters which older results did not store.

        Args:
            values: The stored values; unknown keys, e.g. derived ones, are ignored.
            defaults: Values for parameters missing in `values`, e.g. those an older
                simulation did not store but used.

        Returns:
            The parameters.
        """
        renamed = {LEGACY_KEYS.get(key, key): value for key, value in values.items()}
        merged = {**LEGACY_DEFAULTS, **(defaults or {}), **renamed}
        names = [f.name for f in fields(cls)]
        missing = [name for name in names if name not in merged]
        if missing:
            raise KeyError(
                f"Missing parameters {missing}; give their values as defaults."
            )
        arguments = {name: merged[name] for name in names}
        if isinstance(arguments["experiment_datetime"], str):
            arguments["experiment_datetime"] = datetime.fromisoformat(
                arguments["experiment_datetime"]
            )
        return cls(**arguments)

    def __str__(self) -> str:
        """Return the parameters, one per line."""
        return "\n".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))
