"""Unit tests for elements.py."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from python_pytest_helper.assertions import assert_close
from python_pytest_helper.hypothesis_strategies import physical_axes, position_axes
from qiu_classical_simulation.wave_optics.elements import (
    convex_planar_lens_radius,
    free_space_propagator_phase,
    transparent_plate_phase,
    vacuum_wavenumber,
)
from qiu_signals.algebraic_signal import QuadraticSignal
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import (
    AngularWavenumberAxis,
    AxisDomain,
    PositionAxis,
)


class TestConvexPlanarLensRadius:
    """Test convex_planar_lens_radius."""

    @given(depth_fraction=st.floats(min_value=0.0, max_value=0.99))
    def test_sphere(self, depth_fraction: float):
        """Test that the rim at the depth lies on the sphere of the curvature."""
        curvature, thickness = 5.0, 2.0
        depth = depth_fraction * thickness
        radius = convex_planar_lens_radius(curvature, depth, thickness, False)
        assert_close(radius**2 + (curvature - depth) ** 2, curvature**2)

    @given(depth_fraction=st.floats(min_value=0.0, max_value=0.99))
    def test_fresnel_approximation(self, depth_fraction: float):
        """Test the paraboloid `depth = radius**2 / (2 R)`."""
        curvature, thickness = 5.0, 2.0
        depth = depth_fraction * thickness
        radius = convex_planar_lens_radius(curvature, depth, thickness, True)
        assert_close(radius**2 / (2 * curvature), depth)

    @pytest.mark.parametrize("fresnel_approximation", [False, True])
    @pytest.mark.parametrize("depth", [2.0, 3.0, -0.5])
    def test_outside_the_lens(self, fresnel_approximation: bool, depth: float):
        """Test that the lens has no radius beyond its thickness or before its vertex."""
        assert convex_planar_lens_radius(5.0, depth, 2.0, fresnel_approximation) == 0.0


class TestTransparentPlatePhase:
    """Test transparent_plate_phase."""

    @given(
        axis=position_axes(orderings=st.just(IndexOrdering.NATURAL)),
        radius_fraction=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_phase_within_the_radius(self, axis: PositionAxis, radius_fraction: float):
        """Test the phase `(n - 1) k0 t` within the radius around the window center."""
        radius = radius_fraction * axis.sampling_window_length / 2
        signal = transparent_plate_phase(axis, 1.5, 1e-6, radius, 5e-7, False)
        inside = np.abs(axis.values - axis.sampling_window_length / 2) <= radius
        np.testing.assert_array_equal(
            signal.data, np.where(inside, 0.5 * vacuum_wavenumber(5e-7) * 1e-6, 0.0)
        )

    def test_scale_down(self):
        """Test that scaling down reduces the phase modulo 2 pi."""
        axis = PositionAxis(4, 1.0, IndexOrdering.NATURAL)
        thickness = 1.75 * 1e-6  # a phase of 3.5 pi at n = 1.5 and 5e-7 wavelength
        full = transparent_plate_phase(axis, 1.5, thickness, 10.0, 5e-7, False)
        scaled = transparent_plate_phase(axis, 1.5, thickness, 10.0, 5e-7, True)
        assert_close(full.data, 3.5 * np.pi)
        assert_close(scaled.data, 1.5 * np.pi)
        assert_close(np.exp(1j * scaled.data), np.exp(1j * full.data))


@given(
    axis=physical_axes(AxisDomain.ANGULAR_WAVENUMBER),
    distance=st.floats(min_value=0.0, max_value=1e-3),
)
def test_free_space_propagator_phase(axis: AngularWavenumberAxis, distance: float):
    """Test the paraxial phase `-k**2 dz / (2 k0)`."""
    signal = free_space_propagator_phase(axis, distance, 1e-6)
    assert isinstance(signal, QuadraticSignal)
    assert_close(
        signal.data, -(axis.values**2) * distance / (2 * vacuum_wavenumber(1e-6))
    )
