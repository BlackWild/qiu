"""Unit tests for gaussian_beam.py."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from python_pytest_helper.assertions import assert_close
from python_pytest_helper.hypothesis_strategies import position_axes
from python_signals.physical_axis import PositionAxis
from python_wave_optics.gaussian_beam import (
    beam_after_free_space,
    beam_after_system,
    beam_after_thin_lens,
    gaussian_signal,
    propagation_matrix,
    rayleigh_range,
    thin_lens_matrix,
)

waists = st.floats(min_value=1e-6, max_value=1e-3)
distances = st.floats(min_value=1e-6, max_value=1.0)
wavelengths = st.floats(min_value=1e-7, max_value=1e-5)
indices = st.floats(min_value=1.0, max_value=2.0)


class TestFreeSpace:
    """Test the Gaussian beam after a free propagation from its waist."""

    @given(waist=waists, distance=distances, wavelength=wavelengths, n=indices)
    def test_closed_form(self, waist, distance, wavelength, n):
        """Test `w(z) = w0 sqrt(1 + (z/z_R)**2)` and `R(z) = z (1 + (z_R/z)**2)`."""
        z_r = rayleigh_range(waist, wavelength / n)
        radius, beam_radius = beam_after_free_space(waist, distance, wavelength, n)
        assert_close(beam_radius, waist * np.sqrt(1 + (distance / z_r) ** 2))
        assert_close(radius, distance * (1 + (z_r / distance) ** 2))

    @given(waist=waists, wavelength=wavelengths)
    def test_waist(self, waist, wavelength):
        """Test that the wavefront is flat at the waist."""
        radius, beam_radius = beam_after_system(waist, 0.0, wavelength, np.eye(2))
        assert radius == np.inf
        assert_close(beam_radius, waist)


class TestThinLens:
    """Test the Gaussian beam after a thin lens."""

    @given(waist=waists, focal_length=distances, wavelength=wavelengths)
    def test_focus_of_a_collimated_beam(self, waist, focal_length, wavelength):
        """Test the focused waist `w0 / sqrt(1 + (z_R/f)**2)` of a waist at the lens."""
        z_r = rayleigh_range(waist, wavelength)
        # the focus of a beam with its waist at the lens lies before the focal plane
        focus = focal_length / (1 + (focal_length / z_r) ** 2)
        _, beam_radius = beam_after_thin_lens(waist, focus, focal_length, wavelength, 1)
        assert_close(beam_radius, waist / np.sqrt(1 + (z_r / focal_length) ** 2))

    @given(waist=waists, before=distances, wavelength=wavelengths)
    def test_lens_is_not_a_propagation(self, waist, before, wavelength):
        """Test that right at the lens, only the curvature changes, not the radius."""
        _, radius_before = beam_after_free_space(waist, before, wavelength, 1)
        _, radius_after = beam_after_thin_lens(
            waist, 0.0, 0.1, wavelength, 1, distance_from_waist=before
        )
        assert_close(radius_after, radius_before)

    def test_matrix_order(self):
        """Test that the matrices compose with the last element first."""
        system = propagation_matrix(2.0) @ thin_lens_matrix(1.0)
        np.testing.assert_array_equal(system, [[-1.0, 2.0], [-1.0, 1.0]])


@given(axis=position_axes(), waist=st.floats(min_value=0.1, max_value=10.0))
def test_gaussian_signal(axis: PositionAxis, waist: float):
    """Test the Gaussian field centered at the mean."""
    mean = axis.sampling_window_length / 2
    signal = gaussian_signal(axis, waist, mean)
    assert_close(signal.data, np.exp(-((axis.values - mean) ** 2) / waist**2))
