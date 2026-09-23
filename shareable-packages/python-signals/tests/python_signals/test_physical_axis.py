"""Unit tests for physical_axis.py."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from python_signals.integer_axis import IndexOrdering, IntegerAxis
from python_signals.physical_axis import (
    AngularWavenumberAxis,
    AxisDomain,
    MomentumAxis,
    PhysicalAxis,
    PositionAxis,
    SpatialFrequencyAxis,
    reciprocal_period,
)

sizes = st.integers(min_value=1, max_value=64)
periods = st.floats(min_value=1e-3, max_value=10.0)
orderings = st.sampled_from(list(IndexOrdering))
domains = st.sampled_from(list(AxisDomain))


class TestAxisDomain:
    """Test the AxisDomain enum."""

    def test_values(self):
        """Test the list of the raw values."""
        assert AxisDomain.list() == [
            "position",
            "momentum",
            "angular_wavenumber",
            "spatial_frequency",
        ]

    @pytest.mark.parametrize(
        ("domain", "expected"),
        [
            (AxisDomain.POSITION, False),
            (AxisDomain.MOMENTUM, True),
            (AxisDomain.ANGULAR_WAVENUMBER, True),
            (AxisDomain.SPATIAL_FREQUENCY, True),
        ],
    )
    def test_is_in_fourier_domain(self, domain: AxisDomain, expected: bool):
        """Test which domains are Fourier conjugates of position."""
        assert domain.is_in_fourier_domain is expected


class TestPhysicalAxis:
    """Test the PhysicalAxis."""

    @given(size=sizes, period=periods, ordering=orderings, domain=domains)
    def test_essentials(
        self,
        size: int,
        period: float,
        ordering: IndexOrdering,
        domain: AxisDomain,
    ):
        """Test the essential properties of a physical axis."""
        axis = PhysicalAxis(size, period, ordering, domain)

        assert isinstance(axis, IntegerAxis)
        assert axis.size == size
        assert axis.period == period
        assert axis.ordering is ordering
        assert axis.domain is domain
        assert axis.is_fourier_domain == domain.is_in_fourier_domain
        np.testing.assert_array_equal(axis.index, IntegerAxis(size, ordering).index)
        np.testing.assert_array_equal(axis.values, axis.index * period)
        assert axis.values.dtype == np.float64
        assert axis.sampling_window_length == size * period
        assert type(axis).__name__ in repr(axis)

    def test_accepts_raw_values(self):
        """Test that ordering and domain can be given as raw enum values."""
        axis = PhysicalAxis(4, 1.0, "centered", "momentum")  # type: ignore
        assert axis.ordering is IndexOrdering.CENTERED
        assert axis.domain is AxisDomain.MOMENTUM

    @pytest.mark.parametrize("size", [0, -1])
    def test_invalid_size(self, size: int):
        """Test that non-positive sizes are rejected."""
        with pytest.raises(ValueError, match="size"):
            PhysicalAxis(size, 1.0, IndexOrdering.NATURAL, AxisDomain.POSITION)

    @pytest.mark.parametrize("period", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_period(self, period: float):
        """Test that non-positive or non-finite periods are rejected."""
        with pytest.raises(ValueError, match="period"):
            PhysicalAxis(4, period, IndexOrdering.NATURAL, AxisDomain.POSITION)

    def test_invalid_domain(self):
        """Test that unknown domains are rejected."""
        with pytest.raises(ValueError):
            PhysicalAxis(4, 1.0, IndexOrdering.NATURAL, "unknown")  # type: ignore


class TestReciprocalPeriod:
    """Test the reciprocal_period function."""

    @given(size=sizes, delta_x=periods, hbar=periods)
    def test_formulas(self, size: int, delta_x: float, hbar: float):
        """Test the periods of the conjugate domains."""
        window = size * delta_x
        assert np.isclose(
            reciprocal_period(size, delta_x, AxisDomain.MOMENTUM, hbar=hbar),
            2 * np.pi * hbar / window,
        )
        assert np.isclose(
            reciprocal_period(size, delta_x, AxisDomain.ANGULAR_WAVENUMBER),
            2 * np.pi / window,
        )
        assert np.isclose(
            reciprocal_period(size, delta_x, AxisDomain.SPATIAL_FREQUENCY),
            1 / window,
        )

    def test_hbar_is_required_for_momentum(self):
        """Test that the momentum period cannot be computed without hbar."""
        with pytest.raises(ValueError, match="hbar"):
            reciprocal_period(4, 1.0, AxisDomain.MOMENTUM)

    @pytest.mark.parametrize(
        "domain", [AxisDomain.ANGULAR_WAVENUMBER, AxisDomain.SPATIAL_FREQUENCY]
    )
    def test_hbar_is_rejected_for_other_domains(self, domain: AxisDomain):
        """Test that hbar is not silently ignored for the other domains."""
        with pytest.raises(ValueError, match="hbar"):
            reciprocal_period(4, 1.0, domain, hbar=1.0)

    def test_hbar_is_keyword_only(self):
        """Test that hbar cannot be passed positionally."""
        with pytest.raises(TypeError):
            reciprocal_period(4, 1.0, AxisDomain.MOMENTUM, 1.0)  # type: ignore[misc]

    def test_position_domain_is_rejected(self):
        """Test that the position domain has no reciprocal period."""
        with pytest.raises(ValueError, match="Fourier"):
            reciprocal_period(4, 1.0, AxisDomain.POSITION)


class TestPositionAxis:
    """Test the PositionAxis."""

    @given(size=sizes, delta_x=periods, ordering=orderings)
    def test_essentials(self, size: int, delta_x: float, ordering: IndexOrdering):
        """Test the essential properties of a position axis."""
        axis = PositionAxis(size, delta_x, ordering)
        assert isinstance(axis, PhysicalAxis)
        assert axis.domain is AxisDomain.POSITION
        assert not axis.is_fourier_domain
        assert axis.period == delta_x
        assert axis.ordering is ordering


class TestFourierConjugateAxes:
    """Test the axes conjugate to a position axis."""

    @given(size=sizes, period=periods, ordering=orderings)
    def test_constructors_take_their_own_period(
        self, size: int, period: float, ordering: IndexOrdering
    ):
        """Test that the constructors take the spacing of the conjugate domain."""
        axes = [
            (
                MomentumAxis(size, delta_p=period, ordering=ordering),
                AxisDomain.MOMENTUM,
            ),
            (
                AngularWavenumberAxis(size, delta_k=period, ordering=ordering),
                AxisDomain.ANGULAR_WAVENUMBER,
            ),
            (
                SpatialFrequencyAxis(size, delta_f=period, ordering=ordering),
                AxisDomain.SPATIAL_FREQUENCY,
            ),
        ]
        for axis, domain in axes:
            assert isinstance(axis, PhysicalAxis)
            assert axis.period == period
            assert axis.size == size
            assert axis.ordering is ordering
            assert axis.domain is domain
            assert axis.is_fourier_domain

    def test_hbar_is_required(self):
        """Test that the momentum axis of a position axis needs hbar."""
        x_axis = PositionAxis(4, 1.0, IndexOrdering.NATURAL)
        with pytest.raises(TypeError):
            MomentumAxis.from_position_axis(x_axis)  # type: ignore[call-arg]

    @given(size=sizes, delta_x=periods)
    def test_spatial_frequency_matches_fftfreq(self, size: int, delta_x: float):
        """Test that the spatial frequencies are the ones of numpy.fft.fftfreq."""
        x_axis = PositionAxis(size, delta_x, IndexOrdering.CENTERED)
        f_axis = SpatialFrequencyAxis.from_position_axis(x_axis)

        assert f_axis.domain is AxisDomain.SPATIAL_FREQUENCY
        assert f_axis.ordering is IndexOrdering.FFT
        np.testing.assert_allclose(f_axis.values, np.fft.fftfreq(size, d=delta_x))

    @given(size=sizes, delta_x=periods, hbar=periods)
    def test_momentum_and_angular_wavenumber(
        self, size: int, delta_x: float, hbar: float
    ):
        """Test that p = hbar * k = 2 pi hbar * f sample by sample."""
        x_axis = PositionAxis(size, delta_x, IndexOrdering.NATURAL)
        f = SpatialFrequencyAxis.from_position_axis(x_axis).values
        k_axis = AngularWavenumberAxis.from_position_axis(x_axis)
        p_axis = MomentumAxis.from_position_axis(x_axis, hbar=hbar)

        assert k_axis.domain is AxisDomain.ANGULAR_WAVENUMBER
        assert p_axis.domain is AxisDomain.MOMENTUM
        np.testing.assert_allclose(k_axis.values, 2 * np.pi * f)
        np.testing.assert_allclose(p_axis.values, hbar * k_axis.values)

    @given(size=sizes, delta_x=periods)
    def test_uncertainty_relation_of_the_grids(self, size: int, delta_x: float):
        """Test that the grid spacings satisfy delta_x * delta_k * N = 2 pi."""
        x_axis = PositionAxis(size, delta_x, IndexOrdering.NATURAL)
        k_axis = AngularWavenumberAxis.from_position_axis(x_axis)
        assert np.isclose(delta_x * k_axis.period * size, 2 * np.pi)

    @given(size=sizes, delta_x=periods, ordering=orderings, hbar=periods)
    def test_keep_ordering(
        self, size: int, delta_x: float, ordering: IndexOrdering, hbar: float
    ):
        """Test that from_position_axis keeps the ordering only if asked to."""
        x_axis = PositionAxis(size, delta_x, ordering)
        kept = [
            MomentumAxis.from_position_axis(x_axis, hbar, keep_ordering=True),
            AngularWavenumberAxis.from_position_axis(x_axis, keep_ordering=True),
            SpatialFrequencyAxis.from_position_axis(x_axis, keep_ordering=True),
        ]
        default = [
            MomentumAxis.from_position_axis(x_axis, hbar),
            AngularWavenumberAxis.from_position_axis(x_axis),
            SpatialFrequencyAxis.from_position_axis(x_axis),
        ]
        for axis in kept:
            assert axis.ordering is ordering
            assert axis.size == size
        for axis in default:
            assert axis.ordering is IndexOrdering.FFT

    @given(size=sizes, delta_x=periods)
    def test_dft_of_a_plane_wave_peaks_at_its_frequency(
        self, size: int, delta_x: float
    ):
        """Test that the conjugate axis locates the peak of a DFT spectrum."""
        x_axis = PositionAxis(size, delta_x, IndexOrdering.NATURAL)
        f_axis = SpatialFrequencyAxis.from_position_axis(x_axis)
        m = size // 3  # any frequency on the grid
        f0 = f_axis.values[m]

        spectrum = np.fft.fft(np.exp(2j * np.pi * f0 * x_axis.values))
        assert np.argmax(np.abs(spectrum)) == m
