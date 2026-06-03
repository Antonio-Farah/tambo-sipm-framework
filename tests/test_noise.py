"""Tests for SiPM noise models."""

import numpy as np
import pytest

from tambo_sipm.sipm.noise import (
    afterpulse_amplitude_factor,
    apply_crosstalk_to_amplitudes,
    generate_afterpulses,
    generate_dark_noise,
    sample_crosstalk_multiplicity,
    sample_dcr_times,
    sample_gain_amplitudes,
    validate_probability,
    validate_rate_hz,
)


def test_validate_probability_accepts_valid_values():
    validate_probability(0.0, "p")
    validate_probability(0.5, "p")
    validate_probability(1.0, "p")


def test_validate_probability_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_probability(-0.1, "p")

    with pytest.raises(ValueError):
        validate_probability(1.1, "p")


def test_validate_rate_accepts_non_negative_values():
    validate_rate_hz(0.0)
    validate_rate_hz(1.0)


def test_validate_rate_rejects_negative_values():
    with pytest.raises(ValueError):
        validate_rate_hz(-1.0)


def test_sample_dcr_times_returns_empty_for_zero_dcr():
    rng = np.random.default_rng(123)

    times_ns = sample_dcr_times(
        window_ns=1000.0,
        dcr_hz=0.0,
        rng=rng,
    )

    assert len(times_ns) == 0


def test_sample_dcr_times_are_inside_window_and_sorted():
    rng = np.random.default_rng(123)

    times_ns = sample_dcr_times(
        window_ns=1000.0,
        dcr_hz=1.0e9,
        rng=rng,
    )

    assert np.all(times_ns >= 0.0)
    assert np.all(times_ns < 1000.0)
    assert np.all(np.diff(times_ns) >= 0.0)


def test_sample_dcr_times_is_reproducible():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    times_a = sample_dcr_times(
        window_ns=1000.0,
        dcr_hz=1.0e9,
        rng=rng_a,
    )

    times_b = sample_dcr_times(
        window_ns=1000.0,
        dcr_hz=1.0e9,
        rng=rng_b,
    )

    assert np.array_equal(times_a, times_b)


def test_sample_gain_amplitudes_shape_and_non_negative():
    rng = np.random.default_rng(123)

    amplitudes = sample_gain_amplitudes(
        n_pulses=100,
        mean_pe=1.0,
        sigma_pe=0.1,
        rng=rng,
    )

    assert amplitudes.shape == (100,)
    assert np.all(amplitudes >= 0.0)


def test_sample_crosstalk_multiplicity_no_crosstalk():
    rng = np.random.default_rng(123)

    multiplicity = sample_crosstalk_multiplicity(
        n_primary=10,
        crosstalk_prob=0.0,
        rng=rng,
    )

    assert np.array_equal(multiplicity, np.ones(10, dtype=np.int64))


def test_sample_crosstalk_multiplicity_bounds():
    rng = np.random.default_rng(123)

    multiplicity = sample_crosstalk_multiplicity(
        n_primary=100,
        crosstalk_prob=0.5,
        rng=rng,
        max_multiplicity=5,
    )

    assert np.all(multiplicity >= 1)
    assert np.all(multiplicity <= 5)


def test_apply_crosstalk_to_amplitudes_no_crosstalk():
    rng = np.random.default_rng(123)
    amplitudes = np.array([1.0, 2.0, 3.0])

    result = apply_crosstalk_to_amplitudes(
        amplitudes_pe=amplitudes,
        crosstalk_prob=0.0,
        rng=rng,
    )

    assert np.allclose(result, amplitudes)


def test_afterpulse_amplitude_factor():
    delays_ns = np.array([0.0, 95.0])
    factors = afterpulse_amplitude_factor(
        delay_ns=delays_ns,
        tau_recovery_ns=95.0,
    )

    assert np.isclose(factors[0], 0.0)
    assert np.isclose(factors[1], 1.0 - np.exp(-1.0))


def test_generate_afterpulses_returns_empty_when_probability_zero():
    rng = np.random.default_rng(123)
    primary_times_ns = np.array([10.0, 20.0])
    primary_amplitudes_pe = np.array([1.0, 1.0])

    result = generate_afterpulses(
        primary_times_ns=primary_times_ns,
        primary_amplitudes_pe=primary_amplitudes_pe,
        afterpulse_prob=0.0,
        tau_afterpulse_ns=30.0,
        tau_recovery_ns=95.0,
        window_ns=100.0,
        rng=rng,
    )

    assert len(result.times_ns) == 0
    assert len(result.amplitudes_pe) == 0
    assert result.sources == ()


def test_generate_afterpulses_probability_one_creates_afterpulses_inside_window():
    rng = np.random.default_rng(123)
    primary_times_ns = np.array([10.0, 20.0, 30.0])
    primary_amplitudes_pe = np.array([1.0, 1.0, 1.0])

    result = generate_afterpulses(
        primary_times_ns=primary_times_ns,
        primary_amplitudes_pe=primary_amplitudes_pe,
        afterpulse_prob=1.0,
        tau_afterpulse_ns=5.0,
        tau_recovery_ns=95.0,
        window_ns=100.0,
        rng=rng,
    )

    assert len(result.times_ns) > 0
    assert np.all(result.times_ns < 100.0)
    assert np.all(result.amplitudes_pe >= 0.0)
    assert all(source == "afterpulse" for source in result.sources)


def test_generate_dark_noise_returns_sorted_pulses():
    rng = np.random.default_rng(123)

    result = generate_dark_noise(
        window_ns=1000.0,
        dcr_hz=1.0e9,
        crosstalk_prob=0.1,
        afterpulse_prob=0.1,
        rng=rng,
    )

    assert len(result.times_ns) == len(result.amplitudes_pe)
    assert len(result.times_ns) == len(result.sources)
    assert np.all(result.times_ns >= 0.0)
    assert np.all(result.times_ns < 1000.0)
    assert np.all(np.diff(result.times_ns) >= 0.0)


def test_generate_dark_noise_zero_dcr_returns_empty_result():
    rng = np.random.default_rng(123)

    result = generate_dark_noise(
        window_ns=1000.0,
        dcr_hz=0.0,
        rng=rng,
    )

    assert len(result.times_ns) == 0
    assert len(result.amplitudes_pe) == 0
    assert result.sources == ()