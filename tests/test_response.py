"""Tests for complete SiPM response simulation."""

import numpy as np
import pytest

from tambo_sipm.sipm.response import (
    sample_photoelectron_arrival_times,
    simulate_sipm_response,
    validate_photoelectron_count,
    validate_response_times,
)


def test_validate_photoelectron_count_accepts_non_negative_integer():
    validate_photoelectron_count(0)
    validate_photoelectron_count(10)


def test_validate_photoelectron_count_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_photoelectron_count(-1)

    with pytest.raises(ValueError):
        validate_photoelectron_count(1.5)


def test_validate_response_times_accepts_valid_values():
    validate_response_times(
        window_ns=1000.0,
        dt_ns=0.5,
        event_time_ns=10.0,
        arrival_spread_ns=0.0,
    )


def test_validate_response_times_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_response_times(
            window_ns=0.0,
            dt_ns=0.5,
            event_time_ns=10.0,
            arrival_spread_ns=0.0,
        )

    with pytest.raises(ValueError):
        validate_response_times(
            window_ns=1000.0,
            dt_ns=0.0,
            event_time_ns=10.0,
            arrival_spread_ns=0.0,
        )

    with pytest.raises(ValueError):
        validate_response_times(
            window_ns=1000.0,
            dt_ns=0.5,
            event_time_ns=1000.0,
            arrival_spread_ns=0.0,
        )

    with pytest.raises(ValueError):
        validate_response_times(
            window_ns=1000.0,
            dt_ns=0.5,
            event_time_ns=10.0,
            arrival_spread_ns=-1.0,
        )


def test_sample_photoelectron_arrival_times_no_spread():
    rng = np.random.default_rng(123)

    arrival_times = sample_photoelectron_arrival_times(
        n_pulses=5,
        event_time_ns=10.0,
        arrival_spread_ns=0.0,
        window_ns=100.0,
        rng=rng,
    )

    assert np.allclose(arrival_times, np.full(5, 10.0))


def test_sample_photoelectron_arrival_times_with_spread_stays_inside_window():
    rng = np.random.default_rng(123)

    arrival_times = sample_photoelectron_arrival_times(
        n_pulses=100,
        event_time_ns=10.0,
        arrival_spread_ns=5.0,
        window_ns=100.0,
        rng=rng,
    )

    assert arrival_times.shape == (100,)
    assert np.all(arrival_times >= 0.0)
    assert np.all(arrival_times < 100.0)


def test_simulate_sipm_response_zero_photoelectrons_without_noise():
    rng = np.random.default_rng(123)

    result = simulate_sipm_response(
        photoelectrons=0,
        window_ns=100.0,
        dt_ns=1.0,
        tau_r_ns=2.0,
        tau_f_ns=95.0,
        event_time_ns=10.0,
        include_dark_noise=False,
        rng=rng,
    )

    assert result.input_photoelectrons == 0
    assert result.fired_microcells == 0
    assert result.time_ns.shape == result.total_signal_pe.shape
    assert np.allclose(result.photon_signal_pe, 0.0)
    assert np.allclose(result.dark_noise_pe, 0.0)
    assert np.allclose(result.total_signal_pe, 0.0)


def test_simulate_sipm_response_signal_only():
    rng = np.random.default_rng(123)

    result = simulate_sipm_response(
        photoelectrons=100,
        window_ns=1000.0,
        dt_ns=0.5,
        tau_r_ns=2.0,
        tau_f_ns=95.0,
        event_time_ns=10.0,
        include_dark_noise=False,
        rng=rng,
    )

    assert result.input_photoelectrons == 100
    assert result.fired_microcells > 0
    assert result.time_ns.shape == result.total_signal_pe.shape
    assert np.max(result.photon_signal_pe) > 0.0
    assert np.allclose(result.dark_noise_pe, 0.0)
    assert np.allclose(result.total_signal_pe, result.photon_signal_pe)


def test_simulate_sipm_response_with_dark_noise():
    rng = np.random.default_rng(123)

    result = simulate_sipm_response(
        photoelectrons=0,
        window_ns=1000.0,
        dt_ns=0.5,
        tau_r_ns=2.0,
        tau_f_ns=95.0,
        event_time_ns=10.0,
        include_dark_noise=True,
        dcr_hz=1.0e9,
        rng=rng,
    )

    assert result.input_photoelectrons == 0
    assert result.time_ns.shape == result.total_signal_pe.shape
    assert np.max(result.dark_noise_pe) > 0.0
    assert np.allclose(result.total_signal_pe, result.dark_noise_pe)


def test_simulate_sipm_response_is_reproducible():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result_a = simulate_sipm_response(
        photoelectrons=100,
        window_ns=1000.0,
        dt_ns=0.5,
        tau_r_ns=2.0,
        tau_f_ns=95.0,
        event_time_ns=10.0,
        arrival_spread_ns=2.0,
        include_dark_noise=True,
        dcr_hz=1.0e8,
        crosstalk_prob=0.1,
        afterpulse_prob=0.1,
        rng=rng_a,
    )

    result_b = simulate_sipm_response(
        photoelectrons=100,
        window_ns=1000.0,
        dt_ns=0.5,
        tau_r_ns=2.0,
        tau_f_ns=95.0,
        event_time_ns=10.0,
        arrival_spread_ns=2.0,
        include_dark_noise=True,
        dcr_hz=1.0e8,
        crosstalk_prob=0.1,
        afterpulse_prob=0.1,
        rng=rng_b,
    )

    assert np.array_equal(result_a.time_ns, result_b.time_ns)
    assert np.allclose(result_a.total_signal_pe, result_b.total_signal_pe)
    assert np.array_equal(
        result_a.photon_arrival_times_ns,
        result_b.photon_arrival_times_ns,
    )