"""Tests for SiPM pulse-shape functions."""

import numpy as np
import pytest

from tambo_sipm.sipm.pulse import (
    estimate_fwhm_ns,
    make_time_axis,
    normalized_sipm_pulse,
    raw_sipm_pulse,
    sum_pulses,
)


def test_make_time_axis_has_expected_spacing():
    time_ns = make_time_axis(window_ns=100.0, dt_ns=0.5)

    assert time_ns[0] == 0.0
    assert np.isclose(time_ns[1] - time_ns[0], 0.5)
    assert time_ns[-1] < 100.0


def test_make_time_axis_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        make_time_axis(window_ns=0.0, dt_ns=0.5)

    with pytest.raises(ValueError):
        make_time_axis(window_ns=100.0, dt_ns=0.0)


def test_raw_pulse_is_zero_before_t0():
    time_ns = make_time_axis(window_ns=100.0, dt_ns=1.0)

    pulse = raw_sipm_pulse(
        time_ns=time_ns,
        tau_r_ns=2.0,
        tau_f_ns=95.0,
        amplitude_pe=1.0,
        t0_ns=20.0,
    )

    assert np.all(pulse[time_ns < 20.0] == 0.0)


def test_normalized_pulse_peak_matches_amplitude():
    time_ns = make_time_axis(window_ns=1000.0, dt_ns=0.5)

    pulse = normalized_sipm_pulse(
        time_ns=time_ns,
        tau_r_ns=2.0,
        tau_f_ns=95.0,
        amplitude_pe=3.0,
        t0_ns=10.0,
    )

    assert np.isclose(np.max(pulse), 3.0)


def test_sum_pulses_adds_multiple_arrivals():
    time_ns = make_time_axis(window_ns=1000.0, dt_ns=0.5)
    arrival_times_ns = np.array([10.0, 20.0, 30.0])
    amplitudes_pe = np.array([1.0, 2.0, 3.0])

    waveform = sum_pulses(
        time_ns=time_ns,
        arrival_times_ns=arrival_times_ns,
        amplitudes_pe=amplitudes_pe,
        tau_r_ns=2.0,
        tau_f_ns=95.0,
    )

    assert waveform.shape == time_ns.shape
    assert np.max(waveform) > 3.0


def test_sum_pulses_rejects_shape_mismatch():
    time_ns = make_time_axis(window_ns=100.0, dt_ns=1.0)
    arrival_times_ns = np.array([10.0, 20.0])
    amplitudes_pe = np.array([1.0])

    with pytest.raises(ValueError):
        sum_pulses(
            time_ns=time_ns,
            arrival_times_ns=arrival_times_ns,
            amplitudes_pe=amplitudes_pe,
            tau_r_ns=2.0,
            tau_f_ns=95.0,
        )


def test_estimate_fwhm_returns_positive_width():
    time_ns = make_time_axis(window_ns=1000.0, dt_ns=0.5)

    pulse = normalized_sipm_pulse(
        time_ns=time_ns,
        tau_r_ns=2.0,
        tau_f_ns=95.0,
        amplitude_pe=1.0,
        t0_ns=10.0,
    )

    fwhm_ns = estimate_fwhm_ns(time_ns, pulse)

    assert fwhm_ns > 0.0