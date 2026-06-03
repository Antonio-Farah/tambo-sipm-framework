"""Tests for pulse feature extraction utilities."""

import numpy as np
import pytest

from tambo_sipm.analysis.features import (
    estimate_baseline,
    estimate_noise_rms,
    extract_pulse_features,
    find_threshold_indices,
    integrate_pulse,
    pulse_width_ns,
    subtract_baseline,
    validate_waveform_inputs,
    waveform_rms,
)


def test_validate_waveform_inputs_accepts_valid_waveform():
    time_ns = np.array([0.0, 8.0, 16.0])
    voltage_mV = np.array([0.0, 1.0, 2.0])

    validate_waveform_inputs(time_ns, voltage_mV)


def test_validate_waveform_inputs_rejects_shape_mismatch():
    time_ns = np.array([0.0, 8.0])
    voltage_mV = np.array([0.0])

    with pytest.raises(ValueError):
        validate_waveform_inputs(time_ns, voltage_mV)


def test_validate_waveform_inputs_rejects_non_increasing_time():
    time_ns = np.array([0.0, 8.0, 4.0])
    voltage_mV = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError):
        validate_waveform_inputs(time_ns, voltage_mV)


def test_estimate_baseline_with_median():
    voltage_mV = np.array([10.0, 11.0, 12.0, 50.0])

    baseline_mV = estimate_baseline(
        voltage_mV=voltage_mV,
        n_pretrigger_samples=3,
        method="median",
    )

    assert np.isclose(baseline_mV, 11.0)


def test_estimate_baseline_with_mean():
    voltage_mV = np.array([10.0, 11.0, 12.0, 50.0])

    baseline_mV = estimate_baseline(
        voltage_mV=voltage_mV,
        n_pretrigger_samples=3,
        method="mean",
    )

    assert np.isclose(baseline_mV, 11.0)


def test_estimate_baseline_rejects_invalid_method():
    voltage_mV = np.array([10.0, 11.0, 12.0])

    with pytest.raises(ValueError):
        estimate_baseline(voltage_mV, method="mode")


def test_estimate_noise_rms():
    voltage_mV = np.array([9.0, 10.0, 11.0, 50.0])
    baseline_mV = 10.0

    noise_rms_mV = estimate_noise_rms(
        voltage_mV=voltage_mV,
        baseline_mV=baseline_mV,
        n_pretrigger_samples=3,
    )

    expected = np.sqrt(np.mean(np.array([-1.0, 0.0, 1.0]) ** 2))

    assert np.isclose(noise_rms_mV, expected)


def test_subtract_baseline():
    voltage_mV = np.array([10.0, 20.0, 30.0])

    corrected = subtract_baseline(voltage_mV, baseline_mV=10.0)

    assert np.allclose(corrected, np.array([0.0, 10.0, 20.0]))


def test_find_threshold_indices():
    voltage_mV = np.array([0.0, 5.0, 10.0, 4.0])

    indices = find_threshold_indices(voltage_mV, threshold_mV=5.0)

    assert np.array_equal(indices, np.array([1, 2]))


def test_pulse_width_ns_returns_expected_width():
    time_ns = np.array([0.0, 8.0, 16.0, 24.0])
    indices = np.array([1, 2, 3])

    width_ns = pulse_width_ns(time_ns, indices)

    assert np.isclose(width_ns, 16.0)


def test_pulse_width_ns_returns_zero_for_single_sample():
    time_ns = np.array([0.0, 8.0, 16.0])
    indices = np.array([1])

    width_ns = pulse_width_ns(time_ns, indices)

    assert np.isclose(width_ns, 0.0)


def test_integrate_pulse_returns_expected_integral():
    time_ns = np.array([0.0, 8.0, 16.0, 24.0])
    voltage_mV = np.array([0.0, 10.0, 10.0, 0.0])
    indices = np.array([1, 2])

    integral_mVns = integrate_pulse(time_ns, voltage_mV, indices)

    assert np.isclose(integral_mVns, 80.0)


def test_waveform_rms_full_waveform():
    voltage_mV = np.array([3.0, 4.0])

    rms_mV = waveform_rms(voltage_mV)

    assert np.isclose(rms_mV, np.sqrt(12.5))


def test_waveform_rms_restricted_indices():
    voltage_mV = np.array([0.0, 3.0, 4.0])
    indices = np.array([1, 2])

    rms_mV = waveform_rms(voltage_mV, indices)

    assert np.isclose(rms_mV, np.sqrt(12.5))


def test_extract_pulse_features_valid_pulse():
    time_ns = np.array([0.0, 8.0, 16.0, 24.0, 32.0, 40.0])
    voltage_mV = np.array([10.0, 10.0, 10.0, 30.0, 25.0, 12.0])

    features = extract_pulse_features(
        time_ns=time_ns,
        voltage_mV=voltage_mV,
        threshold_mV=5.0,
        n_pretrigger_samples=3,
    )

    assert features.valid
    assert np.isclose(features.baseline_mV, 10.0)
    assert np.isclose(features.peak_mV, 20.0)
    assert np.isclose(features.peak_time_ns, 24.0)
    assert np.isclose(features.width_ns, 8.0)
    assert features.first_crossing_time_ns == 24.0
    assert features.last_crossing_time_ns == 32.0
    assert features.n_samples_above_threshold == 2


def test_extract_pulse_features_invalid_pulse_below_threshold():
    time_ns = np.array([0.0, 8.0, 16.0, 24.0, 32.0])
    voltage_mV = np.array([10.0, 10.0, 10.0, 12.0, 11.0])

    features = extract_pulse_features(
        time_ns=time_ns,
        voltage_mV=voltage_mV,
        threshold_mV=5.0,
        n_pretrigger_samples=3,
    )

    assert not features.valid
    assert features.width_ns == 0.0
    assert features.integral_mVns == 0.0
    assert features.first_crossing_time_ns is None
    assert features.last_crossing_time_ns is None