"""Tests for complete detector event simulation."""

import numpy as np
import pytest

from tambo_sipm.detector.event import (
    convert_pe_to_mV,
    simulate_detector_event,
    validate_generated_photons,
    validate_voltage_scale,
)


def test_validate_generated_photons_accepts_non_negative_integer():
    validate_generated_photons(0)
    validate_generated_photons(100)


def test_validate_generated_photons_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_generated_photons(-1)

    with pytest.raises(ValueError):
        validate_generated_photons(1.5)


def test_validate_voltage_scale_accepts_non_negative_values():
    validate_voltage_scale(0.0)
    validate_voltage_scale(0.1)


def test_validate_voltage_scale_rejects_negative_values():
    with pytest.raises(ValueError):
        validate_voltage_scale(-0.1)


def test_convert_pe_to_mV():
    signal_pe = np.array([0.0, 1.0, 2.0])

    voltage_mV = convert_pe_to_mV(
        signal_pe=signal_pe,
        voltage_scale_mV_per_pe=0.5,
    )

    assert np.allclose(voltage_mV, np.array([0.0, 0.5, 1.0]))


def test_simulate_detector_event_zero_photons_without_noise():
    rng = np.random.default_rng(123)

    result = simulate_detector_event(
        generated_photons=0,
        transport_efficiency=1.0,
        pde=1.0,
        voltage_scale_mV_per_pe=1.0,
        window_ns=100.0,
        dt_ns=1.0,
        tau_r_ns=2.0,
        tau_f_ns=95.0,
        event_time_ns=10.0,
        include_dark_noise=False,
        rng=rng,
    )

    assert result.photon_transport.generated_photons[0] == 0
    assert result.photon_transport.photoelectrons[0] == 0
    assert result.sipm_response.fired_microcells == 0
    assert np.allclose(result.analog_voltage_mV, 0.0)
    assert np.allclose(result.digitized_voltage_mV, 0.0)


def test_simulate_detector_event_signal_chain():
    rng = np.random.default_rng(123)

    result = simulate_detector_event(
        generated_photons=1000,
        transport_efficiency=1.0,
        pde=1.0,
        voltage_scale_mV_per_pe=0.1,
        window_ns=1000.0,
        dt_ns=0.5,
        tau_r_ns=2.0,
        tau_f_ns=95.0,
        event_time_ns=10.0,
        include_dark_noise=False,
        rng=rng,
    )

    assert result.photon_transport.generated_photons[0] == 1000
    assert result.photon_transport.photons_at_sipm[0] == 1000
    assert result.photon_transport.photoelectrons[0] == 1000
    assert result.sipm_response.fired_microcells > 0
    assert np.max(result.analog_voltage_mV) > 0.0
    assert result.sampled_time_ns.shape == result.adc_code.shape
    assert result.sampled_time_ns.shape == result.digitized_voltage_mV.shape


def test_simulate_detector_event_zero_voltage_scale():
    rng = np.random.default_rng(123)

    result = simulate_detector_event(
        generated_photons=1000,
        transport_efficiency=1.0,
        pde=1.0,
        voltage_scale_mV_per_pe=0.0,
        window_ns=1000.0,
        dt_ns=0.5,
        tau_r_ns=2.0,
        tau_f_ns=95.0,
        event_time_ns=10.0,
        include_dark_noise=False,
        rng=rng,
    )

    assert np.allclose(result.analog_voltage_mV, 0.0)
    assert np.allclose(result.digitized_voltage_mV, 0.0)


def test_simulate_detector_event_with_dark_noise():
    rng = np.random.default_rng(123)

    result = simulate_detector_event(
        generated_photons=0,
        transport_efficiency=1.0,
        pde=1.0,
        voltage_scale_mV_per_pe=1.0,
        window_ns=1000.0,
        dt_ns=0.5,
        tau_r_ns=2.0,
        tau_f_ns=95.0,
        event_time_ns=10.0,
        include_dark_noise=True,
        dcr_hz=1.0e9,
        rng=rng,
    )

    assert result.photon_transport.photoelectrons[0] == 0
    assert np.max(result.analog_voltage_mV) > 0.0
    assert np.max(result.digitized_voltage_mV) > 0.0


def test_simulate_detector_event_is_reproducible():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result_a = simulate_detector_event(
        generated_photons=1300,
        transport_efficiency=0.2,
        pde=0.41,
        voltage_scale_mV_per_pe=0.1,
        window_ns=1000.0,
        dt_ns=0.5,
        tau_r_ns=2.0,
        tau_f_ns=95.0,
        event_time_ns=10.0,
        arrival_spread_ns=2.0,
        crosstalk_prob=0.1,
        include_dark_noise=True,
        dcr_hz=1.0e8,
        rng=rng_a,
    )

    result_b = simulate_detector_event(
        generated_photons=1300,
        transport_efficiency=0.2,
        pde=0.41,
        voltage_scale_mV_per_pe=0.1,
        window_ns=1000.0,
        dt_ns=0.5,
        tau_r_ns=2.0,
        tau_f_ns=95.0,
        event_time_ns=10.0,
        arrival_spread_ns=2.0,
        crosstalk_prob=0.1,
        include_dark_noise=True,
        dcr_hz=1.0e8,
        rng=rng_b,
    )

    assert np.array_equal(
        result_a.photon_transport.photoelectrons,
        result_b.photon_transport.photoelectrons,
    )
    assert np.allclose(
        result_a.sipm_response.total_signal_pe,
        result_b.sipm_response.total_signal_pe,
    )
    assert np.array_equal(result_a.adc_code, result_b.adc_code)