"""Tests for ADC and digitization utilities."""

import numpy as np
import pytest

from tambo_sipm.readout.adc import (
    adc_code_to_voltage,
    adc_lsb_mV,
    digitize_voltage,
    has_adc_saturation,
    sample_and_digitize_waveform,
    sample_waveform,
    voltage_to_adc_code,
)


def test_adc_lsb_matches_red_pitaya_model():
    lsb_mV = adc_lsb_mV(
        adc_bits=14,
        v_min_mV=-1000.0,
        v_max_mV=1000.0,
    )

    assert np.isclose(lsb_mV, 0.1220703125)


def test_adc_lsb_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        adc_lsb_mV(adc_bits=0)

    with pytest.raises(ValueError):
        adc_lsb_mV(v_min_mV=1000.0, v_max_mV=-1000.0)


def test_voltage_to_adc_code_maps_key_values():
    voltage_mV = np.array([-1000.0, 0.0, 1000.0])

    adc_code = voltage_to_adc_code(voltage_mV)

    assert adc_code[0] == 0
    assert adc_code[1] == 8192
    assert adc_code[2] == 16383


def test_voltage_to_adc_code_clips_out_of_range_values():
    voltage_mV = np.array([-2000.0, 2000.0])

    adc_code = voltage_to_adc_code(voltage_mV)

    assert adc_code[0] == 0
    assert adc_code[1] == 16383


def test_adc_code_to_voltage_maps_key_values():
    adc_code = np.array([0, 8192])

    voltage_mV = adc_code_to_voltage(adc_code)

    assert np.isclose(voltage_mV[0], -1000.0)
    assert np.isclose(voltage_mV[1], 0.0)


def test_adc_code_to_voltage_rejects_invalid_codes():
    adc_code = np.array([-1, 0, 16384])

    with pytest.raises(ValueError):
        adc_code_to_voltage(adc_code)


def test_digitize_voltage_preserves_shape():
    voltage_mV = np.linspace(-50.0, 50.0, 100)

    adc_code, quantized_voltage_mV = digitize_voltage(voltage_mV)

    assert adc_code.shape == voltage_mV.shape
    assert quantized_voltage_mV.shape == voltage_mV.shape


def test_sample_waveform_uses_expected_sampling_interval():
    time_ns = np.arange(0.0, 100.0, 0.5)
    voltage_mV = np.sin(time_ns)

    sampled_time_ns, sampled_voltage_mV = sample_waveform(
        time_ns=time_ns,
        voltage_mV=voltage_mV,
        sampling_interval_ns=8.0,
    )

    assert np.isclose(sampled_time_ns[1] - sampled_time_ns[0], 8.0)
    assert sampled_time_ns.shape == sampled_voltage_mV.shape


def test_sample_waveform_rejects_invalid_inputs():
    time_ns = np.array([0.0, 1.0, 0.5])
    voltage_mV = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError):
        sample_waveform(time_ns, voltage_mV)

    with pytest.raises(ValueError):
        sample_waveform(
            np.array([0.0, 1.0]),
            np.array([0.0]),
        )


def test_sample_and_digitize_waveform_returns_consistent_shapes():
    time_ns = np.arange(0.0, 100.0, 0.5)
    voltage_mV = 50.0 * np.exp(-time_ns / 30.0)

    sampled_time_ns, adc_code, quantized_voltage_mV = sample_and_digitize_waveform(
        time_ns=time_ns,
        voltage_mV=voltage_mV,
        sampling_interval_ns=8.0,
    )

    assert sampled_time_ns.shape == adc_code.shape
    assert sampled_time_ns.shape == quantized_voltage_mV.shape


def test_has_adc_saturation_detects_saturation():
    saturated_code = np.array([100, 8192, 16383])
    nonsaturated_code = np.array([100, 8192, 16000])

    assert has_adc_saturation(saturated_code)
    assert not has_adc_saturation(nonsaturated_code)