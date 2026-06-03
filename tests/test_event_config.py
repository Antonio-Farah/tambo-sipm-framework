"""Tests for config-driven detector event simulation."""

import numpy as np
import pytest

from tambo_sipm import load_simulation_config
from tambo_sipm.detector import simulate_detector_event_from_config


def test_simulate_detector_event_from_config_runs_default_config():
    rng = np.random.default_rng(123)
    config = load_simulation_config()

    result = simulate_detector_event_from_config(
        generated_photons=1300,
        config=config,
        rng=rng,
    )

    assert result.photon_transport.generated_photons[0] == 1300
    assert result.sampled_time_ns.shape == result.adc_code.shape
    assert result.sampled_time_ns.shape == result.digitized_voltage_mV.shape


def test_simulate_detector_event_from_config_is_reproducible():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    config = load_simulation_config()

    result_a = simulate_detector_event_from_config(
        generated_photons=1300,
        config=config,
        rng=rng_a,
    )

    result_b = simulate_detector_event_from_config(
        generated_photons=1300,
        config=config,
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


def test_simulate_detector_event_from_config_rejects_missing_section():
    rng = np.random.default_rng(123)
    config = load_simulation_config()
    config.pop("readout")

    with pytest.raises(KeyError):
        simulate_detector_event_from_config(
            generated_photons=1300,
            config=config,
            rng=rng,
        )


def test_simulate_detector_event_from_config_uses_voltage_scale():
    rng = np.random.default_rng(123)
    config = load_simulation_config()
    config["voltage_conversion"]["voltage_scale_mV_per_pe"] = 0.0

    result = simulate_detector_event_from_config(
        generated_photons=1300,
        config=config,
        rng=rng,
    )

    assert np.allclose(result.analog_voltage_mV, 0.0)
    assert np.allclose(result.digitized_voltage_mV, 0.0)