"""Tests for detector simulation calibration utilities."""

import numpy as np
import pandas as pd
import pytest

from tambo_sipm.simulation.calibration import (
    build_calibration_config,
    calibration_grid_rows,
    calibration_score_from_distribution_table,
    feature_error_from_distribution_table,
    run_calibration_grid,
    run_single_calibration_point,
    set_nested_config_value,
    validate_calibration_grid_values,
)


def make_config() -> dict:
    return {
        "photon_transport": {
            "transport_efficiency": 0.2,
            "pde": 0.41,
        },
        "sipm_response": {
            "window_ns": 200.0,
            "dt_ns": 1.0,
            "tau_r_ns": 2.0,
            "tau_f_ns": 95.0,
            "event_time_ns": 16.0,
            "arrival_spread_ns": 2.0,
            "gain_mean_pe": 1.0,
            "gain_sigma_pe": 0.0,
            "crosstalk_prob": 0.0,
            "include_dark_noise": False,
            "dcr_hz": 0.0,
            "afterpulse_prob": 0.0,
            "tau_afterpulse_ns": 30.0,
            "tau_recovery_ns": 95.0,
        },
        "voltage_conversion": {
            "voltage_scale_mV_per_pe": 0.12,
        },
        "readout": {
            "sampling_interval_ns": 8.0,
            "adc_bits": 14,
            "v_min_mV": -1000.0,
            "v_max_mV": 1000.0,
        },
    }


def make_features(scale: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "valid": [True, True, True, True],
            "extraction_status": ["ok", "ok", "ok", "ok"],
            "peak_mV": [
                10.0 * scale,
                20.0 * scale,
                30.0 * scale,
                40.0 * scale,
            ],
            "rms_mV": [
                5.0 * scale,
                10.0 * scale,
                15.0 * scale,
                20.0 * scale,
            ],
            "integral_mVns": [
                100.0 * scale,
                200.0 * scale,
                300.0 * scale,
                400.0 * scale,
            ],
            "width_ns": [
                20.0 * scale,
                40.0 * scale,
                60.0 * scale,
                80.0 * scale,
            ],
        }
    )


def make_photon_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "generated_photons": [1000, 2000, 3000, 4000],
            "particle_name": ["gamma", "e-", "mu-", "gamma"],
            "particle_category": [
                "electromagnetic",
                "electromagnetic",
                "muonic",
                "electromagnetic",
            ],
        }
    )


def make_distribution_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["peak_mV", "rms_mV", "integral_mVns", "width_ns"],
            "mean_relative_difference": [0.1, -0.2, 0.3, -0.4],
        }
    )


def test_set_nested_config_value_does_not_mutate_original():
    config = make_config()

    updated = set_nested_config_value(
        config=config,
        path=("photon_transport", "transport_efficiency"),
        value=0.05,
    )

    assert config["photon_transport"]["transport_efficiency"] == 0.2
    assert updated["photon_transport"]["transport_efficiency"] == 0.05


def test_build_calibration_config_sets_expected_values():
    config = make_config()

    updated = build_calibration_config(
        base_config=config,
        transport_efficiency=0.05,
        tau_f_ns=220.0,
        arrival_spread_ns=20.0,
    )

    assert updated["photon_transport"]["transport_efficiency"] == 0.05
    assert updated["sipm_response"]["tau_f_ns"] == 220.0
    assert updated["sipm_response"]["arrival_spread_ns"] == 20.0
    assert updated["voltage_conversion"]["voltage_scale_mV_per_pe"] == 0.12
    assert updated["photon_transport"]["pde"] == 0.41


def test_validate_calibration_grid_values_accepts_valid_values():
    validate_calibration_grid_values(
        transport_efficiencies=[0.02, 0.05],
        tau_f_values_ns=[95.0, 220.0],
        arrival_spread_values_ns=[2.0, 20.0],
    )


def test_validate_calibration_grid_values_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_calibration_grid_values(
            transport_efficiencies=[],
            tau_f_values_ns=[95.0],
            arrival_spread_values_ns=[2.0],
        )

    with pytest.raises(ValueError):
        validate_calibration_grid_values(
            transport_efficiencies=[1.2],
            tau_f_values_ns=[95.0],
            arrival_spread_values_ns=[2.0],
        )


def test_calibration_grid_rows():
    rows = calibration_grid_rows(
        transport_efficiencies=[0.02, 0.05],
        tau_f_values_ns=[95.0],
        arrival_spread_values_ns=[2.0, 20.0],
    )

    assert len(rows) == 4
    assert rows[0]["transport_efficiency"] == 0.02
    assert rows[0]["tau_f_ns"] == 95.0
    assert rows[0]["arrival_spread_ns"] == 2.0


def test_feature_error_from_distribution_table():
    comparison = make_distribution_comparison()

    error = feature_error_from_distribution_table(
        distribution_comparison=comparison,
        feature="rms_mV",
    )

    assert np.isclose(error, 0.2)


def test_calibration_score_from_distribution_table():
    comparison = make_distribution_comparison()

    score = calibration_score_from_distribution_table(
        distribution_comparison=comparison,
    )

    assert np.isclose(score["peak_mV_error"], 0.1)
    assert np.isclose(score["rms_mV_error"], 0.2)
    assert np.isclose(score["integral_mVns_error"], 0.3)
    assert np.isclose(score["width_ns_error"], 0.4)
    assert np.isclose(score["feature_score"], 0.25)
    assert np.isclose(score["score"], 0.25)


def test_run_single_calibration_point():
    config = make_config()
    real_features = make_features(scale=1.0)
    photon_events = make_photon_events()

    result = run_single_calibration_point(
        real_features=real_features,
        photon_events=photon_events,
        base_config=config,
        transport_efficiency=0.05,
        tau_f_ns=95.0,
        arrival_spread_ns=2.0,
        max_events=4,
        random_seed=123,
        threshold_mV=1.0,
        n_pretrigger_samples=3,
    )

    assert result["transport_efficiency"] == 0.05
    assert result["tau_f_ns"] == 95.0
    assert result["arrival_spread_ns"] == 2.0
    assert "score" in result
    assert "feature_score" in result
    assert "peak_mV_error" in result
    assert "width_ns_error" in result
    assert "real_valid_fraction" in result
    assert "simulated_valid_fraction" in result
    assert "validity_fraction_error" in result


def test_run_calibration_grid():
    config = make_config()
    real_features = make_features(scale=1.0)
    photon_events = make_photon_events()

    results = run_calibration_grid(
        real_features=real_features,
        photon_events=photon_events,
        base_config=config,
        transport_efficiencies=[0.02, 0.05],
        tau_f_values_ns=[95.0],
        arrival_spread_values_ns=[2.0],
        max_events=4,
        random_seed=123,
        threshold_mV=1.0,
        n_pretrigger_samples=3,
    )

    assert len(results) == 2
    assert "score" in results.columns
    assert "feature_score" in results.columns
    assert "validity_fraction_error" in results.columns
    assert "real_valid_fraction" in results.columns
    assert "simulated_valid_fraction" in results.columns
    assert results["score"].is_monotonic_increasing