"""Tests for batch detector simulation utilities."""

import numpy as np
import pandas as pd
import pytest

from tambo_sipm.simulation.batch import (
    require_event_columns,
    select_photon_events,
    simulate_feature_row_from_photon_event,
    simulate_feature_table_from_photon_events,
)


def make_config() -> dict:
    return {
        "photon_transport": {
            "transport_efficiency": 1.0,
            "pde": 1.0,
        },
        "sipm_response": {
            "window_ns": 200.0,
            "dt_ns": 1.0,
            "tau_r_ns": 2.0,
            "tau_f_ns": 50.0,
            "event_time_ns": 16.0,
            "arrival_spread_ns": 0.0,
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
            "voltage_scale_mV_per_pe": 1.0,
        },
        "readout": {
            "sampling_interval_ns": 8.0,
            "adc_bits": 14,
            "v_min_mV": -1000.0,
            "v_max_mV": 1000.0,
        },
    }


def make_photon_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "source_row": [0, 1, 2],
            "detector": [0, 0, 0],
            "shower": [10, 11, 12],
            "particle_id": [1, 3, 6],
            "particle_name": ["gamma", "e-", "mu-"],
            "particle_category": ["electromagnetic", "electromagnetic", "muonic"],
            "generated_photons": [0, 10, 50],
            "energia_detectada_poli_MeV": [0.0, 0.1, 0.2],
        }
    )


def test_require_event_columns_accepts_valid_dataframe():
    require_event_columns(make_photon_events(), ["event_id", "generated_photons"])


def test_require_event_columns_rejects_missing_columns():
    with pytest.raises(ValueError):
        require_event_columns(pd.DataFrame({"event_id": ["a"]}), ["generated_photons"])


def test_select_photon_events_head_without_seed():
    events = make_photon_events()

    selected = select_photon_events(
        photon_events=events,
        max_events=2,
        random_seed=None,
    )

    assert list(selected["event_id"]) == ["a", "b"]


def test_select_photon_events_reproducible_with_seed():
    events = make_photon_events()

    selected_a = select_photon_events(events, max_events=2, random_seed=123)
    selected_b = select_photon_events(events, max_events=2, random_seed=123)

    assert list(selected_a["event_id"]) == list(selected_b["event_id"])


def test_select_photon_events_rejects_invalid_max_events():
    with pytest.raises(ValueError):
        select_photon_events(make_photon_events(), max_events=0)


def test_simulate_feature_row_from_photon_event():
    events = make_photon_events()
    config = make_config()
    rng = np.random.default_rng(123)

    row = simulate_feature_row_from_photon_event(
        event_row=events.iloc[2],
        config=config,
        threshold_mV=1.0,
        n_pretrigger_samples=3,
        rng=rng,
    )

    assert row["event_id"] == "c"
    assert row["generated_photons"] == 50
    assert row["photoelectrons"] == 50
    assert row["fired_microcells"] >= 0
    assert "peak_mV" in row
    assert "integral_mVns" in row


def test_simulate_feature_table_from_photon_events():
    events = make_photon_events()
    config = make_config()

    features = simulate_feature_table_from_photon_events(
        photon_events=events,
        config=config,
        max_events=2,
        random_seed=123,
        threshold_mV=1.0,
        n_pretrigger_samples=3,
    )

    assert len(features) == 2
    assert "event_id" in features.columns
    assert "generated_photons" in features.columns
    assert "photoelectrons" in features.columns
    assert "peak_mV" in features.columns