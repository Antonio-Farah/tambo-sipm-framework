"""Tests for batch waveform feature extraction."""

import numpy as np
import pandas as pd
import pytest

from tambo_sipm.analysis.feature_tables import (
    extract_features_from_pulse_table,
    extract_features_from_waveform_dict,
    extract_waveform_feature_row,
    summarize_feature_table,
)


def make_pulse_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [
                "a",
                "a",
                "a",
                "a",
                "a",
                "b",
                "b",
                "b",
                "b",
                "b",
            ],
            "time_ns": [
                0.0,
                8.0,
                16.0,
                24.0,
                32.0,
                0.0,
                8.0,
                16.0,
                24.0,
                32.0,
            ],
            "voltage_mV": [
                10.0,
                10.0,
                10.0,
                35.0,
                25.0,
                5.0,
                5.0,
                5.0,
                8.0,
                7.0,
            ],
        }
    )


def test_extract_waveform_feature_row_valid_pulse():
    time_ns = np.array([0.0, 8.0, 16.0, 24.0, 32.0])
    voltage_mV = np.array([10.0, 10.0, 10.0, 35.0, 25.0])

    row = extract_waveform_feature_row(
        event_id="a",
        time_ns=time_ns,
        voltage_mV=voltage_mV,
        threshold_mV=10.0,
        n_pretrigger_samples=3,
    )

    assert row["event_id"] == "a"
    assert row["valid"]
    assert row["extraction_status"] == "ok"
    assert np.isclose(row["peak_mV"], 25.0)


def test_extract_waveform_feature_row_too_short():
    time_ns = np.array([0.0, 8.0])
    voltage_mV = np.array([1.0, 2.0])

    row = extract_waveform_feature_row(
        event_id="short",
        time_ns=time_ns,
        voltage_mV=voltage_mV,
        n_pretrigger_samples=3,
    )

    assert not row["valid"]
    assert row["extraction_status"] == "too_short_for_baseline"


def test_extract_features_from_pulse_table():
    pulses = make_pulse_table()

    features = extract_features_from_pulse_table(
        pulses=pulses,
        threshold_mV=10.0,
        n_pretrigger_samples=3,
    )

    assert len(features) == 2
    assert set(features["event_id"]) == {"a", "b"}
    assert "peak_mV" in features.columns
    assert "integral_mVns" in features.columns


def test_extract_features_from_pulse_table_rejects_missing_columns():
    pulses = pd.DataFrame({"event_id": ["a"], "time_ns": [0.0]})

    with pytest.raises(ValueError):
        extract_features_from_pulse_table(pulses)


def test_extract_features_from_waveform_dict():
    waveforms = {
        "a": (
            np.array([0.0, 8.0, 16.0, 24.0, 32.0]),
            np.array([10.0, 10.0, 10.0, 35.0, 25.0]),
        )
    }

    features = extract_features_from_waveform_dict(
        waveforms=waveforms,
        threshold_mV=10.0,
        n_pretrigger_samples=3,
    )

    assert len(features) == 1
    assert features.loc[0, "event_id"] == "a"
    assert features.loc[0, "valid"]


def test_summarize_feature_table():
    pulses = make_pulse_table()

    features = extract_features_from_pulse_table(
        pulses=pulses,
        threshold_mV=10.0,
        n_pretrigger_samples=3,
    )

    summary = summarize_feature_table(features)

    assert "extraction_status" in summary.columns
    assert "valid" in summary.columns
    assert "count" in summary.columns
    assert summary["count"].sum() == 2