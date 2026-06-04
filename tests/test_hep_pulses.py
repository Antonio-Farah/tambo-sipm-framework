"""Tests for HEP measured pulse loading and segmentation."""

import numpy as np
import pandas as pd
import pytest

from tambo_sipm.io.hep_pulses import (
    assign_pulse_ids_from_time_gaps,
    load_hep_timeseries,
    load_segmented_hep_pulses,
    pulse_collection_to_dict,
    pulse_lengths,
    segment_hep_pulses,
)


def make_raw_hep_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "voltaje": [0.001, 0.002, 0.003, 0.010, 0.011],
            "segundos": [0, 0, 0, 0, 0],
            "nanosegundos": [0, 8, 16, 100, 108],
            "tiempo_total_segundos": [
                0.0,
                8.0e-9,
                16.0e-9,
                100.0e-9,
                108.0e-9,
            ],
        }
    )


def test_load_hep_timeseries(tmp_path):
    csv_path = tmp_path / "10min.csv"
    make_raw_hep_dataframe().to_csv(csv_path, index=False)

    timeseries = load_hep_timeseries(csv_path)

    assert "absolute_time_ns" in timeseries.columns
    assert "time_ns" in timeseries.columns
    assert "voltage_V" in timeseries.columns
    assert "voltage_mV" in timeseries.columns
    assert np.allclose(timeseries["voltage_mV"].to_numpy(), [1, 2, 3, 10, 11])


def test_load_hep_timeseries_rejects_missing_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"voltaje": [0.001, 0.002]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_hep_timeseries(csv_path)


def test_assign_pulse_ids_from_time_gaps():
    time_ns = np.array([0.0, 8.0, 16.0, 100.0, 108.0])

    pulse_ids = assign_pulse_ids_from_time_gaps(
        time_ns=time_ns,
        gap_threshold_ns=50.0,
    )

    assert np.array_equal(pulse_ids, np.array([0, 0, 0, 1, 1]))


def test_assign_pulse_ids_rejects_decreasing_time():
    with pytest.raises(ValueError):
        assign_pulse_ids_from_time_gaps(
            time_ns=np.array([0.0, 8.0, 4.0]),
            gap_threshold_ns=50.0,
        )


def test_segment_hep_pulses():
    raw_dataframe = make_raw_hep_dataframe()

    timeseries = raw_dataframe.copy()
    timeseries["voltage_V"] = timeseries["voltaje"]
    timeseries["voltage_mV"] = timeseries["voltaje"] * 1000.0
    timeseries["absolute_time_ns"] = (
        timeseries["tiempo_total_segundos"] * 1.0e9
    )
    timeseries["time_ns"] = timeseries["absolute_time_ns"]

    pulses = segment_hep_pulses(
        timeseries=timeseries,
        gap_threshold_ns=50.0,
        min_samples=2,
    )

    assert pulses["event_id"].nunique() == 2
    assert list(pulses.groupby("event_id").size()) == [3, 2]
    assert np.isclose(pulses[pulses["event_id"] == "real_pulse_00000"]["time_ns"].min(), 0.0)
    assert np.isclose(pulses[pulses["event_id"] == "real_pulse_00001"]["time_ns"].min(), 0.0)


def test_load_segmented_hep_pulses(tmp_path):
    csv_path = tmp_path / "10min.csv"
    make_raw_hep_dataframe().to_csv(csv_path, index=False)

    pulses = load_segmented_hep_pulses(
        path=csv_path,
        gap_threshold_ns=50.0,
        min_samples=2,
    )

    assert pulses["event_id"].nunique() == 2
    assert len(pulses) == 5


def test_pulse_lengths():
    raw_dataframe = make_raw_hep_dataframe()

    timeseries = raw_dataframe.copy()
    timeseries["voltage_V"] = timeseries["voltaje"]
    timeseries["voltage_mV"] = timeseries["voltaje"] * 1000.0
    timeseries["absolute_time_ns"] = (
        timeseries["tiempo_total_segundos"] * 1.0e9
    )
    timeseries["time_ns"] = timeseries["absolute_time_ns"]

    pulses = segment_hep_pulses(timeseries)

    summary = pulse_lengths(pulses)

    assert len(summary) == 2
    assert list(summary["n_samples"]) == [3, 2]


def test_pulse_collection_to_dict():
    raw_dataframe = make_raw_hep_dataframe()

    timeseries = raw_dataframe.copy()
    timeseries["voltage_V"] = timeseries["voltaje"]
    timeseries["voltage_mV"] = timeseries["voltaje"] * 1000.0
    timeseries["absolute_time_ns"] = (
        timeseries["tiempo_total_segundos"] * 1.0e9
    )
    timeseries["time_ns"] = timeseries["absolute_time_ns"]

    pulses = segment_hep_pulses(timeseries)
    waveforms = pulse_collection_to_dict(pulses)

    assert set(waveforms.keys()) == {"real_pulse_00000", "real_pulse_00001"}
    assert np.allclose(waveforms["real_pulse_00000"][0], [0.0, 8.0, 16.0])
    assert np.allclose(waveforms["real_pulse_00001"][0], [0.0, 8.0])