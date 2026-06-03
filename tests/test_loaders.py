"""Tests for CSV data loaders."""

import numpy as np
import pandas as pd
import pytest

from tambo_sipm.io.loaders import (
    load_csv_table,
    load_photon_counts_csv,
    load_single_waveform_csv,
    load_waveform_collection_csv,
    require_columns,
)


def test_load_csv_table_reads_valid_csv(tmp_path):
    csv_path = tmp_path / "table.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)

    dataframe = load_csv_table(csv_path)

    assert list(dataframe.columns) == ["a", "b"]
    assert len(dataframe) == 2


def test_load_csv_table_rejects_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_csv_table(missing_path)


def test_require_columns_accepts_existing_columns():
    dataframe = pd.DataFrame({"a": [1], "b": [2]})

    require_columns(dataframe, ["a", "b"])


def test_require_columns_rejects_missing_columns():
    dataframe = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError):
        require_columns(dataframe, ["a", "b"])


def test_load_photon_counts_csv(tmp_path):
    csv_path = tmp_path / "photons.csv"
    pd.DataFrame({"generated_photons": [0, 10, 1300]}).to_csv(
        csv_path, index=False
    )

    counts = load_photon_counts_csv(csv_path)

    assert np.array_equal(counts, np.array([0, 10, 1300], dtype=np.int64))


def test_load_photon_counts_csv_rejects_negative_counts(tmp_path):
    csv_path = tmp_path / "photons.csv"
    pd.DataFrame({"generated_photons": [10, -1]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_photon_counts_csv(csv_path)


def test_load_photon_counts_csv_rejects_non_integer_counts(tmp_path):
    csv_path = tmp_path / "photons.csv"
    pd.DataFrame({"generated_photons": [10.5, 20.0]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError):
        load_photon_counts_csv(csv_path)


def test_load_single_waveform_csv(tmp_path):
    csv_path = tmp_path / "waveform.csv"
    pd.DataFrame(
        {
            "time_ns": [16.0, 0.0, 8.0],
            "voltage_mV": [2.0, 0.0, 1.0],
        }
    ).to_csv(csv_path, index=False)

    time_ns, voltage_mV = load_single_waveform_csv(csv_path)

    assert np.allclose(time_ns, np.array([0.0, 8.0, 16.0]))
    assert np.allclose(voltage_mV, np.array([0.0, 1.0, 2.0]))


def test_load_single_waveform_csv_rejects_duplicate_time(tmp_path):
    csv_path = tmp_path / "waveform.csv"
    pd.DataFrame(
        {
            "time_ns": [0.0, 8.0, 8.0],
            "voltage_mV": [0.0, 1.0, 2.0],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_single_waveform_csv(csv_path)


def test_load_waveform_collection_csv(tmp_path):
    csv_path = tmp_path / "waveforms.csv"
    pd.DataFrame(
        {
            "event_id": ["a", "a", "b", "b"],
            "time_ns": [0.0, 8.0, 0.0, 8.0],
            "voltage_mV": [0.0, 1.0, 2.0, 3.0],
        }
    ).to_csv(csv_path, index=False)

    waveforms = load_waveform_collection_csv(csv_path)

    assert set(waveforms.keys()) == {"a", "b"}
    assert np.allclose(waveforms["a"][0], np.array([0.0, 8.0]))
    assert np.allclose(waveforms["a"][1], np.array([0.0, 1.0]))
    assert np.allclose(waveforms["b"][1], np.array([2.0, 3.0]))


def test_load_waveform_collection_csv_rejects_short_waveform(tmp_path):
    csv_path = tmp_path / "waveforms.csv"
    pd.DataFrame(
        {
            "event_id": ["a", "b", "b"],
            "time_ns": [0.0, 0.0, 8.0],
            "voltage_mV": [0.0, 2.0, 3.0],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_waveform_collection_csv(csv_path)