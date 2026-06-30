"""Tests for HDF5 inventory utilities."""

import h5py
import numpy as np
import pandas as pd

from tambo_sipm.io.hdf5_inventory import (
    build_hdf5_inventory,
    inspect_hdf5_file,
    parse_test_panel_filename,
    save_hdf5_inventory,
    write_hdf5_summary,
)


def make_synthetic_hdf5(path):
    with h5py.File(path, "w") as hdf5_file:
        hdf5_file.attrs["instrument"] = "synthetic"
        raw = hdf5_file.create_group("raw")
        raw.attrs["channel"] = "A"
        raw.create_dataset(
            "waveforms",
            data=np.arange(12, dtype=np.int16).reshape(3, 4),
            compression="gzip",
        )
        raw.create_dataset("timestamps_ns", data=np.array([0.0, 8.0, 16.0]))


def test_parse_test_panel_filename_with_two_thresholds():
    metadata = parse_test_panel_filename("00A_280426_12h10m_AMP5_C5_thr0_287.h5")

    assert metadata["run_index"] == "00"
    assert metadata["panel"] == "A"
    assert metadata["acquisition_date"] == "2026-04-28"
    assert metadata["duration_label"] == "12h10m"
    assert metadata["duration_minutes"] == 730
    assert metadata["gain"] == "AMP5"
    assert metadata["run_config"] == "C5"
    assert metadata["falling_edge_adc_threshold"] == 0
    assert metadata["rising_edge_adc_threshold"] == 287


def test_parse_test_panel_filename_with_single_threshold():
    metadata = parse_test_panel_filename("03B_040526_1h50m_AMP5_C5_thr0_123.h5")

    assert metadata["panel"] == "B"
    assert metadata["acquisition_date"] == "2026-05-04"
    assert metadata["duration_minutes"] == 110
    assert metadata["falling_edge_adc_threshold"] == 0
    assert metadata["rising_edge_adc_threshold"] == 123


def test_parse_test_panel_filename_reuses_single_threshold_value():
    metadata = parse_test_panel_filename("00A_080426_16h20m_AMP5_thr205.h5")

    assert metadata["falling_edge_adc_threshold"] == 205
    assert metadata["rising_edge_adc_threshold"] == 205


def test_inspect_hdf5_file_records_groups_datasets_attrs_and_samples(tmp_path):
    hdf5_path = tmp_path / "00A_280426_12h10m_AMP5_C5_thr0_287.h5"
    make_synthetic_hdf5(hdf5_path)

    records = inspect_hdf5_file(hdf5_path, tmp_path)
    by_path = {record.object_path: record for record in records}

    assert "/" in by_path
    assert "/raw" in by_path
    assert "/raw/waveforms" in by_path
    assert by_path["/"].object_type == "group"
    assert by_path["/"].n_attrs == 1
    assert by_path["/raw/waveforms"].object_type == "dataset"
    assert by_path["/raw/waveforms"].shape == "3x4"
    assert by_path["/raw/waveforms"].dtype == "int16"
    assert by_path["/raw/waveforms"].compression == "gzip"
    assert by_path["/raw/waveforms"].sample_values == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"


def test_build_and_save_hdf5_inventory(tmp_path):
    make_synthetic_hdf5(tmp_path / "00A_280426_12h10m_AMP5_C5_thr0_287.h5")

    inventory = build_hdf5_inventory(tmp_path)

    assert isinstance(inventory, pd.DataFrame)
    assert set(inventory["object_type"]) == {"group", "dataset"}
    assert "/raw/timestamps_ns" in set(inventory["object_path"])

    csv_path = tmp_path / "outputs" / "inventory.csv"
    summary_path = tmp_path / "outputs" / "summary.md"
    save_hdf5_inventory(inventory, csv_path)
    write_hdf5_summary(inventory, summary_path, readme_notes="Synthetic notes")

    assert csv_path.exists()
    assert summary_path.exists()
    assert "Synthetic notes" in summary_path.read_text(encoding="utf-8")
