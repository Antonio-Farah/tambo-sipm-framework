"""Tests for Test_Panel HDF5 sample loading."""

import h5py
import numpy as np
import pandas as pd
import pytest

from tambo_sipm.io.test_panel_hdf5 import (
    count_feature_valid_pulses,
    estimate_sampling_interval_ns,
    format_test_panel_voltage_unit_report,
    infer_test_panel_voltage_unit,
    load_test_panel_h5_samples,
    pulse_length_distribution,
    sampling_interval_report,
    segment_test_panel_samples,
    segment_test_panel_samples_by_threshold,
    summarize_test_panel_segmentation,
    summarize_test_panel_voltage_units,
)


def make_test_panel_hdf5(path, voltages=None):
    if voltages is None:
        voltages = [0.010, 0.012, 0.014]

    with h5py.File(path, "w") as hdf5_file:
        datos = hdf5_file.create_group("datos")
        datos.create_dataset(
            "axis0",
            data=np.array(
                [
                    b"adc_value",
                    b"voltaje",
                    b"segundos",
                    b"nanosegundos",
                    b"tiempo_total_segundos",
                ]
            ),
        )
        datos.create_dataset("axis1", data=np.array([10, 11, 12], dtype=np.int64))
        datos.create_dataset("block0_items", data=np.array([b"adc_value"]))
        datos.create_dataset(
            "block0_values",
            data=np.array([[120], [130], [140]], dtype=np.int16),
        )
        datos.create_dataset(
            "block1_items",
            data=np.array(
                [
                    b"voltaje",
                    b"segundos",
                    b"nanosegundos",
                    b"tiempo_total_segundos",
                ]
            ),
        )
        datos.create_dataset(
            "block1_values",
            data=np.array(
                [
                    [voltages[0], 1.0, 0.000000000, 1.000000000],
                    [voltages[1], 1.0, 0.000000008, 1.000000008],
                    [voltages[2], 1.0, 0.000000016, 1.000000016],
                ],
                dtype=np.float64,
            ),
        )


def test_load_test_panel_h5_samples_reconstructs_columns(tmp_path):
    hdf5_path = tmp_path / "03A_040526_1h50m_AMP5_C5_thr0_82.h5"
    make_test_panel_hdf5(hdf5_path)

    dataframe = load_test_panel_h5_samples(hdf5_path)

    assert list(dataframe["sample_index"]) == [10, 11, 12]
    assert list(dataframe["adc_value"]) == [120, 130, 140]
    assert np.allclose(dataframe["voltage_raw"], [0.010, 0.012, 0.014])
    assert np.allclose(dataframe["voltage_mV"], [10.0, 12.0, 14.0])
    assert np.allclose(
        dataframe["time_ns"],
        [1_000_000_000.0, 1_000_000_008.0, 1_000_000_016.0],
    )
    assert set(dataframe["source_file"]) == {"03A_040526_1h50m_AMP5_C5_thr0_82.h5"}
    assert set(dataframe["panel"]) == {"A"}
    assert set(dataframe["run_id"]) == {"03A"}
    assert set(dataframe["acquisition_date"]) == {"2026-05-04"}
    assert set(dataframe["acquisition_duration"]) == {"1h50m"}
    assert set(dataframe["threshold_token"]) == {"thr0_82"}
    assert set(dataframe["falling_edge_adc_threshold"]) == {0}
    assert set(dataframe["rising_edge_adc_threshold"]) == {82}
    assert dataframe.attrs["voltage_unit_requested"] == "auto"
    assert dataframe.attrs["voltage_unit_resolved"] == "V"
    assert "median(abs(voltaje))" in dataframe.attrs["voltage_conversion_assumption"]


def test_load_test_panel_h5_samples_can_limit_rows(tmp_path):
    hdf5_path = tmp_path / "03A_040526_1h50m_AMP5_C5_thr0_82.h5"
    make_test_panel_hdf5(hdf5_path)

    dataframe = load_test_panel_h5_samples(hdf5_path, max_rows=2)

    assert len(dataframe) == 2
    assert list(dataframe["sample_index"]) == [10, 11]
    assert list(dataframe["adc_value"]) == [120, 130]


def test_load_test_panel_h5_samples_can_force_voltage_units(tmp_path):
    hdf5_path = tmp_path / "03A_040526_1h50m_AMP5_C5_thr0_82.h5"
    make_test_panel_hdf5(hdf5_path)

    dataframe = load_test_panel_h5_samples(hdf5_path, voltage_unit="mV")

    assert np.allclose(dataframe["voltage_raw"], [0.010, 0.012, 0.014])
    assert np.allclose(dataframe["voltage_mV"], [0.010, 0.012, 0.014])
    assert dataframe.attrs["voltage_unit_requested"] == "mV"
    assert dataframe.attrs["voltage_unit_resolved"] == "mV"
    assert "explicit voltage_unit='mV'" in dataframe.attrs[
        "voltage_conversion_assumption"
    ]


def test_auto_voltage_unit_interprets_tens_or_hundreds_as_mv(tmp_path):
    hdf5_path = tmp_path / "03A_040526_1h50m_AMP5_C5_thr0_82.h5"
    make_test_panel_hdf5(hdf5_path, voltages=[10.0, 12.0, 14.0])

    dataframe = load_test_panel_h5_samples(hdf5_path)

    assert np.allclose(dataframe["voltage_mV"], [10.0, 12.0, 14.0])
    assert dataframe.attrs["voltage_unit_resolved"] == "mV"


def test_voltage_unit_report_includes_stats_and_assumption(tmp_path):
    hdf5_path = tmp_path / "03A_040526_1h50m_AMP5_C5_thr0_82.h5"
    make_test_panel_hdf5(hdf5_path)
    dataframe = load_test_panel_h5_samples(hdf5_path)

    summary = summarize_test_panel_voltage_units(dataframe)
    report = format_test_panel_voltage_unit_report(dataframe)

    assert list(summary["column"]) == ["voltage_raw", "voltage_mV"]
    assert np.allclose(summary["median"], [0.012, 12.0])
    assert "requested voltage_unit: auto" in report
    assert "resolved voltage unit: V" in report
    assert "voltage_raw,0.01,0.012,0.014" in report
    assert "voltage_mV,10,12,14" in report


def test_infer_test_panel_voltage_unit_rule():
    assert infer_test_panel_voltage_unit(np.array([0.1, -0.2, 0.3]))[0] == "V"
    assert infer_test_panel_voltage_unit(np.array([10.0, -20.0, 30.0]))[0] == "mV"


def test_load_test_panel_h5_samples_rejects_bad_voltage_unit(tmp_path):
    hdf5_path = tmp_path / "03A_040526_1h50m_AMP5_C5_thr0_82.h5"
    make_test_panel_hdf5(hdf5_path)

    with pytest.raises(ValueError, match="voltage_unit"):
        load_test_panel_h5_samples(hdf5_path, voltage_unit="ADC")


def test_estimate_sampling_interval_ns_from_time_ns():
    samples = {
        "time_ns": [100.0, 108.0, 116.0, 124.0],
    }

    assert estimate_sampling_interval_ns(samples) == 8.0


def test_estimate_sampling_interval_ns_from_time_total_seconds():
    samples = {
        "time_total_seconds": [1.0, 1.000000004, 1.000000008],
    }

    assert np.isclose(estimate_sampling_interval_ns(samples), 4.0)


def test_estimate_sampling_interval_ns_uses_positive_median():
    samples = {
        "time_ns": [0.0, 8.0, 8.0, 16.0, 40.0],
    }

    assert estimate_sampling_interval_ns(samples) == 8.0


def test_sampling_interval_report_flags_approximately_8_ns():
    report = sampling_interval_report({"time_ns": [0.0, 8.0, 16.0]})

    assert "estimated_sampling_interval_ns: 8" in report
    assert "approximately_reference: True" in report


def test_sampling_interval_report_flags_non_8_ns():
    report = sampling_interval_report({"time_ns": [0.0, 4.0, 8.0]})

    assert "estimated_sampling_interval_ns: 4" in report
    assert "approximately_reference: False" in report
    assert "use the estimated interval for HDF5 time bins" in report


def test_estimate_sampling_interval_ns_rejects_missing_time_column():
    with pytest.raises(ValueError, match="time_ns"):
        estimate_sampling_interval_ns({"voltage_mV": [1.0, 2.0]})


def test_estimate_sampling_interval_ns_rejects_single_sample():
    with pytest.raises(ValueError, match="At least two"):
        estimate_sampling_interval_ns({"time_ns": [1.0]})


def make_loaded_samples_for_segmentation():
    return pd.DataFrame(
        {
            "sample_index": np.arange(8, dtype=np.int64),
            "time_ns": [0.0, 8.0, 16.0, 100.0, 108.0, 116.0, 300.0, 308.0],
            "voltage_mV": [10.0, 10.0, 12.0, 10.0, 30.0, 20.0, 5.0, 6.0],
            "source_file": ["03A_040526_1h50m_AMP5_C5_thr0_82.h5"] * 8,
            "panel": ["A"] * 8,
            "run_id": ["03A"] * 8,
            "acquisition_date": ["2026-05-04"] * 8,
            "acquisition_duration": ["1h50m"] * 8,
            "threshold_token": ["thr0_82"] * 8,
        }
    )


def test_segment_test_panel_samples_by_time_gap_preserves_metadata():
    samples = make_loaded_samples_for_segmentation()

    pulses = segment_test_panel_samples(samples, gap_threshold_ns=50.0, min_samples=2)

    required_columns = [
        "event_id",
        "sample_index",
        "time_ns",
        "voltage_mV",
        "source_file",
        "panel",
        "run_id",
        "acquisition_date",
        "acquisition_duration",
        "threshold_token",
    ]

    assert pulses["event_id"].nunique() == 3
    assert list(pulses["event_id"].drop_duplicates()) == [
        "test_panel_03A_pulse_00000",
        "test_panel_03A_pulse_00001",
        "test_panel_03A_pulse_00002",
    ]
    assert list(pulses.groupby("event_id").size()) == [3, 3, 2]
    assert list(pulses.columns) == [
        "event_id",
        "pulse_id",
        "sample_index",
        "time_ns",
        "voltage_mV",
        "source_file",
        "panel",
        "run_id",
        "acquisition_date",
        "acquisition_duration",
        "threshold_token",
    ]
    assert set(required_columns).issubset(pulses.columns)
    assert set(pulses["panel"]) == {"A"}
    assert np.isclose(
        pulses[pulses["event_id"] == "test_panel_03A_pulse_00001"]["time_ns"].min(),
        0.0,
    )


def test_segment_test_panel_samples_applies_min_samples_without_silent_voltage_filter():
    samples = make_loaded_samples_for_segmentation()

    pulses = segment_test_panel_samples(samples, gap_threshold_ns=50.0, min_samples=3)

    assert pulses["event_id"].nunique() == 2
    assert list(pulses.groupby("event_id").size()) == [3, 3]
    assert pulses["voltage_mV"].min() == 10.0


def test_segment_test_panel_samples_by_threshold_crossings():
    samples = make_loaded_samples_for_segmentation()

    pulses = segment_test_panel_samples_by_threshold(
        samples,
        threshold_mV=20.0,
        pre_samples=1,
        post_samples=1,
        min_samples=2,
    )

    assert pulses["event_id"].nunique() == 1
    assert list(pulses["voltage_mV"]) == [10.0, 30.0, 20.0, 5.0]
    assert list(pulses["sample_index"]) == [0, 1, 2, 3]
    assert set(pulses["threshold_token"]) == {"thr0_82"}


def test_segment_test_panel_samples_by_threshold_rejects_no_crossings():
    samples = make_loaded_samples_for_segmentation()

    with pytest.raises(ValueError, match="No threshold crossings"):
        segment_test_panel_samples_by_threshold(samples, threshold_mV=100.0)


def test_pulse_length_distribution_and_feature_valid_count():
    samples = make_loaded_samples_for_segmentation()
    pulses = segment_test_panel_samples(samples, gap_threshold_ns=50.0, min_samples=2)

    lengths = pulse_length_distribution(pulses)
    valid_count = count_feature_valid_pulses(
        pulses,
        feature_threshold_mV=5.0,
        n_pretrigger_samples=1,
    )

    assert list(lengths["n_samples"]) == [3, 3, 2]
    assert list(lengths["duration_ns"]) == [16.0, 16.0, 8.0]
    assert valid_count == 1


def test_summarize_test_panel_segmentation_reports_required_diagnostics():
    samples = make_loaded_samples_for_segmentation()
    pulses = segment_test_panel_samples(samples, gap_threshold_ns=50.0, min_samples=2)

    summary = summarize_test_panel_segmentation(
        samples,
        pulses,
        feature_threshold_mV=5.0,
        n_pretrigger_samples=1,
    )

    assert summary["total_rows_loaded"] == 8
    assert summary["estimated_sampling_interval_ns"] == 8.0
    assert summary["number_of_segmented_pulses"] == 3
    assert summary["voltage_range_mV"] == {"min": 5.0, "median": 10.0, "max": 30.0}
    assert summary["pulses_passing_feature_threshold"] == 1
    assert list(summary["pulse_length_distribution"]["n_samples"]) == [3, 3, 2]


def test_load_test_panel_h5_samples_supports_panel_b_metadata(tmp_path):
    hdf5_path = tmp_path / "03B_040526_1h50m_AMP5_C5_thr0_123.h5"
    make_test_panel_hdf5(hdf5_path)

    dataframe = load_test_panel_h5_samples(hdf5_path)

    assert set(dataframe["panel"]) == {"B"}
    assert set(dataframe["run_id"]) == {"03B"}
    assert set(dataframe["threshold_token"]) == {"thr0_123"}
    assert set(dataframe["rising_edge_adc_threshold"]) == {123}


def test_load_test_panel_h5_samples_rejects_missing_dataset(tmp_path):
    hdf5_path = tmp_path / "03A_040526_1h50m_AMP5_C5_thr0_82.h5"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.create_group("datos")

    with pytest.raises(ValueError, match="Missing required HDF5 datasets"):
        load_test_panel_h5_samples(hdf5_path)


def test_load_test_panel_h5_samples_rejects_axis0_mismatch(tmp_path):
    hdf5_path = tmp_path / "03A_040526_1h50m_AMP5_C5_thr0_82.h5"
    make_test_panel_hdf5(hdf5_path)
    with h5py.File(hdf5_path, "a") as hdf5_file:
        del hdf5_file["/datos/block1_items"]
        hdf5_file["/datos"].create_dataset(
            "block1_items",
            data=np.array([b"voltaje", b"segundos", b"nanosegundos", b"bad_time"]),
        )

    with pytest.raises(ValueError, match="do not match /datos/axis0"):
        load_test_panel_h5_samples(hdf5_path)


def test_load_test_panel_h5_samples_rejects_missing_logical_column(tmp_path):
    hdf5_path = tmp_path / "03A_040526_1h50m_AMP5_C5_thr0_82.h5"
    make_test_panel_hdf5(hdf5_path)
    with h5py.File(hdf5_path, "a") as hdf5_file:
        del hdf5_file["/datos/axis0"]
        del hdf5_file["/datos/block1_items"]
        labels = np.array([b"adc_value", b"voltaje", b"segundos", b"nanosegundos", b"bad_time"])
        hdf5_file["/datos"].create_dataset("axis0", data=labels)
        hdf5_file["/datos"].create_dataset(
            "block1_items",
            data=np.array([b"voltaje", b"segundos", b"nanosegundos", b"bad_time"]),
        )

    with pytest.raises(ValueError, match="Missing logical columns"):
        load_test_panel_h5_samples(hdf5_path)
