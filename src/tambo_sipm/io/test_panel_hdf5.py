"""Load sample-level Test_Panel HDF5 acquisitions."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd

from tambo_sipm.analysis.features import extract_pulse_features
from tambo_sipm.io.hdf5_inventory import parse_test_panel_filename


_REQUIRED_DATASETS = (
    "/datos/axis0",
    "/datos/axis1",
    "/datos/block0_items",
    "/datos/block0_values",
    "/datos/block1_items",
    "/datos/block1_values",
)
_REQUIRED_LOGICAL_COLUMNS = (
    "adc_value",
    "voltaje",
    "segundos",
    "nanosegundos",
    "tiempo_total_segundos",
)
VoltageUnit = Literal["auto", "V", "mV"]
_AUTO_VOLTAGE_UNIT_THRESHOLD = 5.0
_AUTO_VOLTAGE_UNIT_RULE = (
    "auto uses median(abs(voltaje)): values below 5.0 are interpreted as V "
    "and converted to mV; values at or above 5.0 are interpreted as mV."
)
_DEFAULT_SAMPLING_INTERVAL_NS = 8.0
_TEST_PANEL_METADATA_COLUMNS = [
    "source_file",
    "panel",
    "run_id",
    "acquisition_date",
    "acquisition_duration",
    "threshold_token",
]


def _decode_hdf5_label(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _read_items(dataset: h5py.Dataset) -> list[str]:
    values = np.asarray(dataset[()]).reshape(-1)
    return [_decode_hdf5_label(value) for value in values]


def _read_block(dataset: h5py.Dataset, columns: list[str]) -> pd.DataFrame:
    values = np.asarray(dataset[()])
    return _read_block_from_array(values, columns)


def _read_block_from_array(values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D block values, got shape {values.shape}.")
    if values.shape[1] != len(columns):
        raise ValueError(
            "Block values column count does not match item labels: "
            f"{values.shape[1]} values vs {len(columns)} labels."
        )
    return pd.DataFrame(values, columns=columns)


def _threshold_token(path: Path) -> str | None:
    match = re.search(r"_(thr.+)$", path.stem)
    return match.group(1) if match else None


def _run_id(metadata: dict[str, Any]) -> str | None:
    run_index = metadata.get("run_index")
    panel = metadata.get("panel")
    if run_index is None or panel is None:
        return None
    return f"{run_index}{panel}"


def infer_test_panel_voltage_unit(voltage_raw: pd.Series) -> tuple[Literal["V", "mV"], str]:
    """Infer whether the HDF5 ``voltaje`` column is stored in V or mV.

    The rule is intentionally conservative and visible to callers: if the
    typical absolute value is below a few volts, interpret it as volts; if it is
    already in the tens/hundreds, interpret it as millivolts.
    """
    numeric_voltage = pd.to_numeric(voltage_raw, errors="raise")
    typical_abs = float(np.nanmedian(np.abs(numeric_voltage)))
    if typical_abs < _AUTO_VOLTAGE_UNIT_THRESHOLD:
        return "V", _AUTO_VOLTAGE_UNIT_RULE
    return "mV", _AUTO_VOLTAGE_UNIT_RULE


def convert_test_panel_voltage(
    voltage_raw: pd.Series,
    voltage_unit: VoltageUnit = "auto",
) -> tuple[pd.Series, Literal["V", "mV"], str]:
    """Convert raw Test_Panel voltage values to millivolts."""
    if voltage_unit not in ("auto", "V", "mV"):
        raise ValueError("voltage_unit must be one of: 'auto', 'V', 'mV'.")

    numeric_voltage = pd.to_numeric(voltage_raw, errors="raise")
    if voltage_unit == "auto":
        resolved_unit, rule = infer_test_panel_voltage_unit(numeric_voltage)
    else:
        resolved_unit = voltage_unit
        rule = f"explicit voltage_unit='{voltage_unit}' supplied by caller."

    if resolved_unit == "V":
        return numeric_voltage * 1000.0, resolved_unit, rule
    return numeric_voltage.astype(float), resolved_unit, rule


def summarize_test_panel_voltage_units(samples: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw and converted voltage ranges for notebook reporting."""
    required_columns = ["voltage_raw", "voltage_mV"]
    missing_columns = [column for column in required_columns if column not in samples]
    if missing_columns:
        raise ValueError(f"Missing voltage columns: {missing_columns}")

    rows = []
    for column in required_columns:
        values = pd.to_numeric(samples[column], errors="raise")
        rows.append(
            {
                "column": column,
                "min": float(values.min()),
                "median": float(values.median()),
                "max": float(values.max()),
            }
        )

    summary = pd.DataFrame(rows)
    summary.attrs["voltage_unit_requested"] = samples.attrs.get(
        "voltage_unit_requested"
    )
    summary.attrs["voltage_unit_resolved"] = samples.attrs.get(
        "voltage_unit_resolved"
    )
    summary.attrs["voltage_conversion_assumption"] = samples.attrs.get(
        "voltage_conversion_assumption"
    )
    return summary


def format_test_panel_voltage_unit_report(samples: pd.DataFrame) -> str:
    """Format voltage-unit statistics and conversion assumption for printing."""
    summary = summarize_test_panel_voltage_units(samples)
    lines = [
        "Test_Panel voltage unit report",
        f"requested voltage_unit: {summary.attrs.get('voltage_unit_requested')}",
        f"resolved voltage unit: {summary.attrs.get('voltage_unit_resolved')}",
        "conversion assumption: "
        f"{summary.attrs.get('voltage_conversion_assumption')}",
        "",
        "column,min,median,max",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"{row.column},{row.min:.12g},{row.median:.12g},{row.max:.12g}")
    return "\n".join(lines)


def estimate_sampling_interval_ns(samples: pd.DataFrame) -> float:
    """Estimate the sample spacing in ns from ``time_ns`` or total seconds.

    The estimate uses the median of positive adjacent time differences, which
    is robust to occasional repeated/nonmonotonic rows while still preserving
    the native acquisition cadence for binning decisions.
    """
    if "time_ns" in samples:
        time_ns = np.asarray(
            pd.to_numeric(samples["time_ns"], errors="raise"),
            dtype=np.float64,
        )
    elif "time_total_seconds" in samples:
        time_ns = (
            np.asarray(
                pd.to_numeric(samples["time_total_seconds"], errors="raise"),
                dtype=np.float64,
            )
            * 1_000_000_000.0
        )
    else:
        raise ValueError("samples must contain 'time_ns' or 'time_total_seconds'.")

    if len(time_ns) < 2:
        raise ValueError("At least two time samples are required.")

    diffs = np.diff(time_ns)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if len(positive_diffs) == 0:
        raise ValueError("No positive finite time differences were found.")

    return float(np.median(positive_diffs))


def sampling_interval_report(
    samples: pd.DataFrame,
    expected_ns: float = _DEFAULT_SAMPLING_INTERVAL_NS,
    tolerance_ns: float = 0.25,
) -> str:
    """Format a sampling-interval report for notebook output."""
    interval_ns = estimate_sampling_interval_ns(samples)
    difference_ns = interval_ns - expected_ns
    approximately_expected = abs(difference_ns) <= tolerance_ns
    if approximately_expected:
        decision = (
            f"estimated interval is approximately {expected_ns:g} ns "
            f"(within +/- {tolerance_ns:g} ns)."
        )
    else:
        decision = (
            f"estimated interval is not approximately {expected_ns:g} ns; "
            "use the estimated interval for HDF5 time bins."
        )

    return "\n".join(
        [
            "Test_Panel sampling interval report",
            f"estimated_sampling_interval_ns: {interval_ns:.12g}",
            f"reference_interval_ns: {expected_ns:.12g}",
            f"difference_ns: {difference_ns:.12g}",
            f"approximately_reference: {approximately_expected}",
            f"decision: {decision}",
        ]
    )


def _validate_segmentation_inputs(
    samples: pd.DataFrame,
    min_samples: int,
) -> None:
    required_columns = ["time_ns", "voltage_mV", *_TEST_PANEL_METADATA_COLUMNS]
    missing_columns = [column for column in required_columns if column not in samples]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive.")


def _build_segmented_pulse_table(
    samples: pd.DataFrame,
    segment_ids: np.ndarray,
    min_samples: int,
) -> pd.DataFrame:
    working = samples.copy()
    working["pulse_id_original"] = np.asarray(segment_ids, dtype=np.int64)

    pulse_sizes = working.groupby("pulse_id_original").size()
    valid_original_ids = pulse_sizes[pulse_sizes >= min_samples].index
    working = working[working["pulse_id_original"].isin(valid_original_ids)].copy()

    if working.empty:
        raise ValueError("No pulses remain after applying min_samples.")

    id_mapping = {
        original_id: new_id
        for new_id, original_id in enumerate(sorted(valid_original_ids))
    }
    working["pulse_id"] = working["pulse_id_original"].map(id_mapping)

    rows: list[pd.DataFrame] = []
    run_id = str(working["run_id"].iloc[0])

    for pulse_id, group in working.groupby("pulse_id", sort=True):
        pulse = group.sort_values("time_ns").copy()
        first_time_ns = float(pulse["time_ns"].iloc[0])
        pulse["sample_index"] = np.arange(len(pulse), dtype=np.int64)
        pulse["time_ns"] = pulse["time_ns"] - first_time_ns
        pulse["event_id"] = f"test_panel_{run_id}_pulse_{int(pulse_id):05d}"
        rows.append(pulse)

    pulses = pd.concat(rows, ignore_index=True)

    output_columns = [
        "event_id",
        "pulse_id",
        "sample_index",
        "time_ns",
        "voltage_mV",
        *_TEST_PANEL_METADATA_COLUMNS,
    ]

    return pulses[output_columns]


def segment_test_panel_samples(
    samples: pd.DataFrame,
    gap_threshold_ns: float = 50.0,
    min_samples: int = 2,
) -> pd.DataFrame:
    """Segment Test_Panel samples into pulses using gaps in absolute time.

    This mirrors the original HEP CSV workflow: a new pulse starts when the
    gap between adjacent sample times exceeds ``gap_threshold_ns``. It does not
    filter by voltage or feature quality.
    """
    if gap_threshold_ns <= 0.0:
        raise ValueError("gap_threshold_ns must be positive.")
    _validate_segmentation_inputs(samples, min_samples=min_samples)

    working = samples.sort_values("time_ns").reset_index(drop=True)
    time_ns = pd.to_numeric(working["time_ns"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if len(time_ns) == 0:
        raise ValueError("samples must not be empty.")
    if not np.all(np.isfinite(time_ns)):
        raise ValueError("time_ns must contain finite values.")
    if not np.all(np.diff(time_ns) >= 0.0):
        raise ValueError("time_ns must be non-decreasing after sorting.")

    if len(time_ns) == 1:
        segment_ids = np.array([0], dtype=np.int64)
    else:
        starts_new_pulse = np.concatenate(
            [
                np.array([True]),
                np.diff(time_ns) > gap_threshold_ns,
            ]
        )
        segment_ids = np.cumsum(starts_new_pulse).astype(np.int64) - 1

    return _build_segmented_pulse_table(
        samples=working,
        segment_ids=segment_ids,
        min_samples=min_samples,
    )


def segment_test_panel_samples_by_threshold(
    samples: pd.DataFrame,
    threshold_mV: float,
    pre_samples: int = 3,
    post_samples: int = 3,
    min_samples: int = 2,
    polarity: Literal["positive", "negative"] = "positive",
) -> pd.DataFrame:
    """Segment Test_Panel samples around explicit threshold crossings.

    This is intentionally separate from gap-based segmentation. Crossings are
    found in the continuous ``voltage_mV`` stream and expanded by the requested
    pre/post sample padding before contiguous crossing windows are converted to
    pulse rows.
    """
    _validate_segmentation_inputs(samples, min_samples=min_samples)
    if pre_samples < 0 or post_samples < 0:
        raise ValueError("pre_samples and post_samples must be non-negative.")
    if polarity not in ("positive", "negative"):
        raise ValueError("polarity must be either 'positive' or 'negative'.")

    working = samples.sort_values("time_ns").reset_index(drop=True)
    voltage_mV = pd.to_numeric(working["voltage_mV"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if polarity == "positive":
        crossing_mask = voltage_mV >= threshold_mV
    else:
        crossing_mask = voltage_mV <= -abs(threshold_mV)

    crossing_indices = np.where(crossing_mask)[0]
    if len(crossing_indices) == 0:
        raise ValueError("No threshold crossings were found.")

    selected = np.zeros(len(working), dtype=bool)
    for crossing_index in crossing_indices:
        start = max(0, crossing_index - pre_samples)
        stop = min(len(working), crossing_index + post_samples + 1)
        selected[start:stop] = True

    selected_indices = np.where(selected)[0]
    if len(selected_indices) == 0:
        raise ValueError("No samples were selected around threshold crossings.")

    starts_new_segment = np.concatenate(
        [
            np.array([True]),
            np.diff(selected_indices) > 1,
        ]
    )
    segment_ids_for_selected = np.cumsum(starts_new_segment).astype(np.int64) - 1
    segment_ids = np.full(len(working), -1, dtype=np.int64)
    segment_ids[selected_indices] = segment_ids_for_selected

    threshold_samples = working[selected].reset_index(drop=True)
    threshold_segment_ids = segment_ids[selected]

    return _build_segmented_pulse_table(
        samples=threshold_samples,
        segment_ids=threshold_segment_ids,
        min_samples=min_samples,
    )


def pulse_length_distribution(pulses: pd.DataFrame) -> pd.DataFrame:
    """Summarize segmented pulse lengths without filtering pulses."""
    required_columns = ["event_id", "time_ns", "voltage_mV"]
    missing_columns = [column for column in required_columns if column not in pulses]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return (
        pulses.groupby("event_id", sort=True)
        .agg(
            n_samples=("voltage_mV", "size"),
            duration_ns=("time_ns", lambda values: float(values.max() - values.min())),
        )
        .reset_index()
    )


def count_feature_valid_pulses(
    pulses: pd.DataFrame,
    feature_threshold_mV: float,
    n_pretrigger_samples: int = 3,
    baseline_method: str = "median",
) -> int:
    """Count pulses passing the existing feature-extraction validity rule."""
    valid_count = 0
    for _, group in pulses.groupby("event_id", sort=True):
        sorted_group = group.sort_values("time_ns")
        if len(sorted_group) < max(2, n_pretrigger_samples):
            continue
        features = extract_pulse_features(
            time_ns=sorted_group["time_ns"].to_numpy(dtype=np.float64),
            voltage_mV=sorted_group["voltage_mV"].to_numpy(dtype=np.float64),
            threshold_mV=feature_threshold_mV,
            n_pretrigger_samples=n_pretrigger_samples,
            baseline_method=baseline_method,
        )
        if features.valid:
            valid_count += 1
    return valid_count


def summarize_test_panel_segmentation(
    samples: pd.DataFrame,
    pulses: pd.DataFrame,
    feature_threshold_mV: float,
    n_pretrigger_samples: int = 3,
) -> dict[str, object]:
    """Build notebook-ready segmentation diagnostics without filtering pulses."""
    voltage_mV = pd.to_numeric(samples["voltage_mV"], errors="raise")
    lengths = pulse_length_distribution(pulses)
    return {
        "total_rows_loaded": int(len(samples)),
        "estimated_sampling_interval_ns": estimate_sampling_interval_ns(samples),
        "number_of_segmented_pulses": int(pulses["event_id"].nunique()),
        "pulse_length_distribution": lengths,
        "voltage_range_mV": {
            "min": float(voltage_mV.min()),
            "median": float(voltage_mV.median()),
            "max": float(voltage_mV.max()),
        },
        "feature_threshold_mV": float(feature_threshold_mV),
        "pulses_passing_feature_threshold": count_feature_valid_pulses(
            pulses,
            feature_threshold_mV=feature_threshold_mV,
            n_pretrigger_samples=n_pretrigger_samples,
        ),
    }


def load_test_panel_h5_samples(
    path: str | Path,
    voltage_unit: VoltageUnit = "auto",
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load a Test_Panel HDF5 file into a sample-level DataFrame.

    The current Test_Panel files store a Pandas/PyTables-style frame under
    ``/datos``. This loader reconstructs the logical columns from the block
    item labels and block value arrays; it does not segment pulses.

    Args:
        path: Path to one Test_Panel ``.h5`` file.
        voltage_unit: Unit for the stored ``voltaje`` column. ``"auto"`` uses
            the documented median-absolute-value rule and records the resolved
            assumption in ``DataFrame.attrs``.
        max_rows: Optional number of leading rows to load. This is useful for
            deadline/debug notebooks that intentionally work on a bounded sample
            instead of materializing the full HDF5 run.

    Returns:
        DataFrame with one row per stored sample.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If required datasets or logical columns are missing.
    """
    hdf5_path = Path(path)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided.")

    metadata = parse_test_panel_filename(hdf5_path)

    with h5py.File(hdf5_path, "r") as hdf5_file:
        missing = [
            dataset_path
            for dataset_path in _REQUIRED_DATASETS
            if dataset_path not in hdf5_file
        ]
        if missing:
            raise ValueError(f"Missing required HDF5 datasets: {missing}")

        block0_items = _read_items(hdf5_file["/datos/block0_items"])
        block1_items = _read_items(hdf5_file["/datos/block1_items"])
        axis0_items = _read_items(hdf5_file["/datos/axis0"])
        row_slice = slice(None) if max_rows is None else slice(0, max_rows)
        block0_values = np.asarray(hdf5_file["/datos/block0_values"][row_slice])
        block1_values = np.asarray(hdf5_file["/datos/block1_values"][row_slice])
        block0 = _read_block_from_array(block0_values, block0_items)
        block1 = _read_block_from_array(block1_values, block1_items)

        if len(block0) != len(block1):
            raise ValueError(
                "HDF5 value blocks have different row counts: "
                f"{len(block0)} and {len(block1)}."
            )

        sample_index = np.asarray(hdf5_file["/datos/axis1"][row_slice]).reshape(-1)
        if len(sample_index) != len(block0):
            raise ValueError(
                "Sample index length does not match value block row count: "
                f"{len(sample_index)} and {len(block0)}."
            )

    dataframe = pd.concat([block0, block1], axis=1)
    if list(dataframe.columns) != axis0_items:
        raise ValueError(
            "Reconstructed HDF5 columns do not match /datos/axis0 labels: "
            f"{list(dataframe.columns)} vs {axis0_items}."
        )

    missing_columns = [
        column for column in _REQUIRED_LOGICAL_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing logical columns: {missing_columns}")

    voltage_raw = pd.to_numeric(dataframe["voltaje"], errors="raise")
    voltage_mV, resolved_voltage_unit, voltage_rule = convert_test_panel_voltage(
        voltage_raw,
        voltage_unit=voltage_unit,
    )

    output = pd.DataFrame(
        {
            "sample_index": sample_index.astype(np.int64, copy=False),
            "adc_value": dataframe["adc_value"].to_numpy(dtype=np.int16),
            "voltage_raw": voltage_raw,
            "voltage_mV": voltage_mV,
            "seconds": pd.to_numeric(dataframe["segundos"], errors="raise"),
            "nanoseconds": pd.to_numeric(dataframe["nanosegundos"], errors="raise"),
            "time_total_seconds": pd.to_numeric(
                dataframe["tiempo_total_segundos"],
                errors="raise",
            ),
        }
    )
    output["time_ns"] = output["time_total_seconds"] * 1_000_000_000.0
    output["source_file"] = hdf5_path.name
    output["panel"] = metadata["panel"]
    output["run_id"] = _run_id(metadata)
    output["acquisition_date"] = metadata["acquisition_date"]
    output["acquisition_duration"] = metadata["duration_label"]
    output["acquisition_duration_minutes"] = metadata["duration_minutes"]
    output["threshold_token"] = _threshold_token(hdf5_path)
    output["falling_edge_adc_threshold"] = metadata["falling_edge_adc_threshold"]
    output["rising_edge_adc_threshold"] = metadata["rising_edge_adc_threshold"]
    output.attrs["voltage_unit_requested"] = voltage_unit
    output.attrs["voltage_unit_resolved"] = resolved_voltage_unit
    output.attrs["voltage_conversion_assumption"] = voltage_rule

    return output
