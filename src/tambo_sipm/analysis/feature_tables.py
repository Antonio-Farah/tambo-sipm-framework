"""Batch feature extraction for waveform collections.

This module converts long-format pulse tables or waveform dictionaries into
feature tables suitable for measured-versus-simulated comparisons.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tambo_sipm.analysis.features import extract_pulse_features


def _require_columns(dataframe: pd.DataFrame, columns: list[str]) -> None:
    """Validate required DataFrame columns.

    Args:
        dataframe: Input DataFrame.
        columns: Required columns.

    Raises:
        ValueError: If at least one column is missing.
    """
    missing_columns = [column for column in columns if column not in dataframe.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _base_waveform_metadata(
    event_id: str,
    time_ns: NDArray[np.float64],
    voltage_mV: NDArray[np.float64],
    threshold_mV: float,
    n_pretrigger_samples: int,
) -> dict[str, Any]:
    """Build metadata shared by successful and failed feature rows."""
    n_samples = int(len(time_ns))

    duration_ns = (
        float(np.max(time_ns) - np.min(time_ns))
        if n_samples >= 2
        else 0.0
    )

    raw_peak_mV = float(np.max(voltage_mV)) if n_samples > 0 else np.nan
    raw_min_mV = float(np.min(voltage_mV)) if n_samples > 0 else np.nan

    return {
        "event_id": event_id,
        "n_samples": n_samples,
        "duration_ns": duration_ns,
        "raw_peak_mV": raw_peak_mV,
        "raw_min_mV": raw_min_mV,
        "threshold_mV": float(threshold_mV),
        "n_pretrigger_samples": int(n_pretrigger_samples),
    }


def _empty_feature_values() -> dict[str, Any]:
    """Return empty feature values for failed extractions."""
    return {
        "valid": False,
        "baseline_mV": np.nan,
        "noise_rms_mV": np.nan,
        "peak_mV": np.nan,
        "peak_time_ns": np.nan,
        "width_ns": np.nan,
        "integral_mVns": np.nan,
        "rms_mV": np.nan,
        "first_crossing_time_ns": np.nan,
        "last_crossing_time_ns": np.nan,
        "n_samples_above_threshold": 0,
    }


def extract_waveform_feature_row(
    event_id: str,
    time_ns: NDArray[np.float64],
    voltage_mV: NDArray[np.float64],
    threshold_mV: float = 18.0,
    n_pretrigger_samples: int = 3,
    baseline_method: str = "median",
) -> dict[str, Any]:
    """Extract one feature row from a waveform.

    Args:
        event_id: Event identifier.
        time_ns: Time array in ns.
        voltage_mV: Voltage array in mV.
        threshold_mV: Threshold after baseline subtraction.
        n_pretrigger_samples: Number of samples used for baseline estimation.
        baseline_method: Baseline estimator, usually "median" or "mean".

    Returns:
        Dictionary containing waveform metadata and extracted features.
    """
    time_array = np.asarray(time_ns, dtype=np.float64)
    voltage_array = np.asarray(voltage_mV, dtype=np.float64)

    row = _base_waveform_metadata(
        event_id=event_id,
        time_ns=time_array,
        voltage_mV=voltage_array,
        threshold_mV=threshold_mV,
        n_pretrigger_samples=n_pretrigger_samples,
    )

    if len(time_array) < n_pretrigger_samples:
        row.update(_empty_feature_values())
        row["extraction_status"] = "too_short_for_baseline"
        return row

    try:
        features = extract_pulse_features(
            time_ns=time_array,
            voltage_mV=voltage_array,
            threshold_mV=threshold_mV,
            n_pretrigger_samples=n_pretrigger_samples,
            baseline_method=baseline_method,
        )

        feature_values = asdict(features)

        for key, value in feature_values.items():
            if value is None:
                feature_values[key] = np.nan

        row.update(feature_values)
        row["extraction_status"] = "ok"

    except ValueError as exc:
        row.update(_empty_feature_values())
        row["extraction_status"] = f"error: {exc}"

    return row


def extract_features_from_pulse_table(
    pulses: pd.DataFrame,
    event_id_column: str = "event_id",
    time_column: str = "time_ns",
    voltage_column: str = "voltage_mV",
    threshold_mV: float = 18.0,
    n_pretrigger_samples: int = 3,
    baseline_method: str = "median",
) -> pd.DataFrame:
    """Extract pulse features from a long-format pulse table.

    Args:
        pulses: Long-format pulse table.
        event_id_column: Event identifier column.
        time_column: Time column in ns.
        voltage_column: Voltage column in mV.
        threshold_mV: Threshold after baseline subtraction.
        n_pretrigger_samples: Number of samples used for baseline estimation.
        baseline_method: Baseline estimator.

    Returns:
        Feature table with one row per event.
    """
    _require_columns(pulses, [event_id_column, time_column, voltage_column])

    rows: list[dict[str, Any]] = []

    for event_id, group in pulses.groupby(event_id_column, sort=True):
        sorted_group = group.sort_values(time_column)

        row = extract_waveform_feature_row(
            event_id=str(event_id),
            time_ns=sorted_group[time_column].to_numpy(dtype=np.float64),
            voltage_mV=sorted_group[voltage_column].to_numpy(dtype=np.float64),
            threshold_mV=threshold_mV,
            n_pretrigger_samples=n_pretrigger_samples,
            baseline_method=baseline_method,
        )

        rows.append(row)

    return pd.DataFrame(rows)


def extract_features_from_waveform_dict(
    waveforms: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    threshold_mV: float = 18.0,
    n_pretrigger_samples: int = 3,
    baseline_method: str = "median",
) -> pd.DataFrame:
    """Extract features from a dictionary of waveforms.

    Args:
        waveforms: Mapping from event ID to (time_ns, voltage_mV).
        threshold_mV: Threshold after baseline subtraction.
        n_pretrigger_samples: Number of samples used for baseline estimation.
        baseline_method: Baseline estimator.

    Returns:
        Feature table with one row per event.
    """
    rows = [
        extract_waveform_feature_row(
            event_id=event_id,
            time_ns=time_ns,
            voltage_mV=voltage_mV,
            threshold_mV=threshold_mV,
            n_pretrigger_samples=n_pretrigger_samples,
            baseline_method=baseline_method,
        )
        for event_id, (time_ns, voltage_mV) in waveforms.items()
    ]

    return pd.DataFrame(rows)


def summarize_feature_table(features: pd.DataFrame) -> pd.DataFrame:
    """Summarize feature extraction status.

    Args:
        features: Feature table.

    Returns:
        Summary table by extraction status and validity.
    """
    _require_columns(features, ["extraction_status", "valid"])

    summary = (
        features.groupby(["extraction_status", "valid"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    return summary