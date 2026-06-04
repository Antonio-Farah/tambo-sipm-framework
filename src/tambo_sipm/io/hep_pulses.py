"""Load and segment HEP/PUCP measured SiPM pulses.

This module handles the raw 10min.csv file measured by the HEP group at PUCP.

The raw file is interpreted as a time series. Pulses are segmented using time
gaps between consecutive samples. The standard output format is long-format:

    event_id, pulse_id, sample_index, time_ns, absolute_time_ns, voltage_mV

Conventions:
    - Input voltage is in V.
    - Output voltage is in mV.
    - Input time is taken from tiempo_total_segundos.
    - Output time is in ns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


REQUIRED_HEP_COLUMNS = [
    "voltaje",
    "tiempo_total_segundos",
]


def load_hep_timeseries(
    path: str | Path,
    voltage_column: str = "voltaje",
    time_seconds_column: str = "tiempo_total_segundos",
) -> pd.DataFrame:
    """Load the raw HEP SiPM time series.

    Args:
        path: CSV file path.
        voltage_column: Column containing voltage in V.
        time_seconds_column: Column containing total time in seconds.

    Returns:
        DataFrame with absolute_time_ns, time_ns, voltage_V and voltage_mV.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or invalid.
    """
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)

    missing_columns = [
        column
        for column in [voltage_column, time_seconds_column]
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if dataframe.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    voltage_v = pd.to_numeric(dataframe[voltage_column], errors="raise")
    time_seconds = pd.to_numeric(dataframe[time_seconds_column], errors="raise")

    if voltage_v.isna().any():
        raise ValueError("Voltage column contains NaN values.")
    if time_seconds.isna().any():
        raise ValueError("Time column contains NaN values.")

    result = dataframe.copy()
    result["voltage_V"] = voltage_v.astype(float)
    result["voltage_mV"] = result["voltage_V"] * 1000.0
    result["absolute_time_ns"] = time_seconds.astype(float) * 1.0e9

    result = result.sort_values("absolute_time_ns").reset_index(drop=True)

    if len(result) < 2:
        raise ValueError("Time series must contain at least two samples.")

    if not np.all(np.diff(result["absolute_time_ns"].to_numpy()) >= 0.0):
        raise ValueError("absolute_time_ns must be non-decreasing.")

    first_time_ns = float(result["absolute_time_ns"].iloc[0])
    result["time_ns"] = result["absolute_time_ns"] - first_time_ns

    return result


def assign_pulse_ids_from_time_gaps(
    time_ns: NDArray[np.float64],
    gap_threshold_ns: float = 50.0,
) -> NDArray[np.int64]:
    """Assign pulse IDs using gaps between consecutive samples.

    Args:
        time_ns: Time array in ns.
        gap_threshold_ns: New pulse threshold in ns.

    Returns:
        Pulse ID array.

    Raises:
        ValueError: If inputs are invalid.
    """
    if gap_threshold_ns <= 0.0:
        raise ValueError("gap_threshold_ns must be positive.")

    time_array = np.asarray(time_ns, dtype=np.float64)

    if time_array.ndim != 1:
        raise ValueError("time_ns must be one-dimensional.")
    if len(time_array) == 0:
        raise ValueError("time_ns must not be empty.")
    if not np.all(np.isfinite(time_array)):
        raise ValueError("time_ns must contain finite values.")
    if not np.all(np.diff(time_array) >= 0.0):
        raise ValueError("time_ns must be non-decreasing.")

    if len(time_array) == 1:
        return np.array([0], dtype=np.int64)

    gaps_ns = np.diff(time_array)
    starts_new_pulse = np.concatenate(
        [
            np.array([True]),
            gaps_ns > gap_threshold_ns,
        ]
    )

    return np.cumsum(starts_new_pulse).astype(np.int64) - 1


def segment_hep_pulses(
    timeseries: pd.DataFrame,
    gap_threshold_ns: float = 50.0,
    min_samples: int = 2,
) -> pd.DataFrame:
    """Segment the HEP time series into individual pulses.

    Args:
        timeseries: DataFrame from load_hep_timeseries.
        gap_threshold_ns: New pulse threshold in ns.
        min_samples: Minimum number of samples required to keep a pulse.

    Returns:
        Long-format pulse table.

    Raises:
        ValueError: If required columns are missing or min_samples is invalid.
    """
    required_columns = ["absolute_time_ns", "time_ns", "voltage_mV"]
    missing_columns = [
        column for column in required_columns if column not in timeseries.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if min_samples <= 0:
        raise ValueError("min_samples must be positive.")

    working_dataframe = timeseries.sort_values("absolute_time_ns").reset_index(
        drop=True
    )

    pulse_ids = assign_pulse_ids_from_time_gaps(
        time_ns=working_dataframe["time_ns"].to_numpy(dtype=np.float64),
        gap_threshold_ns=gap_threshold_ns,
    )

    working_dataframe["pulse_id_original"] = pulse_ids

    pulse_sizes = working_dataframe.groupby("pulse_id_original").size()
    valid_original_ids = pulse_sizes[pulse_sizes >= min_samples].index

    working_dataframe = working_dataframe[
        working_dataframe["pulse_id_original"].isin(valid_original_ids)
    ].copy()

    if working_dataframe.empty:
        raise ValueError("No pulses remain after applying min_samples.")

    id_mapping = {
        original_id: new_id
        for new_id, original_id in enumerate(sorted(valid_original_ids))
    }

    working_dataframe["pulse_id"] = working_dataframe["pulse_id_original"].map(
        id_mapping
    )

    rows: list[pd.DataFrame] = []

    for pulse_id, group in working_dataframe.groupby("pulse_id", sort=True):
        pulse = group.sort_values("absolute_time_ns").copy()
        first_absolute_time_ns = float(pulse["absolute_time_ns"].iloc[0])

        pulse["sample_index"] = np.arange(len(pulse), dtype=np.int64)
        pulse["time_ns"] = pulse["absolute_time_ns"] - first_absolute_time_ns
        pulse["event_id"] = f"real_pulse_{int(pulse_id):05d}"

        rows.append(pulse)

    pulses = pd.concat(rows, ignore_index=True)

    output_columns = [
        "event_id",
        "pulse_id",
        "sample_index",
        "time_ns",
        "absolute_time_ns",
        "voltage_mV",
    ]

    optional_columns = [
        "voltage_V",
        "segundos",
        "nanosegundos",
        "tiempo_total_segundos",
    ]

    for column in optional_columns:
        if column in pulses.columns:
            output_columns.append(column)

    return pulses[output_columns]


def load_segmented_hep_pulses(
    path: str | Path,
    gap_threshold_ns: float = 50.0,
    min_samples: int = 2,
) -> pd.DataFrame:
    """Load and segment HEP measured pulses from CSV.

    Args:
        path: CSV file path.
        gap_threshold_ns: New pulse threshold in ns.
        min_samples: Minimum number of samples required to keep a pulse.

    Returns:
        Long-format pulse table.
    """
    timeseries = load_hep_timeseries(path)

    return segment_hep_pulses(
        timeseries=timeseries,
        gap_threshold_ns=gap_threshold_ns,
        min_samples=min_samples,
    )


def pulse_lengths(pulses: pd.DataFrame) -> pd.DataFrame:
    """Compute pulse lengths in samples and ns.

    Args:
        pulses: Long-format pulse table.

    Returns:
        DataFrame with one row per pulse.
    """
    required_columns = ["event_id", "time_ns", "voltage_mV"]
    missing_columns = [
        column for column in required_columns if column not in pulses.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    summary = (
        pulses.groupby("event_id")
        .agg(
            n_samples=("voltage_mV", "size"),
            duration_ns=("time_ns", lambda values: float(values.max() - values.min())),
            max_voltage_mV=("voltage_mV", "max"),
            min_voltage_mV=("voltage_mV", "min"),
        )
        .reset_index()
    )

    return summary


def pulse_collection_to_dict(
    pulses: pd.DataFrame,
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Convert a long-format pulse table to a waveform dictionary.

    Args:
        pulses: Long-format pulse table.

    Returns:
        Dictionary mapping event_id to (time_ns, voltage_mV).
    """
    required_columns = ["event_id", "time_ns", "voltage_mV"]
    missing_columns = [
        column for column in required_columns if column not in pulses.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    waveforms: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}

    for event_id, group in pulses.groupby("event_id", sort=True):
        sorted_group = group.sort_values("time_ns")

        waveforms[str(event_id)] = (
            sorted_group["time_ns"].to_numpy(dtype=np.float64),
            sorted_group["voltage_mV"].to_numpy(dtype=np.float64),
        )

    return waveforms