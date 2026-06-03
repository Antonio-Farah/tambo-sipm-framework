"""Input/output loaders for TAMBO simulation data.

This module provides lightweight CSV loading utilities for photon-count data
and waveform data.

Conventions:
    - Time is expressed in ns.
    - Voltage is expressed in mV.
    - Photon counts are non-negative integers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def load_csv_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV file as a pandas DataFrame.

    Args:
        path: CSV file path.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty.
    """
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)

    if dataframe.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    return dataframe


def require_columns(dataframe: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate that a DataFrame contains required columns.

    Args:
        dataframe: Input DataFrame.
        required_columns: Required column names.

    Raises:
        ValueError: If at least one required column is missing.
    """
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def load_photon_counts_csv(
    path: str | Path,
    photon_count_column: str = "generated_photons",
) -> NDArray[np.int64]:
    """Load generated photon counts from a CSV file.

    Args:
        path: CSV file path.
        photon_count_column: Column containing generated photon counts.

    Returns:
        Photon counts as a NumPy int64 array.

    Raises:
        ValueError: If the photon-count column is invalid.
    """
    dataframe = load_csv_table(path)
    require_columns(dataframe, [photon_count_column])

    values = dataframe[photon_count_column]

    if np.any(pd.isna(values)):
        raise ValueError("Photon count column contains NaN values.")

    numeric_values = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)

    if np.any(numeric_values < 0):
        raise ValueError("Photon counts must be non-negative.")

    if not np.all(np.equal(numeric_values, np.floor(numeric_values))):
        raise ValueError("Photon counts must be integer-valued.")

    return numeric_values.astype(np.int64)


def load_single_waveform_csv(
    path: str | Path,
    time_column: str = "time_ns",
    voltage_column: str = "voltage_mV",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load a single waveform from a CSV file.

    Args:
        path: CSV file path.
        time_column: Column containing time values in ns.
        voltage_column: Column containing voltage values in mV.

    Returns:
        Tuple containing:
            - Time array in ns.
            - Voltage array in mV.

    Raises:
        ValueError: If columns are missing, contain NaN, or time is invalid.
    """
    dataframe = load_csv_table(path)
    require_columns(dataframe, [time_column, voltage_column])

    time_ns = pd.to_numeric(dataframe[time_column], errors="raise").to_numpy(
        dtype=np.float64
    )
    voltage_mV = pd.to_numeric(
        dataframe[voltage_column], errors="raise"
    ).to_numpy(dtype=np.float64)

    if np.any(np.isnan(time_ns)) or np.any(np.isnan(voltage_mV)):
        raise ValueError("Waveform columns must not contain NaN values.")

    order = np.argsort(time_ns)
    time_ns = time_ns[order]
    voltage_mV = voltage_mV[order]

    if len(time_ns) < 2:
        raise ValueError("Waveform must contain at least two samples.")

    if not np.all(np.diff(time_ns) > 0):
        raise ValueError("Time values must be strictly increasing.")

    return time_ns, voltage_mV


def load_waveform_collection_csv(
    path: str | Path,
    event_column: str = "event_id",
    time_column: str = "time_ns",
    voltage_column: str = "voltage_mV",
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Load multiple waveforms from a long-format CSV file.

    The expected format is one row per sample, with columns identifying the
    event, time, and voltage.

    Args:
        path: CSV file path.
        event_column: Column identifying the waveform/event.
        time_column: Column containing time values in ns.
        voltage_column: Column containing voltage values in mV.

    Returns:
        Dictionary mapping event IDs to (time_ns, voltage_mV) arrays.
    """
    dataframe = load_csv_table(path)
    require_columns(dataframe, [event_column, time_column, voltage_column])

    waveforms: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}

    for event_id, group in dataframe.groupby(event_column):
        sorted_group = group.sort_values(time_column)

        time_ns = pd.to_numeric(
            sorted_group[time_column], errors="raise"
        ).to_numpy(dtype=np.float64)
        voltage_mV = pd.to_numeric(
            sorted_group[voltage_column], errors="raise"
        ).to_numpy(dtype=np.float64)

        if len(time_ns) < 2:
            raise ValueError(f"Waveform {event_id} must contain at least two samples.")

        if np.any(np.isnan(time_ns)) or np.any(np.isnan(voltage_mV)):
            raise ValueError(f"Waveform {event_id} contains NaN values.")

        if not np.all(np.diff(time_ns) > 0):
            raise ValueError(f"Waveform {event_id} time values must increase strictly.")

        waveforms[str(event_id)] = (time_ns, voltage_mV)

    if not waveforms:
        raise ValueError("No waveforms were loaded.")

    return waveforms