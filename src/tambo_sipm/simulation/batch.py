"""Batch simulation utilities for TAMBO detector events.

This module runs the detector simulation chain over a table of photon events
and extracts pulse features from the digitized simulated waveforms.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tambo_sipm.analysis.feature_tables import extract_waveform_feature_row
from tambo_sipm.detector.event import simulate_detector_event_from_config


def require_event_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Validate required columns in a photon-event table.

    Args:
        dataframe: Input event table.
        required_columns: Required column names.

    Raises:
        ValueError: If required columns are missing.
    """
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def select_photon_events(
    photon_events: pd.DataFrame,
    max_events: int | None = None,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Select photon events for batch simulation.

    Args:
        photon_events: Input photon-event table.
        max_events: Optional maximum number of events to simulate.
        random_seed: Optional random seed for reproducible random sampling.

    Returns:
        Selected photon-event table.

    Raises:
        ValueError: If max_events is invalid.
    """
    if max_events is None:
        return photon_events.copy().reset_index(drop=True)

    if max_events <= 0:
        raise ValueError("max_events must be positive.")

    if max_events >= len(photon_events):
        return photon_events.copy().reset_index(drop=True)

    if random_seed is None:
        return photon_events.head(max_events).copy().reset_index(drop=True)

    return (
        photon_events.sample(n=max_events, random_state=random_seed)
        .reset_index(drop=True)
    )


def _copy_metadata(
    event_row: pd.Series,
    metadata_columns: list[str],
) -> dict[str, Any]:
    """Copy selected metadata columns from one event row."""
    metadata: dict[str, Any] = {}

    for column in metadata_columns:
        if column in event_row.index:
            metadata[column] = event_row[column]

    return metadata


def simulate_feature_row_from_photon_event(
    event_row: pd.Series,
    config: dict[str, Any],
    threshold_mV: float = 18.0,
    n_pretrigger_samples: int = 3,
    baseline_method: str = "median",
    rng: np.random.Generator | None = None,
    metadata_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Simulate one photon event and extract pulse features.

    Args:
        event_row: Row from a photon-event table.
        config: Detector-event simulation configuration dictionary.
        threshold_mV: Threshold after baseline subtraction.
        n_pretrigger_samples: Number of pre-trigger samples for baseline.
        baseline_method: Baseline estimator.
        rng: Optional NumPy random generator.
        metadata_columns: Metadata columns copied into the output row.

    Returns:
        Dictionary with metadata, simulation metadata, and extracted features.
    """
    if "generated_photons" not in event_row.index:
        raise ValueError("event_row must contain generated_photons.")

    if "event_id" in event_row.index:
        event_id = str(event_row["event_id"])
    else:
        event_id = "simulated_event"

    if rng is None:
        rng = np.random.default_rng()

    generated_photons = int(event_row["generated_photons"])

    simulation_result = simulate_detector_event_from_config(
        generated_photons=generated_photons,
        config=config,
        rng=rng,
    )

    feature_row = extract_waveform_feature_row(
        event_id=event_id,
        time_ns=simulation_result.sampled_time_ns,
        voltage_mV=simulation_result.digitized_voltage_mV,
        threshold_mV=threshold_mV,
        n_pretrigger_samples=n_pretrigger_samples,
        baseline_method=baseline_method,
    )

    if metadata_columns is None:
        metadata_columns = [
            "source_row",
            "detector",
            "shower",
            "particle_id",
            "particle_name",
            "particle_name_original",
            "particle_category",
            "generated_photons",
            "energia_detectada_poli_MeV",
            "tiempo_deteccion",
            "x",
            "y",
            "ek",
        ]

    metadata = _copy_metadata(event_row, metadata_columns)

    simulation_metadata = {
        "photons_at_sipm": int(
            simulation_result.photon_transport.photons_at_sipm[0]
        ),
        "photoelectrons": int(
            simulation_result.photon_transport.photoelectrons[0]
        ),
        "fired_microcells": int(simulation_result.sipm_response.fired_microcells),
        "voltage_scale_mV_per_pe": float(
            simulation_result.voltage_scale_mV_per_pe
        ),
        "max_digitized_voltage_mV": float(
            np.max(simulation_result.digitized_voltage_mV)
        ),
        "min_digitized_voltage_mV": float(
            np.min(simulation_result.digitized_voltage_mV)
        ),
    }

    output_row = {
        **metadata,
        **simulation_metadata,
        **feature_row,
    }

    return output_row


def simulate_feature_table_from_photon_events(
    photon_events: pd.DataFrame,
    config: dict[str, Any],
    max_events: int | None = None,
    random_seed: int | None = None,
    threshold_mV: float = 18.0,
    n_pretrigger_samples: int = 3,
    baseline_method: str = "median",
) -> pd.DataFrame:
    """Simulate a feature table from photon events.

    Args:
        photon_events: Photon-event table with generated_photons.
        config: Detector-event simulation configuration dictionary.
        max_events: Optional maximum number of events to simulate.
        random_seed: Optional seed for event selection and simulation.
        threshold_mV: Threshold after baseline subtraction.
        n_pretrigger_samples: Number of pre-trigger samples for baseline.
        baseline_method: Baseline estimator.

    Returns:
        Feature table with one row per simulated event.
    """
    require_event_columns(photon_events, ["event_id", "generated_photons"])

    selected_events = select_photon_events(
        photon_events=photon_events,
        max_events=max_events,
        random_seed=random_seed,
    )

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []

    for _, event_row in selected_events.iterrows():
        rows.append(
            simulate_feature_row_from_photon_event(
                event_row=event_row,
                config=config,
                threshold_mV=threshold_mV,
                n_pretrigger_samples=n_pretrigger_samples,
                baseline_method=baseline_method,
                rng=rng,
            )
        )

    return pd.DataFrame(rows)