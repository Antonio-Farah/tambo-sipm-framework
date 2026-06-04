"""Calibration utilities for TAMBO detector simulations.

This module performs grid-search calibration over physically meaningful
simulation parameters.

Current calibration strategy:
    Fixed:
        - voltage_scale_mV_per_pe
        - PDE

    Swept:
        - transport_efficiency
        - tau_f_ns
        - arrival_spread_ns

The calibration score is computed from real-vs-simulated feature differences
for:
    - peak_mV
    - rms_mV
    - integral_mVns
    - width_ns

The valid-event fraction is reported as a diagnostic, but it is not included
in the main score. This avoids mixing calibration with trigger/selection
efficiency effects.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from tambo_sipm.analysis.comparison import compare_real_sim_features
from tambo_sipm.simulation.batch import simulate_feature_table_from_photon_events


DEFAULT_CALIBRATION_FEATURES = [
    "peak_mV",
    "rms_mV",
    "integral_mVns",
    "width_ns",
]

DEFAULT_CALIBRATION_WEIGHTS = {
    "peak_mV": 1.0,
    "rms_mV": 1.0,
    "integral_mVns": 1.0,
    "width_ns": 1.0,
}


def set_nested_config_value(
    config: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> dict[str, Any]:
    """Set a nested value in a copied configuration dictionary.

    Args:
        config: Input configuration dictionary.
        path: Nested key path.
        value: Value to set.

    Returns:
        New configuration dictionary with the updated value.

    Raises:
        ValueError: If path is empty.
        KeyError: If an intermediate section is missing.
    """
    if len(path) == 0:
        raise ValueError("path must not be empty.")

    updated_config = deepcopy(config)
    current = updated_config

    for key in path[:-1]:
        if key not in current:
            raise KeyError(f"Missing configuration section: {key}")

        if not isinstance(current[key], dict):
            raise ValueError(f"Configuration section is not a dictionary: {key}")

        current = current[key]

    current[path[-1]] = value

    return updated_config


def build_calibration_config(
    base_config: dict[str, Any],
    transport_efficiency: float,
    tau_f_ns: float,
    arrival_spread_ns: float,
) -> dict[str, Any]:
    """Build a calibrated simulation configuration.

    Args:
        base_config: Base detector simulation configuration.
        transport_efficiency: Optical transport efficiency.
        tau_f_ns: SiPM fall-time constant in ns.
        arrival_spread_ns: Photon/photoelectron arrival-time spread in ns.

    Returns:
        New configuration dictionary.
    """
    config = set_nested_config_value(
        config=base_config,
        path=("photon_transport", "transport_efficiency"),
        value=float(transport_efficiency),
    )

    config = set_nested_config_value(
        config=config,
        path=("sipm_response", "tau_f_ns"),
        value=float(tau_f_ns),
    )

    config = set_nested_config_value(
        config=config,
        path=("sipm_response", "arrival_spread_ns"),
        value=float(arrival_spread_ns),
    )

    return config


def validate_calibration_grid_values(
    transport_efficiencies: list[float],
    tau_f_values_ns: list[float],
    arrival_spread_values_ns: list[float],
) -> None:
    """Validate calibration grid values.

    Args:
        transport_efficiencies: Transport efficiency values.
        tau_f_values_ns: Fall-time constants in ns.
        arrival_spread_values_ns: Arrival-time spread values in ns.

    Raises:
        ValueError: If any grid is empty or contains invalid values.
    """
    if len(transport_efficiencies) == 0:
        raise ValueError("transport_efficiencies must not be empty.")
    if len(tau_f_values_ns) == 0:
        raise ValueError("tau_f_values_ns must not be empty.")
    if len(arrival_spread_values_ns) == 0:
        raise ValueError("arrival_spread_values_ns must not be empty.")

    if any(value < 0.0 or value > 1.0 for value in transport_efficiencies):
        raise ValueError("transport_efficiencies must be in [0, 1].")

    if any(value <= 0.0 for value in tau_f_values_ns):
        raise ValueError("tau_f_values_ns must contain positive values.")

    if any(value < 0.0 for value in arrival_spread_values_ns):
        raise ValueError(
            "arrival_spread_values_ns must contain non-negative values."
        )


def feature_error_from_distribution_table(
    distribution_comparison: pd.DataFrame,
    feature: str,
    metric_column: str = "mean_relative_difference",
) -> float:
    """Extract an absolute feature error from a comparison table.

    Args:
        distribution_comparison: Output of compare_feature_distributions.
        feature: Feature name.
        metric_column: Metric used to define the error.

    Returns:
        Absolute error value.

    Raises:
        ValueError: If the feature or metric column is missing.
    """
    if metric_column not in distribution_comparison.columns:
        raise ValueError(f"Missing metric column: {metric_column}")

    feature_rows = distribution_comparison[
        distribution_comparison["feature"] == feature
    ]

    if feature_rows.empty:
        raise ValueError(f"Feature not found in comparison table: {feature}")

    value = float(feature_rows.iloc[0][metric_column])

    return abs(value)


def calibration_score_from_distribution_table(
    distribution_comparison: pd.DataFrame,
    feature_columns: list[str] | None = None,
    feature_weights: dict[str, float] | None = None,
    metric_column: str = "mean_relative_difference",
) -> dict[str, float]:
    """Compute calibration score from distribution comparison metrics.

    Args:
        distribution_comparison: Feature distribution comparison table.
        feature_columns: Features used in the score.
        feature_weights: Feature weights.
        metric_column: Metric used to define each feature error.

    Returns:
        Dictionary containing per-feature errors and total feature score.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_CALIBRATION_FEATURES

    if feature_weights is None:
        feature_weights = DEFAULT_CALIBRATION_WEIGHTS

    errors: dict[str, float] = {}
    weighted_errors: list[float] = []
    weights: list[float] = []

    for feature in feature_columns:
        error = feature_error_from_distribution_table(
            distribution_comparison=distribution_comparison,
            feature=feature,
            metric_column=metric_column,
        )

        weight = float(feature_weights.get(feature, 1.0))

        errors[f"{feature}_error"] = error
        weighted_errors.append(weight * error)
        weights.append(weight)

    feature_score = float(np.sum(weighted_errors) / np.sum(weights))

    errors["feature_score"] = feature_score
    errors["score"] = feature_score

    return errors


def calibration_grid_rows(
    transport_efficiencies: list[float],
    tau_f_values_ns: list[float],
    arrival_spread_values_ns: list[float],
) -> list[dict[str, float]]:
    """Build calibration grid rows.

    Args:
        transport_efficiencies: Transport efficiency values.
        tau_f_values_ns: Fall-time constants in ns.
        arrival_spread_values_ns: Arrival-time spread values in ns.

    Returns:
        List of parameter dictionaries.
    """
    validate_calibration_grid_values(
        transport_efficiencies=transport_efficiencies,
        tau_f_values_ns=tau_f_values_ns,
        arrival_spread_values_ns=arrival_spread_values_ns,
    )

    rows: list[dict[str, float]] = []

    for transport_efficiency, tau_f_ns, arrival_spread_ns in product(
        transport_efficiencies,
        tau_f_values_ns,
        arrival_spread_values_ns,
    ):
        rows.append(
            {
                "transport_efficiency": float(transport_efficiency),
                "tau_f_ns": float(tau_f_ns),
                "arrival_spread_ns": float(arrival_spread_ns),
            }
        )

    return rows


def _extract_valid_fraction(
    validity_summary: pd.DataFrame,
    dataset: str,
) -> float:
    """Extract valid fraction for one dataset from a validity summary."""
    dataset_rows = validity_summary[validity_summary["dataset"] == dataset]

    if dataset_rows.empty:
        raise ValueError(f"Dataset not found in validity summary: {dataset}")

    return float(dataset_rows["valid_fraction"].iloc[0])


def run_single_calibration_point(
    real_features: pd.DataFrame,
    photon_events: pd.DataFrame,
    base_config: dict[str, Any],
    transport_efficiency: float,
    tau_f_ns: float,
    arrival_spread_ns: float,
    max_events: int | None = None,
    random_seed: int | None = None,
    threshold_mV: float = 18.0,
    n_pretrigger_samples: int = 3,
    feature_columns: list[str] | None = None,
    feature_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run one calibration point and return score metrics.

    Args:
        real_features: Real measured feature table.
        photon_events: Photon-event table from CORSIKA row-level events.
        base_config: Base detector simulation configuration.
        transport_efficiency: Optical transport efficiency.
        tau_f_ns: SiPM fall-time constant in ns.
        arrival_spread_ns: Photon/photoelectron arrival-time spread in ns.
        max_events: Optional number of simulated events.
        random_seed: Random seed.
        threshold_mV: Feature extraction threshold.
        n_pretrigger_samples: Number of samples used for baseline.
        feature_columns: Feature columns used for comparison.
        feature_weights: Feature weights for scoring.

    Returns:
        Dictionary with calibration parameters and score metrics.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_CALIBRATION_FEATURES

    config = build_calibration_config(
        base_config=base_config,
        transport_efficiency=transport_efficiency,
        tau_f_ns=tau_f_ns,
        arrival_spread_ns=arrival_spread_ns,
    )

    simulated_features = simulate_feature_table_from_photon_events(
        photon_events=photon_events,
        config=config,
        max_events=max_events,
        random_seed=random_seed,
        threshold_mV=threshold_mV,
        n_pretrigger_samples=n_pretrigger_samples,
    )

    comparison = compare_real_sim_features(
        real_features=real_features,
        simulated_features=simulated_features,
        feature_columns=feature_columns,
        require_valid=True,
    )

    score_values = calibration_score_from_distribution_table(
        distribution_comparison=comparison.distribution_comparison,
        feature_columns=feature_columns,
        feature_weights=feature_weights,
    )

    real_valid_fraction = _extract_valid_fraction(
        validity_summary=comparison.validity_summary,
        dataset="real",
    )
    simulated_valid_fraction = _extract_valid_fraction(
        validity_summary=comparison.validity_summary,
        dataset="simulated",
    )
    validity_fraction_error = abs(real_valid_fraction - simulated_valid_fraction)

    row: dict[str, Any] = {
        "transport_efficiency": float(transport_efficiency),
        "tau_f_ns": float(tau_f_ns),
        "arrival_spread_ns": float(arrival_spread_ns),
        "real_valid_fraction": real_valid_fraction,
        "simulated_valid_fraction": simulated_valid_fraction,
        "validity_fraction_error": float(validity_fraction_error),
        "n_simulated_events": int(len(simulated_features)),
    }

    row.update(score_values)

    return row


def run_calibration_grid(
    real_features: pd.DataFrame,
    photon_events: pd.DataFrame,
    base_config: dict[str, Any],
    transport_efficiencies: list[float],
    tau_f_values_ns: list[float],
    arrival_spread_values_ns: list[float],
    max_events: int | None = None,
    random_seed: int | None = None,
    threshold_mV: float = 18.0,
    n_pretrigger_samples: int = 3,
    feature_columns: list[str] | None = None,
    feature_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Run a calibration grid search.

    Args:
        real_features: Real measured feature table.
        photon_events: Photon-event table from CORSIKA row-level events.
        base_config: Base detector simulation configuration.
        transport_efficiencies: Transport efficiency values to test.
        tau_f_values_ns: Fall-time constants to test.
        arrival_spread_values_ns: Arrival spread values to test.
        max_events: Optional number of simulated events per point.
        random_seed: Random seed.
        threshold_mV: Feature extraction threshold.
        n_pretrigger_samples: Number of samples used for baseline.
        feature_columns: Feature columns used for comparison.
        feature_weights: Feature weights for scoring.

    Returns:
        Calibration results sorted by increasing feature score.
    """
    grid = calibration_grid_rows(
        transport_efficiencies=transport_efficiencies,
        tau_f_values_ns=tau_f_values_ns,
        arrival_spread_values_ns=arrival_spread_values_ns,
    )

    rows: list[dict[str, Any]] = []

    for index, parameters in enumerate(grid):
        seed = None if random_seed is None else random_seed + index

        rows.append(
            run_single_calibration_point(
                real_features=real_features,
                photon_events=photon_events,
                base_config=base_config,
                transport_efficiency=parameters["transport_efficiency"],
                tau_f_ns=parameters["tau_f_ns"],
                arrival_spread_ns=parameters["arrival_spread_ns"],
                max_events=max_events,
                random_seed=seed,
                threshold_mV=threshold_mV,
                n_pretrigger_samples=n_pretrigger_samples,
                feature_columns=feature_columns,
                feature_weights=feature_weights,
            )
        )

    results = pd.DataFrame(rows)

    return results.sort_values("score", ascending=True).reset_index(drop=True)