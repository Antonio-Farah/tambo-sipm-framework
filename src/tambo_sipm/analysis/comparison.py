"""Real-versus-simulated feature comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tambo_sipm.analysis.metrics import (
    compare_feature_correlations,
    compare_feature_distributions,
)


DEFAULT_FEATURE_COLUMNS = [
    "peak_mV",
    "rms_mV",
    "integral_mVns",
    "width_ns",
]

DEFAULT_RELATIONSHIPS = [
    ("peak_mV", "integral_mVns"),
    ("width_ns", "integral_mVns"),
]


@dataclass(frozen=True)
class FeatureComparisonResult:
    """Container for real-versus-simulated comparison tables."""

    distribution_comparison: pd.DataFrame
    correlation_comparison: pd.DataFrame
    validity_summary: pd.DataFrame


def filter_comparable_features(
    features: pd.DataFrame,
    require_valid: bool = True,
) -> pd.DataFrame:
    """Filter a feature table for real-versus-simulated comparison.

    Args:
        features: Feature table.
        require_valid: If True, keep only rows with valid == True.

    Returns:
        Filtered feature table.

    Raises:
        ValueError: If the valid column is missing when require_valid is True.
    """
    filtered = features.copy()

    if require_valid:
        if "valid" not in filtered.columns:
            raise ValueError("features must contain a 'valid' column.")

        filtered = filtered[filtered["valid"]].copy()

    return filtered.reset_index(drop=True)


def summarize_validity(
    features: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Summarize extraction status and validity for a feature table.

    Args:
        features: Feature table.
        label: Dataset label, for example "real" or "simulated".

    Returns:
        Validity summary table.
    """
    required_columns = ["valid", "extraction_status"]
    missing_columns = [
        column for column in required_columns if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    total_events = int(len(features))
    valid_events = int(features["valid"].sum())
    invalid_events = total_events - valid_events

    status_summary = (
        features.groupby(["extraction_status", "valid"])
        .size()
        .reset_index(name="count")
    )

    status_summary["dataset"] = label
    status_summary["total_events"] = total_events
    status_summary["valid_events"] = valid_events
    status_summary["invalid_events"] = invalid_events
    status_summary["valid_fraction"] = (
        valid_events / total_events if total_events > 0 else 0.0
    )

    columns = [
        "dataset",
        "extraction_status",
        "valid",
        "count",
        "total_events",
        "valid_events",
        "invalid_events",
        "valid_fraction",
    ]

    return status_summary[columns]


def compare_real_sim_features(
    real_features: pd.DataFrame,
    simulated_features: pd.DataFrame,
    feature_columns: list[str] | None = None,
    relationships: list[tuple[str, str]] | None = None,
    require_valid: bool = True,
) -> FeatureComparisonResult:
    """Compare real and simulated feature tables.

    Args:
        real_features: Real measured feature table.
        simulated_features: Simulated feature table.
        feature_columns: One-dimensional feature columns to compare.
        relationships: Two-dimensional relationships to compare.
        require_valid: If True, compare only valid pulses.

    Returns:
        FeatureComparisonResult with distribution, correlation, and validity
        tables.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS

    if relationships is None:
        relationships = DEFAULT_RELATIONSHIPS

    real_validity = summarize_validity(real_features, label="real")
    simulated_validity = summarize_validity(simulated_features, label="simulated")

    validity_summary = pd.concat(
        [real_validity, simulated_validity],
        ignore_index=True,
    )

    real_comparable = filter_comparable_features(
        real_features,
        require_valid=require_valid,
    )
    simulated_comparable = filter_comparable_features(
        simulated_features,
        require_valid=require_valid,
    )

    distribution_comparison = compare_feature_distributions(
        reference_dataframe=real_comparable,
        candidate_dataframe=simulated_comparable,
        feature_columns=feature_columns,
    )

    correlation_comparison = compare_feature_correlations(
        reference_dataframe=real_comparable,
        candidate_dataframe=simulated_comparable,
        relationships=relationships,
    )

    return FeatureComparisonResult(
        distribution_comparison=distribution_comparison,
        correlation_comparison=correlation_comparison,
        validity_summary=validity_summary,
    )


def save_feature_comparison_result(
    result: FeatureComparisonResult,
    output_dir: str,
) -> None:
    """Save comparison result tables as CSV files.

    Args:
        result: FeatureComparisonResult.
        output_dir: Output directory path.
    """
    from pathlib import Path

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    result.distribution_comparison.to_csv(
        path / "feature_distribution_comparison.csv",
        index=False,
    )
    result.correlation_comparison.to_csv(
        path / "feature_correlation_comparison.csv",
        index=False,
    )
    result.validity_summary.to_csv(
        path / "feature_validity_summary.csv",
        index=False,
    )