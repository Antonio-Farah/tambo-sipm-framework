"""Matched feature comparison utilities.

This module selects a subset of simulated events that best matches a set of
real measured events in feature space.

This workflow is separate from physical calibration. Calibration should use
representative/random CORSIKA events. Matching is intended for post-calibration
visual and statistical comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


DEFAULT_MATCHING_FEATURES = [
    "peak_mV",
    "rms_mV",
    "integral_mVns",
    "width_ns",
]


@dataclass(frozen=True)
class MatchedFeatureResult:
    """Container for matched real and simulated feature tables."""

    matched_real_features: pd.DataFrame
    matched_simulated_features: pd.DataFrame
    match_table: pd.DataFrame


def require_matching_columns(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    require_valid: bool = True,
) -> None:
    """Validate columns needed for feature matching.

    Args:
        dataframe: Input feature table.
        feature_columns: Feature columns used for matching.
        require_valid: If True, require a valid column.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = list(feature_columns)

    if require_valid:
        required_columns.append("valid")

    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def filter_matchable_features(
    features: pd.DataFrame,
    feature_columns: list[str] | None = None,
    require_valid: bool = True,
) -> pd.DataFrame:
    """Filter a feature table to rows usable for matching.

    Args:
        features: Input feature table.
        feature_columns: Feature columns used for matching.
        require_valid: If True, keep only rows with valid == True.

    Returns:
        Filtered feature table.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_MATCHING_FEATURES

    require_matching_columns(
        dataframe=features,
        feature_columns=feature_columns,
        require_valid=require_valid,
    )

    filtered = features.copy()

    if require_valid:
        filtered = filtered[filtered["valid"]].copy()

    finite_mask = np.ones(len(filtered), dtype=bool)

    for feature in feature_columns:
        values = pd.to_numeric(filtered[feature], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite_mask &= np.isfinite(values)

    filtered = filtered[finite_mask].copy()

    if filtered.empty:
        raise ValueError("No matchable feature rows remain after filtering.")

    return filtered.reset_index(drop=True)


def reference_center_and_scale(
    reference_features: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute reference center and scale for standardized matching.

    The center is the reference median and the scale is the reference
    interquartile range. Zero IQR values are replaced by standard deviation,
    and remaining zero scales are replaced by 1.

    Args:
        reference_features: Reference feature table.
        feature_columns: Feature columns used for matching.

    Returns:
        Tuple of center and scale arrays.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_MATCHING_FEATURES

    values = reference_features[feature_columns].to_numpy(dtype=np.float64)

    center = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = q75 - q25

    std = np.nanstd(values, axis=0, ddof=1)
    scale = np.where(scale > 0.0, scale, std)
    scale = np.where(scale > 0.0, scale, 1.0)

    return center.astype(np.float64), scale.astype(np.float64)


def standardized_feature_matrix(
    features: pd.DataFrame,
    feature_columns: list[str],
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert feature columns to a standardized matrix.

    Args:
        features: Feature table.
        feature_columns: Feature columns used for matching.
        center: Center values.
        scale: Scale values.

    Returns:
        Standardized feature matrix.
    """
    values = features[feature_columns].to_numpy(dtype=np.float64)

    return (values - center) / scale


def weighted_euclidean_distances(
    reference_vector: NDArray[np.float64],
    candidate_matrix: NDArray[np.float64],
    feature_weights: dict[str, float] | None = None,
    feature_columns: list[str] | None = None,
) -> NDArray[np.float64]:
    """Compute weighted Euclidean distances.

    Args:
        reference_vector: One standardized reference vector.
        candidate_matrix: Standardized candidate matrix.
        feature_weights: Optional feature weights.
        feature_columns: Feature column names corresponding to matrix columns.

    Returns:
        Distance array.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_MATCHING_FEATURES

    if feature_weights is None:
        weights = np.ones(len(feature_columns), dtype=np.float64)
    else:
        weights = np.array(
            [float(feature_weights.get(feature, 1.0)) for feature in feature_columns],
            dtype=np.float64,
        )

    if np.any(weights < 0.0):
        raise ValueError("feature_weights must be non-negative.")

    if np.sum(weights) == 0.0:
        raise ValueError("At least one feature weight must be positive.")

    differences = candidate_matrix - reference_vector

    return np.sqrt(np.sum(weights * differences**2, axis=1) / np.sum(weights))


def match_simulated_to_real_features(
    real_features: pd.DataFrame,
    simulated_features: pd.DataFrame,
    feature_columns: list[str] | None = None,
    n_matches: int | None = None,
    feature_weights: dict[str, float] | None = None,
    require_valid: bool = True,
) -> MatchedFeatureResult:
    """Match simulated feature rows to real feature rows.

    Matching is done greedily without replacement. Each real pulse is matched
    to the nearest remaining simulated pulse in standardized feature space.

    Args:
        real_features: Real measured feature table.
        simulated_features: Simulated feature table.
        feature_columns: Feature columns used for matching.
        n_matches: Number of matches. If None, uses all valid real rows if
            enough simulated rows are available.
        feature_weights: Optional feature weights.
        require_valid: If True, use only valid rows from both tables.

    Returns:
        MatchedFeatureResult containing matched real rows, matched simulated
        rows, and a match table.

    Raises:
        ValueError: If not enough matchable rows are available.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_MATCHING_FEATURES

    real_matchable = filter_matchable_features(
        features=real_features,
        feature_columns=feature_columns,
        require_valid=require_valid,
    )
    simulated_matchable = filter_matchable_features(
        features=simulated_features,
        feature_columns=feature_columns,
        require_valid=require_valid,
    )

    if n_matches is None:
        n_matches = len(real_matchable)

    if n_matches <= 0:
        raise ValueError("n_matches must be positive.")

    if n_matches > len(real_matchable):
        raise ValueError("n_matches cannot exceed matchable real rows.")

    if n_matches > len(simulated_matchable):
        raise ValueError("n_matches cannot exceed matchable simulated rows.")

    real_matchable = real_matchable.head(n_matches).copy().reset_index(drop=True)

    center, scale = reference_center_and_scale(
        reference_features=real_matchable,
        feature_columns=feature_columns,
    )

    real_matrix = standardized_feature_matrix(
        features=real_matchable,
        feature_columns=feature_columns,
        center=center,
        scale=scale,
    )
    simulated_matrix = standardized_feature_matrix(
        features=simulated_matchable,
        feature_columns=feature_columns,
        center=center,
        scale=scale,
    )

    available_indices = list(range(len(simulated_matchable)))
    selected_simulated_indices: list[int] = []
    match_rows: list[dict[str, object]] = []

    for reference_index in range(n_matches):
        reference_vector = real_matrix[reference_index]
        candidate_matrix = simulated_matrix[available_indices]

        distances = weighted_euclidean_distances(
            reference_vector=reference_vector,
            candidate_matrix=candidate_matrix,
            feature_weights=feature_weights,
            feature_columns=feature_columns,
        )

        local_best_index = int(np.argmin(distances))
        best_global_index = available_indices.pop(local_best_index)
        best_distance = float(distances[local_best_index])

        selected_simulated_indices.append(best_global_index)

        real_event_id = (
            str(real_matchable.iloc[reference_index]["event_id"])
            if "event_id" in real_matchable.columns
            else str(reference_index)
        )
        simulated_event_id = (
            str(simulated_matchable.iloc[best_global_index]["event_id"])
            if "event_id" in simulated_matchable.columns
            else str(best_global_index)
        )

        row: dict[str, object] = {
            "match_id": reference_index,
            "real_index": reference_index,
            "simulated_index": best_global_index,
            "real_event_id": real_event_id,
            "simulated_event_id": simulated_event_id,
            "matching_distance": best_distance,
        }

        for feature in feature_columns:
            row[f"real_{feature}"] = float(real_matchable.iloc[reference_index][feature])
            row[f"simulated_{feature}"] = float(
                simulated_matchable.iloc[best_global_index][feature]
            )
            row[f"delta_{feature}"] = (
                row[f"simulated_{feature}"] - row[f"real_{feature}"]
            )

        match_rows.append(row)

    matched_real = real_matchable.copy()
    matched_simulated = simulated_matchable.iloc[selected_simulated_indices].copy()
    matched_simulated = matched_simulated.reset_index(drop=True)

    match_table = pd.DataFrame(match_rows)

    matched_real["match_id"] = match_table["match_id"].to_numpy(dtype=np.int64)
    matched_simulated["match_id"] = match_table["match_id"].to_numpy(dtype=np.int64)
    matched_simulated["matched_real_event_id"] = match_table[
        "real_event_id"
    ].to_numpy()
    matched_simulated["matching_distance"] = match_table[
        "matching_distance"
    ].to_numpy(dtype=np.float64)

    return MatchedFeatureResult(
        matched_real_features=matched_real,
        matched_simulated_features=matched_simulated,
        match_table=match_table,
    )


def save_matched_feature_result(
    result: MatchedFeatureResult,
    output_dir: str,
) -> None:
    """Save matched feature tables.

    Args:
        result: MatchedFeatureResult.
        output_dir: Output directory path.
    """
    from pathlib import Path

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    result.matched_real_features.to_csv(
        path / "matched_real_features.csv",
        index=False,
    )
    result.matched_simulated_features.to_csv(
        path / "matched_simulated_features.csv",
        index=False,
    )
    result.match_table.to_csv(
        path / "matched_pairs.csv",
        index=False,
    )