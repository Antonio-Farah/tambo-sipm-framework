"""Distribution and correlation metrics for TAMBO waveform observables.

This module provides quantitative tools to compare measured and simulated
pulse-feature distributions and two-dimensional relationships.

The metrics are designed around the validation plots used in the TAMBO SiPM
response study:
    - peak voltage
    - RMS voltage
    - integrated voltage
    - pulse width
    - point-wise voltage distribution
    - integrated signal versus peak voltage
    - integrated signal versus pulse width
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import ks_2samp, linregress, pearsonr, spearmanr
from scipy.stats import wasserstein_distance


@dataclass(frozen=True)
class DistributionSummary:
    """Summary statistics for a one-dimensional distribution."""

    n: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    p05: float
    p25: float
    p75: float
    p95: float
    iqr: float
    data_range: float


@dataclass(frozen=True)
class DistributionComparison:
    """Comparison metrics between two one-dimensional distributions."""

    n_reference: int
    n_candidate: int
    mean_reference: float
    mean_candidate: float
    median_reference: float
    median_candidate: float
    std_reference: float
    std_candidate: float
    iqr_reference: float
    iqr_candidate: float
    range_reference: float
    range_candidate: float
    ks_statistic: float
    ks_pvalue: float
    wasserstein_distance: float
    mean_relative_difference: float
    median_relative_difference: float
    std_ratio: float
    iqr_ratio: float
    range_coverage: float


@dataclass(frozen=True)
class CorrelationSummary:
    """Summary of a two-dimensional relationship."""

    n: int
    pearson_r: float
    pearson_pvalue: float
    spearman_r: float
    spearman_pvalue: float
    linear_slope: float
    linear_intercept: float
    linear_r2: float
    quadratic_a: float
    quadratic_b: float
    quadratic_c: float
    quadratic_r2: float


@dataclass(frozen=True)
class CorrelationComparison:
    """Comparison of two two-dimensional relationships."""

    reference: CorrelationSummary
    candidate: CorrelationSummary
    pearson_delta: float
    spearman_delta: float
    linear_slope_relative_difference: float
    linear_r2_delta: float
    quadratic_r2_delta: float


def validate_1d_numeric_array(
    values: NDArray[np.float64] | list[float],
    name: str,
) -> NDArray[np.float64]:
    """Validate and convert input values to a one-dimensional numeric array.

    Args:
        values: Input values.
        name: Name used in error messages.

    Returns:
        One-dimensional NumPy array with dtype float64.

    Raises:
        ValueError: If the input is empty, not one-dimensional, or non-finite.
    """
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def validate_xy_arrays(
    x_values: NDArray[np.float64] | list[float],
    y_values: NDArray[np.float64] | list[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate paired x-y arrays.

    Args:
        x_values: x values.
        y_values: y values.

    Returns:
        Tuple of validated x and y arrays.

    Raises:
        ValueError: If arrays are invalid or have different lengths.
    """
    x_array = validate_1d_numeric_array(x_values, "x_values")
    y_array = validate_1d_numeric_array(y_values, "y_values")

    if len(x_array) != len(y_array):
        raise ValueError("x_values and y_values must have the same length.")
    if len(x_array) < 2:
        raise ValueError("At least two paired points are required.")

    return x_array, y_array


def summarize_distribution(
    values: NDArray[np.float64] | list[float],
) -> DistributionSummary:
    """Compute summary statistics for a distribution.

    Args:
        values: Input distribution.

    Returns:
        DistributionSummary object.
    """
    array = validate_1d_numeric_array(values, "values")

    p05 = float(np.percentile(array, 5))
    p25 = float(np.percentile(array, 25))
    p75 = float(np.percentile(array, 75))
    p95 = float(np.percentile(array, 95))

    minimum = float(np.min(array))
    maximum = float(np.max(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0

    return DistributionSummary(
        n=int(len(array)),
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        std=std,
        minimum=minimum,
        maximum=maximum,
        p05=p05,
        p25=p25,
        p75=p75,
        p95=p95,
        iqr=float(p75 - p25),
        data_range=float(maximum - minimum),
    )


def relative_difference(reference: float, candidate: float) -> float:
    """Compute relative difference with respect to a reference value.

    Args:
        reference: Reference value.
        candidate: Candidate value.

    Returns:
        (candidate - reference) / abs(reference). If reference is zero and
        candidate is zero, returns 0. If reference is zero and candidate is
        nonzero, returns infinity.
    """
    if reference == 0.0:
        if candidate == 0.0:
            return 0.0
        return float(np.inf)

    return float((candidate - reference) / abs(reference))


def safe_ratio(numerator: float, denominator: float) -> float:
    """Compute a ratio while handling zero denominators.

    Args:
        numerator: Numerator.
        denominator: Denominator.

    Returns:
        numerator / denominator, 0 if both are zero, or infinity if the
        denominator is zero and the numerator is nonzero.
    """
    if denominator == 0.0:
        if numerator == 0.0:
            return 0.0
        return float(np.inf)

    return float(numerator / denominator)


def range_coverage(reference_min: float, reference_max: float, candidate_min: float, candidate_max: float) -> float:
    """Compute how much of the reference range is covered by the candidate range.

    Args:
        reference_min: Minimum reference value.
        reference_max: Maximum reference value.
        candidate_min: Minimum candidate value.
        candidate_max: Maximum candidate value.

    Returns:
        Fraction of the reference range covered by the candidate range.
    """
    reference_range = reference_max - reference_min

    if reference_range == 0.0:
        return 1.0 if candidate_min <= reference_min <= candidate_max else 0.0

    overlap_min = max(reference_min, candidate_min)
    overlap_max = min(reference_max, candidate_max)
    overlap = max(0.0, overlap_max - overlap_min)

    return float(overlap / reference_range)


def compare_distributions(
    reference: NDArray[np.float64] | list[float],
    candidate: NDArray[np.float64] | list[float],
) -> DistributionComparison:
    """Compare two one-dimensional distributions.

    Args:
        reference: Reference distribution, typically experimental data.
        candidate: Candidate distribution, typically simulated data.

    Returns:
        DistributionComparison object.
    """
    reference_array = validate_1d_numeric_array(reference, "reference")
    candidate_array = validate_1d_numeric_array(candidate, "candidate")

    reference_summary = summarize_distribution(reference_array)
    candidate_summary = summarize_distribution(candidate_array)

    ks_result = ks_2samp(reference_array, candidate_array, method="auto")
    wasserstein = wasserstein_distance(reference_array, candidate_array)

    return DistributionComparison(
        n_reference=reference_summary.n,
        n_candidate=candidate_summary.n,
        mean_reference=reference_summary.mean,
        mean_candidate=candidate_summary.mean,
        median_reference=reference_summary.median,
        median_candidate=candidate_summary.median,
        std_reference=reference_summary.std,
        std_candidate=candidate_summary.std,
        iqr_reference=reference_summary.iqr,
        iqr_candidate=candidate_summary.iqr,
        range_reference=reference_summary.data_range,
        range_candidate=candidate_summary.data_range,
        ks_statistic=float(ks_result.statistic),
        ks_pvalue=float(ks_result.pvalue),
        wasserstein_distance=float(wasserstein),
        mean_relative_difference=relative_difference(
            reference_summary.mean,
            candidate_summary.mean,
        ),
        median_relative_difference=relative_difference(
            reference_summary.median,
            candidate_summary.median,
        ),
        std_ratio=safe_ratio(candidate_summary.std, reference_summary.std),
        iqr_ratio=safe_ratio(candidate_summary.iqr, reference_summary.iqr),
        range_coverage=range_coverage(
            reference_min=reference_summary.minimum,
            reference_max=reference_summary.maximum,
            candidate_min=candidate_summary.minimum,
            candidate_max=candidate_summary.maximum,
        ),
    )


def percentile_differences(
    reference: NDArray[np.float64] | list[float],
    candidate: NDArray[np.float64] | list[float],
    percentiles: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0),
) -> dict[str, float]:
    """Compute candidate-reference percentile differences.

    Args:
        reference: Reference distribution.
        candidate: Candidate distribution.
        percentiles: Percentiles to compare.

    Returns:
        Dictionary with keys like "p05", "p25", etc.
    """
    reference_array = validate_1d_numeric_array(reference, "reference")
    candidate_array = validate_1d_numeric_array(candidate, "candidate")

    differences: dict[str, float] = {}

    for percentile in percentiles:
        if percentile < 0.0 or percentile > 100.0:
            raise ValueError("percentiles must be in the interval [0, 100].")

        key = f"p{int(percentile):02d}"
        differences[key] = float(
            np.percentile(candidate_array, percentile)
            - np.percentile(reference_array, percentile)
        )

    return differences


def empirical_cdf(
    values: NDArray[np.float64] | list[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the empirical cumulative distribution function.

    Args:
        values: Input distribution.

    Returns:
        Tuple containing sorted values and empirical cumulative probabilities.
    """
    array = validate_1d_numeric_array(values, "values")

    sorted_values = np.sort(array)
    cumulative_probability = np.arange(1, len(sorted_values) + 1) / len(
        sorted_values
    )

    return sorted_values, cumulative_probability.astype(np.float64)


def common_bin_edges(
    reference: NDArray[np.float64] | list[float],
    candidate: NDArray[np.float64] | list[float],
    n_bins: int = 50,
    data_range: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """Create common histogram bin edges for two distributions.

    Args:
        reference: Reference distribution.
        candidate: Candidate distribution.
        n_bins: Number of bins.
        data_range: Optional fixed range as (min, max).

    Returns:
        Common bin edges.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    reference_array = validate_1d_numeric_array(reference, "reference")
    candidate_array = validate_1d_numeric_array(candidate, "candidate")

    if data_range is None:
        combined = np.concatenate([reference_array, candidate_array])
        lower = float(np.min(combined))
        upper = float(np.max(combined))
    else:
        lower, upper = data_range

    if upper < lower:
        raise ValueError("data_range upper bound must be >= lower bound.")

    if upper == lower:
        lower -= 0.5
        upper += 0.5

    return np.linspace(lower, upper, n_bins + 1, dtype=np.float64)


def _safe_pearson(x_values: NDArray[np.float64], y_values: NDArray[np.float64]) -> tuple[float, float]:
    """Compute Pearson correlation safely."""
    if np.std(x_values) == 0.0 or np.std(y_values) == 0.0:
        return float(np.nan), float(np.nan)

    result = pearsonr(x_values, y_values)

    return float(result.statistic), float(result.pvalue)


def _safe_spearman(x_values: NDArray[np.float64], y_values: NDArray[np.float64]) -> tuple[float, float]:
    """Compute Spearman correlation safely."""
    if np.std(x_values) == 0.0 or np.std(y_values) == 0.0:
        return float(np.nan), float(np.nan)

    result = spearmanr(x_values, y_values)

    return float(result.statistic), float(result.pvalue)


def _r2_score(y_true: NDArray[np.float64], y_predicted: NDArray[np.float64]) -> float:
    """Compute R squared."""
    residual_sum = float(np.sum((y_true - y_predicted) ** 2))
    total_sum = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if total_sum == 0.0:
        return float(np.nan)

    return float(1.0 - residual_sum / total_sum)


def summarize_xy_relationship(
    x_values: NDArray[np.float64] | list[float],
    y_values: NDArray[np.float64] | list[float],
) -> CorrelationSummary:
    """Summarize a two-dimensional relationship.

    Args:
        x_values: x values.
        y_values: y values.

    Returns:
        CorrelationSummary object.
    """
    x_array, y_array = validate_xy_arrays(x_values, y_values)

    pearson_r, pearson_pvalue = _safe_pearson(x_array, y_array)
    spearman_r, spearman_pvalue = _safe_spearman(x_array, y_array)

    if np.std(x_array) == 0.0:
        linear_slope = float(np.nan)
        linear_intercept = float(np.nan)
        linear_r2 = float(np.nan)
    else:
        linear_result = linregress(x_array, y_array)
        linear_slope = float(linear_result.slope)
        linear_intercept = float(linear_result.intercept)
        y_linear = linear_slope * x_array + linear_intercept
        linear_r2 = _r2_score(y_array, y_linear)

    if len(x_array) >= 3 and len(np.unique(x_array)) >= 3:
        quadratic_coefficients = np.polyfit(x_array, y_array, deg=2)
        quadratic_a = float(quadratic_coefficients[0])
        quadratic_b = float(quadratic_coefficients[1])
        quadratic_c = float(quadratic_coefficients[2])
        y_quadratic = (
            quadratic_a * x_array**2
            + quadratic_b * x_array
            + quadratic_c
        )
        quadratic_r2 = _r2_score(y_array, y_quadratic)
    else:
        quadratic_a = float(np.nan)
        quadratic_b = float(np.nan)
        quadratic_c = float(np.nan)
        quadratic_r2 = float(np.nan)

    return CorrelationSummary(
        n=int(len(x_array)),
        pearson_r=pearson_r,
        pearson_pvalue=pearson_pvalue,
        spearman_r=spearman_r,
        spearman_pvalue=spearman_pvalue,
        linear_slope=linear_slope,
        linear_intercept=linear_intercept,
        linear_r2=linear_r2,
        quadratic_a=quadratic_a,
        quadratic_b=quadratic_b,
        quadratic_c=quadratic_c,
        quadratic_r2=quadratic_r2,
    )


def compare_xy_relationships(
    reference_x: NDArray[np.float64] | list[float],
    reference_y: NDArray[np.float64] | list[float],
    candidate_x: NDArray[np.float64] | list[float],
    candidate_y: NDArray[np.float64] | list[float],
) -> CorrelationComparison:
    """Compare two two-dimensional relationships.

    Args:
        reference_x: Reference x values.
        reference_y: Reference y values.
        candidate_x: Candidate x values.
        candidate_y: Candidate y values.

    Returns:
        CorrelationComparison object.
    """
    reference_summary = summarize_xy_relationship(reference_x, reference_y)
    candidate_summary = summarize_xy_relationship(candidate_x, candidate_y)

    return CorrelationComparison(
        reference=reference_summary,
        candidate=candidate_summary,
        pearson_delta=float(
            candidate_summary.pearson_r - reference_summary.pearson_r
        ),
        spearman_delta=float(
            candidate_summary.spearman_r - reference_summary.spearman_r
        ),
        linear_slope_relative_difference=relative_difference(
            reference_summary.linear_slope,
            candidate_summary.linear_slope,
        ),
        linear_r2_delta=float(
            candidate_summary.linear_r2 - reference_summary.linear_r2
        ),
        quadratic_r2_delta=float(
            candidate_summary.quadratic_r2 - reference_summary.quadratic_r2
        ),
    )


def compare_feature_distributions(
    reference_dataframe: pd.DataFrame,
    candidate_dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Compare multiple feature distributions from two DataFrames.

    Args:
        reference_dataframe: DataFrame with reference feature values.
        candidate_dataframe: DataFrame with candidate feature values.
        feature_columns: Feature columns to compare.

    Returns:
        DataFrame with one row per feature.
    """
    rows: list[dict[str, float | int | str]] = []

    for feature in feature_columns:
        if feature not in reference_dataframe.columns:
            raise ValueError(f"Missing feature in reference_dataframe: {feature}")
        if feature not in candidate_dataframe.columns:
            raise ValueError(f"Missing feature in candidate_dataframe: {feature}")

        reference_values = reference_dataframe[feature].dropna().to_numpy(
            dtype=np.float64
        )
        candidate_values = candidate_dataframe[feature].dropna().to_numpy(
            dtype=np.float64
        )

        comparison = compare_distributions(reference_values, candidate_values)
        percentile_delta = percentile_differences(reference_values, candidate_values)

        row = {
            "feature": feature,
            "n_reference": comparison.n_reference,
            "n_candidate": comparison.n_candidate,
            "mean_reference": comparison.mean_reference,
            "mean_candidate": comparison.mean_candidate,
            "median_reference": comparison.median_reference,
            "median_candidate": comparison.median_candidate,
            "std_reference": comparison.std_reference,
            "std_candidate": comparison.std_candidate,
            "iqr_reference": comparison.iqr_reference,
            "iqr_candidate": comparison.iqr_candidate,
            "range_reference": comparison.range_reference,
            "range_candidate": comparison.range_candidate,
            "ks_statistic": comparison.ks_statistic,
            "ks_pvalue": comparison.ks_pvalue,
            "wasserstein_distance": comparison.wasserstein_distance,
            "mean_relative_difference": comparison.mean_relative_difference,
            "median_relative_difference": comparison.median_relative_difference,
            "std_ratio": comparison.std_ratio,
            "iqr_ratio": comparison.iqr_ratio,
            "range_coverage": comparison.range_coverage,
        }

        row.update({f"delta_{key}": value for key, value in percentile_delta.items()})
        rows.append(row)

    return pd.DataFrame(rows)


def compare_feature_correlations(
    reference_dataframe: pd.DataFrame,
    candidate_dataframe: pd.DataFrame,
    relationships: list[tuple[str, str]],
) -> pd.DataFrame:
    """Compare multiple two-dimensional feature relationships.

    Args:
        reference_dataframe: DataFrame with reference feature values.
        candidate_dataframe: DataFrame with candidate feature values.
        relationships: List of (x_column, y_column) relationships.

    Returns:
        DataFrame with one row per relationship.
    """
    rows: list[dict[str, float | int | str]] = []

    for x_column, y_column in relationships:
        for dataframe_name, dataframe in (
            ("reference_dataframe", reference_dataframe),
            ("candidate_dataframe", candidate_dataframe),
        ):
            if x_column not in dataframe.columns:
                raise ValueError(f"Missing {x_column} in {dataframe_name}.")
            if y_column not in dataframe.columns:
                raise ValueError(f"Missing {y_column} in {dataframe_name}.")

        reference_pairs = reference_dataframe[[x_column, y_column]].dropna()
        candidate_pairs = candidate_dataframe[[x_column, y_column]].dropna()

        comparison = compare_xy_relationships(
            reference_x=reference_pairs[x_column].to_numpy(dtype=np.float64),
            reference_y=reference_pairs[y_column].to_numpy(dtype=np.float64),
            candidate_x=candidate_pairs[x_column].to_numpy(dtype=np.float64),
            candidate_y=candidate_pairs[y_column].to_numpy(dtype=np.float64),
        )

        rows.append(
            {
                "relationship": f"{y_column}_vs_{x_column}",
                "x_column": x_column,
                "y_column": y_column,
                "n_reference": comparison.reference.n,
                "n_candidate": comparison.candidate.n,
                "pearson_reference": comparison.reference.pearson_r,
                "pearson_candidate": comparison.candidate.pearson_r,
                "pearson_delta": comparison.pearson_delta,
                "spearman_reference": comparison.reference.spearman_r,
                "spearman_candidate": comparison.candidate.spearman_r,
                "spearman_delta": comparison.spearman_delta,
                "linear_slope_reference": comparison.reference.linear_slope,
                "linear_slope_candidate": comparison.candidate.linear_slope,
                "linear_slope_relative_difference": (
                    comparison.linear_slope_relative_difference
                ),
                "linear_r2_reference": comparison.reference.linear_r2,
                "linear_r2_candidate": comparison.candidate.linear_r2,
                "linear_r2_delta": comparison.linear_r2_delta,
                "quadratic_r2_reference": comparison.reference.quadratic_r2,
                "quadratic_r2_candidate": comparison.candidate.quadratic_r2,
                "quadratic_r2_delta": comparison.quadratic_r2_delta,
            }
        )

    return pd.DataFrame(rows)