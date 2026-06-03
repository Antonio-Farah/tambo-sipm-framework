"""Tests for distribution and correlation comparison metrics."""

import numpy as np
import pandas as pd
import pytest

from tambo_sipm.analysis.metrics import (
    common_bin_edges,
    compare_distributions,
    compare_feature_correlations,
    compare_feature_distributions,
    compare_xy_relationships,
    empirical_cdf,
    percentile_differences,
    range_coverage,
    relative_difference,
    safe_ratio,
    summarize_distribution,
    summarize_xy_relationship,
    validate_1d_numeric_array,
    validate_xy_arrays,
)


def test_validate_1d_numeric_array_accepts_valid_array():
    array = validate_1d_numeric_array([1.0, 2.0, 3.0], "values")

    assert np.array_equal(array, np.array([1.0, 2.0, 3.0]))


def test_validate_1d_numeric_array_rejects_empty_array():
    with pytest.raises(ValueError):
        validate_1d_numeric_array([], "values")


def test_validate_1d_numeric_array_rejects_nan():
    with pytest.raises(ValueError):
        validate_1d_numeric_array([1.0, np.nan], "values")


def test_validate_xy_arrays_accepts_valid_pairs():
    x_values, y_values = validate_xy_arrays([1.0, 2.0], [3.0, 4.0])

    assert np.array_equal(x_values, np.array([1.0, 2.0]))
    assert np.array_equal(y_values, np.array([3.0, 4.0]))


def test_validate_xy_arrays_rejects_length_mismatch():
    with pytest.raises(ValueError):
        validate_xy_arrays([1.0, 2.0], [1.0])


def test_summarize_distribution():
    summary = summarize_distribution([1.0, 2.0, 3.0])

    assert summary.n == 3
    assert np.isclose(summary.mean, 2.0)
    assert np.isclose(summary.median, 2.0)
    assert np.isclose(summary.minimum, 1.0)
    assert np.isclose(summary.maximum, 3.0)
    assert np.isclose(summary.iqr, 1.0)
    assert np.isclose(summary.data_range, 2.0)


def test_relative_difference_standard_case():
    result = relative_difference(reference=10.0, candidate=12.0)

    assert np.isclose(result, 0.2)


def test_relative_difference_zero_reference():
    assert relative_difference(reference=0.0, candidate=0.0) == 0.0
    assert np.isinf(relative_difference(reference=0.0, candidate=1.0))


def test_safe_ratio():
    assert np.isclose(safe_ratio(2.0, 4.0), 0.5)
    assert safe_ratio(0.0, 0.0) == 0.0
    assert np.isinf(safe_ratio(1.0, 0.0))


def test_range_coverage_full_overlap():
    coverage = range_coverage(
        reference_min=0.0,
        reference_max=10.0,
        candidate_min=-1.0,
        candidate_max=11.0,
    )

    assert np.isclose(coverage, 1.0)


def test_range_coverage_partial_overlap():
    coverage = range_coverage(
        reference_min=0.0,
        reference_max=10.0,
        candidate_min=5.0,
        candidate_max=15.0,
    )

    assert np.isclose(coverage, 0.5)


def test_common_bin_edges_length():
    edges = common_bin_edges(
        reference=[0.0, 1.0],
        candidate=[2.0, 3.0],
        n_bins=3,
    )

    assert len(edges) == 4
    assert np.isclose(edges[0], 0.0)
    assert np.isclose(edges[-1], 3.0)


def test_common_bin_edges_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        common_bin_edges(
            reference=[0.0, 1.0],
            candidate=[2.0, 3.0],
            n_bins=0,
        )

    with pytest.raises(ValueError):
        common_bin_edges(
            reference=[0.0, 1.0],
            candidate=[2.0, 3.0],
            data_range=(3.0, 1.0),
        )


def test_empirical_cdf():
    x_values, y_values = empirical_cdf([3.0, 1.0, 2.0])

    assert np.allclose(x_values, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(y_values, np.array([1 / 3, 2 / 3, 1.0]))


def test_percentile_differences_identical_distributions():
    differences = percentile_differences(
        reference=[1.0, 2.0, 3.0],
        candidate=[1.0, 2.0, 3.0],
    )

    assert all(np.isclose(value, 0.0) for value in differences.values())


def test_compare_distributions_identical_arrays():
    comparison = compare_distributions(
        reference=[1.0, 2.0, 3.0],
        candidate=[1.0, 2.0, 3.0],
    )

    assert np.isclose(comparison.ks_statistic, 0.0)
    assert np.isclose(comparison.wasserstein_distance, 0.0)
    assert np.isclose(comparison.mean_relative_difference, 0.0)
    assert np.isclose(comparison.std_ratio, 1.0)
    assert np.isclose(comparison.iqr_ratio, 1.0)
    assert np.isclose(comparison.range_coverage, 1.0)


def test_compare_distributions_shifted_arrays():
    comparison = compare_distributions(
        reference=[1.0, 2.0, 3.0],
        candidate=[2.0, 3.0, 4.0],
    )

    assert comparison.wasserstein_distance > 0.0
    assert comparison.mean_candidate > comparison.mean_reference


def test_summarize_xy_relationship_linear_case():
    summary = summarize_xy_relationship(
        x_values=[1.0, 2.0, 3.0, 4.0],
        y_values=[2.0, 4.0, 6.0, 8.0],
    )

    assert summary.n == 4
    assert np.isclose(summary.pearson_r, 1.0)
    assert np.isclose(summary.spearman_r, 1.0)
    assert np.isclose(summary.linear_slope, 2.0)
    assert np.isclose(summary.linear_intercept, 0.0)
    assert np.isclose(summary.linear_r2, 1.0)
    assert np.isclose(summary.quadratic_r2, 1.0)


def test_compare_xy_relationships():
    comparison = compare_xy_relationships(
        reference_x=[1.0, 2.0, 3.0, 4.0],
        reference_y=[2.0, 4.0, 6.0, 8.0],
        candidate_x=[1.0, 2.0, 3.0, 4.0],
        candidate_y=[3.0, 6.0, 9.0, 12.0],
    )

    assert np.isclose(comparison.reference.linear_slope, 2.0)
    assert np.isclose(comparison.candidate.linear_slope, 3.0)
    assert comparison.linear_slope_relative_difference > 0.0


def test_compare_feature_distributions():
    reference_dataframe = pd.DataFrame(
        {
            "peak_mV": [1.0, 2.0, 3.0],
            "width_ns": [10.0, 20.0, 30.0],
        }
    )
    candidate_dataframe = pd.DataFrame(
        {
            "peak_mV": [1.0, 2.0, 4.0],
            "width_ns": [10.0, 25.0, 35.0],
        }
    )

    result = compare_feature_distributions(
        reference_dataframe=reference_dataframe,
        candidate_dataframe=candidate_dataframe,
        feature_columns=["peak_mV", "width_ns"],
    )

    assert list(result["feature"]) == ["peak_mV", "width_ns"]
    assert "ks_statistic" in result.columns
    assert "wasserstein_distance" in result.columns
    assert "std_ratio" in result.columns
    assert "range_coverage" in result.columns


def test_compare_feature_distributions_rejects_missing_column():
    reference_dataframe = pd.DataFrame({"peak_mV": [1.0, 2.0, 3.0]})
    candidate_dataframe = pd.DataFrame({"width_ns": [10.0, 20.0, 30.0]})

    with pytest.raises(ValueError):
        compare_feature_distributions(
            reference_dataframe=reference_dataframe,
            candidate_dataframe=candidate_dataframe,
            feature_columns=["peak_mV"],
        )


def test_compare_feature_correlations():
    reference_dataframe = pd.DataFrame(
        {
            "peak_mV": [1.0, 2.0, 3.0, 4.0],
            "integral_mVns": [2.0, 4.0, 6.0, 8.0],
            "width_ns": [10.0, 20.0, 30.0, 40.0],
        }
    )
    candidate_dataframe = pd.DataFrame(
        {
            "peak_mV": [1.0, 2.0, 3.0, 4.0],
            "integral_mVns": [3.0, 6.0, 9.0, 12.0],
            "width_ns": [10.0, 20.0, 30.0, 40.0],
        }
    )

    result = compare_feature_correlations(
        reference_dataframe=reference_dataframe,
        candidate_dataframe=candidate_dataframe,
        relationships=[
            ("peak_mV", "integral_mVns"),
            ("width_ns", "integral_mVns"),
        ],
    )

    assert list(result["relationship"]) == [
        "integral_mVns_vs_peak_mV",
        "integral_mVns_vs_width_ns",
    ]
    assert "pearson_reference" in result.columns
    assert "linear_slope_relative_difference" in result.columns
    assert "quadratic_r2_candidate" in result.columns


def test_compare_feature_correlations_rejects_missing_column():
    reference_dataframe = pd.DataFrame({"peak_mV": [1.0, 2.0, 3.0]})
    candidate_dataframe = pd.DataFrame({"peak_mV": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError):
        compare_feature_correlations(
            reference_dataframe=reference_dataframe,
            candidate_dataframe=candidate_dataframe,
            relationships=[("peak_mV", "integral_mVns")],
        )