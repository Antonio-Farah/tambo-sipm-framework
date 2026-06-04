"""Tests for matched feature comparison utilities."""

import numpy as np
import pandas as pd
import pytest

from tambo_sipm.analysis.matching import (
    filter_matchable_features,
    match_simulated_to_real_features,
    reference_center_and_scale,
    standardized_feature_matrix,
    weighted_euclidean_distances,
)


def make_real_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["r0", "r1", "r2"],
            "valid": [True, True, False],
            "peak_mV": [10.0, 20.0, 30.0],
            "rms_mV": [5.0, 10.0, 15.0],
            "integral_mVns": [100.0, 200.0, 300.0],
            "width_ns": [20.0, 40.0, 60.0],
        }
    )


def make_simulated_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["s0", "s1", "s2", "s3"],
            "valid": [True, True, True, False],
            "peak_mV": [11.0, 19.0, 100.0, 20.0],
            "rms_mV": [5.5, 9.5, 50.0, 10.0],
            "integral_mVns": [110.0, 190.0, 1000.0, 200.0],
            "width_ns": [22.0, 38.0, 200.0, 40.0],
        }
    )


def test_filter_matchable_features_keeps_valid_finite_rows():
    features = make_real_features()

    filtered = filter_matchable_features(features)

    assert len(filtered) == 2
    assert filtered["valid"].all()


def test_filter_matchable_features_rejects_missing_columns():
    features = pd.DataFrame({"event_id": ["a"], "valid": [True]})

    with pytest.raises(ValueError):
        filter_matchable_features(features)


def test_reference_center_and_scale():
    features = filter_matchable_features(make_real_features())

    center, scale = reference_center_and_scale(features)

    assert center.shape == (4,)
    assert scale.shape == (4,)
    assert np.all(scale > 0.0)


def test_standardized_feature_matrix():
    features = filter_matchable_features(make_real_features())
    center, scale = reference_center_and_scale(features)

    matrix = standardized_feature_matrix(
        features=features,
        feature_columns=["peak_mV", "rms_mV", "integral_mVns", "width_ns"],
        center=center,
        scale=scale,
    )

    assert matrix.shape == (2, 4)


def test_weighted_euclidean_distances():
    reference_vector = np.array([0.0, 0.0])
    candidate_matrix = np.array([[0.0, 0.0], [3.0, 4.0]])

    distances = weighted_euclidean_distances(
        reference_vector=reference_vector,
        candidate_matrix=candidate_matrix,
        feature_columns=["x", "y"],
    )

    assert np.isclose(distances[0], 0.0)
    assert distances[1] > 0.0


def test_match_simulated_to_real_features():
    result = match_simulated_to_real_features(
        real_features=make_real_features(),
        simulated_features=make_simulated_features(),
    )

    assert len(result.matched_real_features) == 2
    assert len(result.matched_simulated_features) == 2
    assert len(result.match_table) == 2
    assert set(result.match_table["real_event_id"]) == {"r0", "r1"}
    assert "matching_distance" in result.match_table.columns


def test_match_simulated_to_real_features_rejects_too_many_matches():
    with pytest.raises(ValueError):
        match_simulated_to_real_features(
            real_features=make_real_features(),
            simulated_features=make_simulated_features(),
            n_matches=10,
        )