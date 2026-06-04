"""Tests for real-versus-simulated feature comparison workflows."""

import pandas as pd
import pytest

from tambo_sipm.analysis.comparison import (
    compare_real_sim_features,
    filter_comparable_features,
    save_feature_comparison_result,
    summarize_validity,
)


def make_features(label: str) -> pd.DataFrame:
    scale = 1.0 if label == "real" else 1.2

    return pd.DataFrame(
        {
            "event_id": [f"{label}_0", f"{label}_1", f"{label}_2", f"{label}_3"],
            "valid": [True, True, True, False],
            "extraction_status": ["ok", "ok", "ok", "ok"],
            "peak_mV": [10.0 * scale, 20.0 * scale, 30.0 * scale, 5.0],
            "rms_mV": [5.0 * scale, 10.0 * scale, 15.0 * scale, 2.0],
            "integral_mVns": [
                100.0 * scale,
                200.0 * scale,
                300.0 * scale,
                0.0,
            ],
            "width_ns": [20.0 * scale, 40.0 * scale, 60.0 * scale, 0.0],
        }
    )


def test_filter_comparable_features_keeps_only_valid_rows():
    features = make_features("real")

    filtered = filter_comparable_features(features, require_valid=True)

    assert len(filtered) == 3
    assert filtered["valid"].all()


def test_filter_comparable_features_can_keep_all_rows():
    features = make_features("real")

    filtered = filter_comparable_features(features, require_valid=False)

    assert len(filtered) == 4


def test_filter_comparable_features_rejects_missing_valid_column():
    features = pd.DataFrame({"peak_mV": [1.0, 2.0]})

    with pytest.raises(ValueError):
        filter_comparable_features(features, require_valid=True)


def test_summarize_validity():
    features = make_features("real")

    summary = summarize_validity(features, label="real")

    assert "dataset" in summary.columns
    assert summary["total_events"].iloc[0] == 4
    assert summary["valid_events"].iloc[0] == 3
    assert summary["invalid_events"].iloc[0] == 1


def test_compare_real_sim_features():
    real_features = make_features("real")
    simulated_features = make_features("simulated")

    result = compare_real_sim_features(
        real_features=real_features,
        simulated_features=simulated_features,
        require_valid=True,
    )

    assert len(result.distribution_comparison) == 4
    assert len(result.correlation_comparison) == 2
    assert set(result.validity_summary["dataset"]) == {"real", "simulated"}


def test_save_feature_comparison_result(tmp_path):
    real_features = make_features("real")
    simulated_features = make_features("simulated")

    result = compare_real_sim_features(
        real_features=real_features,
        simulated_features=simulated_features,
        require_valid=True,
    )

    save_feature_comparison_result(
        result=result,
        output_dir=str(tmp_path),
    )

    assert (tmp_path / "feature_distribution_comparison.csv").exists()
    assert (tmp_path / "feature_correlation_comparison.csv").exists()
    assert (tmp_path / "feature_validity_summary.csv").exists()