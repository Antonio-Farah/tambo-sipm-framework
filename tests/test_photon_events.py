"""Tests for processed CORSIKA photon-event utilities."""

import numpy as np
import pandas as pd
import pytest

from tambo_sipm.io.photon_events import (
    add_particle_metadata,
    build_photon_events,
    build_row_level_photon_events,
    detector_summary,
    filter_detector,
    load_romero_photon_table,
    particle_category_from_corsika_id,
    particle_name_from_corsika_id,
    particle_summary,
    select_detector_with_most_rows,
)


def make_test_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Detector": [0, 0, 0, 1, 1],
            "shower": [10, 10, 11, 10, 12],
            "id": [1, 3, 5, 14, 8],
            "particula": ["foton", "electron", "muon-", "proton", "pion+"],
            "num_fotones": [100, 200, 300, 400, 500],
            "energia_detectada_poli_MeV": [
                0.0125,
                0.025,
                0.0375,
                0.05,
                0.0625,
            ],
            "tiempo_deteccion": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )


def test_load_romero_photon_table(tmp_path):
    csv_path = tmp_path / "romero.csv"
    make_test_dataframe().to_csv(csv_path, index=False)

    dataframe = load_romero_photon_table(csv_path)

    assert len(dataframe) == 5
    assert "num_fotones" in dataframe.columns
    assert dataframe["num_fotones"].dtype == np.int64


def test_load_romero_photon_table_rejects_missing_column(tmp_path):
    csv_path = tmp_path / "romero.csv"
    pd.DataFrame({"Detector": [0], "num_fotones": [100]}).to_csv(
        csv_path,
        index=False,
    )

    with pytest.raises(ValueError):
        load_romero_photon_table(csv_path)


def test_particle_name_from_corsika_id():
    assert particle_name_from_corsika_id(1) == "gamma"
    assert particle_name_from_corsika_id(5) == "mu+"
    assert particle_name_from_corsika_id(999) == "unknown"


def test_particle_category_from_corsika_id():
    assert particle_category_from_corsika_id(1) == "electromagnetic"
    assert particle_category_from_corsika_id(5) == "muonic"
    assert particle_category_from_corsika_id(14) == "hadronic"
    assert particle_category_from_corsika_id(999) == "other"


def test_add_particle_metadata():
    dataframe = make_test_dataframe()

    result = add_particle_metadata(dataframe)

    assert "particle_id" in result.columns
    assert "particle_name" in result.columns
    assert "particle_name_original" in result.columns
    assert "particle_category" in result.columns
    assert result.loc[0, "particle_name"] == "gamma"
    assert result.loc[0, "particle_name_original"] == "foton"
    assert result.loc[1, "particle_category"] == "electromagnetic"
    assert result.loc[2, "particle_category"] == "muonic"
    assert result.loc[3, "particle_category"] == "hadronic"


def test_detector_summary_sorts_by_rows():
    dataframe = make_test_dataframe()

    summary = detector_summary(dataframe)

    assert list(summary["Detector"]) == [0, 1]
    assert list(summary["rows"]) == [3, 2]
    assert list(summary["unique_showers"]) == [2, 2]


def test_particle_summary():
    dataframe = make_test_dataframe()

    summary = particle_summary(dataframe)

    assert "particle_category" in summary.columns
    assert "particle_name" in summary.columns
    assert "rows" in summary.columns
    assert summary["rows"].sum() == 5

def test_particle_summary_accepts_standardized_events():
    dataframe = make_test_dataframe()

    events = build_row_level_photon_events(
        dataframe=dataframe,
        detector=0,
    )

    summary = particle_summary(events)

    assert "particle_category" in summary.columns
    assert "particle_name" in summary.columns
    assert "rows" in summary.columns
    assert summary["rows"].sum() == 3

def test_particle_summary_preserves_standardized_particle_metadata():
    dataframe = make_test_dataframe()

    events = build_row_level_photon_events(
        dataframe=dataframe,
        detector=0,
    )

    summary = particle_summary(events)

    assert set(summary["particle_name"]) == {"gamma", "e-", "mu+"}
    assert set(summary["particle_category"]) == {
        "electromagnetic",
        "muonic",
    }

def test_select_detector_with_most_rows():
    dataframe = make_test_dataframe()

    detector = select_detector_with_most_rows(dataframe)

    assert detector == 0


def test_filter_detector():
    dataframe = make_test_dataframe()

    filtered = filter_detector(dataframe, detector=0)

    assert len(filtered) == 3
    assert set(filtered["Detector"]) == {0}


def test_filter_detector_rejects_missing_detector():
    dataframe = make_test_dataframe()

    with pytest.raises(ValueError):
        filter_detector(dataframe, detector=9)


def test_build_row_level_photon_events_detector_zero():
    dataframe = make_test_dataframe()

    events = build_row_level_photon_events(
        dataframe=dataframe,
        detector=0,
    )

    assert len(events) == 3
    assert list(events["generated_photons"]) == [100, 200, 300]
    assert np.all(events["n_hits"] == 1)
    assert set(events["detector"]) == {0}
    assert "particle_id" in events.columns
    assert "particle_name" in events.columns
    assert "particle_name_original" in events.columns
    assert "particle_category" in events.columns
    assert "energia_detectada_poli_MeV" in events.columns


def test_build_photon_events_default_row_level_detector_zero():
    dataframe = make_test_dataframe()

    events = build_photon_events(dataframe)

    assert len(events) == 3
    assert set(events["detector"]) == {0}


def test_build_photon_events_rejects_invalid_mode():
    dataframe = make_test_dataframe()

    with pytest.raises(ValueError):
        build_photon_events(
            dataframe=dataframe,
            detector=0,
            mode="shower_detector",
        )