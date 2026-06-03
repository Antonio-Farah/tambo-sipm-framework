"""Tests for SiPM saturation models."""

import numpy as np
import pytest

from tambo_sipm.sipm.saturation import (
    MICROFC_60035_C_SERIES_MICROCELLS,
    estimate_photoelectrons_from_fired_microcells,
    expected_fired_microcells,
    sample_fired_microcells,
    saturation_factor,
    validate_microcell_count,
    validate_non_negative_integer_counts,
)


def test_microfc_60035_c_series_microcell_constant():
    assert MICROFC_60035_C_SERIES_MICROCELLS == 18_980


def test_validate_microcell_count_accepts_positive_value():
    validate_microcell_count(1)


def test_validate_microcell_count_rejects_non_positive_value():
    with pytest.raises(ValueError):
        validate_microcell_count(0)

    with pytest.raises(ValueError):
        validate_microcell_count(-1)


def test_validate_non_negative_integer_counts_accepts_integer_array():
    validate_non_negative_integer_counts(
        np.array([0, 1, 2], dtype=np.int64),
        name="counts",
    )


def test_validate_non_negative_integer_counts_rejects_negative_values():
    with pytest.raises(ValueError):
        validate_non_negative_integer_counts(
            np.array([0, -1, 2], dtype=np.int64),
            name="counts",
        )


def test_validate_non_negative_integer_counts_rejects_float_values():
    with pytest.raises(ValueError):
        validate_non_negative_integer_counts(
            np.array([0.0, 1.0, 2.0]),
            name="counts",
        )


def test_expected_fired_microcells_zero_photoelectrons():
    photoelectrons = np.array([0], dtype=np.int64)

    fired = expected_fired_microcells(
        photoelectrons=photoelectrons,
        n_microcells=100,
    )

    assert np.allclose(fired, np.array([0.0]))


def test_expected_fired_microcells_low_occupancy_is_nearly_linear():
    photoelectrons = np.array([1, 2, 3], dtype=np.int64)

    fired = expected_fired_microcells(
        photoelectrons=photoelectrons,
        n_microcells=1_000_000,
    )

    assert np.allclose(fired, photoelectrons.astype(float), rtol=1e-5)


def test_expected_fired_microcells_never_exceeds_microcell_count():
    photoelectrons = np.array([0, 100, 1_000, 10_000], dtype=np.int64)

    fired = expected_fired_microcells(
        photoelectrons=photoelectrons,
        n_microcells=100,
    )

    assert np.all(fired <= 100)


def test_saturation_factor_is_one_for_zero_photoelectrons():
    photoelectrons = np.array([0], dtype=np.int64)

    factor = saturation_factor(
        photoelectrons=photoelectrons,
        n_microcells=100,
    )

    assert np.allclose(factor, np.array([1.0]))


def test_saturation_factor_decreases_at_high_occupancy():
    photoelectrons = np.array([10, 1000], dtype=np.int64)

    factor = saturation_factor(
        photoelectrons=photoelectrons,
        n_microcells=100,
    )

    assert factor[1] < factor[0]


def test_sample_fired_microcells_is_reproducible():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    photoelectrons = np.array([10, 100, 1000], dtype=np.int64)

    sample_a = sample_fired_microcells(
        photoelectrons=photoelectrons,
        n_microcells=100,
        rng=rng_a,
    )

    sample_b = sample_fired_microcells(
        photoelectrons=photoelectrons,
        n_microcells=100,
        rng=rng_b,
    )

    assert np.array_equal(sample_a, sample_b)


def test_sample_fired_microcells_bounds():
    rng = np.random.default_rng(123)
    photoelectrons = np.array([10, 100, 1000], dtype=np.int64)

    fired = sample_fired_microcells(
        photoelectrons=photoelectrons,
        n_microcells=100,
        rng=rng,
    )

    assert np.all(fired >= 0)
    assert np.all(fired <= 100)


def test_estimate_photoelectrons_from_fired_microcells_zero():
    fired_microcells = np.array([0], dtype=np.int64)

    estimate = estimate_photoelectrons_from_fired_microcells(
        fired_microcells=fired_microcells,
        n_microcells=100,
    )

    assert np.allclose(estimate, np.array([0.0]))


def test_estimate_photoelectrons_from_fired_microcells_inverse_model():
    photoelectrons = np.array([1, 10, 50], dtype=np.int64)

    fired_mean = expected_fired_microcells(
        photoelectrons=photoelectrons,
        n_microcells=1000,
    )

    fired_integer = np.rint(fired_mean).astype(np.int64)

    estimate = estimate_photoelectrons_from_fired_microcells(
        fired_microcells=fired_integer,
        n_microcells=1000,
    )

    assert np.allclose(estimate, photoelectrons.astype(float), atol=1.0)


def test_estimate_photoelectrons_from_fired_microcells_returns_inf_at_full_saturation():
    fired_microcells = np.array([100], dtype=np.int64)

    estimate = estimate_photoelectrons_from_fired_microcells(
        fired_microcells=fired_microcells,
        n_microcells=100,
    )

    assert np.isinf(estimate[0])


def test_estimate_photoelectrons_from_fired_microcells_rejects_too_many_fired_cells():
    fired_microcells = np.array([101], dtype=np.int64)

    with pytest.raises(ValueError):
        estimate_photoelectrons_from_fired_microcells(
            fired_microcells=fired_microcells,
            n_microcells=100,
        )