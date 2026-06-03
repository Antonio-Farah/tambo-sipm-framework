"""Tests for photon transport models."""

import numpy as np
import pytest

from tambo_sipm.detector.photon_transport import (
    detection_probability_at_least_one,
    expected_photoelectrons,
    expected_photons_at_sipm,
    sample_photoelectrons,
    sample_photons_at_sipm,
    simulate_photon_transport,
    validate_photon_counts,
    validate_probability,
)


def test_validate_probability_accepts_valid_values():
    validate_probability(0.0, "p")
    validate_probability(0.5, "p")
    validate_probability(1.0, "p")


def test_validate_probability_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_probability(-0.1, "p")

    with pytest.raises(ValueError):
        validate_probability(1.1, "p")


def test_validate_photon_counts_accepts_integer_counts():
    validate_photon_counts(np.array([0, 1, 2], dtype=np.int64))


def test_validate_photon_counts_rejects_negative_counts():
    with pytest.raises(ValueError):
        validate_photon_counts(np.array([0, -1, 2], dtype=np.int64))


def test_validate_photon_counts_rejects_float_counts():
    with pytest.raises(ValueError):
        validate_photon_counts(np.array([0.0, 1.0, 2.0]))


def test_expected_photons_at_sipm():
    generated_photons = np.array([100, 200, 300], dtype=np.int64)

    expected = expected_photons_at_sipm(
        generated_photons=generated_photons,
        transport_efficiency=0.1,
    )

    assert np.allclose(expected, np.array([10.0, 20.0, 30.0]))


def test_expected_photoelectrons():
    generated_photons = np.array([100, 200, 300], dtype=np.int64)

    expected = expected_photoelectrons(
        generated_photons=generated_photons,
        transport_efficiency=0.1,
        pde=0.4,
    )

    assert np.allclose(expected, np.array([4.0, 8.0, 12.0]))


def test_sample_photons_at_sipm_is_reproducible():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    generated_photons = np.array([100, 200, 300], dtype=np.int64)

    sample_a = sample_photons_at_sipm(
        generated_photons=generated_photons,
        transport_efficiency=0.2,
        rng=rng_a,
    )

    sample_b = sample_photons_at_sipm(
        generated_photons=generated_photons,
        transport_efficiency=0.2,
        rng=rng_b,
    )

    assert np.array_equal(sample_a, sample_b)


def test_sample_photoelectrons_is_reproducible():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    photons_at_sipm = np.array([100, 200, 300], dtype=np.int64)

    sample_a = sample_photoelectrons(
        photons_at_sipm=photons_at_sipm,
        pde=0.4,
        rng=rng_a,
    )

    sample_b = sample_photoelectrons(
        photons_at_sipm=photons_at_sipm,
        pde=0.4,
        rng=rng_b,
    )

    assert np.array_equal(sample_a, sample_b)


def test_detection_probability_at_least_one():
    photons_at_sipm = np.array([0, 1, 2], dtype=np.int64)

    probability = detection_probability_at_least_one(
        photons_at_sipm=photons_at_sipm,
        pde=0.5,
    )

    assert np.allclose(probability, np.array([0.0, 0.5, 0.75]))


def test_simulate_photon_transport_returns_consistent_shapes():
    rng = np.random.default_rng(123)
    generated_photons = np.array([100, 200, 300], dtype=np.int64)

    result = simulate_photon_transport(
        generated_photons=generated_photons,
        transport_efficiency=0.2,
        pde=0.4,
        rng=rng,
    )

    assert result.generated_photons.shape == generated_photons.shape
    assert result.photons_at_sipm.shape == generated_photons.shape
    assert result.photoelectrons.shape == generated_photons.shape
    assert result.transport_efficiency == 0.2
    assert result.pde == 0.4


def test_simulate_photon_transport_bounds():
    rng = np.random.default_rng(123)
    generated_photons = np.array([100, 200, 300], dtype=np.int64)

    result = simulate_photon_transport(
        generated_photons=generated_photons,
        transport_efficiency=0.2,
        pde=0.4,
        rng=rng,
    )

    assert np.all(result.photons_at_sipm <= result.generated_photons)
    assert np.all(result.photoelectrons <= result.photons_at_sipm)