"""Photon transport and photoelectron generation models.

This module models the stochastic conversion chain:

    generated scintillation photons
        -> photons reaching the SiPM
        -> detected photoelectrons

Conventions:
    - Photon counts are integer values.
    - Efficiencies and PDE values are probabilities in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PhotonTransportResult:
    """Container for photon transport simulation outputs."""

    generated_photons: NDArray[np.int64]
    photons_at_sipm: NDArray[np.int64]
    photoelectrons: NDArray[np.int64]
    transport_efficiency: float
    pde: float


def validate_probability(value: float, name: str) -> None:
    """Validate that a value is a probability.

    Args:
        value: Probability value.
        name: Name used in the error message.

    Raises:
        ValueError: If value is outside [0, 1].
    """
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in the interval [0, 1].")


def validate_photon_counts(
    photon_counts: int | NDArray[np.int64],
    name: str = "photon_counts",
) -> None:
    """Validate photon count input.

    Args:
        photon_counts: Scalar or array of photon counts.
        name: Name used in the error message.

    Raises:
        ValueError: If counts are negative or non-integer.
    """
    counts = np.asarray(photon_counts)

    if np.any(counts < 0):
        raise ValueError(f"{name} must contain non-negative values.")

    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError(f"{name} must contain integer values.")


def expected_photons_at_sipm(
    generated_photons: int | NDArray[np.int64],
    transport_efficiency: float,
) -> NDArray[np.float64]:
    """Compute the expected number of photons reaching the SiPM.

    Args:
        generated_photons: Number of generated scintillation photons.
        transport_efficiency: Effective probability that a generated photon
            reaches the SiPM active area.

    Returns:
        Expected number of photons reaching the SiPM.
    """
    validate_photon_counts(generated_photons, name="generated_photons")
    validate_probability(transport_efficiency, name="transport_efficiency")

    return np.asarray(generated_photons, dtype=np.float64) * transport_efficiency


def expected_photoelectrons(
    generated_photons: int | NDArray[np.int64],
    transport_efficiency: float,
    pde: float,
) -> NDArray[np.float64]:
    """Compute the expected number of detected photoelectrons.

    Args:
        generated_photons: Number of generated scintillation photons.
        transport_efficiency: Effective probability that a generated photon
            reaches the SiPM active area.
        pde: Photon detection efficiency of the SiPM.

    Returns:
        Expected number of photoelectrons.
    """
    validate_probability(pde, name="pde")

    photons_at_sipm_mean = expected_photons_at_sipm(
        generated_photons=generated_photons,
        transport_efficiency=transport_efficiency,
    )

    return photons_at_sipm_mean * pde


def sample_photons_at_sipm(
    generated_photons: int | NDArray[np.int64],
    transport_efficiency: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Sample photons reaching the SiPM from generated photons.

    The number of photons reaching the SiPM is modeled as a binomial process:

        N_sipm ~ Binomial(N_generated, transport_efficiency)

    Args:
        generated_photons: Number of generated scintillation photons.
        transport_efficiency: Effective probability that a generated photon
            reaches the SiPM active area.
        rng: Optional NumPy random generator.

    Returns:
        Sampled number of photons reaching the SiPM.
    """
    validate_photon_counts(generated_photons, name="generated_photons")
    validate_probability(transport_efficiency, name="transport_efficiency")

    if rng is None:
        rng = np.random.default_rng()

    return rng.binomial(
        n=np.asarray(generated_photons, dtype=np.int64),
        p=transport_efficiency,
    ).astype(np.int64)


def sample_photoelectrons(
    photons_at_sipm: int | NDArray[np.int64],
    pde: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Sample detected photoelectrons from photons reaching the SiPM.

    The number of photoelectrons is modeled as a binomial process:

        N_pe ~ Binomial(N_sipm, PDE)

    Args:
        photons_at_sipm: Number of photons reaching the SiPM active area.
        pde: Photon detection efficiency of the SiPM.
        rng: Optional NumPy random generator.

    Returns:
        Sampled number of photoelectrons.
    """
    validate_photon_counts(photons_at_sipm, name="photons_at_sipm")
    validate_probability(pde, name="pde")

    if rng is None:
        rng = np.random.default_rng()

    return rng.binomial(
        n=np.asarray(photons_at_sipm, dtype=np.int64),
        p=pde,
    ).astype(np.int64)


def detection_probability_at_least_one(
    photons_at_sipm: int | NDArray[np.int64],
    pde: float,
) -> NDArray[np.float64]:
    """Compute probability of detecting at least one photon.

    For n photons reaching the SiPM, the probability of detecting at least one
    photon is:

        P(detection) = 1 - (1 - PDE)**n

    Args:
        photons_at_sipm: Number of photons reaching the SiPM active area.
        pde: Photon detection efficiency of the SiPM.

    Returns:
        Probability of at least one detected photoelectron.
    """
    validate_photon_counts(photons_at_sipm, name="photons_at_sipm")
    validate_probability(pde, name="pde")

    counts = np.asarray(photons_at_sipm, dtype=np.float64)

    return 1.0 - (1.0 - pde) ** counts


def simulate_photon_transport(
    generated_photons: int | NDArray[np.int64],
    transport_efficiency: float,
    pde: float,
    rng: np.random.Generator | None = None,
) -> PhotonTransportResult:
    """Simulate photon transport and photoelectron generation.

    Args:
        generated_photons: Number of generated scintillation photons.
        transport_efficiency: Effective probability that a generated photon
            reaches the SiPM active area.
        pde: Photon detection efficiency of the SiPM.
        rng: Optional NumPy random generator.

    Returns:
        PhotonTransportResult containing generated photons, photons at SiPM,
        and detected photoelectrons.
    """
    validate_photon_counts(generated_photons, name="generated_photons")
    validate_probability(transport_efficiency, name="transport_efficiency")
    validate_probability(pde, name="pde")

    generated_photons_array = np.asarray(generated_photons, dtype=np.int64)

    if rng is None:
        rng = np.random.default_rng()

    photons_at_sipm = sample_photons_at_sipm(
        generated_photons=generated_photons_array,
        transport_efficiency=transport_efficiency,
        rng=rng,
    )

    photoelectrons = sample_photoelectrons(
        photons_at_sipm=photons_at_sipm,
        pde=pde,
        rng=rng,
    )

    return PhotonTransportResult(
        generated_photons=generated_photons_array,
        photons_at_sipm=photons_at_sipm,
        photoelectrons=photoelectrons,
        transport_efficiency=transport_efficiency,
        pde=pde,
    )