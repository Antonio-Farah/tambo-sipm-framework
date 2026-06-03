"""SiPM microcell saturation models.

This module models the finite number of available SiPM microcells.

Conventions:
    - Photoelectron counts are integer values.
    - Fired microcell counts are integer values.
    - The default number of microcells for the onsemi MicroFC-60035 C-Series
      sensor is 18,980.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


MICROFC_60035_C_SERIES_MICROCELLS = 18_980


def validate_microcell_count(n_microcells: int) -> None:
    """Validate the number of SiPM microcells.

    Args:
        n_microcells: Total number of available microcells.

    Raises:
        ValueError: If n_microcells is not positive.
    """
    if n_microcells <= 0:
        raise ValueError("n_microcells must be positive.")


def validate_non_negative_integer_counts(
    counts: int | NDArray[np.int64],
    name: str,
) -> None:
    """Validate non-negative integer count input.

    Args:
        counts: Scalar or array of counts.
        name: Name used in the error message.

    Raises:
        ValueError: If counts are negative or non-integer.
    """
    values = np.asarray(counts)

    if np.any(values < 0):
        raise ValueError(f"{name} must contain non-negative values.")

    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must contain integer values.")


def expected_fired_microcells(
    photoelectrons: int | NDArray[np.int64],
    n_microcells: int = MICROFC_60035_C_SERIES_MICROCELLS,
) -> NDArray[np.float64]:
    """Compute expected fired microcells from photoelectrons.

    The model assumes that photoelectrons are randomly distributed across the
    available microcells. The expected number of fired microcells is:

        N_fired = N_cells * [1 - (1 - 1 / N_cells)**N_pe]

    For large N_cells, this approaches:

        N_fired ≈ N_cells * [1 - exp(-N_pe / N_cells)]

    Args:
        photoelectrons: Number of generated photoelectrons.
        n_microcells: Total number of available microcells.

    Returns:
        Expected number of fired microcells.
    """
    validate_microcell_count(n_microcells)
    validate_non_negative_integer_counts(photoelectrons, name="photoelectrons")

    photoelectrons_array = np.asarray(photoelectrons, dtype=np.float64)

    return n_microcells * (
        1.0 - (1.0 - 1.0 / n_microcells) ** photoelectrons_array
    )


def saturation_factor(
    photoelectrons: int | NDArray[np.int64],
    n_microcells: int = MICROFC_60035_C_SERIES_MICROCELLS,
) -> NDArray[np.float64]:
    """Compute the saturation factor N_fired / N_pe.

    Args:
        photoelectrons: Number of generated photoelectrons.
        n_microcells: Total number of available microcells.

    Returns:
        Saturation factor. For zero photoelectrons, the factor is defined as
        1.0 to avoid division by zero.
    """
    validate_microcell_count(n_microcells)
    validate_non_negative_integer_counts(photoelectrons, name="photoelectrons")

    photoelectrons_array = np.asarray(photoelectrons, dtype=np.float64)
    fired_mean = expected_fired_microcells(
        photoelectrons=np.asarray(photoelectrons, dtype=np.int64),
        n_microcells=n_microcells,
    )

    factor = np.ones_like(photoelectrons_array, dtype=np.float64)
    mask = photoelectrons_array > 0.0
    factor[mask] = fired_mean[mask] / photoelectrons_array[mask]

    return factor


def sample_fired_microcells(
    photoelectrons: int | NDArray[np.int64],
    n_microcells: int = MICROFC_60035_C_SERIES_MICROCELLS,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Sample the number of fired microcells.

    For a given number of photoelectrons, the probability that a given microcell
    fires at least once is:

        p_fire = 1 - (1 - 1 / N_cells)**N_pe

    The total number of fired cells is then sampled as:

        N_fired ~ Binomial(N_cells, p_fire)

    Args:
        photoelectrons: Number of generated photoelectrons.
        n_microcells: Total number of available microcells.
        rng: Optional NumPy random generator.

    Returns:
        Sampled number of fired microcells.
    """
    validate_microcell_count(n_microcells)
    validate_non_negative_integer_counts(photoelectrons, name="photoelectrons")

    if rng is None:
        rng = np.random.default_rng()

    photoelectrons_array = np.asarray(photoelectrons, dtype=np.float64)

    p_fire = 1.0 - (1.0 - 1.0 / n_microcells) ** photoelectrons_array

    return rng.binomial(n=n_microcells, p=p_fire).astype(np.int64)


def estimate_photoelectrons_from_fired_microcells(
    fired_microcells: int | NDArray[np.int64],
    n_microcells: int = MICROFC_60035_C_SERIES_MICROCELLS,
) -> NDArray[np.float64]:
    """Estimate incident photoelectrons from fired microcells.

    This is the inverse of the expected saturation model:

        N_pe = ln(1 - N_fired / N_cells) / ln(1 - 1 / N_cells)

    If N_fired equals N_cells, the estimate diverges and is returned as inf.

    Args:
        fired_microcells: Number of fired microcells.
        n_microcells: Total number of available microcells.

    Returns:
        Estimated number of incident photoelectrons.

    Raises:
        ValueError: If fired_microcells exceeds n_microcells.
    """
    validate_microcell_count(n_microcells)
    validate_non_negative_integer_counts(fired_microcells, name="fired_microcells")

    fired_array = np.asarray(fired_microcells, dtype=np.float64)

    if np.any(fired_array > n_microcells):
        raise ValueError("fired_microcells cannot exceed n_microcells.")

    estimate = np.full_like(fired_array, fill_value=np.inf, dtype=np.float64)

    unsaturated_mask = fired_array < n_microcells
    estimate[unsaturated_mask] = np.log(
        1.0 - fired_array[unsaturated_mask] / n_microcells
    ) / np.log(1.0 - 1.0 / n_microcells)

    return estimate