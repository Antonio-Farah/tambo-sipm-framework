"""SiPM noise models.

This module implements first-order SiPM noise models used in the TAMBO
simulation framework.

Conventions:
    - Time is expressed in ns.
    - Rates are expressed in Hz.
    - Amplitudes are expressed in photoelectron-equivalent units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NoisePulses:
    """Container for generated SiPM noise pulses."""

    times_ns: NDArray[np.float64]
    amplitudes_pe: NDArray[np.float64]
    sources: tuple[str, ...]


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


def validate_rate_hz(rate_hz: float, name: str = "rate_hz") -> None:
    """Validate a non-negative rate.

    Args:
        rate_hz: Rate in Hz.
        name: Name used in the error message.

    Raises:
        ValueError: If the rate is negative.
    """
    if rate_hz < 0.0:
        raise ValueError(f"{name} must be non-negative.")


def validate_positive_time(value_ns: float, name: str) -> None:
    """Validate a positive time value in ns.

    Args:
        value_ns: Time value in ns.
        name: Name used in the error message.

    Raises:
        ValueError: If the time value is not positive.
    """
    if value_ns <= 0.0:
        raise ValueError(f"{name} must be positive.")


def sample_dcr_times(
    window_ns: float,
    dcr_hz: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Sample dark count times inside an acquisition window.

    Dark counts are modeled as a Poisson process. Therefore, the time
    difference between consecutive dark counts follows an exponential
    distribution with mean 1 / DCR.

    Args:
        window_ns: Acquisition window in ns.
        dcr_hz: Dark Count Rate in Hz.
        rng: Optional NumPy random generator.

    Returns:
        Sorted dark count times in ns.
    """
    validate_positive_time(window_ns, "window_ns")
    validate_rate_hz(dcr_hz, "dcr_hz")

    if rng is None:
        rng = np.random.default_rng()

    if dcr_hz == 0.0:
        return np.array([], dtype=np.float64)

    mean_interval_ns = 1.0e9 / dcr_hz
    times: list[float] = []

    current_time_ns = float(rng.exponential(mean_interval_ns))

    while current_time_ns < window_ns:
        times.append(current_time_ns)
        current_time_ns += float(rng.exponential(mean_interval_ns))

    return np.asarray(times, dtype=np.float64)


def sample_gain_amplitudes(
    n_pulses: int,
    mean_pe: float = 1.0,
    sigma_pe: float = 0.1,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Sample SiPM pulse amplitudes in p.e.

    The SiPM framework paper models the single-photoelectron amplitude with a
    Gaussian distribution around 1 p.e. This function clips negative values to
    zero to avoid unphysical negative pulse amplitudes.

    Args:
        n_pulses: Number of amplitudes to sample.
        mean_pe: Mean pulse amplitude in p.e.
        sigma_pe: Standard deviation of pulse amplitude in p.e.
        rng: Optional NumPy random generator.

    Returns:
        Pulse amplitudes in p.e.

    Raises:
        ValueError: If inputs are invalid.
    """
    if n_pulses < 0:
        raise ValueError("n_pulses must be non-negative.")
    if mean_pe < 0.0:
        raise ValueError("mean_pe must be non-negative.")
    if sigma_pe < 0.0:
        raise ValueError("sigma_pe must be non-negative.")

    if rng is None:
        rng = np.random.default_rng()

    if n_pulses == 0:
        return np.array([], dtype=np.float64)

    amplitudes = rng.normal(loc=mean_pe, scale=sigma_pe, size=n_pulses)

    return np.clip(amplitudes, a_min=0.0, a_max=None).astype(np.float64)


def sample_crosstalk_multiplicity(
    n_primary: int,
    crosstalk_prob: float,
    rng: np.random.Generator | None = None,
    max_multiplicity: int = 100,
) -> NDArray[np.int64]:
    """Sample crosstalk multiplicity for primary avalanches.

    The multiplicity includes the primary avalanche. A value of 1 means no
    additional crosstalk avalanche occurred.

    The model is a simple geometric chain: after each avalanche, an additional
    avalanche is generated with probability crosstalk_prob.

    Args:
        n_primary: Number of primary avalanches.
        crosstalk_prob: Crosstalk probability.
        rng: Optional NumPy random generator.
        max_multiplicity: Safety cap for the chain length.

    Returns:
        Multiplicity array with values >= 1.
    """
    if n_primary < 0:
        raise ValueError("n_primary must be non-negative.")
    validate_probability(crosstalk_prob, "crosstalk_prob")
    if max_multiplicity <= 0:
        raise ValueError("max_multiplicity must be positive.")

    if rng is None:
        rng = np.random.default_rng()

    multiplicity = np.ones(n_primary, dtype=np.int64)

    if n_primary == 0 or crosstalk_prob == 0.0:
        return multiplicity

    active = rng.random(n_primary) < crosstalk_prob

    while np.any(active):
        multiplicity[active] += 1

        still_allowed = multiplicity < max_multiplicity
        active = active & still_allowed & (rng.random(n_primary) < crosstalk_prob)

    return multiplicity


def apply_crosstalk_to_amplitudes(
    amplitudes_pe: NDArray[np.float64],
    crosstalk_prob: float,
    rng: np.random.Generator | None = None,
    max_multiplicity: int = 100,
) -> NDArray[np.float64]:
    """Apply crosstalk multiplicity to pulse amplitudes.

    Args:
        amplitudes_pe: Primary pulse amplitudes in p.e.
        crosstalk_prob: Crosstalk probability.
        rng: Optional NumPy random generator.
        max_multiplicity: Safety cap for the chain length.

    Returns:
        Amplitudes after crosstalk contribution.
    """
    multiplicity = sample_crosstalk_multiplicity(
        n_primary=len(amplitudes_pe),
        crosstalk_prob=crosstalk_prob,
        rng=rng,
        max_multiplicity=max_multiplicity,
    )

    return amplitudes_pe.astype(np.float64) * multiplicity.astype(np.float64)


def afterpulse_amplitude_factor(
    delay_ns: float | NDArray[np.float64],
    tau_recovery_ns: float,
) -> NDArray[np.float64]:
    """Compute afterpulse amplitude recovery factor.

    The model follows:

        A_AP(t) = 1 - exp(-t / tau_recovery)

    Args:
        delay_ns: Delay after the primary avalanche in ns.
        tau_recovery_ns: Microcell recovery time constant in ns.

    Returns:
        Afterpulse amplitude factor in [0, 1].
    """
    validate_positive_time(tau_recovery_ns, "tau_recovery_ns")

    delay_array = np.asarray(delay_ns, dtype=np.float64)

    if np.any(delay_array < 0.0):
        raise ValueError("delay_ns must be non-negative.")

    return 1.0 - np.exp(-delay_array / tau_recovery_ns)


def generate_afterpulses(
    primary_times_ns: NDArray[np.float64],
    primary_amplitudes_pe: NDArray[np.float64],
    afterpulse_prob: float,
    tau_afterpulse_ns: float,
    tau_recovery_ns: float,
    window_ns: float,
    rng: np.random.Generator | None = None,
) -> NoisePulses:
    """Generate afterpulses associated with primary avalanches.

    This first-order implementation creates at most one afterpulse per primary
    avalanche. The delay is sampled from an exponential distribution with mean
    tau_afterpulse_ns, and the amplitude is reduced by the microcell recovery
    factor.

    Args:
        primary_times_ns: Primary pulse times in ns.
        primary_amplitudes_pe: Primary pulse amplitudes in p.e.
        afterpulse_prob: Probability of generating an afterpulse per primary.
        tau_afterpulse_ns: Mean afterpulse delay in ns.
        tau_recovery_ns: Microcell recovery time constant in ns.
        window_ns: Acquisition window in ns.
        rng: Optional NumPy random generator.

    Returns:
        NoisePulses containing afterpulse times and amplitudes.
    """
    if primary_times_ns.shape != primary_amplitudes_pe.shape:
        raise ValueError("primary_times_ns and primary_amplitudes_pe must match.")

    validate_probability(afterpulse_prob, "afterpulse_prob")
    validate_positive_time(tau_afterpulse_ns, "tau_afterpulse_ns")
    validate_positive_time(tau_recovery_ns, "tau_recovery_ns")
    validate_positive_time(window_ns, "window_ns")

    if rng is None:
        rng = np.random.default_rng()

    if len(primary_times_ns) == 0 or afterpulse_prob == 0.0:
        return NoisePulses(
            times_ns=np.array([], dtype=np.float64),
            amplitudes_pe=np.array([], dtype=np.float64),
            sources=(),
        )

    create_afterpulse = rng.random(len(primary_times_ns)) < afterpulse_prob
    selected_times = primary_times_ns[create_afterpulse]
    selected_amplitudes = primary_amplitudes_pe[create_afterpulse]

    if len(selected_times) == 0:
        return NoisePulses(
            times_ns=np.array([], dtype=np.float64),
            amplitudes_pe=np.array([], dtype=np.float64),
            sources=(),
        )

    delays_ns = rng.exponential(scale=tau_afterpulse_ns, size=len(selected_times))
    afterpulse_times_ns = selected_times + delays_ns

    inside_window = afterpulse_times_ns < window_ns

    afterpulse_times_ns = afterpulse_times_ns[inside_window]
    delays_ns = delays_ns[inside_window]
    selected_amplitudes = selected_amplitudes[inside_window]

    amplitude_factors = afterpulse_amplitude_factor(
        delay_ns=delays_ns,
        tau_recovery_ns=tau_recovery_ns,
    )

    afterpulse_amplitudes_pe = selected_amplitudes * amplitude_factors

    return NoisePulses(
        times_ns=afterpulse_times_ns.astype(np.float64),
        amplitudes_pe=afterpulse_amplitudes_pe.astype(np.float64),
        sources=tuple("afterpulse" for _ in range(len(afterpulse_times_ns))),
    )


def generate_dark_noise(
    window_ns: float,
    dcr_hz: float,
    gain_mean_pe: float = 1.0,
    gain_sigma_pe: float = 0.1,
    crosstalk_prob: float = 0.0,
    afterpulse_prob: float = 0.0,
    tau_afterpulse_ns: float = 30.0,
    tau_recovery_ns: float = 95.0,
    rng: np.random.Generator | None = None,
) -> NoisePulses:
    """Generate dark noise pulses including DCR, crosstalk, and afterpulsing.

    Args:
        window_ns: Acquisition window in ns.
        dcr_hz: Dark Count Rate in Hz.
        gain_mean_pe: Mean primary dark pulse amplitude in p.e.
        gain_sigma_pe: Gain spread in p.e.
        crosstalk_prob: Crosstalk probability.
        afterpulse_prob: Afterpulsing probability.
        tau_afterpulse_ns: Mean afterpulse delay in ns.
        tau_recovery_ns: Microcell recovery time constant in ns.
        rng: Optional NumPy random generator.

    Returns:
        NoisePulses containing sorted pulse times, amplitudes, and sources.
    """
    validate_positive_time(window_ns, "window_ns")
    validate_rate_hz(dcr_hz, "dcr_hz")
    validate_probability(crosstalk_prob, "crosstalk_prob")
    validate_probability(afterpulse_prob, "afterpulse_prob")
    validate_positive_time(tau_afterpulse_ns, "tau_afterpulse_ns")
    validate_positive_time(tau_recovery_ns, "tau_recovery_ns")

    if rng is None:
        rng = np.random.default_rng()

    dcr_times_ns = sample_dcr_times(
        window_ns=window_ns,
        dcr_hz=dcr_hz,
        rng=rng,
    )

    dcr_amplitudes_pe = sample_gain_amplitudes(
        n_pulses=len(dcr_times_ns),
        mean_pe=gain_mean_pe,
        sigma_pe=gain_sigma_pe,
        rng=rng,
    )

    dcr_amplitudes_pe = apply_crosstalk_to_amplitudes(
        amplitudes_pe=dcr_amplitudes_pe,
        crosstalk_prob=crosstalk_prob,
        rng=rng,
    )

    afterpulses = generate_afterpulses(
        primary_times_ns=dcr_times_ns,
        primary_amplitudes_pe=dcr_amplitudes_pe,
        afterpulse_prob=afterpulse_prob,
        tau_afterpulse_ns=tau_afterpulse_ns,
        tau_recovery_ns=tau_recovery_ns,
        window_ns=window_ns,
        rng=rng,
    )

    all_times = np.concatenate([dcr_times_ns, afterpulses.times_ns])
    all_amplitudes = np.concatenate([dcr_amplitudes_pe, afterpulses.amplitudes_pe])
    all_sources = tuple("dcr" for _ in range(len(dcr_times_ns))) + afterpulses.sources

    if len(all_times) == 0:
        return NoisePulses(
            times_ns=np.array([], dtype=np.float64),
            amplitudes_pe=np.array([], dtype=np.float64),
            sources=(),
        )

    order = np.argsort(all_times)

    return NoisePulses(
        times_ns=all_times[order].astype(np.float64),
        amplitudes_pe=all_amplitudes[order].astype(np.float64),
        sources=tuple(all_sources[index] for index in order),
    )