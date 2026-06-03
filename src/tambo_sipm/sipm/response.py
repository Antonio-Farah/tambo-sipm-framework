"""SiPM response simulation.

This module combines the intrinsic SiPM pulse model, finite microcell
saturation, gain spread, crosstalk, and optional dark noise into a complete
SiPM waveform in photoelectron-equivalent units.

Conventions:
    - Time is expressed in ns.
    - Signal amplitude is expressed in p.e.
    - Rates are expressed in Hz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tambo_sipm.sipm.noise import (
    NoisePulses,
    apply_crosstalk_to_amplitudes,
    generate_dark_noise,
    sample_gain_amplitudes,
)
from tambo_sipm.sipm.pulse import make_time_axis, sum_pulses
from tambo_sipm.sipm.saturation import (
    MICROFC_60035_C_SERIES_MICROCELLS,
    sample_fired_microcells,
    validate_microcell_count,
)


@dataclass(frozen=True)
class SiPMResponseResult:
    """Container for SiPM response simulation outputs."""

    time_ns: NDArray[np.float64]
    photon_signal_pe: NDArray[np.float64]
    dark_noise_pe: NDArray[np.float64]
    total_signal_pe: NDArray[np.float64]
    input_photoelectrons: int
    fired_microcells: int
    photon_arrival_times_ns: NDArray[np.float64]
    photon_amplitudes_pe: NDArray[np.float64]
    noise_pulses: NoisePulses


def validate_photoelectron_count(photoelectrons: int) -> None:
    """Validate a photoelectron count.

    Args:
        photoelectrons: Number of photoelectrons.

    Raises:
        ValueError: If photoelectrons is negative or non-integer.
    """
    if not isinstance(photoelectrons, (int, np.integer)):
        raise ValueError("photoelectrons must be an integer.")
    if photoelectrons < 0:
        raise ValueError("photoelectrons must be non-negative.")


def validate_response_times(
    window_ns: float,
    dt_ns: float,
    event_time_ns: float,
    arrival_spread_ns: float,
) -> None:
    """Validate time parameters for SiPM response simulation.

    Args:
        window_ns: Acquisition window in ns.
        dt_ns: Time step in ns.
        event_time_ns: Primary signal time in ns.
        arrival_spread_ns: Standard deviation of photon arrival times in ns.

    Raises:
        ValueError: If any time parameter is invalid.
    """
    if window_ns <= 0.0:
        raise ValueError("window_ns must be positive.")
    if dt_ns <= 0.0:
        raise ValueError("dt_ns must be positive.")
    if event_time_ns < 0.0 or event_time_ns >= window_ns:
        raise ValueError("event_time_ns must lie inside the acquisition window.")
    if arrival_spread_ns < 0.0:
        raise ValueError("arrival_spread_ns must be non-negative.")


def sample_photoelectron_arrival_times(
    n_pulses: int,
    event_time_ns: float,
    arrival_spread_ns: float,
    window_ns: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Sample photoelectron arrival times.

    If arrival_spread_ns is zero, all pulses are placed at event_time_ns.
    Otherwise, arrival times are sampled from a Gaussian distribution centered
    at event_time_ns and clipped to the acquisition window.

    Args:
        n_pulses: Number of pulse arrival times to sample.
        event_time_ns: Mean event time in ns.
        arrival_spread_ns: Standard deviation of arrival times in ns.
        window_ns: Acquisition window in ns.
        rng: Optional NumPy random generator.

    Returns:
        Arrival times in ns.

    Raises:
        ValueError: If inputs are invalid.
    """
    if n_pulses < 0:
        raise ValueError("n_pulses must be non-negative.")

    validate_response_times(
        window_ns=window_ns,
        dt_ns=1.0,
        event_time_ns=event_time_ns,
        arrival_spread_ns=arrival_spread_ns,
    )

    if rng is None:
        rng = np.random.default_rng()

    if n_pulses == 0:
        return np.array([], dtype=np.float64)

    if arrival_spread_ns == 0.0:
        return np.full(n_pulses, event_time_ns, dtype=np.float64)

    arrival_times_ns = rng.normal(
        loc=event_time_ns,
        scale=arrival_spread_ns,
        size=n_pulses,
    )

    return np.clip(arrival_times_ns, 0.0, np.nextafter(window_ns, 0.0)).astype(
        np.float64
    )


def simulate_sipm_response(
    photoelectrons: int,
    window_ns: float,
    dt_ns: float,
    tau_r_ns: float,
    tau_f_ns: float,
    event_time_ns: float = 0.0,
    arrival_spread_ns: float = 0.0,
    n_microcells: int = MICROFC_60035_C_SERIES_MICROCELLS,
    gain_mean_pe: float = 1.0,
    gain_sigma_pe: float = 0.1,
    crosstalk_prob: float = 0.0,
    include_dark_noise: bool = False,
    dcr_hz: float = 0.0,
    afterpulse_prob: float = 0.0,
    tau_afterpulse_ns: float = 30.0,
    tau_recovery_ns: float = 95.0,
    rng: np.random.Generator | None = None,
) -> SiPMResponseResult:
    """Simulate a complete SiPM waveform in p.e.

    Args:
        photoelectrons: Number of signal photoelectrons.
        window_ns: Acquisition window in ns.
        dt_ns: Time step in ns.
        tau_r_ns: SiPM pulse rise-time constant in ns.
        tau_f_ns: SiPM pulse fall-time constant in ns.
        event_time_ns: Mean signal time in ns.
        arrival_spread_ns: Standard deviation of signal arrival times in ns.
        n_microcells: Number of available SiPM microcells.
        gain_mean_pe: Mean single-cell amplitude in p.e.
        gain_sigma_pe: Standard deviation of single-cell amplitude in p.e.
        crosstalk_prob: Crosstalk probability applied to signal and dark pulses.
        include_dark_noise: If True, add DCR, crosstalk, and afterpulse noise.
        dcr_hz: Dark Count Rate in Hz.
        afterpulse_prob: Afterpulsing probability for dark pulses.
        tau_afterpulse_ns: Mean afterpulse delay in ns.
        tau_recovery_ns: Microcell recovery time constant in ns.
        rng: Optional NumPy random generator.

    Returns:
        SiPMResponseResult containing the waveform components and metadata.
    """
    validate_photoelectron_count(photoelectrons)
    validate_response_times(
        window_ns=window_ns,
        dt_ns=dt_ns,
        event_time_ns=event_time_ns,
        arrival_spread_ns=arrival_spread_ns,
    )
    validate_microcell_count(n_microcells)

    if rng is None:
        rng = np.random.default_rng()

    time_ns = make_time_axis(window_ns=window_ns, dt_ns=dt_ns)

    if photoelectrons == 0:
        fired_microcells = 0
    else:
        fired_microcells = int(
            sample_fired_microcells(
                photoelectrons=np.array([photoelectrons], dtype=np.int64),
                n_microcells=n_microcells,
                rng=rng,
            )[0]
        )

    photon_arrival_times_ns = sample_photoelectron_arrival_times(
        n_pulses=fired_microcells,
        event_time_ns=event_time_ns,
        arrival_spread_ns=arrival_spread_ns,
        window_ns=window_ns,
        rng=rng,
    )

    photon_amplitudes_pe = sample_gain_amplitudes(
        n_pulses=fired_microcells,
        mean_pe=gain_mean_pe,
        sigma_pe=gain_sigma_pe,
        rng=rng,
    )

    photon_amplitudes_pe = apply_crosstalk_to_amplitudes(
        amplitudes_pe=photon_amplitudes_pe,
        crosstalk_prob=crosstalk_prob,
        rng=rng,
    )

    photon_signal_pe = sum_pulses(
        time_ns=time_ns,
        arrival_times_ns=photon_arrival_times_ns,
        amplitudes_pe=photon_amplitudes_pe,
        tau_r_ns=tau_r_ns,
        tau_f_ns=tau_f_ns,
        normalize_peak=True,
    )

    if include_dark_noise:
        noise_pulses = generate_dark_noise(
            window_ns=window_ns,
            dcr_hz=dcr_hz,
            gain_mean_pe=gain_mean_pe,
            gain_sigma_pe=gain_sigma_pe,
            crosstalk_prob=crosstalk_prob,
            afterpulse_prob=afterpulse_prob,
            tau_afterpulse_ns=tau_afterpulse_ns,
            tau_recovery_ns=tau_recovery_ns,
            rng=rng,
        )
    else:
        noise_pulses = NoisePulses(
            times_ns=np.array([], dtype=np.float64),
            amplitudes_pe=np.array([], dtype=np.float64),
            sources=(),
        )

    dark_noise_pe = sum_pulses(
        time_ns=time_ns,
        arrival_times_ns=noise_pulses.times_ns,
        amplitudes_pe=noise_pulses.amplitudes_pe,
        tau_r_ns=tau_r_ns,
        tau_f_ns=tau_f_ns,
        normalize_peak=True,
    )

    total_signal_pe = photon_signal_pe + dark_noise_pe

    return SiPMResponseResult(
        time_ns=time_ns,
        photon_signal_pe=photon_signal_pe,
        dark_noise_pe=dark_noise_pe,
        total_signal_pe=total_signal_pe,
        input_photoelectrons=int(photoelectrons),
        fired_microcells=fired_microcells,
        photon_arrival_times_ns=photon_arrival_times_ns,
        photon_amplitudes_pe=photon_amplitudes_pe,
        noise_pulses=noise_pulses,
    )