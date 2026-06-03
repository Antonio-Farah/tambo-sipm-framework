"""SiPM pulse-shape models.

This module defines the elementary single-photoelectron response used by the
TAMBO SiPM simulation framework. All time quantities are expressed in
nanoseconds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_time_axis(window_ns: float, dt_ns: float) -> NDArray[np.float64]:
    """Create a uniformly sampled time axis.

    Args:
        window_ns: Total acquisition window in nanoseconds.
        dt_ns: Sampling step in nanoseconds.

    Returns:
        Time array from 0 to window_ns, excluding the endpoint.

    Raises:
        ValueError: If window_ns or dt_ns are not positive.
    """
    if window_ns <= 0:
        raise ValueError("window_ns must be positive.")
    if dt_ns <= 0:
        raise ValueError("dt_ns must be positive.")

    return np.arange(0.0, window_ns, dt_ns, dtype=np.float64)


def raw_sipm_pulse(
    time_ns: NDArray[np.float64],
    tau_r_ns: float,
    tau_f_ns: float,
    amplitude_pe: float = 1.0,
    t0_ns: float = 0.0,
) -> NDArray[np.float64]:
    """Compute the raw analytical SiPM pulse.

    The analytical model is:

        h(t) = A * (1 - exp(-(t - t0) / tau_r)) * exp(-(t - t0) / tau_f)

    for t >= t0, and h(t) = 0 for t < t0.

    In this raw form, amplitude_pe is the analytical scale parameter A, not
    necessarily the maximum value of the pulse.

    Args:
        time_ns: Time array in nanoseconds.
        tau_r_ns: Rise-time constant in nanoseconds.
        tau_f_ns: Fall-time constant in nanoseconds.
        amplitude_pe: Analytical amplitude scale in photoelectrons.
        t0_ns: Pulse start time in nanoseconds.

    Returns:
        Pulse amplitude array in photoelectron-equivalent units.

    Raises:
        ValueError: If tau_r_ns, tau_f_ns, or amplitude_pe are invalid.
    """
    if tau_r_ns <= 0:
        raise ValueError("tau_r_ns must be positive.")
    if tau_f_ns <= 0:
        raise ValueError("tau_f_ns must be positive.")
    if amplitude_pe < 0:
        raise ValueError("amplitude_pe must be non-negative.")

    shifted_time = time_ns - t0_ns
    pulse = np.zeros_like(time_ns, dtype=np.float64)

    mask = shifted_time >= 0.0
    t = shifted_time[mask]

    pulse[mask] = amplitude_pe * (1.0 - np.exp(-t / tau_r_ns)) * np.exp(
        -t / tau_f_ns
    )

    return pulse


def normalized_sipm_pulse(
    time_ns: NDArray[np.float64],
    tau_r_ns: float,
    tau_f_ns: float,
    amplitude_pe: float = 1.0,
    t0_ns: float = 0.0,
) -> NDArray[np.float64]:
    """Compute a SiPM pulse normalized so its peak equals amplitude_pe.

    This function uses the same analytical pulse shape as raw_sipm_pulse, but
    rescales the result so that the maximum value is exactly amplitude_pe.
    This is useful when amplitude_pe is intended to represent the observed
    peak amplitude in photoelectron-equivalent units.

    Args:
        time_ns: Time array in nanoseconds.
        tau_r_ns: Rise-time constant in nanoseconds.
        tau_f_ns: Fall-time constant in nanoseconds.
        amplitude_pe: Desired peak amplitude in photoelectrons.
        t0_ns: Pulse start time in nanoseconds.

    Returns:
        Pulse amplitude array in photoelectron-equivalent units.

    Raises:
        ValueError: If amplitude_pe is invalid.
    """
    if amplitude_pe < 0:
        raise ValueError("amplitude_pe must be non-negative.")
    if amplitude_pe == 0:
        return np.zeros_like(time_ns, dtype=np.float64)

    unit_pulse = raw_sipm_pulse(
        time_ns=time_ns,
        tau_r_ns=tau_r_ns,
        tau_f_ns=tau_f_ns,
        amplitude_pe=1.0,
        t0_ns=t0_ns,
    )

    peak = float(np.max(unit_pulse))
    if peak <= 0:
        return np.zeros_like(time_ns, dtype=np.float64)

    return amplitude_pe * unit_pulse / peak


def sum_pulses(
    time_ns: NDArray[np.float64],
    arrival_times_ns: NDArray[np.float64],
    amplitudes_pe: NDArray[np.float64],
    tau_r_ns: float,
    tau_f_ns: float,
    normalize_peak: bool = True,
) -> NDArray[np.float64]:
    """Sum multiple SiPM pulses at different arrival times.

    Args:
        time_ns: Time array in nanoseconds.
        arrival_times_ns: Pulse start times in nanoseconds.
        amplitudes_pe: Pulse amplitudes in photoelectron-equivalent units.
        tau_r_ns: Rise-time constant in nanoseconds.
        tau_f_ns: Fall-time constant in nanoseconds.
        normalize_peak: If True, each individual pulse peak is normalized to
            its amplitude_pe value.

    Returns:
        Total waveform in photoelectron-equivalent units.

    Raises:
        ValueError: If arrival_times_ns and amplitudes_pe have different sizes.
    """
    if arrival_times_ns.shape != amplitudes_pe.shape:
        raise ValueError("arrival_times_ns and amplitudes_pe must have same shape.")

    waveform = np.zeros_like(time_ns, dtype=np.float64)

    pulse_function = normalized_sipm_pulse if normalize_peak else raw_sipm_pulse

    for arrival_time_ns, amplitude_pe in zip(arrival_times_ns, amplitudes_pe):
        waveform += pulse_function(
            time_ns=time_ns,
            tau_r_ns=tau_r_ns,
            tau_f_ns=tau_f_ns,
            amplitude_pe=float(amplitude_pe),
            t0_ns=float(arrival_time_ns),
        )

    return waveform


def estimate_fwhm_ns(
    time_ns: NDArray[np.float64],
    pulse: NDArray[np.float64],
) -> float:
    """Estimate the full width at half maximum of a pulse.

    Args:
        time_ns: Time array in nanoseconds.
        pulse: Pulse amplitude array.

    Returns:
        Estimated FWHM in nanoseconds. Returns 0 if the pulse is empty or
        never reaches half maximum.
    """
    if time_ns.shape != pulse.shape:
        raise ValueError("time_ns and pulse must have the same shape.")

    peak = float(np.max(pulse))
    if peak <= 0:
        return 0.0

    half_max = 0.5 * peak
    indices = np.where(pulse >= half_max)[0]

    if len(indices) < 2:
        return 0.0

    return float(time_ns[indices[-1]] - time_ns[indices[0]])