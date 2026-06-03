"""Pulse feature extraction utilities.

This module provides feature extraction tools for TAMBO SiPM waveforms.

Conventions:
    - Time is expressed in ns.
    - Voltage is expressed in mV.
    - Baseline is estimated from pre-trigger samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PulseFeatures:
    """Container for pulse-level observables."""

    valid: bool
    baseline_mV: float
    noise_rms_mV: float
    peak_mV: float
    peak_time_ns: float
    width_ns: float
    integral_mVns: float
    rms_mV: float
    first_crossing_time_ns: float | None
    last_crossing_time_ns: float | None
    n_samples_above_threshold: int


def validate_waveform_inputs(
    time_ns: NDArray[np.float64],
    voltage_mV: NDArray[np.float64],
) -> None:
    """Validate waveform input arrays.

    Args:
        time_ns: Time array in ns.
        voltage_mV: Voltage array in mV.

    Raises:
        ValueError: If the arrays are invalid.
    """
    if time_ns.shape != voltage_mV.shape:
        raise ValueError("time_ns and voltage_mV must have the same shape.")
    if len(time_ns) < 2:
        raise ValueError("waveform must contain at least two samples.")
    if not np.all(np.diff(time_ns) > 0):
        raise ValueError("time_ns must be strictly increasing.")


def estimate_baseline(
    voltage_mV: NDArray[np.float64],
    n_pretrigger_samples: int = 3,
    method: str = "median",
) -> float:
    """Estimate the waveform baseline from pre-trigger samples.

    Args:
        voltage_mV: Voltage waveform in mV.
        n_pretrigger_samples: Number of initial samples used for baseline.
        method: Baseline estimator. Supported values are "median" and "mean".

    Returns:
        Estimated baseline in mV.

    Raises:
        ValueError: If inputs are invalid.
    """
    if n_pretrigger_samples <= 0:
        raise ValueError("n_pretrigger_samples must be positive.")
    if n_pretrigger_samples > len(voltage_mV):
        raise ValueError("n_pretrigger_samples cannot exceed waveform length.")

    pretrigger = voltage_mV[:n_pretrigger_samples]

    if method == "median":
        return float(np.median(pretrigger))
    if method == "mean":
        return float(np.mean(pretrigger))

    raise ValueError("method must be either 'median' or 'mean'.")


def estimate_noise_rms(
    voltage_mV: NDArray[np.float64],
    baseline_mV: float,
    n_pretrigger_samples: int = 3,
) -> float:
    """Estimate pre-trigger noise RMS.

    Args:
        voltage_mV: Voltage waveform in mV.
        baseline_mV: Baseline estimate in mV.
        n_pretrigger_samples: Number of initial samples used for noise estimate.

    Returns:
        Pre-trigger RMS noise in mV.

    Raises:
        ValueError: If n_pretrigger_samples is invalid.
    """
    if n_pretrigger_samples <= 0:
        raise ValueError("n_pretrigger_samples must be positive.")
    if n_pretrigger_samples > len(voltage_mV):
        raise ValueError("n_pretrigger_samples cannot exceed waveform length.")

    pretrigger = voltage_mV[:n_pretrigger_samples] - baseline_mV

    return float(np.sqrt(np.mean(pretrigger**2)))


def subtract_baseline(
    voltage_mV: NDArray[np.float64],
    baseline_mV: float,
) -> NDArray[np.float64]:
    """Subtract a scalar baseline from a voltage waveform.

    Args:
        voltage_mV: Voltage waveform in mV.
        baseline_mV: Baseline estimate in mV.

    Returns:
        Baseline-corrected voltage waveform in mV.
    """
    return voltage_mV.astype(np.float64) - baseline_mV


def find_threshold_indices(
    voltage_mV: NDArray[np.float64],
    threshold_mV: float,
) -> NDArray[np.int64]:
    """Find indices where the waveform is above or equal to threshold.

    Args:
        voltage_mV: Baseline-corrected voltage waveform in mV.
        threshold_mV: Threshold in mV.

    Returns:
        Array of indices satisfying voltage_mV >= threshold_mV.
    """
    return np.where(voltage_mV >= threshold_mV)[0].astype(np.int64)


def pulse_width_ns(
    time_ns: NDArray[np.float64],
    indices_above_threshold: NDArray[np.int64],
) -> float:
    """Compute pulse width from threshold-crossing indices.

    Args:
        time_ns: Time array in ns.
        indices_above_threshold: Indices where the waveform is above threshold.

    Returns:
        Width in ns. Returns 0 if fewer than two samples are above threshold.
    """
    if len(indices_above_threshold) < 2:
        return 0.0

    first_index = indices_above_threshold[0]
    last_index = indices_above_threshold[-1]

    return float(time_ns[last_index] - time_ns[first_index])


def integrate_pulse(
    time_ns: NDArray[np.float64],
    voltage_mV: NDArray[np.float64],
    indices_above_threshold: NDArray[np.int64],
) -> float:
    """Integrate the pulse between first and last threshold crossing.

    Args:
        time_ns: Time array in ns.
        voltage_mV: Baseline-corrected voltage waveform in mV.
        indices_above_threshold: Indices where the waveform is above threshold.

    Returns:
        Pulse integral in mV ns. Returns 0 if fewer than two samples are above
        threshold.
    """
    if len(indices_above_threshold) < 2:
        return 0.0

    first_index = indices_above_threshold[0]
    last_index = indices_above_threshold[-1]

    return float(
        np.trapezoid(
            voltage_mV[first_index : last_index + 1],
            time_ns[first_index : last_index + 1],
        )
    )


def waveform_rms(
    voltage_mV: NDArray[np.float64],
    indices_above_threshold: NDArray[np.int64] | None = None,
) -> float:
    """Compute RMS voltage.

    Args:
        voltage_mV: Baseline-corrected voltage waveform in mV.
        indices_above_threshold: Optional indices used to restrict the RMS
            calculation.

    Returns:
        RMS voltage in mV.
    """
    if indices_above_threshold is None:
        values = voltage_mV
    else:
        if len(indices_above_threshold) == 0:
            return 0.0
        values = voltage_mV[indices_above_threshold]

    return float(np.sqrt(np.mean(values**2)))


def extract_pulse_features(
    time_ns: NDArray[np.float64],
    voltage_mV: NDArray[np.float64],
    threshold_mV: float,
    n_pretrigger_samples: int = 3,
    baseline_method: str = "median",
) -> PulseFeatures:
    """Extract pulse features from a waveform.

    The waveform is first baseline-corrected using the first pre-trigger
    samples. A pulse is considered valid only if it crosses the threshold in at
    least two samples.

    Args:
        time_ns: Time array in ns.
        voltage_mV: Voltage waveform in mV.
        threshold_mV: Threshold applied after baseline subtraction.
        n_pretrigger_samples: Number of initial samples used for baseline.
        baseline_method: Baseline estimator. Supported values are "median"
            and "mean".

    Returns:
        PulseFeatures object with pulse observables.

    Raises:
        ValueError: If the waveform inputs are invalid.
    """
    validate_waveform_inputs(time_ns, voltage_mV)

    baseline_mV = estimate_baseline(
        voltage_mV=voltage_mV,
        n_pretrigger_samples=n_pretrigger_samples,
        method=baseline_method,
    )

    noise_rms_mV = estimate_noise_rms(
        voltage_mV=voltage_mV,
        baseline_mV=baseline_mV,
        n_pretrigger_samples=n_pretrigger_samples,
    )

    corrected_voltage_mV = subtract_baseline(
        voltage_mV=voltage_mV,
        baseline_mV=baseline_mV,
    )

    peak_index = int(np.argmax(corrected_voltage_mV))
    peak_mV = float(corrected_voltage_mV[peak_index])
    peak_time_ns = float(time_ns[peak_index])

    indices_above_threshold = find_threshold_indices(
        voltage_mV=corrected_voltage_mV,
        threshold_mV=threshold_mV,
    )

    n_samples_above_threshold = int(len(indices_above_threshold))
    valid = n_samples_above_threshold >= 2

    if not valid:
        return PulseFeatures(
            valid=False,
            baseline_mV=baseline_mV,
            noise_rms_mV=noise_rms_mV,
            peak_mV=peak_mV,
            peak_time_ns=peak_time_ns,
            width_ns=0.0,
            integral_mVns=0.0,
            rms_mV=waveform_rms(corrected_voltage_mV),
            first_crossing_time_ns=None,
            last_crossing_time_ns=None,
            n_samples_above_threshold=n_samples_above_threshold,
        )

    first_crossing_time_ns = float(time_ns[indices_above_threshold[0]])
    last_crossing_time_ns = float(time_ns[indices_above_threshold[-1]])

    width_ns = pulse_width_ns(
        time_ns=time_ns,
        indices_above_threshold=indices_above_threshold,
    )

    integral_mVns = integrate_pulse(
        time_ns=time_ns,
        voltage_mV=corrected_voltage_mV,
        indices_above_threshold=indices_above_threshold,
    )

    rms_mV = waveform_rms(
        voltage_mV=corrected_voltage_mV,
        indices_above_threshold=indices_above_threshold,
    )

    return PulseFeatures(
        valid=True,
        baseline_mV=baseline_mV,
        noise_rms_mV=noise_rms_mV,
        peak_mV=peak_mV,
        peak_time_ns=peak_time_ns,
        width_ns=width_ns,
        integral_mVns=integral_mVns,
        rms_mV=rms_mV,
        first_crossing_time_ns=first_crossing_time_ns,
        last_crossing_time_ns=last_crossing_time_ns,
        n_samples_above_threshold=n_samples_above_threshold,
    )