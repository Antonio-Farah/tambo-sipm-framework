"""ADC and waveform digitization utilities.

This module implements sampling and digitization utilities for the
TAMBO Red Pitaya STEMlab 125-14 readout model.

Conventions:
    - Time is expressed in ns.
    - Voltage is expressed in mV.
    - ADC codes are integer values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def adc_lsb_mV(
    adc_bits: int = 14,
    v_min_mV: float = -1000.0,
    v_max_mV: float = 1000.0,
) -> float:
    """Compute the ADC least significant bit in mV.

    The default configuration follows the TAMBO-4DAQ documentation for the
    Red Pitaya STEMlab 125-14 readout: 14-bit ADC, 125 MS/s sampling,
    8 ns sampling interval, and approximately 122 uV sensitivity.

    Args:
        adc_bits: Number of ADC bits.
        v_min_mV: Minimum input voltage in mV.
        v_max_mV: Maximum input voltage in mV.

    Returns:
        ADC least significant bit in mV.

    Raises:
        ValueError: If the ADC configuration is invalid.
    """
    if adc_bits <= 0:
        raise ValueError("adc_bits must be positive.")
    if v_max_mV <= v_min_mV:
        raise ValueError("v_max_mV must be greater than v_min_mV.")

    return (v_max_mV - v_min_mV) / (2**adc_bits)


def voltage_to_adc_code(
    voltage_mV: NDArray[np.float64],
    adc_bits: int = 14,
    v_min_mV: float = -1000.0,
    v_max_mV: float = 1000.0,
) -> NDArray[np.int64]:
    """Convert voltage values in mV to ADC integer codes.

    Values outside the configured ADC range are clipped.

    Args:
        voltage_mV: Input voltage array in mV.
        adc_bits: Number of ADC bits.
        v_min_mV: Minimum input voltage in mV.
        v_max_mV: Maximum input voltage in mV.

    Returns:
        ADC code array as integers in [0, 2**adc_bits - 1].
    """
    lsb_mV = adc_lsb_mV(
        adc_bits=adc_bits,
        v_min_mV=v_min_mV,
        v_max_mV=v_max_mV,
    )

    max_code = 2**adc_bits - 1
    raw_code = np.rint((voltage_mV - v_min_mV) / lsb_mV)
    clipped_code = np.clip(raw_code, 0, max_code)

    return clipped_code.astype(np.int64)


def adc_code_to_voltage(
    adc_code: NDArray[np.int64],
    adc_bits: int = 14,
    v_min_mV: float = -1000.0,
    v_max_mV: float = 1000.0,
) -> NDArray[np.float64]:
    """Convert ADC integer codes back to quantized voltages in mV.

    Args:
        adc_code: ADC code array.
        adc_bits: Number of ADC bits.
        v_min_mV: Minimum input voltage in mV.
        v_max_mV: Maximum input voltage in mV.

    Returns:
        Quantized voltage array in mV.

    Raises:
        ValueError: If ADC codes are outside the valid range.
    """
    max_code = 2**adc_bits - 1

    if np.any(adc_code < 0) or np.any(adc_code > max_code):
        raise ValueError("adc_code contains values outside the valid ADC range.")

    lsb_mV = adc_lsb_mV(
        adc_bits=adc_bits,
        v_min_mV=v_min_mV,
        v_max_mV=v_max_mV,
    )

    return v_min_mV + adc_code.astype(np.float64) * lsb_mV


def digitize_voltage(
    voltage_mV: NDArray[np.float64],
    adc_bits: int = 14,
    v_min_mV: float = -1000.0,
    v_max_mV: float = 1000.0,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Digitize a voltage waveform.

    Args:
        voltage_mV: Input voltage waveform in mV.
        adc_bits: Number of ADC bits.
        v_min_mV: Minimum input voltage in mV.
        v_max_mV: Maximum input voltage in mV.

    Returns:
        Tuple containing:
            - ADC integer codes.
            - Quantized voltage waveform in mV.
    """
    adc_code = voltage_to_adc_code(
        voltage_mV=voltage_mV,
        adc_bits=adc_bits,
        v_min_mV=v_min_mV,
        v_max_mV=v_max_mV,
    )

    quantized_voltage_mV = adc_code_to_voltage(
        adc_code=adc_code,
        adc_bits=adc_bits,
        v_min_mV=v_min_mV,
        v_max_mV=v_max_mV,
    )

    return adc_code, quantized_voltage_mV


def sample_waveform(
    time_ns: NDArray[np.float64],
    voltage_mV: NDArray[np.float64],
    sampling_interval_ns: float = 8.0,
    start_time_ns: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample a waveform at a fixed sampling interval.

    The Red Pitaya STEMlab 125-14 used in TAMBO samples at 125 MS/s,
    corresponding to 8 ns per sample. This function uses linear
    interpolation to evaluate a high-resolution waveform on the readout grid.

    Args:
        time_ns: Original time array in ns.
        voltage_mV: Original voltage waveform in mV.
        sampling_interval_ns: Sampling interval in ns.
        start_time_ns: First sampled time. If None, uses time_ns[0].

    Returns:
        Tuple containing:
            - Sampled time array in ns.
            - Sampled voltage array in mV.

    Raises:
        ValueError: If inputs are invalid.
    """
    if time_ns.shape != voltage_mV.shape:
        raise ValueError("time_ns and voltage_mV must have the same shape.")
    if len(time_ns) < 2:
        raise ValueError("time_ns must contain at least two samples.")
    if sampling_interval_ns <= 0:
        raise ValueError("sampling_interval_ns must be positive.")
    if not np.all(np.diff(time_ns) > 0):
        raise ValueError("time_ns must be strictly increasing.")

    if start_time_ns is None:
        start_time_ns = float(time_ns[0])

    if start_time_ns < time_ns[0] or start_time_ns > time_ns[-1]:
        raise ValueError("start_time_ns must lie inside the input time range.")

    sampled_time_ns = np.arange(
        start_time_ns,
        float(time_ns[-1]) + 1e-12,
        sampling_interval_ns,
        dtype=np.float64,
    )

    sampled_voltage_mV = np.interp(sampled_time_ns, time_ns, voltage_mV)

    return sampled_time_ns, sampled_voltage_mV


def sample_and_digitize_waveform(
    time_ns: NDArray[np.float64],
    voltage_mV: NDArray[np.float64],
    sampling_interval_ns: float = 8.0,
    adc_bits: int = 14,
    v_min_mV: float = -1000.0,
    v_max_mV: float = 1000.0,
    start_time_ns: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Sample and digitize a waveform.

    Args:
        time_ns: Original time array in ns.
        voltage_mV: Original voltage waveform in mV.
        sampling_interval_ns: Sampling interval in ns.
        adc_bits: Number of ADC bits.
        v_min_mV: Minimum input voltage in mV.
        v_max_mV: Maximum input voltage in mV.
        start_time_ns: First sampled time. If None, uses time_ns[0].

    Returns:
        Tuple containing:
            - Sampled time array in ns.
            - ADC integer codes.
            - Quantized sampled voltage waveform in mV.
    """
    sampled_time_ns, sampled_voltage_mV = sample_waveform(
        time_ns=time_ns,
        voltage_mV=voltage_mV,
        sampling_interval_ns=sampling_interval_ns,
        start_time_ns=start_time_ns,
    )

    adc_code, quantized_voltage_mV = digitize_voltage(
        voltage_mV=sampled_voltage_mV,
        adc_bits=adc_bits,
        v_min_mV=v_min_mV,
        v_max_mV=v_max_mV,
    )

    return sampled_time_ns, adc_code, quantized_voltage_mV


def has_adc_saturation(
    adc_code: NDArray[np.int64],
    adc_bits: int = 14,
) -> bool:
    """Check whether an ADC waveform contains saturated samples.

    Args:
        adc_code: ADC code array.
        adc_bits: Number of ADC bits.

    Returns:
        True if the waveform contains at least one minimum-code or maximum-code
        sample.
    """
    max_code = 2**adc_bits - 1

    return bool(np.any((adc_code == 0) | (adc_code == max_code)))