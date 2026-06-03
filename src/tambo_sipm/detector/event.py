"""Detector event simulation chain.

This module connects the detector-level photon transport model with the SiPM
response model and the Red Pitaya digitization model.

Conventions:
    - Time is expressed in ns.
    - SiPM signal amplitude is expressed in p.e.
    - Voltage is expressed in mV.
    - The voltage scale in mV/p.e. must be calibrated from data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tambo_sipm.detector.photon_transport import (
    PhotonTransportResult,
    simulate_photon_transport,
)
from tambo_sipm.readout.adc import sample_and_digitize_waveform
from tambo_sipm.sipm.response import SiPMResponseResult, simulate_sipm_response


@dataclass(frozen=True)
class DetectorEventResult:
    """Container for a complete detector event simulation."""

    photon_transport: PhotonTransportResult
    sipm_response: SiPMResponseResult
    analog_voltage_mV: NDArray[np.float64]
    sampled_time_ns: NDArray[np.float64]
    adc_code: NDArray[np.int64]
    digitized_voltage_mV: NDArray[np.float64]
    voltage_scale_mV_per_pe: float


def validate_generated_photons(generated_photons: int) -> None:
    """Validate the number of generated scintillation photons.

    Args:
        generated_photons: Number of generated scintillation photons.

    Raises:
        ValueError: If generated_photons is negative or non-integer.
    """
    if not isinstance(generated_photons, (int, np.integer)):
        raise ValueError("generated_photons must be an integer.")
    if generated_photons < 0:
        raise ValueError("generated_photons must be non-negative.")


def validate_voltage_scale(voltage_scale_mV_per_pe: float) -> None:
    """Validate the p.e. to voltage conversion scale.

    Args:
        voltage_scale_mV_per_pe: Conversion factor from p.e. to mV.

    Raises:
        ValueError: If the conversion scale is negative.
    """
    if voltage_scale_mV_per_pe < 0.0:
        raise ValueError("voltage_scale_mV_per_pe must be non-negative.")


def convert_pe_to_mV(
    signal_pe: NDArray[np.float64],
    voltage_scale_mV_per_pe: float,
) -> NDArray[np.float64]:
    """Convert a signal from p.e. to mV.

    Args:
        signal_pe: Signal waveform in p.e.
        voltage_scale_mV_per_pe: Conversion factor from p.e. to mV.

    Returns:
        Signal waveform in mV.
    """
    validate_voltage_scale(voltage_scale_mV_per_pe)

    return signal_pe.astype(np.float64) * voltage_scale_mV_per_pe


def simulate_detector_event(
    generated_photons: int,
    transport_efficiency: float,
    pde: float,
    voltage_scale_mV_per_pe: float,
    window_ns: float,
    dt_ns: float,
    tau_r_ns: float,
    tau_f_ns: float,
    event_time_ns: float = 0.0,
    arrival_spread_ns: float = 0.0,
    sampling_interval_ns: float = 8.0,
    adc_bits: int = 14,
    v_min_mV: float = -1000.0,
    v_max_mV: float = 1000.0,
    gain_mean_pe: float = 1.0,
    gain_sigma_pe: float = 0.1,
    crosstalk_prob: float = 0.0,
    include_dark_noise: bool = False,
    dcr_hz: float = 0.0,
    afterpulse_prob: float = 0.0,
    tau_afterpulse_ns: float = 30.0,
    tau_recovery_ns: float = 95.0,
    rng: np.random.Generator | None = None,
) -> DetectorEventResult:
    """Simulate a complete detector event.

    Args:
        generated_photons: Number of scintillation photons generated in the bar.
        transport_efficiency: Effective probability that a generated photon
            reaches the SiPM active area.
        pde: Photon detection efficiency of the SiPM.
        voltage_scale_mV_per_pe: Conversion factor from p.e. to mV.
        window_ns: High-resolution simulation window in ns.
        dt_ns: High-resolution simulation time step in ns.
        tau_r_ns: SiPM pulse rise-time constant in ns.
        tau_f_ns: SiPM pulse fall-time constant in ns.
        event_time_ns: Mean event time in ns.
        arrival_spread_ns: Standard deviation of signal arrival times in ns.
        sampling_interval_ns: Readout sampling interval in ns.
        adc_bits: Number of ADC bits.
        v_min_mV: Minimum ADC input voltage in mV.
        v_max_mV: Maximum ADC input voltage in mV.
        gain_mean_pe: Mean single-cell amplitude in p.e.
        gain_sigma_pe: Standard deviation of single-cell amplitude in p.e.
        crosstalk_prob: Crosstalk probability.
        include_dark_noise: If True, includes DCR and correlated dark noise.
        dcr_hz: Dark Count Rate in Hz.
        afterpulse_prob: Afterpulsing probability.
        tau_afterpulse_ns: Mean afterpulse delay in ns.
        tau_recovery_ns: Microcell recovery time constant in ns.
        rng: Optional NumPy random generator.

    Returns:
        DetectorEventResult containing photon transport, SiPM response,
        analog voltage, sampled times, ADC codes, and digitized voltage.
    """
    validate_generated_photons(generated_photons)
    validate_voltage_scale(voltage_scale_mV_per_pe)

    if rng is None:
        rng = np.random.default_rng()

    photon_transport = simulate_photon_transport(
        generated_photons=np.array([generated_photons], dtype=np.int64),
        transport_efficiency=transport_efficiency,
        pde=pde,
        rng=rng,
    )

    photoelectrons = int(photon_transport.photoelectrons[0])

    sipm_response = simulate_sipm_response(
        photoelectrons=photoelectrons,
        window_ns=window_ns,
        dt_ns=dt_ns,
        tau_r_ns=tau_r_ns,
        tau_f_ns=tau_f_ns,
        event_time_ns=event_time_ns,
        arrival_spread_ns=arrival_spread_ns,
        gain_mean_pe=gain_mean_pe,
        gain_sigma_pe=gain_sigma_pe,
        crosstalk_prob=crosstalk_prob,
        include_dark_noise=include_dark_noise,
        dcr_hz=dcr_hz,
        afterpulse_prob=afterpulse_prob,
        tau_afterpulse_ns=tau_afterpulse_ns,
        tau_recovery_ns=tau_recovery_ns,
        rng=rng,
    )

    analog_voltage_mV = convert_pe_to_mV(
        signal_pe=sipm_response.total_signal_pe,
        voltage_scale_mV_per_pe=voltage_scale_mV_per_pe,
    )

    sampled_time_ns, adc_code, digitized_voltage_mV = sample_and_digitize_waveform(
        time_ns=sipm_response.time_ns,
        voltage_mV=analog_voltage_mV,
        sampling_interval_ns=sampling_interval_ns,
        adc_bits=adc_bits,
        v_min_mV=v_min_mV,
        v_max_mV=v_max_mV,
    )

    return DetectorEventResult(
        photon_transport=photon_transport,
        sipm_response=sipm_response,
        analog_voltage_mV=analog_voltage_mV,
        sampled_time_ns=sampled_time_ns,
        adc_code=adc_code,
        digitized_voltage_mV=digitized_voltage_mV,
        voltage_scale_mV_per_pe=voltage_scale_mV_per_pe,
    )