"""Detector-level models for the TAMBO simulation framework."""

from tambo_sipm.detector.event import (
    DetectorEventResult,
    convert_pe_to_mV,
    simulate_detector_event,
    simulate_detector_event_from_config,
    validate_generated_photons,
    validate_voltage_scale,
)
from tambo_sipm.detector.photon_transport import (
    PhotonTransportResult,
    detection_probability_at_least_one,
    expected_photoelectrons,
    expected_photons_at_sipm,
    sample_photoelectrons,
    sample_photons_at_sipm,
    simulate_photon_transport,
    validate_photon_counts,
    validate_probability,
)

__all__ = [
    "DetectorEventResult",
    "PhotonTransportResult",
    "convert_pe_to_mV",
    "detection_probability_at_least_one",
    "expected_photoelectrons",
    "expected_photons_at_sipm",
    "sample_photoelectrons",
    "sample_photons_at_sipm",
    "simulate_detector_event",
    "simulate_photon_transport",
    "validate_generated_photons",
    "validate_photon_counts",
    "validate_probability",
    "validate_voltage_scale",
    "simulate_detector_event_from_config",
]