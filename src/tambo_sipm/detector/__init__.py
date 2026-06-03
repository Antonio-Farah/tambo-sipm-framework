"""Detector-level models for the TAMBO simulation framework."""

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
    "PhotonTransportResult",
    "detection_probability_at_least_one",
    "expected_photoelectrons",
    "expected_photons_at_sipm",
    "sample_photoelectrons",
    "sample_photons_at_sipm",
    "simulate_photon_transport",
    "validate_photon_counts",
    "validate_probability",
]