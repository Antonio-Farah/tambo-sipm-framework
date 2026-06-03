"""SiPM models for the TAMBO simulation framework."""

from tambo_sipm.sipm.pulse import (
    estimate_fwhm_ns,
    make_time_axis,
    normalized_sipm_pulse,
    raw_sipm_pulse,
    sum_pulses,
)
from tambo_sipm.sipm.saturation import (
    MICROFC_60035_C_SERIES_MICROCELLS,
    estimate_photoelectrons_from_fired_microcells,
    expected_fired_microcells,
    sample_fired_microcells,
    saturation_factor,
    validate_microcell_count,
    validate_non_negative_integer_counts,
)

__all__ = [
    "MICROFC_60035_C_SERIES_MICROCELLS",
    "estimate_fwhm_ns",
    "estimate_photoelectrons_from_fired_microcells",
    "expected_fired_microcells",
    "make_time_axis",
    "normalized_sipm_pulse",
    "raw_sipm_pulse",
    "sample_fired_microcells",
    "saturation_factor",
    "sum_pulses",
    "validate_microcell_count",
    "validate_non_negative_integer_counts",
]