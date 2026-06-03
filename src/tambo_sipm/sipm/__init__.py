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
from tambo_sipm.sipm.noise import (
    NoisePulses,
    afterpulse_amplitude_factor,
    apply_crosstalk_to_amplitudes,
    generate_afterpulses,
    generate_dark_noise,
    sample_crosstalk_multiplicity,
    sample_dcr_times,
    sample_gain_amplitudes,
)
from tambo_sipm.sipm.response import (
    SiPMResponseResult,
    sample_photoelectron_arrival_times,
    simulate_sipm_response,
    validate_photoelectron_count,
    validate_response_times,
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
    "NoisePulses",
    "afterpulse_amplitude_factor",
    "apply_crosstalk_to_amplitudes",
    "generate_afterpulses",
    "generate_dark_noise",
    "sample_crosstalk_multiplicity",
    "sample_dcr_times",
    "sample_gain_amplitudes",
    "SiPMResponseResult",
    "sample_photoelectron_arrival_times",
    "simulate_sipm_response",
    "validate_photoelectron_count",
    "validate_response_times",
]