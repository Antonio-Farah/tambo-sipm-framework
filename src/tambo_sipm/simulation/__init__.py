"""Simulation utilities for TAMBO detector studies."""

from tambo_sipm.simulation.batch import (
    require_event_columns,
    select_photon_events,
    simulate_feature_row_from_photon_event,
    simulate_feature_table_from_photon_events,
)
from tambo_sipm.simulation.calibration import (
    DEFAULT_CALIBRATION_FEATURES,
    DEFAULT_CALIBRATION_WEIGHTS,
    build_calibration_config,
    calibration_grid_rows,
    calibration_score_from_distribution_table,
    feature_error_from_distribution_table,
    run_calibration_grid,
    run_single_calibration_point,
    set_nested_config_value,
    validate_calibration_grid_values,
)

__all__ = [
    "DEFAULT_CALIBRATION_FEATURES",
    "DEFAULT_CALIBRATION_WEIGHTS",
    "build_calibration_config",
    "calibration_grid_rows",
    "calibration_score_from_distribution_table",
    "feature_error_from_distribution_table",
    "require_event_columns",
    "run_calibration_grid",
    "run_single_calibration_point",
    "select_photon_events",
    "set_nested_config_value",
    "simulate_feature_row_from_photon_event",
    "simulate_feature_table_from_photon_events",
    "validate_calibration_grid_values",
]