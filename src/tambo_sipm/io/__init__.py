"""I/O utilities for TAMBO simulation data."""

from tambo_sipm.io.hep_pulses import (
    assign_pulse_ids_from_time_gaps,
    load_hep_timeseries,
    load_segmented_hep_pulses,
    pulse_collection_to_dict,
    pulse_lengths,
    segment_hep_pulses,
)
from tambo_sipm.io.inventory import (
    FileInventoryRecord,
    build_data_inventory,
    inspect_csv_file,
    inspect_generic_file,
    list_files,
    save_inventory,
)
from tambo_sipm.io.loaders import (
    load_csv_table,
    load_photon_counts_csv,
    load_single_waveform_csv,
    load_waveform_collection_csv,
    require_columns,
)
from tambo_sipm.io.photon_events import (
    add_particle_metadata,
    build_photon_events,
    build_row_level_photon_events,
    detector_summary,
    filter_detector,
    load_romero_photon_table,
    particle_category_from_corsika_id,
    particle_name_from_corsika_id,
    particle_summary,
    select_detector_with_most_rows,
)

__all__ = [
    "FileInventoryRecord",
    "add_particle_metadata",
    "assign_pulse_ids_from_time_gaps",
    "build_data_inventory",
    "build_photon_events",
    "build_row_level_photon_events",
    "detector_summary",
    "filter_detector",
    "inspect_csv_file",
    "inspect_generic_file",
    "list_files",
    "load_csv_table",
    "load_hep_timeseries",
    "load_photon_counts_csv",
    "load_romero_photon_table",
    "load_segmented_hep_pulses",
    "load_single_waveform_csv",
    "load_waveform_collection_csv",
    "particle_category_from_corsika_id",
    "particle_name_from_corsika_id",
    "particle_summary",
    "pulse_collection_to_dict",
    "pulse_lengths",
    "require_columns",
    "save_inventory",
    "segment_hep_pulses",
    "select_detector_with_most_rows",
]