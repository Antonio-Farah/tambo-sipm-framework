"""Configuration loading utilities.

This module provides YAML loading tools for the TAMBO SiPM simulation
framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to a YAML file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML file is empty or does not contain a dictionary.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a dictionary at top level.")

    return config


def project_root() -> Path:
    """Return the project root directory.

    This assumes the package is installed in editable mode from:

        project_root/src/tambo_sipm/config.py

    Returns:
        Project root path.
    """
    return Path(__file__).resolve().parents[2]


def config_path(*parts: str) -> Path:
    """Build a path inside the configs directory.

    Args:
        *parts: Path components inside the configs directory.

    Returns:
        Full path to the requested config file.
    """
    return project_root() / "configs" / Path(*parts)


def load_sipm_config(name: str = "microfc_60035_smt_cseries.yaml") -> dict[str, Any]:
    """Load a SiPM configuration file.

    Args:
        name: File name inside configs/sipm.

    Returns:
        SiPM configuration dictionary.
    """
    return load_yaml_config(config_path("sipm", name))


def load_readout_config(name: str = "red_pitaya_125_14.yaml") -> dict[str, Any]:
    """Load a readout configuration file.

    Args:
        name: File name inside configs/readout.

    Returns:
        Readout configuration dictionary.
    """
    return load_yaml_config(config_path("readout", name))


def load_simulation_config(
    name: str = "default_detector_event.yaml",
) -> dict[str, Any]:
    """Load a detector-event simulation configuration file.

    Args:
        name: File name inside configs/simulation.

    Returns:
        Simulation configuration dictionary.
    """
    return load_yaml_config(config_path("simulation", name))