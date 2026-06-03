"""Tests for configuration loading utilities."""

from pathlib import Path

import pytest

from tambo_sipm.config import (
    config_path,
    load_readout_config,
    load_simulation_config,
    load_sipm_config,
    load_yaml_config,
    project_root,
)


def test_project_root_exists():
    root = project_root()

    assert root.exists()
    assert (root / "pyproject.toml").exists()


def test_config_path_points_inside_configs():
    path = config_path("sipm", "microfc_60035_smt_cseries.yaml")

    assert "configs" in path.parts
    assert path.name == "microfc_60035_smt_cseries.yaml"


def test_load_yaml_config_rejects_missing_file():
    missing_path = Path("missing_config.yaml")

    with pytest.raises(FileNotFoundError):
        load_yaml_config(missing_path)


def test_load_sipm_config():
    config = load_sipm_config()

    assert config["name"] == "onsemi MicroFC-60035-SMT C-Series"
    assert config["n_microcells"] == 18980
    assert config["pde"]["vbr_plus_5p0"] == 0.41


def test_load_readout_config():
    config = load_readout_config()

    assert config["name"] == "Red Pitaya STEMlab 125-14"
    assert config["adc_bits"] == 14
    assert config["sampling_interval_ns"] == 8.0
    assert config["pretrigger_samples"] == 3


def test_load_simulation_config():
    config = load_simulation_config()

    assert config["name"] == "Default TAMBO detector event simulation"
    assert "photon_transport" in config
    assert "sipm_response" in config
    assert "readout" in config