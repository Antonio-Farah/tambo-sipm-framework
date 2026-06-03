"""Tests for data inventory utilities."""

from pathlib import Path

import pandas as pd
import pytest

from tambo_sipm.io.inventory import (
    build_data_inventory,
    inspect_csv_file,
    inspect_generic_file,
    list_files,
    save_inventory,
)


def test_list_files_returns_files(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello", encoding="utf-8")

    files = list_files(tmp_path)

    assert len(files) == 2
    assert all(path.is_file() for path in files)


def test_list_files_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_files(tmp_path / "missing")


def test_inspect_csv_file_reads_metadata(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)

    record = inspect_csv_file(csv_path, tmp_path)

    assert record.readable
    assert record.relative_path == "data.csv"
    assert record.suffix == ".csv"
    assert record.n_rows == 2
    assert record.n_columns == 2
    assert record.columns == "a, b"
    assert record.error is None


def test_inspect_generic_file(tmp_path):
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("hello", encoding="utf-8")

    record = inspect_generic_file(txt_path, tmp_path)

    assert record.readable
    assert record.relative_path == "notes.txt"
    assert record.suffix == ".txt"
    assert record.n_rows is None
    assert record.n_columns is None


def test_build_data_inventory(tmp_path):
    pd.DataFrame({"a": [1, 2]}).to_csv(tmp_path / "data.csv", index=False)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")

    inventory = build_data_inventory(tmp_path)

    assert len(inventory) == 2
    assert "relative_path" in inventory.columns
    assert "size_bytes" in inventory.columns


def test_save_inventory_creates_csv(tmp_path):
    inventory = pd.DataFrame(
        {
            "relative_path": ["data.csv"],
            "size_bytes": [10],
        }
    )

    output_path = tmp_path / "outputs" / "inventory.csv"
    save_inventory(inventory, output_path)

    assert output_path.exists()