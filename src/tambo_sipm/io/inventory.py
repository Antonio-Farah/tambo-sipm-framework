"""Data inventory utilities for TAMBO input files.

This module inspects files inside data directories and summarizes basic
metadata such as file size, extension, CSV rows, columns, and read errors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FileInventoryRecord:
    """Container for file inventory metadata."""

    path: str
    relative_path: str
    suffix: str
    size_bytes: int
    readable: bool
    n_rows: int | None
    n_columns: int | None
    columns: str | None
    error: str | None


def list_files(
    root_dir: str | Path,
    recursive: bool = True,
) -> list[Path]:
    """List files inside a directory.

    Args:
        root_dir: Directory to inspect.
        recursive: If True, search recursively.

    Returns:
        Sorted list of file paths.

    Raises:
        FileNotFoundError: If root_dir does not exist.
        NotADirectoryError: If root_dir is not a directory.
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_path}")

    pattern = "**/*" if recursive else "*"

    return sorted(path for path in root_path.glob(pattern) if path.is_file())


def inspect_csv_file(
    file_path: str | Path,
    root_dir: str | Path,
) -> FileInventoryRecord:
    """Inspect a CSV file.

    Args:
        file_path: CSV file path.
        root_dir: Root directory used to compute relative path.

    Returns:
        FileInventoryRecord.
    """
    path = Path(file_path)
    root_path = Path(root_dir)

    try:
        dataframe = pd.read_csv(path, nrows=1000)
        readable = True
        n_rows = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        n_columns = int(len(dataframe.columns))
        columns = ", ".join(str(column) for column in dataframe.columns)
        error = None
    except Exception as exc:
        readable = False
        n_rows = None
        n_columns = None
        columns = None
        error = f"{type(exc).__name__}: {exc}"

    return FileInventoryRecord(
        path=str(path),
        relative_path=str(path.relative_to(root_path)),
        suffix=path.suffix.lower(),
        size_bytes=path.stat().st_size,
        readable=readable,
        n_rows=n_rows,
        n_columns=n_columns,
        columns=columns,
        error=error,
    )


def inspect_generic_file(
    file_path: str | Path,
    root_dir: str | Path,
) -> FileInventoryRecord:
    """Inspect a non-CSV file.

    Args:
        file_path: File path.
        root_dir: Root directory used to compute relative path.

    Returns:
        FileInventoryRecord.
    """
    path = Path(file_path)
    root_path = Path(root_dir)

    return FileInventoryRecord(
        path=str(path),
        relative_path=str(path.relative_to(root_path)),
        suffix=path.suffix.lower(),
        size_bytes=path.stat().st_size,
        readable=True,
        n_rows=None,
        n_columns=None,
        columns=None,
        error=None,
    )


def build_data_inventory(
    root_dir: str | Path,
    recursive: bool = True,
) -> pd.DataFrame:
    """Build a file inventory table.

    Args:
        root_dir: Directory to inspect.
        recursive: If True, search recursively.

    Returns:
        DataFrame with one row per file.
    """
    root_path = Path(root_dir)
    records: list[FileInventoryRecord] = []

    for path in list_files(root_path, recursive=recursive):
        if path.suffix.lower() == ".csv":
            record = inspect_csv_file(path, root_path)
        else:
            record = inspect_generic_file(path, root_path)

        records.append(record)

    return pd.DataFrame([asdict(record) for record in records])


def save_inventory(
    inventory: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save an inventory table as CSV.

    Args:
        inventory: Inventory DataFrame.
        output_path: Output CSV path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(path, index=False)