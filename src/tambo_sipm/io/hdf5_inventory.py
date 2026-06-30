"""HDF5 inspection utilities for exploratory raw-data inventories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
from typing import Any

import h5py
import numpy as np
import pandas as pd


_RUN_NAME_PATTERN = re.compile(
    r"^(?P<run>\d+)(?P<panel>[AB])_"
    r"(?P<date>\d{6})_"
    r"(?P<duration>\d+h(?:\d+m)?)_"
    r"(?P<gain>AMP\d+)"
    r"(?:_(?P<config>C\d+))?"
    r"_(?P<threshold>thr.+)$"
)


@dataclass(frozen=True)
class HDF5InventoryRecord:
    """Container for one HDF5 file object inventory row."""

    file_name: str
    relative_file_path: str
    file_size_bytes: int
    run_index: str | None
    panel: str | None
    acquisition_date: str | None
    duration_label: str | None
    duration_minutes: int | None
    gain: str | None
    run_config: str | None
    falling_edge_adc_threshold: int | None
    rising_edge_adc_threshold: int | None
    object_path: str
    object_name: str
    object_type: str
    parent_path: str
    shape: str | None
    ndim: int | None
    dtype: str | None
    size: int | None
    chunks: str | None
    compression: str | None
    n_attrs: int
    attributes: str
    sample_values: str | None
    error: str | None


def parse_test_panel_filename(file_path: str | Path) -> dict[str, Any]:
    """Parse run metadata encoded in Test_Panel HDF5 file names.

    The convention is documented in ``data/raw/Test_Panel/Readme.docx``:
    panel letter, date as DDMMYY, run duration, amplifier gain, optional
    configuration token, and ADC threshold values.
    """
    path = Path(file_path)
    match = _RUN_NAME_PATTERN.match(path.stem)
    if match is None:
        return {
            "run_index": None,
            "panel": None,
            "acquisition_date": None,
            "duration_label": None,
            "duration_minutes": None,
            "gain": None,
            "run_config": None,
            "falling_edge_adc_threshold": None,
            "rising_edge_adc_threshold": None,
        }

    groups = match.groupdict()
    threshold_values = [int(value) for value in re.findall(r"\d+", groups["threshold"])]
    falling_threshold = threshold_values[0] if threshold_values else None
    rising_threshold = (
        threshold_values[1]
        if len(threshold_values) > 1
        else falling_threshold
    )

    date_token = groups["date"]
    acquisition_date = (
        f"20{date_token[4:6]}-{date_token[2:4]}-{date_token[0:2]}"
    )

    duration_label = groups["duration"]
    duration_match = re.match(r"(?P<hours>\d+)h(?:(?P<minutes>\d+)m)?", duration_label)
    duration_minutes = None
    if duration_match is not None:
        hours = int(duration_match.group("hours"))
        minutes = int(duration_match.group("minutes") or 0)
        duration_minutes = hours * 60 + minutes

    return {
        "run_index": groups["run"],
        "panel": groups["panel"],
        "acquisition_date": acquisition_date,
        "duration_label": duration_label,
        "duration_minutes": duration_minutes,
        "gain": groups["gain"],
        "run_config": groups["config"],
        "falling_edge_adc_threshold": falling_threshold,
        "rising_edge_adc_threshold": rising_threshold,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _to_json_text(value: Any) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=True, sort_keys=True)


def _markdown_table(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "_No rows._"

    string_frame = dataframe.fillna("").astype(str)
    headers = list(string_frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in string_frame.itertuples(index=False, name=None):
        escaped = [value.replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def _format_shape(shape: tuple[int, ...] | None) -> str | None:
    if shape is None:
        return None
    return "scalar" if shape == () else "x".join(str(dim) for dim in shape)


def _safe_dataset_sample(dataset: h5py.Dataset, max_items: int = 10) -> str | None:
    if dataset.shape is None:
        return None

    try:
        if dataset.shape == ():
            sample = dataset[()]
        else:
            selection = tuple(slice(0, min(5, dim)) for dim in dataset.shape)
            sample = dataset[selection]

        array = np.asarray(sample)
        if array.shape == ():
            values: Any = array.item()
        else:
            values = array.reshape(-1)[:max_items].tolist()
        return _to_json_text(values)
    except Exception as exc:
        return _to_json_text({"sample_error": f"{type(exc).__name__}: {exc}"})


def _object_record(
    file_path: Path,
    root_dir: Path,
    object_path: str,
    hdf5_object: h5py.Group | h5py.Dataset,
    metadata: dict[str, Any],
) -> HDF5InventoryRecord:
    attrs = {str(key): value for key, value in hdf5_object.attrs.items()}
    object_name = "/" if object_path == "/" else Path(object_path).name
    parent_path = "" if object_path == "/" else str(Path(object_path).parent)
    if parent_path == ".":
        parent_path = "/"

    if isinstance(hdf5_object, h5py.Dataset):
        object_type = "dataset"
        shape = _format_shape(hdf5_object.shape)
        ndim = int(hdf5_object.ndim)
        dtype = str(hdf5_object.dtype)
        size = int(hdf5_object.size)
        chunks = _format_shape(hdf5_object.chunks)
        compression = str(hdf5_object.compression) if hdf5_object.compression else None
        sample_values = _safe_dataset_sample(hdf5_object)
    else:
        object_type = "group"
        shape = None
        ndim = None
        dtype = None
        size = None
        chunks = None
        compression = None
        sample_values = None

    return HDF5InventoryRecord(
        file_name=file_path.name,
        relative_file_path=str(file_path.relative_to(root_dir)),
        file_size_bytes=file_path.stat().st_size,
        object_path=object_path,
        object_name=object_name,
        object_type=object_type,
        parent_path=parent_path,
        shape=shape,
        ndim=ndim,
        dtype=dtype,
        size=size,
        chunks=chunks,
        compression=compression,
        n_attrs=len(attrs),
        attributes=_to_json_text(attrs),
        sample_values=sample_values,
        error=None,
        **metadata,
    )


def inspect_hdf5_file(
    file_path: str | Path,
    root_dir: str | Path,
) -> list[HDF5InventoryRecord]:
    """Recursively inspect groups, datasets, attributes, and safe samples."""
    path = Path(file_path)
    root_path = Path(root_dir)
    metadata = parse_test_panel_filename(path)

    records: list[HDF5InventoryRecord] = []
    try:
        with h5py.File(path, "r") as hdf5_file:
            records.append(_object_record(path, root_path, "/", hdf5_file, metadata))

            def visitor(name: str, hdf5_object: h5py.Group | h5py.Dataset) -> None:
                records.append(
                    _object_record(
                        path,
                        root_path,
                        f"/{name}",
                        hdf5_object,
                        metadata,
                    )
                )

            hdf5_file.visititems(visitor)
    except Exception as exc:
        records.append(
            HDF5InventoryRecord(
                file_name=path.name,
                relative_file_path=str(path.relative_to(root_path)),
                file_size_bytes=path.stat().st_size,
                object_path="/",
                object_name="/",
                object_type="file_error",
                parent_path="",
                shape=None,
                ndim=None,
                dtype=None,
                size=None,
                chunks=None,
                compression=None,
                n_attrs=0,
                attributes="{}",
                sample_values=None,
                error=f"{type(exc).__name__}: {exc}",
                **metadata,
            )
        )

    return records


def build_hdf5_inventory(
    root_dir: str | Path,
    pattern: str = "*.h5",
) -> pd.DataFrame:
    """Build a recursive HDF5 inventory table for files under ``root_dir``."""
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_path}")

    records: list[HDF5InventoryRecord] = []
    for path in sorted(root_path.glob(pattern)):
        if path.is_file():
            records.extend(inspect_hdf5_file(path, root_path))

    return pd.DataFrame([asdict(record) for record in records])


def save_hdf5_inventory(inventory: pd.DataFrame, output_path: str | Path) -> None:
    """Save an HDF5 inventory table as CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(path, index=False)


def write_hdf5_summary(
    inventory: pd.DataFrame,
    output_path: str | Path,
    readme_notes: str | None = None,
) -> None:
    """Write a compact Markdown summary for a generated HDF5 inventory."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_rows = (
        inventory[
            [
                "file_name",
                "panel",
                "acquisition_date",
                "duration_label",
                "duration_minutes",
                "gain",
                "run_config",
                "falling_edge_adc_threshold",
                "rising_edge_adc_threshold",
                "file_size_bytes",
            ]
        ]
        .drop_duplicates()
        .sort_values(["panel", "acquisition_date", "file_name"])
    )
    dataset_rows = inventory[inventory["object_type"] == "dataset"].copy()

    lines = [
        "# Test Panel HDF5 Inventory Summary",
        "",
        "## README notes",
        "",
    ]
    if readme_notes:
        lines.extend(readme_notes.strip().splitlines())
    else:
        lines.append("No README notes were provided to the summary writer.")

    lines.extend(["", "## Files", ""])
    lines.append(_markdown_table(file_rows))

    lines.extend(["", "## Dataset structure", ""])
    if dataset_rows.empty:
        lines.append("No datasets were found.")
    else:
        dataset_summary = dataset_rows[
            [
                "file_name",
                "object_path",
                "shape",
                "dtype",
                "n_attrs",
                "sample_values",
            ]
        ].sort_values(["file_name", "object_path"])
        lines.append(_markdown_table(dataset_summary))

    lines.extend(["", "## Object counts", ""])
    counts = (
        inventory.groupby(["file_name", "object_type"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["file_name", "object_type"])
    )
    lines.append(_markdown_table(counts))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
