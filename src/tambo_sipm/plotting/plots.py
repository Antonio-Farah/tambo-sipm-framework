"""Plotting utilities for TAMBO SiPM waveform analysis.

This module provides reusable plotting functions for measured-versus-simulated
waveform comparisons.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from tambo_sipm.analysis.metrics import common_bin_edges, empirical_cdf


def save_figure(
    figure: plt.Figure,
    output_path: str | Path,
    dpi: int = 300,
) -> None:
    """Save a Matplotlib figure.

    Args:
        figure: Matplotlib figure.
        output_path: Output file path.
        dpi: Figure resolution.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_distribution_comparison(
    reference: NDArray[np.float64] | list[float],
    candidate: NDArray[np.float64] | list[float],
    xlabel: str,
    ylabel: str = "Count",
    reference_label: str = "Real",
    candidate_label: str = "Simulated",
    n_bins: int = 50,
    density: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot a one-dimensional distribution comparison with common bins.

    Args:
        reference: Reference distribution.
        candidate: Candidate distribution.
        xlabel: x-axis label.
        ylabel: y-axis label.
        reference_label: Reference legend label.
        candidate_label: Candidate legend label.
        n_bins: Number of common bins.
        density: If True, normalize histograms as probability densities.
        ax: Optional Matplotlib axis.

    Returns:
        Matplotlib axis.
    """
    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)

    bins = common_bin_edges(reference_array, candidate_array, n_bins=n_bins)

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(
        reference_array,
        bins=bins,
        histtype="step",
        density=density,
        label=reference_label,
    )
    ax.hist(
        candidate_array,
        bins=bins,
        histtype="step",
        density=density,
        label=candidate_label,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_ecdf_comparison(
    reference: NDArray[np.float64] | list[float],
    candidate: NDArray[np.float64] | list[float],
    xlabel: str,
    ylabel: str = "ECDF",
    reference_label: str = "Real",
    candidate_label: str = "Simulated",
    ax: Axes | None = None,
) -> Axes:
    """Plot empirical cumulative distribution functions.

    Args:
        reference: Reference distribution.
        candidate: Candidate distribution.
        xlabel: x-axis label.
        ylabel: y-axis label.
        reference_label: Reference legend label.
        candidate_label: Candidate legend label.
        ax: Optional Matplotlib axis.

    Returns:
        Matplotlib axis.
    """
    reference_x, reference_y = empirical_cdf(reference)
    candidate_x, candidate_y = empirical_cdf(candidate)

    if ax is None:
        _, ax = plt.subplots()

    ax.step(reference_x, reference_y, where="post", label=reference_label)
    ax.step(candidate_x, candidate_y, where="post", label=candidate_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_waveform_overlay(
    waveforms: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    xlabel: str = "Time [ns]",
    ylabel: str = "Voltage [mV]",
    label: str | None = None,
    alpha: float = 0.25,
    ax: Axes | None = None,
) -> Axes:
    """Plot an overlay of multiple waveforms.

    Args:
        waveforms: List of (time_ns, voltage_mV) waveform tuples.
        xlabel: x-axis label.
        ylabel: y-axis label.
        label: Optional legend label for the first waveform.
        alpha: Line transparency.
        ax: Optional Matplotlib axis.

    Returns:
        Matplotlib axis.
    """
    if ax is None:
        _, ax = plt.subplots()

    for index, (time_ns, voltage_mV) in enumerate(waveforms):
        line_label = label if index == 0 else None
        ax.plot(time_ns, voltage_mV, alpha=alpha, label=line_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if label is not None:
        ax.legend()

    return ax


def plot_mean_waveform_with_percentiles(
    time_ns: NDArray[np.float64],
    waveform_matrix: NDArray[np.float64],
    lower_percentile: float = 10.0,
    upper_percentile: float = 90.0,
    xlabel: str = "Time [ns]",
    ylabel: str = "Voltage [mV]",
    label: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot mean waveform with percentile band.

    Args:
        time_ns: Common time grid.
        waveform_matrix: Matrix with shape (n_waveforms, n_samples).
        lower_percentile: Lower percentile for the band.
        upper_percentile: Upper percentile for the band.
        xlabel: x-axis label.
        ylabel: y-axis label.
        label: Mean waveform label.
        ax: Optional Matplotlib axis.

    Returns:
        Matplotlib axis.

    Raises:
        ValueError: If waveform_matrix has invalid shape.
    """
    if waveform_matrix.ndim != 2:
        raise ValueError("waveform_matrix must be two-dimensional.")
    if waveform_matrix.shape[1] != len(time_ns):
        raise ValueError("waveform_matrix columns must match time_ns length.")

    mean_waveform = np.mean(waveform_matrix, axis=0)
    lower_band = np.percentile(waveform_matrix, lower_percentile, axis=0)
    upper_band = np.percentile(waveform_matrix, upper_percentile, axis=0)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(time_ns, mean_waveform, label=label)
    ax.fill_between(time_ns, lower_band, upper_band, alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if label is not None:
        ax.legend()

    return ax


def plot_pointwise_voltage_distribution(
    reference_waveform_matrix: NDArray[np.float64],
    candidate_waveform_matrix: NDArray[np.float64],
    xlabel: str = "Voltage [mV]",
    ylabel: str = "Count",
    reference_label: str = "Real",
    candidate_label: str = "Simulated",
    n_bins: int = 80,
    ax: Axes | None = None,
) -> Axes:
    """Plot point-wise voltage distribution from waveform matrices.

    Args:
        reference_waveform_matrix: Reference waveform matrix.
        candidate_waveform_matrix: Candidate waveform matrix.
        xlabel: x-axis label.
        ylabel: y-axis label.
        reference_label: Reference legend label.
        candidate_label: Candidate legend label.
        n_bins: Number of common bins.
        ax: Optional Matplotlib axis.

    Returns:
        Matplotlib axis.
    """
    reference_values = np.ravel(reference_waveform_matrix)
    candidate_values = np.ravel(candidate_waveform_matrix)

    return plot_distribution_comparison(
        reference=reference_values,
        candidate=candidate_values,
        xlabel=xlabel,
        ylabel=ylabel,
        reference_label=reference_label,
        candidate_label=candidate_label,
        n_bins=n_bins,
        ax=ax,
    )


def plot_xy_histogram(
    x_values: NDArray[np.float64] | list[float],
    y_values: NDArray[np.float64] | list[float],
    xlabel: str,
    ylabel: str,
    bins: int = 50,
    ax: Axes | None = None,
) -> Axes:
    """Plot a two-dimensional histogram.

    Args:
        x_values: x values.
        y_values: y values.
        xlabel: x-axis label.
        ylabel: y-axis label.
        bins: Number of bins in each dimension.
        ax: Optional Matplotlib axis.

    Returns:
        Matplotlib axis.
    """
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)

    if ax is None:
        _, ax = plt.subplots()

    image = ax.hist2d(x_array, y_array, bins=bins)
    plt.colorbar(image[3], ax=ax, label="Count")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)

    return ax


def plot_xy_scatter(
    x_values: NDArray[np.float64] | list[float],
    y_values: NDArray[np.float64] | list[float],
    xlabel: str,
    ylabel: str,
    label: str | None = None,
    alpha: float = 0.6,
    ax: Axes | None = None,
) -> Axes:
    """Plot a two-dimensional scatter plot.

    Args:
        x_values: x values.
        y_values: y values.
        xlabel: x-axis label.
        ylabel: y-axis label.
        label: Optional legend label.
        alpha: Marker transparency.
        ax: Optional Matplotlib axis.

    Returns:
        Matplotlib axis.
    """
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(x_array, y_array, alpha=alpha, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if label is not None:
        ax.legend()

    return ax