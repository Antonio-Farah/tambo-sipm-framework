"""Tests for plotting utilities."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tambo_sipm.plotting import (
    plot_distribution_comparison,
    plot_ecdf_comparison,
    plot_mean_waveform_with_percentiles,
    plot_pointwise_voltage_distribution,
    plot_waveform_overlay,
    plot_xy_histogram,
    plot_xy_scatter,
    save_figure,
)


def test_plot_distribution_comparison_returns_axis():
    ax = plot_distribution_comparison(
        reference=[1.0, 2.0, 3.0],
        candidate=[2.0, 3.0, 4.0],
        xlabel="Peak [mV]",
    )

    assert ax.get_xlabel() == "Peak [mV]"
    plt.close(ax.figure)


def test_plot_ecdf_comparison_returns_axis():
    ax = plot_ecdf_comparison(
        reference=[1.0, 2.0, 3.0],
        candidate=[2.0, 3.0, 4.0],
        xlabel="Width [ns]",
    )

    assert ax.get_ylabel() == "ECDF"
    plt.close(ax.figure)


def test_plot_waveform_overlay_returns_axis():
    time_ns = np.array([0.0, 8.0, 16.0])
    voltage_mV = np.array([0.0, 1.0, 0.0])

    ax = plot_waveform_overlay(
        waveforms=[(time_ns, voltage_mV), (time_ns, 2.0 * voltage_mV)],
        label="Real",
    )

    assert ax.get_xlabel() == "Time [ns]"
    plt.close(ax.figure)


def test_plot_mean_waveform_with_percentiles_returns_axis():
    time_ns = np.array([0.0, 8.0, 16.0])
    waveform_matrix = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )

    ax = plot_mean_waveform_with_percentiles(
        time_ns=time_ns,
        waveform_matrix=waveform_matrix,
        label="Mean",
    )

    assert ax.get_xlabel() == "Time [ns]"
    plt.close(ax.figure)


def test_plot_pointwise_voltage_distribution_returns_axis():
    reference_matrix = np.array([[0.0, 1.0], [2.0, 3.0]])
    candidate_matrix = np.array([[0.0, 2.0], [4.0, 6.0]])

    ax = plot_pointwise_voltage_distribution(
        reference_waveform_matrix=reference_matrix,
        candidate_waveform_matrix=candidate_matrix,
    )

    assert ax.get_xlabel() == "Voltage [mV]"
    plt.close(ax.figure)


def test_plot_xy_histogram_returns_axis():
    ax = plot_xy_histogram(
        x_values=[1.0, 2.0, 3.0],
        y_values=[2.0, 4.0, 6.0],
        xlabel="Peak [mV]",
        ylabel="Integral [mV ns]",
    )

    assert ax.get_ylabel() == "Integral [mV ns]"
    plt.close(ax.figure)


def test_plot_xy_scatter_returns_axis():
    ax = plot_xy_scatter(
        x_values=[1.0, 2.0, 3.0],
        y_values=[2.0, 4.0, 6.0],
        xlabel="Width [ns]",
        ylabel="Integral [mV ns]",
    )

    assert ax.get_xlabel() == "Width [ns]"
    plt.close(ax.figure)


def test_save_figure_creates_file(tmp_path):
    figure, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])

    output_path = tmp_path / "figure.png"
    save_figure(figure, output_path)

    assert output_path.exists()
    plt.close(figure)