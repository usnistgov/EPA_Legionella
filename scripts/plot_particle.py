#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Analysis Plotting Functions
=====================================

This module provides specialized plotting functions for particle decay and
emission analysis in the EPA Legionella project.

Key Functions:
    - plot_particle_decay_event: Individual event decay curves per bin
    - plot_penetration_summary: Bar chart of penetration factors
    - plot_deposition_summary: Bar chart of deposition rates
    - plot_emission_summary: Bar chart of emission rates

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plot_style import (
    COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    LINE_WIDTH_DATA,
    SENSOR_COLORS,
    apply_style,
    create_figure,
    format_datetime_axis,
    save_figure,
)


def plot_particle_decay_event(
    particle_data: pd.DataFrame,
    event: Dict,
    bin_num: int,
    bin_info: Dict,
    result: Dict,
    output_path: Path,
    event_number: int,
) -> None:
    """
    Plot particle concentration decay for a single event and bin.

    Shows inside and outside concentrations with marked analysis windows.

    Parameters:
        particle_data: DataFrame with particle concentrations
        event: Event timing dictionary
        bin_num: Particle bin number (0-6)
        bin_info: Bin information dictionary
        result: Analysis results for this event
        output_path: Path to save the figure
        event_number: Event number for title
    """
    apply_style()

    col_inside = f"{bin_info['column']}_inside"
    col_outside = f"{bin_info['column']}_outside"

    # Extract data for plotting window (2 hours before shower to 3 hours after)
    plot_start = event["shower_on"] - timedelta(hours=2)
    plot_end = event["deposition_end"] + timedelta(hours=1)

    mask = (particle_data["datetime"] >= plot_start) & (
        particle_data["datetime"] <= plot_end
    )
    plot_data = particle_data[mask].copy()

    if plot_data.empty:
        print(f"    Warning: No data for event {event_number}, bin {bin_num}")
        return

    fig, ax = create_figure(figsize=(10, 6))

    # Plot concentrations
    ax.plot(
        plot_data["datetime"],
        plot_data[col_inside],
        label="Inside (Bedroom)",
        color=COLORS["bedroom"],
        linewidth=LINE_WIDTH_DATA,
    )
    ax.plot(
        plot_data["datetime"],
        plot_data[col_outside],
        label="Outside",
        color=COLORS["outside"],
        linewidth=LINE_WIDTH_DATA,
        alpha=0.7,
    )

    # Add vertical markers for key times
    ax.axvline(
        event["penetration_start"],
        color=COLORS["grid"],
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Penetration window",
    )
    ax.axvline(
        event["shower_on"],
        color=COLORS["shower_on"],
        linestyle="--",
        linewidth=1.5,
        label="Shower ON",
    )
    ax.axvline(
        event["shower_off"],
        color=COLORS["shower_off"],
        linestyle="--",
        linewidth=1.5,
        label="Shower OFF",
    )
    ax.axvline(
        event["deposition_end"],
        color=COLORS["grid"],
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Deposition window",
    )

    # Formatting
    ax.set_xlabel("Time", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Particle Concentration (#/cm³)", fontsize=FONT_SIZE_LABEL)

    bin_name = bin_info["name"]
    title = (
        f"Event {event_number}: Particle Decay - Bin {bin_num} ({bin_name} µm)\n"
        f"{event['shower_on'].strftime('%Y-%m-%d %H:%M')}"
    )
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="bold")

    # Add results text box
    p_mean = result.get(f"bin{bin_num}_p_mean", np.nan)
    beta_mean = result.get(f"bin{bin_num}_beta_mean", np.nan)
    E_mean = result.get(f"bin{bin_num}_E_mean", np.nan)
    lambda_ach = result.get("lambda_ach", np.nan)

    textstr = f"λ = {lambda_ach:.4f} h⁻¹\n"
    textstr += f"p = {p_mean:.3f}\n" if not np.isnan(p_mean) else "p = N/A\n"
    textstr += f"β = {beta_mean:.3f} h⁻¹\n" if not np.isnan(beta_mean) else "β = N/A\n"
    textstr += f"E = {E_mean:.2e} #/min" if not np.isnan(E_mean) else "E = N/A"

    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        verticalalignment="top",
        bbox=props,
    )

    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    format_datetime_axis(ax)

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_penetration_summary(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
) -> None:
    """
    Create bar chart summarizing penetration factors across all bins.

    Parameters:
        results_df: DataFrame with analysis results
        particle_bins: Dictionary of particle bin information
        output_path: Path to save the figure
    """
    apply_style()

    bin_nums = list(particle_bins.keys())
    bin_labels = [particle_bins[i]["name"] for i in bin_nums]

    # Calculate mean and std for each bin
    p_means = []
    p_stds = []
    for bin_num in bin_nums:
        col = f"bin{bin_num}_p_mean"
        valid_values = results_df[col].dropna()
        p_means.append(valid_values.mean() if len(valid_values) > 0 else 0)
        p_stds.append(valid_values.std() if len(valid_values) > 0 else 0)

    fig, ax = create_figure(figsize=(10, 6))

    x = np.arange(len(bin_nums))
    bars = ax.bar(
        x,
        p_means,
        yerr=p_stds,
        capsize=5,
        color=SENSOR_COLORS[0],
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, p_means, p_stds)):
        height = bar.get_height()
        if mean > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + std + 0.02,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_TICK,
            )

    ax.set_xlabel("Particle Size Bin (µm)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Penetration Factor (p)", fontsize=FONT_SIZE_LABEL)
    ax.set_title(
        "Penetration Factor by Particle Size\n(Mean ± Std Dev)",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.7, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Expected range")
    ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_LEGEND)

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_deposition_summary(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
) -> None:
    """
    Create bar chart summarizing deposition rates across all bins.

    Parameters:
        results_df: DataFrame with analysis results
        particle_bins: Dictionary of particle bin information
        output_path: Path to save the figure
    """
    apply_style()

    bin_nums = list(particle_bins.keys())
    bin_labels = [particle_bins[i]["name"] for i in bin_nums]

    # Calculate mean and std for each bin
    beta_means = []
    beta_stds = []
    for bin_num in bin_nums:
        col = f"bin{bin_num}_beta_mean"
        valid_values = results_df[col].dropna()
        beta_means.append(valid_values.mean() if len(valid_values) > 0 else 0)
        beta_stds.append(valid_values.std() if len(valid_values) > 0 else 0)

    fig, ax = create_figure(figsize=(10, 6))

    x = np.arange(len(bin_nums))
    bars = ax.bar(
        x,
        beta_means,
        yerr=beta_stds,
        capsize=5,
        color=SENSOR_COLORS[1],
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, beta_means, beta_stds)):
        height = bar.get_height()
        if mean > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + std + 0.1,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_TICK,
            )

    ax.set_xlabel("Particle Size Bin (µm)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Deposition Rate β (h⁻¹)", fontsize=FONT_SIZE_LABEL)
    ax.set_title(
        "Deposition Rate by Particle Size\n(Mean ± Std Dev)",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")

    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_emission_summary(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
) -> None:
    """
    Create bar chart summarizing emission rates across all bins.

    Parameters:
        results_df: DataFrame with analysis results
        particle_bins: Dictionary of particle bin information
        output_path: Path to save the figure
    """
    apply_style()

    bin_nums = list(particle_bins.keys())
    bin_labels = [particle_bins[i]["name"] for i in bin_nums]

    # Calculate mean and std for each bin
    E_means = []
    E_stds = []
    for bin_num in bin_nums:
        col = f"bin{bin_num}_E_mean"
        valid_values = results_df[col].dropna()
        E_means.append(valid_values.mean() if len(valid_values) > 0 else 0)
        E_stds.append(valid_values.std() if len(valid_values) > 0 else 0)

    fig, ax = create_figure(figsize=(10, 6))

    x = np.arange(len(bin_nums))
    bars = ax.bar(
        x,
        E_means,
        yerr=E_stds,
        capsize=5,
        color=SENSOR_COLORS[2],
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels on bars (scientific notation)
    for i, (bar, mean, std) in enumerate(zip(bars, E_means, E_stds)):
        height = bar.get_height()
        if mean > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + std,
                f"{mean:.1e}",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_TICK - 1,
                rotation=0,
            )

    ax.set_xlabel("Particle Size Bin (µm)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Emission Rate E (#/min)", fontsize=FONT_SIZE_LABEL)
    ax.set_title(
        "Shower Emission Rate by Particle Size\n(Mean ± Std Dev)",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")

    # Use log scale if range is large
    if max(E_means) > 0 and max(E_means) / min([m for m in E_means if m > 0] + [1]) > 100:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_size_distribution_summary(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
) -> None:
    """
    Create multi-panel figure showing all three metrics vs particle size.

    Parameters:
        results_df: DataFrame with analysis results
        particle_bins: Dictionary of particle bin information
        output_path: Path to save the figure
    """
    apply_style()

    bin_nums = list(particle_bins.keys())
    bin_labels = [particle_bins[i]["name"] for i in bin_nums]
    bin_centers = [
        (particle_bins[i]["min"] + particle_bins[i]["max"]) / 2 for i in bin_nums
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Penetration factor
    p_means = []
    p_stds = []
    for bin_num in bin_nums:
        col = f"bin{bin_num}_p_mean"
        valid_values = results_df[col].dropna()
        p_means.append(valid_values.mean() if len(valid_values) > 0 else np.nan)
        p_stds.append(valid_values.std() if len(valid_values) > 0 else np.nan)

    axes[0].errorbar(
        bin_centers,
        p_means,
        yerr=p_stds,
        marker="o",
        markersize=8,
        capsize=5,
        color=SENSOR_COLORS[0],
        linewidth=LINE_WIDTH_DATA,
    )
    axes[0].set_xlabel("Particle Size (µm)", fontsize=FONT_SIZE_LABEL)
    axes[0].set_ylabel("Penetration Factor (p)", fontsize=FONT_SIZE_LABEL)
    axes[0].set_title("(a) Penetration Factor", fontsize=FONT_SIZE_TITLE)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.1)

    # Panel 2: Deposition rate
    beta_means = []
    beta_stds = []
    for bin_num in bin_nums:
        col = f"bin{bin_num}_beta_mean"
        valid_values = results_df[col].dropna()
        beta_means.append(valid_values.mean() if len(valid_values) > 0 else np.nan)
        beta_stds.append(valid_values.std() if len(valid_values) > 0 else np.nan)

    axes[1].errorbar(
        bin_centers,
        beta_means,
        yerr=beta_stds,
        marker="s",
        markersize=8,
        capsize=5,
        color=SENSOR_COLORS[1],
        linewidth=LINE_WIDTH_DATA,
    )
    axes[1].set_xlabel("Particle Size (µm)", fontsize=FONT_SIZE_LABEL)
    axes[1].set_ylabel("Deposition Rate β (h⁻¹)", fontsize=FONT_SIZE_LABEL)
    axes[1].set_title("(b) Deposition Rate", fontsize=FONT_SIZE_TITLE)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Emission rate
    E_means = []
    E_stds = []
    for bin_num in bin_nums:
        col = f"bin{bin_num}_E_mean"
        valid_values = results_df[col].dropna()
        E_means.append(valid_values.mean() if len(valid_values) > 0 else np.nan)
        E_stds.append(valid_values.std() if len(valid_values) > 0 else np.nan)

    axes[2].errorbar(
        bin_centers,
        E_means,
        yerr=E_stds,
        marker="^",
        markersize=8,
        capsize=5,
        color=SENSOR_COLORS[2],
        linewidth=LINE_WIDTH_DATA,
    )
    axes[2].set_xlabel("Particle Size (µm)", fontsize=FONT_SIZE_LABEL)
    axes[2].set_ylabel("Emission Rate E (#/min)", fontsize=FONT_SIZE_LABEL)
    axes[2].set_title("(c) Emission Rate", fontsize=FONT_SIZE_TITLE)
    axes[2].grid(True, alpha=0.3)

    # Apply log scale if needed
    if max([e for e in E_means if not np.isnan(e)] + [0]) > 0:
        valid_E = [e for e in E_means if not np.isnan(e) and e > 0]
        if valid_E and max(valid_E) / min(valid_E) > 100:
            axes[2].set_yscale("log")

    for ax in axes:
        ax.tick_params(labelsize=FONT_SIZE_TICK)

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
