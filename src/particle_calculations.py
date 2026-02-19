#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Calculation Functions
==============================

Pure computation functions for particle penetration, deposition, emission,
and concentration prediction. These functions perform the numerical analysis
for the particle decay study without any I/O or event management logic.

Key Calculations:
    - Penetration factor (p): C_inside / C_outside ratio averaged over
      before/after windows relative to the shower event
    - Deposition rate (beta): Numerical step-by-step estimation from the
      discrete mass balance during post-shower decay; beta solved at each
      time step and averaged
    - Emission rate (E): Mass balance solved numerically during
      shower-on-to-peak period
    - Ct prediction: Forward Euler simulation of indoor concentration
      covering the emission phase (shower_on to peak) and decay phase
      (peak to deposition_end) as a single continuous simulation

Step-by-step methodology:

    Step 1 - Penetration factor (p):
        p = C_inside / C_outside averaged over before and after windows,
        zeros excluded, result capped at 1.

    Step 2 - Air change rate (lambda):
        Loaded from CO2 decay analysis results (h⁻¹).

    Step 3 - Deposition rate (beta, numerical):
        Using the 2-hour post-shower window (peak concentration to end):
            (C_{t+1} - C_t)/dt = p*lambda*C_out,t - lambda*C_t - beta*C_t
        Rearranged to solve for beta at each time step:
            beta = 1/dt - lambda - C_{t+1}/(C_t*dt) + p*lambda*(C_out,t/C_t)
        All estimates <= MAX_DEPOSITION_RATE are collected (no lower bound,
        so negative values from noise or rising-concentration steps are
        included to avoid upward bias).  A 5th-95th percentile trim then
        removes extreme outliers on both sides symmetrically.  Mean and std
        of the trimmed set are reported (beta_raw_mean = unclamped mean;
        beta = max(beta_raw_mean, 0)).  R² computed from a forward Euler
        simulation using the mean beta and time-varying outdoor concentration.

    Step 4 - Emission rate (E):
        Using the shower-on-to-peak window:
            E_t = V*(C_{t+1} - C_t)/dt - p*lambda*V*C_out,t
                  + lambda*V*C_t + beta*V*C_t
        Mean and std reported from positive E_t values.

    Step 5 - Predicted concentration (Ct, forward Euler):
        Window: shower_on to 2 hours after shower_off.
        Single continuous simulation:
            C_{t+1} = C_t + dt*[p*lambda*C_out,t - C_t*(lambda + beta) + E_t/V]
        E_t = E_mean for t <= peak_time, E_t = 0 for t > peak_time.
        Emission phase (shower_on to peak_time) and decay phase (peak_time
        to deposition_end) are returned as separate arrays that form a
        continuous prediction curve.

    Step 6 - Total emission (E_total):
        Area under the E_t vs. time curve (trapezoidal rule):
            E_total = dt * sum[(E_t + E_{t+1}) / 2]
        Summed over all time steps from shower_on to peak_time.

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

# =============================================================================
# Configuration Constants
# =============================================================================

# Particle size bin definitions (µm) - Alphasense OPC-N3
PARTICLE_BINS = {
    0: {"name": "0.35-0.46", "min": 0.35, "max": 0.46, "column": "opc_bin0"},
    1: {"name": "0.46-0.66", "min": 0.46, "max": 0.66, "column": "opc_bin1"},
    2: {"name": "0.66-1.0", "min": 0.66, "max": 1.0, "column": "opc_bin2"},
    3: {"name": "1.0-1.3", "min": 1.0, "max": 1.3, "column": "opc_bin3"},
    4: {"name": "1.3-1.7", "min": 1.3, "max": 1.7, "column": "opc_bin4"},
    5: {"name": "1.7-2.3", "min": 1.7, "max": 2.3, "column": "opc_bin5"},
    6: {"name": "2.3-3.0", "min": 2.3, "max": 3.0, "column": "opc_bin6"},
}

# Physical parameters
BEDROOM_VOLUME_M3 = 36.1  # Bedroom volume in cubic meters (36.10859771 m³ from CAD)
CM3_PER_M3 = 1e6  # Conversion factor: cubic centimeters per cubic meter

# Analysis timing parameters
DEPOSITION_WINDOW_HOURS = 2.0  # Hours after shower for β calculation
TIME_STEP_MINUTES = 1.0  # Time resolution for numerical calculations

# Smoothing parameters (set to 0 to disable)
ROLLING_WINDOW_MIN = 0  # Rolling average window in minutes (0 = no smoothing)

# Validation thresholds
# MAX_DEPOSITION_RATE: upper cap on individual per-step beta estimates before
# percentile-trimming.  Values above 15 h⁻¹ correspond to a half-life of
# ~4 min for gravitational settling alone, which is physically unreasonable
# for particles < 3 µm; these are almost certainly noise spikes.
MAX_DEPOSITION_RATE = 15.0  # Maximum reasonable β (h⁻¹)

# Minimum data point requirements
# MIN_POINTS_PENETRATION: penetration windows are ~5 h long at 1-min resolution
# (~300 data points); 10 is a very loose lower bound that only rejects windows
# with severe data dropouts.
MIN_POINTS_PENETRATION = 10  # Minimum points for penetration calculation
# MIN_POINTS_DEPOSITION: the 2-hr decay window contains ~120 points; 10 ensures
# at least a few minutes of continuous decay data for a meaningful beta estimate.
MIN_POINTS_DEPOSITION = 10  # Minimum points for deposition calculation
# MIN_POINTS_EMISSION: the shower-on-to-peak window can be as short as 2-3 min;
# requiring at least 3 consecutive valid steps avoids single-point estimates.
MIN_POINTS_EMISSION = 3  # Minimum points for emission calculation


# =============================================================================
# Penetration Factor Functions
# =============================================================================


def get_penetration_windows(
    shower_on: datetime,
    time_of_day: str,
) -> List[tuple]:
    """
    Calculate penetration factor averaging windows based on shower time and time of day.

    For Night events:
        Before: 9pm (day before) to 2am (day of)
        After:  9am (day of) to 2pm (day of)

    For Day events:
        Before: 9am (day of) to 2pm (day of)
        After:  9pm (day of) to 2am (next day)

    Parameters:
        shower_on (datetime): Shower start time
        time_of_day (str): "Night" or "Day"

    Returns:
        List of (window_start, window_end) tuples for before and after windows
    """
    shower_date = shower_on.replace(hour=0, minute=0, second=0, microsecond=0)

    # Classify as night or day event
    is_night_event = time_of_day == "Night" or (
        time_of_day == "" and shower_on.hour < 12
    )

    if is_night_event:
        # 3am event: before = 9pm (day before) to 2am (day of)
        #             after  = 9am (day of) to 2pm (day of)
        before_start = shower_date - timedelta(hours=3)  # 9pm day before
        before_end = shower_date + timedelta(hours=2)  # 2am day of
        after_start = shower_date + timedelta(hours=9)  # 9am day of
        after_end = shower_date + timedelta(hours=14)  # 2pm day of
    else:
        # 3pm event: before = 9am (day of) to 2pm (day of)
        #             after  = 9pm (day of) to 2am (next day)
        before_start = shower_date + timedelta(hours=9)  # 9am day of
        before_end = shower_date + timedelta(hours=14)  # 2pm day of
        after_start = shower_date + timedelta(hours=21)  # 9pm day of
        after_end = shower_date + timedelta(hours=26)  # 2am next day

    return [(before_start, before_end), (after_start, after_end)]


def _calculate_p_for_window(
    particle_data: pd.DataFrame,
    window_start: datetime,
    window_end: datetime,
    bin_num: int,
) -> Dict:
    """
    Calculate penetration factor (p) for a single window.

    p = C_inside / C_outside averaged over the window, excluding zero values.

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        window_start (datetime): Start of analysis window
        window_end (datetime): End of analysis window
        bin_num (int): Particle bin number (0-6)

    Returns:
        Dict with p_mean, p_std, n_points, and optional skip_reason
    """
    bin_info = PARTICLE_BINS[bin_num]
    col_inside = f"{bin_info['column']}_inside"
    col_outside = f"{bin_info['column']}_outside"

    # Filter to window
    mask = (particle_data["datetime"] >= window_start) & (
        particle_data["datetime"] <= window_end
    )
    window_data = particle_data[mask].copy()

    if len(window_data) < MIN_POINTS_PENETRATION:
        return {
            "p_mean": np.nan,
            "p_std": np.nan,
            "n_points": len(window_data),
            "skip_reason": f"Insufficient data: {len(window_data)} points (minimum {MIN_POINTS_PENETRATION} required)",
        }

    c_inside = np.asarray(window_data[col_inside].values, dtype=np.float64)
    c_outside = np.asarray(window_data[col_outside].values, dtype=np.float64)

    # Remove invalid points: exclude zeros and NaNs
    valid_mask = (
        (c_inside > 0)
        & (c_outside > 0)
        & (~np.isnan(c_inside))
        & (~np.isnan(c_outside))
    )

    if np.sum(valid_mask) < MIN_POINTS_PENETRATION:
        return {
            "p_mean": np.nan,
            "p_std": np.nan,
            "n_points": int(np.sum(valid_mask)),
            "skip_reason": f"Insufficient valid points: {np.sum(valid_mask)} (minimum {MIN_POINTS_PENETRATION} required)",
        }

    c_inside_valid = c_inside[valid_mask]
    c_outside_valid = c_outside[valid_mask]

    # Calculate p for each point
    p_values = c_inside_valid / c_outside_valid

    return {
        "p_mean": float(np.mean(p_values)),
        "p_std": float(np.std(p_values)),
        "n_points": len(p_values),
    }


def calculate_penetration_factor(
    particle_data: pd.DataFrame,
    shower_on: datetime,
    time_of_day: str,
    bin_num: int,
) -> Dict:
    """
    Calculate penetration factor (p) for a particle bin using before/after windows.

    p = average of C_inside / C_outside from the before and after windows.
    Zero concentration values are excluded. Values above 1 are capped at 1.

    QA filters applied:
        - MIN_POINTS_PENETRATION (10): minimum valid points per window; each
          window spans ~5 h at 1-min resolution (~300 points), so this only
          rejects windows with severe data dropouts.
        - Zero/NaN concentrations excluded before computing ratios.
        - p capped at 1.0 (physical upper bound for infiltration factor).

    Window timing (see get_penetration_windows for exact offsets):
        Night events: before = 9 pm (day before) → 2 am (day of);
                      after  = 9 am → 2 pm (day of)
        Day events:   before = 9 am → 2 pm (day of);
                      after  = 9 pm (day of) → 2 am (next day)

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        shower_on (datetime): Shower start time
        time_of_day (str): "Day" or "Night" time classification
        bin_num (int): Particle bin number (0-6)

    Returns:
        Dict: Dictionary with p_mean, p_std, n_points, n_windows; or
              p_mean=NaN and skip_reason on failure
    """
    windows = get_penetration_windows(shower_on, time_of_day)

    window_p_values = []
    total_points = 0
    skip_reasons = []

    for i, (w_start, w_end) in enumerate(windows):
        label = "before" if i == 0 else "after"
        result = _calculate_p_for_window(particle_data, w_start, w_end, bin_num)

        if not np.isnan(result.get("p_mean", np.nan)):
            window_p_values.append(result["p_mean"])
            total_points += result["n_points"]
        else:
            skip_reasons.append(f"{label}: {result.get('skip_reason', 'Unknown')}")

    if not window_p_values:
        return {
            "p_mean": np.nan,
            "p_std": np.nan,
            "c_inside_mean": np.nan,
            "c_outside_mean": np.nan,
            "n_points": total_points,
            "skip_reason": "; ".join(skip_reasons),
        }

    # Average across available windows, then cap at 1
    p_avg = float(np.mean(window_p_values))
    p_capped = min(p_avg, 1.0)

    return {
        "p_mean": p_capped,
        "p_std": float(np.std(window_p_values)) if len(window_p_values) > 1 else 0.0,
        "c_inside_mean": np.nan,
        "c_outside_mean": np.nan,
        "n_points": total_points,
        "n_windows": len(window_p_values),
    }


# =============================================================================
# Deposition Rate Functions
# =============================================================================


def calculate_deposition_rate(
    particle_data: pd.DataFrame,
    window_start: datetime,
    window_end: datetime,
    bin_num: int,
    p: float,
    lambda_ach: float,
) -> Dict:
    """
    Calculate deposition rate (beta) using a numerical step-by-step approach.

    At each consecutive time step in the post-shower decay window the mass
    balance (with E = 0) is rearranged to solve for beta:

        (C_{t+1} - C_t)/dt = p*lambda*C_out,t - lambda*C_t - beta*C_t

        => beta = 1/dt - lambda - C_{t+1}/(C_t*dt) + p*lambda*(C_out,t/C_t)

    where dt is the time step in hours and C_out,t is the measured outdoor
    concentration at time t (time-varying, not averaged).

    All per-step estimates at or below MAX_DEPOSITION_RATE are collected
    (no lower bound — negative values from noise or brief concentration
    rises are retained to avoid upward bias).  A symmetric 5th–95th
    percentile trim then removes extreme outliers on both sides.  R² is
    computed from a forward Euler simulation using the trimmed-mean beta
    and the measured (time-varying) outdoor concentration.

    QA filters applied:
        - MIN_POINTS_DEPOSITION (10): minimum valid data points both in the
          full window and after peak; the 2-hr window at 1-min resolution
          yields ~120 points, so 10 rejects only severe dropouts.
        - MAX_DEPOSITION_RATE (15.0 h⁻¹): upper cap on per-step estimates;
          values above this are physically unreasonable for sub-3 µm particles
          and indicate measurement noise spikes.
        - 5th–95th percentile trim on the retained estimates to remove
          symmetric extreme outliers without introducing directional bias.
        - Negative trimmed-mean beta is clamped to 0 (beta_raw_mean stores the
          unclamped value for transparency).  For small bins where indoor ≈
          outdoor concentration the signal is buried in noise; clamping to 0
          allows Ct prediction and emission calculations to continue rather
          than discarding the event entirely.
        - c_t <= 0 steps skipped (division by c_t unstable).
        - DEPOSITION_WINDOW_HOURS (2.0): decay window length; 2 h is long
          enough for measurable decay of larger bins and short enough to
          avoid environmental drift dominating the signal.

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        window_start (datetime): Start of deposition window (shower_off)
        window_end (datetime): End of deposition window
        bin_num (int): Particle bin number (0-6)
        p (float): Penetration factor
        lambda_ach (float): Air change rate (h⁻¹)

    Returns:
        Dict: Dictionary with beta (clamped ≥ 0), beta_raw_mean (unclamped
              trimmed mean for transparency), beta_std, beta_r_squared,
              n_points, c_steady_state, and peak_time; or NaN values +
              skip_reason on failure
    """
    bin_info = PARTICLE_BINS[bin_num]
    col_inside = f"{bin_info['column']}_inside"
    col_outside = f"{bin_info['column']}_outside"

    _nan_result = {
        "beta": np.nan,
        "beta_raw_mean": np.nan,
        "beta_std": np.nan,
        "beta_r_squared": np.nan,
        "n_points": 0,
        "c_steady_state": np.nan,
        "peak_time": None,
    }

    # Filter to full deposition window first
    mask = (particle_data["datetime"] >= window_start) & (
        particle_data["datetime"] <= window_end
    )
    window_data = particle_data[mask].copy()

    if len(window_data) < MIN_POINTS_DEPOSITION:
        return {
            **_nan_result,
            "n_points": len(window_data),
            "skip_reason": (
                f"Insufficient data: {len(window_data)} points "
                f"(minimum {MIN_POINTS_DEPOSITION} required)"
            ),
        }

    # Find peak concentration within the deposition window for this bin
    c_inside_full = np.asarray(window_data[col_inside].values, dtype=np.float64)

    valid_mask_full = ~np.isnan(c_inside_full)
    if not np.any(valid_mask_full):
        return {
            **_nan_result,
            "skip_reason": "No valid concentration data in window",
        }

    # Get peak index within the full window
    peak_idx = np.nanargmax(c_inside_full)
    peak_time = pd.Timestamp(window_data["datetime"].iloc[peak_idx])

    # Filter data from peak to end of window for decay calculation
    decay_data = window_data.iloc[peak_idx:].copy()

    if len(decay_data) < MIN_POINTS_DEPOSITION:
        return {
            **_nan_result,
            "n_points": len(decay_data),
            "peak_time": peak_time,
            "skip_reason": (
                f"Insufficient data after peak: {len(decay_data)} points "
                f"(minimum {MIN_POINTS_DEPOSITION} required)"
            ),
        }

    c_inside = np.asarray(decay_data[col_inside].values, dtype=np.float64)
    c_outside = np.asarray(decay_data[col_outside].values, dtype=np.float64)
    c_outside_mean = float(np.nanmean(c_outside))

    # Numerical approach: solve for beta at each consecutive time step.
    # From the discrete mass balance (E = 0):
    #   beta = 1/dt_h - lambda - C_{t+1}/(C_t * dt_h) + p*lambda*(C_out,t/C_t)
    # where dt_h is the time step in hours.
    #
    # All estimates at or below MAX_DEPOSITION_RATE are collected (no lower
    # bound, so negative values from noisy or rising-concentration steps are
    # included).  A 5th–95th percentile trim is then applied to remove
    # extreme outliers on both sides without introducing directional bias.
    dt_h = TIME_STEP_MINUTES / 60.0  # time step in hours
    beta_raw = []
    for i in range(len(c_inside) - 1):
        c_t = c_inside[i]
        c_t_next = c_inside[i + 1]
        c_out_t = c_outside[i]

        # Skip invalid or near-zero concentrations (division by c_t unstable)
        if np.isnan(c_t) or np.isnan(c_t_next) or np.isnan(c_out_t) or c_t <= 0:
            continue

        beta_t = (
            (1.0 / dt_h)
            - lambda_ach
            - (c_t_next / (c_t * dt_h))
            + p * lambda_ach * (c_out_t / c_t)
        )

        # Reject only unphysically large positive outliers
        if beta_t <= MAX_DEPOSITION_RATE:
            beta_raw.append(beta_t)

    if len(beta_raw) < MIN_POINTS_DEPOSITION:
        return {
            **_nan_result,
            "n_points": len(beta_raw),
            "peak_time": peak_time,
            "skip_reason": (
                f"Insufficient valid beta estimates: {len(beta_raw)} "
                f"(minimum {MIN_POINTS_DEPOSITION} required)"
            ),
        }

    # Percentile trim (5th–95th) to remove outliers without directional bias
    beta_arr = np.array(beta_raw, dtype=np.float64)
    p5 = float(np.percentile(beta_arr, 5))
    p95 = float(np.percentile(beta_arr, 95))
    beta_trimmed = beta_arr[(beta_arr >= p5) & (beta_arr <= p95)]

    if len(beta_trimmed) == 0:
        beta_trimmed = beta_arr  # fall back to full set if trim removes everything

    # Clamp the trimmed-mean beta to 0.  A negative mean indicates the
    # concentration gradient was too small to measure deposition above noise
    # (common for small bins where indoor ≈ outdoor in high-ACH rooms).
    # We clamp rather than returning NaN so that the Ct prediction and emission
    # calculations can still run; the raw (possibly negative) mean is stored as
    # beta_raw_mean so callers can see when clamping occurred.
    beta_mean = float(np.mean(beta_trimmed))
    beta_val = max(beta_mean, 0.0)
    beta_std_val = float(np.std(beta_trimmed))

    # Compute R² from a forward Euler simulation using mean beta and
    # the measured (time-varying) outdoor concentration.
    valid = ~np.isnan(c_inside) & ~np.isnan(c_outside)
    c_inside_valid = c_inside[valid]
    c_outside_valid = c_outside[valid]

    if len(c_inside_valid) >= 2:
        sim = np.zeros(len(c_inside_valid))
        sim[0] = c_inside_valid[0]
        for j in range(len(sim) - 1):
            c_t = sim[j]
            c_out_t = c_outside_valid[j]
            dCdt = p * lambda_ach * c_out_t - c_t * (lambda_ach + beta_val)
            sim[j + 1] = max(c_t + dt_h * dCdt, 0.0)

        ss_res = float(np.sum((c_inside_valid - sim) ** 2))
        ss_tot = float(np.sum((c_inside_valid - np.mean(c_inside_valid)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        r_squared = np.nan

    # Steady-state concentration (using mean outdoor concentration)
    total_loss = lambda_ach + beta_val
    c_steady_state = (
        p * lambda_ach * c_outside_mean / total_loss if total_loss > 0 else 0.0
    )

    return {
        "beta": beta_val,
        "beta_raw_mean": beta_mean,
        "beta_std": beta_std_val,
        "beta_r_squared": r_squared,
        "n_points": len(beta_trimmed),
        "c_steady_state": float(c_steady_state),
        "peak_time": peak_time,
    }


# =============================================================================
# Emission Rate Functions
# =============================================================================


def calculate_emission_rate(
    particle_data: pd.DataFrame,
    shower_on: datetime,
    peak_time: datetime,
    bin_num: int,
    p: float,
    lambda_ach: float,
    beta: float,
) -> Dict:
    """
    Calculate emission rate (E) from shower start to peak concentration.

    Solves for E_t at each time step by rearranging the mass balance equation:
        (C_{t+1} - C_t)/Δt = pλC_out,t - λC_t - β_deposition C_t + E_t/V
        E_t = V(C_{t+1} - C_t)/Δt - pλVC_out,t + λVC_t + β_deposition VC_t

    During the shower phase the concentration is rising (C_{t+1} > C_t), so
    the first term V(C_{t+1} - C_t)/Δt is large and positive, correctly
    yielding a positive emission rate.

    E_total is the area under the E_t vs. time curve (trapezoidal rule),
    integrating only non-negative contributions:
        E_total = Δt × Σ max(E_t, 0) dt  (trapezoidal, clipped at 0)

    QA filters applied:
        - MIN_POINTS_EMISSION (3): minimum valid steps in the emission window;
          the shower-on-to-peak window can be as short as 2–3 min, so requiring
          3 consecutive steps avoids single-point emission estimates.
        - NaN/invalid concentration steps are skipped; the window must contain
          at least MIN_POINTS_EMISSION non-NaN consecutive pairs.
        - E_total clips negative per-step contributions to 0 before trapezoid
          integration; brief noisy dips should not subtract from the total count.
        - E_mean and E_std are computed from positive E_t values only; zero or
          negative steps (concentration briefly flat or dipping) are excluded
          from the mean but included in E_total (as 0).

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        shower_on (datetime): Shower start time
        peak_time (datetime): Time of peak inside concentration
        bin_num (int): Particle bin number (0-6)
        p (float): Penetration factor
        lambda_ach (float): Air change rate (h⁻¹)
        beta (float): Deposition rate (h⁻¹)

    Returns:
        Dict: Dictionary with E_mean, E_std, E_total statistics (#/min, #);
              also E_times (list of timestamps) and E_per_step (list of all
              per-step E_t values aligned with E_times) for visualization
    """
    bin_info = PARTICLE_BINS[bin_num]
    col_inside = f"{bin_info['column']}_inside"
    col_outside = f"{bin_info['column']}_outside"

    # Filter to emission window: shower_on to peak_time
    mask = (particle_data["datetime"] >= shower_on) & (
        particle_data["datetime"] <= peak_time
    )
    shower_data = particle_data[mask].copy()

    if len(shower_data) < MIN_POINTS_EMISSION:
        return {
            "E_mean": np.nan,
            "E_std": np.nan,
            "E_median": np.nan,
            "E_total": np.nan,
            "n_points": len(shower_data),
            "skip_reason": (
                f"Insufficient data: {len(shower_data)} points "
                f"(minimum {MIN_POINTS_EMISSION} required)"
            ),
        }

    c_inside = np.asarray(shower_data[col_inside].values, dtype=np.float64)
    c_outside = np.asarray(shower_data[col_outside].values, dtype=np.float64)
    datetimes_arr = shower_data["datetime"].values  # timestamps for per-step output

    V = (
        BEDROOM_VOLUME_M3 * CM3_PER_M3
    )  # Convert m³ to cm³ for concentration units (#/cm³)
    dt_minutes = TIME_STEP_MINUTES  # minutes

    # Calculate E for each time step
    E_values_all = []  # All valid E values (for trapezoidal E_total)
    E_times_list = []  # Timestamps corresponding to each entry in E_values_all
    E_values = []  # Positive E values only (for mean/std/median)

    # Convert λ and β from h⁻¹ to min⁻¹ once
    lambda_per_min = lambda_ach / 60.0
    beta_per_min = beta / 60.0

    for i in range(len(c_inside) - 1):
        c_t = c_inside[i]
        c_t_next = c_inside[i + 1]
        c_out_t = c_outside[i]

        # Skip invalid points
        if np.isnan(c_t) or np.isnan(c_t_next) or np.isnan(c_out_t):
            continue

        # Solve for E_t from the mass balance equation:
        # (C_{t+1} - C_t)/Δt = pλC_out,t - λC_t - β C_t + E_t/V
        # Rearranging for E_t:
        # E_t = V(C_{t+1} - C_t)/Δt - pλVC_out,t + λVC_t + βVC_t
        # During the shower C_{t+1} > C_t, so term1 is large positive → E > 0
        term1 = V * (c_t_next - c_t) / dt_minutes
        term2 = -p * lambda_per_min * V * c_out_t
        term3 = lambda_per_min * V * c_t
        term4 = beta_per_min * V * c_t

        E = term1 + term2 + term3 + term4

        E_values_all.append(E)
        E_times_list.append(datetimes_arr[i])

        # Only accumulate positive emission rates for statistics
        if E > 0:
            E_values.append(E)

    if len(E_values) == 0:
        return {
            "E_mean": np.nan,
            "E_std": np.nan,
            "E_median": np.nan,
            "E_total": np.nan,
            "n_points": 0,
            "skip_reason": "No positive emission values calculated",
        }

    # Calculate total emission using trapezoidal rule, clipping negatives to
    # zero so that brief noisy dips do not subtract from the total count.
    # E_total = Δt × trapezoid(max(E_t, 0))
    if len(E_values_all) >= 2:
        E_arr = np.maximum(np.array(E_values_all, dtype=np.float64), 0.0)
        E_total = float(trapezoid(E_arr) * dt_minutes)
    else:
        E_total = float(max(0.0, E_values_all[0]) * dt_minutes) if E_values_all else 0.0

    return {
        "E_mean": float(np.mean(E_values)),
        "E_std": float(np.std(E_values)),
        "E_median": float(np.median(E_values)),
        "E_total": float(E_total),
        "n_points": len(E_values),
        "E_times": E_times_list,
        "E_per_step": E_values_all,
    }


# =============================================================================
# Ct Prediction (Forward Euler)
# =============================================================================


def calculate_ct_prediction(
    particle_data: pd.DataFrame,
    shower_on: datetime,
    shower_off: datetime,
    deposition_end: datetime,
    bin_num: int,
    p: float,
    lambda_ach: float,
    beta: float,
    E_mean: float,
    peak_time: datetime,
) -> Dict:
    """
    Simulate indoor particle concentration using forward Euler method.

    A single continuous simulation runs from shower_on to deposition_end:

        C_{t+1} = C_t + dt * [p*lambda*C_out,t - C_t*(lambda + beta) + E_t/V]

    where E_t = E_mean (converted to h⁻¹ units) for t <= peak_time and
    E_t = 0 for t > peak_time.  Pass E_mean = 0 to compute a decay-only
    prediction (useful when no emission estimate is available).

    The result is split at peak_time into two continuous segments that
    together form the full prediction curve:

        Emission phase: shower_on to peak_time  (E = E_mean)
        Decay phase:    peak_time to deposition_end  (E = 0)

    The decay phase starts from the *predicted* concentration at peak_time
    (i.e., the end of the emission phase), not from the measured peak.

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        shower_on (datetime): Shower start time (simulation start)
        shower_off (datetime): Shower end time
        deposition_end (datetime): End of deposition window (simulation end)
        bin_num (int): Particle bin number (0-6)
        p (float): Penetration factor
        lambda_ach (float): Air change rate (h⁻¹)
        beta (float): Deposition rate (h⁻¹)
        E_mean (float): Mean emission rate during shower (#/min); use 0.0
                        to compute a decay-only prediction
        peak_time (datetime): Time of peak concentration (E=0 after this)

    Returns:
        Dict with keys 'datetimes', 'predicted_ct', 'emission_datetimes',
        'emission_predicted', 'decay_datetimes', 'decay_predicted',
        'E_r_squared'; or 'skip_reason' on failure.
        E_r_squared is the R² of the emission-phase forward Euler prediction
        vs. measured indoor concentration from shower_on to peak_time.
    """
    bin_info = PARTICLE_BINS[bin_num]
    col_inside = f"{bin_info['column']}_inside"
    col_outside = f"{bin_info['column']}_outside"

    # Simulation window: shower_on to deposition_end (shower_off + 2h)
    mask = (particle_data["datetime"] >= shower_on) & (
        particle_data["datetime"] <= deposition_end
    )
    sim_data = particle_data[mask].copy()

    if len(sim_data) < 2:
        return {"skip_reason": "Insufficient data for Ct prediction"}

    c_outside = np.asarray(sim_data[col_outside].values, dtype=np.float64)
    c_inside = np.asarray(sim_data[col_inside].values, dtype=np.float64)
    datetimes = sim_data["datetime"].values

    # Volume in cm³ (concentration units are #/cm³)
    V = BEDROOM_VOLUME_M3 * CM3_PER_M3

    # Time step in hours (data is at 1-minute intervals after resampling)
    dt_hours = TIME_STEP_MINUTES / 60.0

    # Convert E from #/min to #/hour for consistent units with lambda and beta
    E_per_hour = E_mean * 60.0

    # Initial condition: observed concentration at shower_on
    c_0 = c_inside[0]
    if np.isnan(c_0):
        return {"skip_reason": "Initial concentration is NaN"}

    # Forward Euler simulation (single continuous run)
    predicted = np.zeros(len(sim_data))
    predicted[0] = c_0

    peak_ts = pd.Timestamp(peak_time)
    datetimes_pd = pd.to_datetime(datetimes)

    for i in range(len(predicted) - 1):
        c_t = predicted[i]
        c_out_t = c_outside[i] if not np.isnan(c_outside[i]) else 0.0

        # E is active from shower_on to peak_time, then zero
        E_active = E_per_hour if datetimes_pd[i] <= peak_ts else 0.0

        # C_{t+1} = C_t + dt * [p*lambda*C_out - C_t*(lambda + beta) + E/V]
        dCdt = p * lambda_ach * c_out_t - c_t * (lambda_ach + beta) + E_active / V
        predicted[i + 1] = max(c_t + dt_hours * dCdt, 0.0)

    # Split the continuous simulation at peak_time into emission and decay phases.
    # Both arrays overlap by one point at peak_time so they form a seamless curve.
    emission_mask = datetimes_pd <= peak_ts
    decay_mask = datetimes_pd >= peak_ts

    # Emission R²: how well does the emission-phase forward Euler prediction
    # match the measured indoor concentration from shower_on to peak_time?
    c_inside_emission = c_inside[emission_mask]
    predicted_emission = predicted[emission_mask]
    valid_em = ~np.isnan(c_inside_emission)
    if np.sum(valid_em) >= 2:
        c_em_valid = c_inside_emission[valid_em]
        pred_em_valid = predicted_emission[valid_em]
        ss_res_em = float(np.sum((c_em_valid - pred_em_valid) ** 2))
        ss_tot_em = float(np.sum((c_em_valid - np.mean(c_em_valid)) ** 2))
        E_r_squared = 1.0 - ss_res_em / ss_tot_em if ss_tot_em > 0 else np.nan
    else:
        E_r_squared = np.nan

    return {
        "datetimes": datetimes,
        "predicted_ct": predicted,
        "emission_datetimes": datetimes[emission_mask],
        "emission_predicted": predicted[emission_mask],
        "decay_datetimes": datetimes[decay_mask],
        "decay_predicted": predicted[decay_mask],
        "E_r_squared": E_r_squared,
    }
