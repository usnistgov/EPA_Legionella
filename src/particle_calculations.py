#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Calculation Functions
==============================

Pure computation functions for particle penetration, deposition, emission,
and concentration prediction. These functions perform the numerical analysis
for the particle decay study without any I/O or event management logic.

Key Calculations:
    - Penetration factor (p): C_inside / C_outside ratio
    - Deposition rate (beta): Numerical solution from decay data
    - Emission rate (E): Mass balance during shower-to-peak period
    - Ct prediction: Forward Euler simulation of indoor concentration

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

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
MAX_DEPOSITION_RATE = 15.0  # Maximum reasonable β (h⁻¹)
MIN_CONCENTRATION_RATIO = (
    1.05  # Minimum C_inside/C_outside during decay
)

# Minimum data point requirements
MIN_POINTS_PENETRATION = 10  # Minimum points for penetration calculation
MIN_POINTS_DEPOSITION = (
    10  # Minimum points for deposition calculation
)
MIN_POINTS_EMISSION = 3  # Minimum points for emission calculation
MIN_VALID_BETA = 5  # Minimum valid β values required


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

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        shower_on (datetime): Shower start time
        time_of_day (str): "Day" or "Night" time classification
        bin_num (int): Particle bin number (0-6)

    Returns:
        Dict: Dictionary with p value and statistics
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


def _calculate_linearized_fit(
    c_inside: np.ndarray,
    c_outside: np.ndarray,
    datetimes: np.ndarray,
    p: float,
    lambda_ach: float,
    beta_numerical: float,
) -> Dict:
    """
    Calculate linearized regression fit for particle decay.

    Uses the linearized form: y = -ln[(C(t) - C_ss) / (C_0 - C_ss)] = (λ + β) * t
    where C_ss = pλC_out_avg / (λ + β) is the steady-state concentration.

    Parameters:
        c_inside: Inside concentration array
        c_outside: Outside concentration array
        datetimes: Datetime array
        p: Penetration factor
        lambda_ach: Air change rate (h⁻¹)
        beta_numerical: Numerically calculated β (h⁻¹)

    Returns:
        Dict with fit parameters: beta_fit, r_squared, _t_values, _y_values, c_steady_state
    """
    from scipy import stats

    # Calculate steady-state concentration using numerical β
    c_outside_mean = float(np.nanmean(c_outside))
    total_loss_rate = lambda_ach + beta_numerical

    if total_loss_rate <= 0:
        return {
            "beta_fit": np.nan,
            "r_squared": np.nan,
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": np.nan,
        }

    c_steady_state = (p * lambda_ach * c_outside_mean) / total_loss_rate

    # Get initial concentration
    c_0 = c_inside[0]
    delta_c_0 = c_0 - c_steady_state

    if delta_c_0 <= 0:
        return {
            "beta_fit": np.nan,
            "r_squared": np.nan,
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": c_steady_state,
        }

    # Calculate time in hours from start
    t0 = datetimes[0]
    t_hours = (datetimes - t0).astype("timedelta64[s]").astype(float) / 3600.0

    # Calculate linearized y values: y = -ln[(C(t) - C_ss) / (C_0 - C_ss)]
    t_values = []
    y_values = []

    for i in range(len(c_inside)):
        c_t = c_inside[i]
        if np.isnan(c_t) or c_t <= c_steady_state:
            continue

        delta_c_t = c_t - c_steady_state
        ratio = delta_c_t / delta_c_0

        if ratio > 0 and ratio <= 1:
            y = -np.log(ratio)
            t_values.append(t_hours[i])
            y_values.append(y)

    if len(t_values) < 5:
        return {
            "beta_fit": np.nan,
            "r_squared": np.nan,
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": c_steady_state,
        }

    # Linear regression: y = slope * t + intercept
    t_arr = np.array(t_values)
    y_arr = np.array(y_values)

    slope, intercept, r_value, p_value, std_err = stats.linregress(t_arr, y_arr)
    r_squared = r_value**2

    # slope = λ + β, so β_fit = slope - λ
    beta_fit = slope - lambda_ach

    return {
        "beta_fit": float(beta_fit) if beta_fit >= 0 else np.nan,
        "r_squared": float(r_squared),
        "_t_values": t_values,
        "_y_values": y_values,
        "c_steady_state": float(c_steady_state),
        "_slope": float(slope),
        "_intercept": float(intercept),
    }


def calculate_deposition_rate(
    particle_data: pd.DataFrame,
    window_start: datetime,
    window_end: datetime,
    bin_num: int,
    p: float,
    lambda_ach: float,
) -> Dict:
    """
    Calculate deposition rate (β_deposition) using numerical approach and linearized fit.

    The decay calculation starts from the peak concentration within the deposition
    window (not from window_start), as the peak may occur after shower_off.

    Numerical method solves: β_deposition = 1/Δt - λ - C_t(i+1)/(C_t Δt) + (pλC_out,t)/C_t

    Also performs linearized regression fit for plotting:
    y = -ln[(C(t) - C_ss) / (C_0 - C_ss)] = (λ + β) * t

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        window_start (datetime): Start of deposition window (shower_off)
        window_end (datetime): End of deposition window
        bin_num (int): Particle bin number (0-6)
        p (float): Penetration factor
        lambda_ach (float): Air change rate (h⁻¹)

    Returns:
        Dict: Dictionary with β statistics, R², fit data for plotting, and peak_time
    """
    bin_info = PARTICLE_BINS[bin_num]
    col_inside = f"{bin_info['column']}_inside"
    col_outside = f"{bin_info['column']}_outside"

    # Filter to full deposition window first
    mask = (particle_data["datetime"] >= window_start) & (
        particle_data["datetime"] <= window_end
    )
    window_data = particle_data[mask].copy()

    if len(window_data) < MIN_POINTS_DEPOSITION:
        return {
            "beta_mean": np.nan,
            "beta_std": np.nan,
            "beta_median": np.nan,
            "beta_r_squared": np.nan,
            "n_points": len(window_data),
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": np.nan,
            "peak_time": None,
            "skip_reason": f"Insufficient data: {len(window_data)} points (minimum {MIN_POINTS_DEPOSITION} required)",
        }

    # Find peak concentration within the deposition window for this bin
    c_inside_full = np.asarray(window_data[col_inside].values, dtype=np.float64)

    # Find index of maximum concentration (ignoring NaN values)
    valid_mask = ~np.isnan(c_inside_full)
    if not np.any(valid_mask):
        return {
            "beta_mean": np.nan,
            "beta_std": np.nan,
            "beta_median": np.nan,
            "beta_r_squared": np.nan,
            "n_points": 0,
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": np.nan,
            "peak_time": None,
            "skip_reason": "No valid concentration data in window",
        }

    # Get peak index within the full window
    peak_idx = np.nanargmax(c_inside_full)
    peak_time = pd.Timestamp(window_data["datetime"].iloc[peak_idx])

    # Now filter data from peak to end of window for decay calculation
    decay_data = window_data.iloc[peak_idx:].copy()

    if len(decay_data) < MIN_POINTS_DEPOSITION:
        return {
            "beta_mean": np.nan,
            "beta_std": np.nan,
            "beta_median": np.nan,
            "beta_r_squared": np.nan,
            "n_points": len(decay_data),
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": np.nan,
            "peak_time": peak_time,
            "skip_reason": f"Insufficient data after peak: {len(decay_data)} points (minimum {MIN_POINTS_DEPOSITION} required)",
        }

    c_inside = np.asarray(decay_data[col_inside].values, dtype=np.float64)
    c_outside = np.asarray(decay_data[col_outside].values, dtype=np.float64)
    datetimes = decay_data["datetime"].values

    # Check for sufficient concentration difference (now using peak concentration)
    c_outside_mean = np.mean(c_outside)
    c_ratio = c_inside[0] / c_outside_mean if c_outside_mean > 0 else 0
    if c_ratio < MIN_CONCENTRATION_RATIO:
        return {
            "beta_mean": np.nan,
            "beta_std": np.nan,
            "beta_median": np.nan,
            "beta_r_squared": np.nan,
            "n_points": 0,
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": np.nan,
            "peak_time": peak_time,
            "skip_reason": (
                f"Insufficient concentration ratio at peak: {c_ratio:.3f} "
                f"(minimum {MIN_CONCENTRATION_RATIO}). "
                f"C_peak={c_inside[0]:.1f}, C_outside_mean={c_outside_mean:.1f}"
            ),
        }

    # Calculate β for each time step (numerical method) starting from peak
    dt_hours = TIME_STEP_MINUTES / 60.0  # Convert to hours
    beta_values = []

    for i in range(len(c_inside) - 1):
        c_t = c_inside[i]
        c_t_next = c_inside[i + 1]
        c_out_t = c_outside[i]

        # Skip invalid points
        if c_t <= 0 or np.isnan(c_t) or np.isnan(c_t_next) or np.isnan(c_out_t):
            continue

        # Calculate β_deposition
        # β = 1/Δt - λ - C_t(i+1)/(C_t Δt) + (pλC_out,t)/C_t
        term1 = 1.0 / dt_hours
        term2 = -lambda_ach
        term3 = -c_t_next / (c_t * dt_hours)
        term4 = (p * lambda_ach * c_out_t) / c_t

        beta = term1 + term2 + term3 + term4

        # Filter unreasonable values
        if 0 <= beta <= MAX_DEPOSITION_RATE:
            beta_values.append(beta)

    if len(beta_values) < MIN_VALID_BETA:
        return {
            "beta_mean": np.nan,
            "beta_std": np.nan,
            "beta_median": np.nan,
            "beta_r_squared": np.nan,
            "n_points": len(beta_values),
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": np.nan,
            "peak_time": peak_time,
            "skip_reason": (
                f"Insufficient valid β values: {len(beta_values)} "
                f"(minimum {MIN_VALID_BETA} required)"
            ),
        }

    beta_mean = float(np.mean(beta_values))

    # Calculate linearized fit for plotting and R² (using data from peak onward)
    fit_result = _calculate_linearized_fit(
        c_inside, c_outside, datetimes, p, lambda_ach, beta_mean
    )

    return {
        "beta_mean": beta_mean,
        "beta_std": float(np.std(beta_values)),
        "beta_median": float(np.median(beta_values)),
        "beta_r_squared": fit_result.get("r_squared", np.nan),
        "beta_fit": fit_result.get("beta_fit", np.nan),  # β from linearized regression
        "_fit_slope": fit_result.get("_slope", np.nan),  # Actual slope from regression
        "_fit_intercept": fit_result.get(
            "_intercept", 0.0
        ),  # Intercept from regression
        "n_points": len(beta_values),
        "_t_values": fit_result.get("_t_values", []),
        "_y_values": fit_result.get("_y_values", []),
        "c_steady_state": fit_result.get("c_steady_state", np.nan),
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

    Solves: E = pλVC_out,t + V(C_t - C_t(i+1))/Δt - λVC_t - β_deposition VC_t

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        shower_on (datetime): Shower start time
        peak_time (datetime): Time of peak inside concentration
        bin_num (int): Particle bin number (0-6)
        p (float): Penetration factor
        lambda_ach (float): Air change rate (h⁻¹)
        beta (float): Deposition rate (h⁻¹)

    Returns:
        Dict: Dictionary with E statistics (particles/minute)
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

    V = (
        BEDROOM_VOLUME_M3 * CM3_PER_M3
    )  # Convert m³ to cm³ for concentration units (#/cm³)
    dt_minutes = TIME_STEP_MINUTES  # minutes

    # Calculate E for each time step
    E_values = []

    for i in range(len(c_inside) - 1):
        c_t = c_inside[i]
        c_t_next = c_inside[i + 1]
        c_out_t = c_outside[i]

        # Skip invalid points
        if np.isnan(c_t) or np.isnan(c_t_next) or np.isnan(c_out_t):
            continue

        # Calculate E (particles/minute)
        # E = pλVC_out,t + V(C_t - C_t(i+1))/Δt - λVC_t - β_deposition VC_t
        # Convert λ and β from h⁻¹ to min⁻¹
        lambda_per_min = lambda_ach / 60.0
        beta_per_min = beta / 60.0

        term1 = p * lambda_per_min * V * c_out_t
        term2 = V * (c_t - c_t_next) / dt_minutes
        term3 = -lambda_per_min * V * c_t
        term4 = -beta_per_min * V * c_t

        E = term1 + term2 + term3 + term4

        # Only keep positive emission rates
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

    # Calculate total emission over shower duration
    E_total = np.sum(E_values) * dt_minutes  # Total particles emitted

    return {
        "E_mean": float(np.mean(E_values)),
        "E_std": float(np.std(E_values)),
        "E_median": float(np.median(E_values)),
        "E_total": float(E_total),
        "n_points": len(E_values),
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

    Forward Euler scheme:
        C_t(i+1) = C_t + Δt * [p·λ·C_out,t - C_t·(λ + β_deposition) + E/V]

    E = E_mean for t <= peak_time, E = 0 for t > peak_time.

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        shower_on (datetime): Shower start time (simulation start)
        shower_off (datetime): Shower end time
        deposition_end (datetime): End of deposition window (simulation end)
        bin_num (int): Particle bin number (0-6)
        p (float): Penetration factor
        lambda_ach (float): Air change rate (h⁻¹)
        beta (float): Deposition rate (h⁻¹)
        E_mean (float): Mean emission rate during shower (#/min)
        peak_time (datetime): Time of peak concentration (E=0 after this)

    Returns:
        Dict with 'datetimes' and 'predicted_ct' arrays, or 'skip_reason'
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

    # Convert E from #/min to #/hour for consistent units with λ and β (h⁻¹)
    E_per_hour = E_mean * 60.0

    # Initial condition: observed concentration at shower_on
    c_0 = c_inside[0]
    if np.isnan(c_0):
        return {"skip_reason": "Initial concentration is NaN"}

    # Forward Euler simulation
    predicted = np.zeros(len(sim_data))
    predicted[0] = c_0

    for i in range(len(predicted) - 1):
        c_t = predicted[i]
        c_out_t = c_outside[i] if not np.isnan(c_outside[i]) else 0.0
        current_time = pd.Timestamp(datetimes[i])

        # E is active from shower_on to peak_time, then zero
        if current_time <= peak_time:
            E_active = E_per_hour
        else:
            E_active = 0.0

        # C(i+1) = C(i) + dt * [p*λ*C_out - C(i)*(λ + β) + E/V]
        dCdt = (
            p * lambda_ach * c_out_t
            - c_t * (lambda_ach + beta)
            + E_active / V
        )
        predicted[i + 1] = c_t + dt_hours * dCdt

        # Ensure non-negative concentration
        if predicted[i + 1] < 0:
            predicted[i + 1] = 0.0

    # Decay-only simulation: starts from measured peak at peak_time, E=0
    peak_ts = pd.Timestamp(peak_time)
    decay_mask = pd.to_datetime(datetimes) >= peak_ts
    decay_indices = np.where(decay_mask)[0]

    decay_datetimes = []
    decay_predicted = []

    if len(decay_indices) > 1:
        start_idx = decay_indices[0]
        c_peak_measured = c_inside[start_idx]
        if not np.isnan(c_peak_measured):
            n_decay = len(decay_indices)
            decay_pred = np.zeros(n_decay)
            decay_pred[0] = c_peak_measured

            for j in range(n_decay - 1):
                data_idx = decay_indices[j]
                c_t = decay_pred[j]
                c_out_t = c_outside[data_idx] if not np.isnan(c_outside[data_idx]) else 0.0

                dCdt = (
                    p * lambda_ach * c_out_t
                    - c_t * (lambda_ach + beta)
                )
                decay_pred[j + 1] = c_t + dt_hours * dCdt
                if decay_pred[j + 1] < 0:
                    decay_pred[j + 1] = 0.0

            decay_datetimes = datetimes[decay_mask]
            decay_predicted = decay_pred

    return {
        "datetimes": datetimes,
        "predicted_ct": predicted,
        "decay_datetimes": decay_datetimes,
        "decay_predicted": decay_predicted,
    }
