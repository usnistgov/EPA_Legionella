# NIST-EPA Legionella Study

Repository for NIST-EPA Legionella study Python tools. Includes scripts for aerosol particle size distribution analysis, emission-rate calculations, data validation, visualization, and statistical modeling of shower-generated aerosols under varying temperature, humidity, and exhaust-fan conditions.

## Project Overview

This project analyzes aerosol characteristics from shower-generated aerosols with a focus on potential Legionella transmission risks from shower fixtures. The research investigates:

- Aerosol particle size distribution analysis
- Emission-rate calculations using numerical approaches
- CO2 decay analysis for air change rate determination
- Temperature and relative humidity monitoring
- data validation, visualization, and statistical modeling

## Disclaimer

Certain commercial equipment, instruments, software, or materials are identified in this repository in order to specify the experimental and analytical procedures adequately. Such identification is not intended to imply recommendation or endorsement of any product or service by NIST, nor is it intended to imply that the materials or equipment identified are necessarily the best available for the purpose.

## Measurement Uncertainty

The analysis procedures in this repository involve multiple stages of data processing and mathematical modeling, each contributing to the overall uncertainty in the final results. Key sources of uncertainty include:

**CO₂-derived Air Change Rate (λ) Uncertainty:**
- Statistical uncertainty from linear regression of log-transformed decay data (reported as standard error of the slope)
- Systematic uncertainty from assuming a well-mixed zone and constant infiltration fractions (α, β)
- Uncertainty in source concentration characterization evaluated through three methods: weighted average (α·C_outside + β·C_entry), outside-only, and entry-only
- Data quality filters including minimum concentration difference requirements (50 ppm) and sufficient data point thresholds
- Temporal alignment uncertainty between CO₂ injection timing and decay window selection

**Particle Analysis Uncertainty:**
- Penetration factor (p) uncertainty from temporal variability in indoor/outdoor concentration ratios during before/after windows
- Other process rate (β) uncertainty from numerical differentiation of concentration data, mitigated by 5th–95th percentile trimming and R²-based selection criteria
- Emission rate (E) uncertainty from propagation of errors in the mass balance equation and concentration time derivatives
- Forward Euler simulation uncertainty from accumulation of integration errors over the 2-hour prediction window
- Size-bin dependent uncertainties due to varying signal-to-noise ratios across the 0.35–10.0 µm range

**General Measurement Uncertainties:**
- Sensor-specific uncertainties: Aranet4 CO₂ sensors (±50 ppm + 3% of reading), QuantAQ MODULAIR-PM particle counters, HOBO UX100 temperature/relative humidity sensors
- Temporal synchronization uncertainty between different sensor systems sampling at 1-minute intervals
- Spatial representativeness limitations of point measurements in heterogeneous indoor environments
- Model structural uncertainty from assuming first-order decay processes and uniform mixing in the mass balance equations

Uncertainty quantification is implemented throughout the analysis pipeline, with standard deviations, standard errors, and confidence intervals reported where applicable. The three-source-method approach for λ calculation provides explicit assessment of structural uncertainty in ventilation rate determination. Where measurements fall below detection thresholds or fail quality criteria, results are appropriately flagged or excluded to prevent overinterpretation.

## Repository Structure

```
NIST_EPA_Legionella/
├── .vscode/                          # VS Code configuration
│   └── settings.json
├── .env                              # Environment variables (API keys for QuantAQ, gitignored)
├── .gitignore
├── CODEMETA.yaml                     # NIST Software Portal metadata
├── CODEOWNERS                        # Repository ownership for PR reviews
├── LICENSE.md                        # NIST software licensing statement
├── README.md
├── data_config.json                  # Active configuration (gitignored)
├── data_config.template.json         # Configuration template
├── epa_mh.yml                        # Conda environment specification
├── run_analysis_workflow.py          # Convenience script to run the full analysis pipeline
│
├── src/                              # Core analysis modules (imported by scripts)
│   ├── __init__.py                   # Package initialization (re-exports data_paths)
│   ├── data_paths.py                 # Portable data access via data_config.json
│   ├── env_data_loader.py            # Unified environmental sensor data loader
│   ├── event_manager.py              # Event filtering, naming (MMDD_Temp_ToD_RNN), logging
│   ├── event_matching.py             # Match CO2 injections to shower events by timing
│   ├── particle_data_loader.py       # QuantAQ particle data loading and preparation
│   ├── particle_calculations.py      # Penetration, deposition & emission calculations
│   ├── plot_co2.py                   # CO2 decay visualization functions
│   ├── plot_environmental.py         # RH, temperature, wind visualization
│   ├── plot_particle.py              # Particle decay visualization functions
│   ├── plot_style.py                 # Consistent matplotlib styling constants
│   ├── plot_utils.py                 # Central plotting re-exports (imports all plot_* modules)
│   ├── quantaq_utils.py              # QuantAQ API client utilities
│   ├── sig_figs.py                   # Significant figures rounding and formatting
│   └── deprecated/                   # Deprecated/archived code
│       └── co2_decay_analysis.py     # Previous version of CO2 analysis
│
├── scripts/                          # Executable analysis scripts (run directly)
│   ├── __init__.py
│   │
│   │ # Data Download & Processing
│   ├── download_quantaq_data.py      # Download QuantAQ sensor data from API
│   ├── process_quantaq_data.py       # Combine weekly chunks and process raw/final QuantAQ data
│   │
│   │ # Log Processing
│   ├── process_co2_log.py            # Consolidate daily CO2 injection logs → state-change log
│   ├── process_shower_log.py         # Consolidate daily shower logs → state-change log
│   ├── fix_log_files.py              # Repair corrupted/empty log files
│   ├── analyze_log_files.py          # Diagnose log file issues (unmatched events, etc.)
│   │
│   │ # Event Management
│   ├── event_registry.py             # Unified event registry with synthetic event creation
│   │
│   │ # Analysis
│   ├── co2_decay_analysis.py         # CO2 decay & air change rate (λ) analysis
│   ├── particle_decay_analysis.py    # Particle penetration, deposition & emission analysis
│   ├── rh_temp_other_analysis.py     # RH, temperature & wind condition analysis
│   └── export_event_timeseries.py    # Export per-event predicted Ct for all size bins to CSV (--event N)
│
├── testing/                          # Testing and exploratory scripts
│   └── co2 plot.py                   # Aranet4 CO2 plotting script
│
└── docs/                             # Documentation & instrument references
    ├── Data Analysis.docx            # Data analysis planning notes
    ├── MH cDAQ channels MAP and wiring notes 123125.xlsx
    ├── MH_weather_sta_to_DAQ_connections sketch.pdf
    ├── decay_fit_step_change_note.txt        # Explanation of beta=NaN step-change fix
    ├── python_script_style.txt               # Code style conventions for this project
    ├── instrument_informed_sig_figs_prompt.txt  # Prompt for instrument-accuracy sig figs
    ├── notes.txt                             # Project notes
    │
    │ # Instrument Manuals & Data Sheets
    ├── AIO-2-9800-Manual-Rev-G.pdf   # Met One AIO 2 weather sensor manual
    ├── AIO2-1.pdf                    # Met One AIO 2 data sheet
    ├── Aranet_Datasheet_TDSPC003_Aranet4_PRO_1.pdf
    ├── aranet4_user_manual_v25_web.pdf
    ├── NI cDAQ-9171_9174_9178 User Manual - National Instruments.pdf
    ├── ni-9201_getting_started_2-1-2024.pdf
    ├── Setra_Model_264_Data_Sheet.pdf
    ├── Vaisala HMP155_User_Guide_in_English.pdf
    └── Vaisala HMP45AD-User-Guide-U274EN.pdf
```

## Installation

### Prerequisites

- Python 3.14+
- Conda (recommended for environment management)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/usnistgov/NIST_EPA_Legionella.git
   cd NIST_EPA_Legionella
   ```

2. Create the conda environment:
   ```bash
   conda env create -f epa_mh.yml
   conda activate epa_mh
   ```

3. Configure data paths:
   ```bash
   cp data_config.template.json data_config.json
   ```
   Edit `data_config.json` to set the correct `data_root` path for your machine.

4. Set up API credentials (for QuantAQ data):
   Create a `.env` file with your QuantAQ API key:
   ```
   QUANTAQ_API_KEY=your_api_key_here
   ```

## Usage

### Data Access

The `src/data_paths.py` module provides portable data access functions:

```python
from src import get_instrument_path, get_instrument_file, get_instrument_files_for_date_range

# Get instrument data directory
path = get_instrument_path("Aranet4")

# Get data file for a specific date
file = get_instrument_file("Aranet4", "2026-01-15")

# Get files for a date range
files = get_instrument_files_for_date_range("QuantAQ", "2026-01-05", "2026-01-15")
```

### QuantAQ Data Pipeline

1. **Download data** from the QuantAQ API:
   ```bash
   python scripts/download_quantaq_data.py
   ```

2. **Process data** (parse nested structures, merge raw/final data):
   ```bash
   python scripts/process_quantaq_data.py
   ```

### Visualization Tools

Plotting modules in `src/` generate publication-quality figures used by the analysis scripts:

- **`src/plot_co2.py`** - CO2 decay curves and air change rate analysis
- **`src/plot_particle.py`** - Particle concentration decay and emission rates
- **`src/plot_environmental.py`** - RH, temperature, and wind time series
- **`src/plot_style.py`** - Consistent matplotlib styling constants and helpers
- **`src/plot_utils.py`** - Re-exports all plot_* functions for backward compatibility

All plot modules are imported by analysis scripts through `src/plot_utils.py`.

## Configured Instruments

| Instrument | Model | Purpose | Data Location |
|------------|-------|---------|---------------|
| AIO2 | Met One AIO 2 | Outdoor weather (10m tower) | MH DAQ/weather_station |
| Aranet4 | Aranet4 PRO | CO2 monitoring | CO2 |
| HOBO_UX100 | Onset HOBO UX100 | RH/Temperature (6 locations) | HOBOs |
| QuantAQ | MODULAIR-PM | PM monitoring (indoor/outdoor) | QuantAQ |
| Setra_264 | Setra 264 | Differential pressure | MH DAQ/indoor_daq |
| Vaisala_HMP155 | Vaisala HMP155 | RH/Temperature | MH DAQ/indoor_daq |
| Vaisala_HMP45A | Vaisala HMP45A | RH/Temperature | MH DAQ/indoor_daq |

### DAQ System

- **Chassis:** NI cDAQ-9178 (8-slot USB)
- **Modules:** 5x NI 9201 (8-channel analog input, +/-10V)

## Analysis Workflow

The complete data analysis pipeline follows this sequence. To run all steps automatically:
```bash
python run_analysis_workflow.py
```
Each step's output is logged to `output/logs/<script_name>.log`. Or run steps individually:

1. **Data Collection**: Download sensor data from the QuantAQ API
   ```bash
   python scripts/download_quantaq_data.py
   ```

2. **Data Processing**: Process and merge raw QuantAQ sensor data
   ```bash
   python scripts/process_quantaq_data.py
   ```

3. **Log Processing**: Consolidate daily 1-second log files into state-change logs
   ```bash
   python scripts/process_co2_log.py
   python scripts/process_shower_log.py
   ```

4. **Event Management**: Run the event manager to match, name, and register events
   ```bash
   python scripts/event_registry.py [--force] [--no-co2] [--output-dir DIR]
   ```
   | Flag | Description |
   |------|-------------|
   | `--force` | Regenerate registry even if it already exists |
   | `--no-co2` | Skip CO2 analysis (registry will have no lambda values or PM exclusions) |
   | `--output-dir DIR` | Output directory (default: `data_root/output`) |

   This must be run **before** the analysis scripts so that events have consistent names
   (see [Event Naming Convention](#event-naming-convention) below). Produces `event_log.csv`.

5. **CO2 Analysis**: Determine air change rates (λ) from CO2 decay
   ```bash
   python scripts/co2_decay_analysis.py [--alpha FLOAT] [--beta FLOAT] [--output-dir DIR] [--no-plot] [--no-sig-figs]
   ```
   | Flag | Description |
   |------|-------------|
   | `--alpha FLOAT` | Fraction of infiltration from outside (default: 0.5) |
   | `--beta FLOAT` | Fraction of infiltration from entry zone (default: 0.5) |
   | `--output-dir DIR` | Output directory (default: `data_root/output`) |
   | `--no-plot` | Disable plot generation |
   | `--no-sig-figs` | Disable significant figure rounding (default: 3 sig figs for files, 2 for figures) |
   | `--entry-stop` | Truncate decay window when C_bedroom ≤ 100 + (C_entry + C_outside) / 2 — stops the fit once the remaining bedroom decay signal is within 100 ppm of the entry/outside background mean |

6. **Environmental Analysis**: Characterize RH, temperature, and wind conditions
   ```bash
   python scripts/rh_temp_other_analysis.py [--output-dir DIR] [--no-plots] [--max-plot-events N] [--events LIST] [--no-sig-figs]
   ```
   | Flag | Description |
   |------|-------------|
   | `--output-dir DIR` | Output directory (default: `data_root/output`) |
   | `--no-plots` | Skip plot generation |
   | `--max-plot-events N` | Limit time series plots to first N events (default: all) |
   | `--events LIST` | Comma-separated event numbers to plot, e.g. `1,3,5` (overrides `--max-plot-events`) |
   | `--no-sig-figs` | Disable significant figure rounding (default: 3 sig figs for files, 2 for figures) |

7. **Particle Analysis**: Calculate penetration, deposition, and emission rates (requires step 5 results)
   ```bash
   python scripts/particle_decay_analysis.py [--output-dir DIR] [--no-plot] [--no-sig-figs]
   ```
   | Flag | Description |
   |------|-------------|
   | `--output-dir DIR` | Output directory (default: `data_root/output`) |
   | `--no-plot` | Disable plot generation |
   | `--no-sig-figs` | Disable significant figure rounding (default: 3 sig figs for files, 2 for figures) |

8. **Export Event Time Series** *(optional utility)*: Export predicted Ct for all size bins for a single event to CSV:
   ```bash
   python scripts/export_event_timeseries.py --event N [--output PATH] [--output-dir DIR] [--no-sig-figs]
   ```
   | Flag | Description |
   |------|-------------|
   | `--event INT` | Event number to export (required) |
   | `--output PATH` | Output CSV file path (default: `output/event_N_timeseries.csv`) |
   | `--output-dir DIR` | Analysis output directory (default: `data_root/output`) |
   | `--no-sig-figs` | Disable significant figure rounding |

Each analysis module produces CSV/Excel summaries and optional visualizations in the `output/` directory.

## Event Naming Convention

Events are named using a structured format generated by `scripts/event_registry.py`. This naming convention must be applied (step 4 in the workflow above) **before** running the analysis scripts.

**Format:** `MMDD_W##[_ShowerHead[_SprayPattern]][_Mannequin]_DoorPos[_Fan]_RNN`

| Component | Description | Values |
|-----------|-------------|--------|
| `MMDD` | Month and day of the event | e.g., `0115` for January 15 |
| `W##` | Water temperature code (°C) | `W11`, `W14`, `W22`, `W23`, `W25`, `W30`, `W37`, `W38`, `W40`, `W43`, `W48`, `W49`, `W52`, `W53` |
| `ShowerHead` | Shower head type; omitted for Standard head | `Pepco` |
| `SprayPattern` | Spray pattern (Pepco head only; omitted if not applicable) | `Wide`, `Narrow`, `Mid` |
| `Mannequin` | Appended only when a mannequin was present | `Mannequin` |
| `DoorPos` | Bathroom door position | `Open`, `Closed`, `Partial` |
| `Fan` | Appended only when the bath fan ran during the test | `Fan` |
| `RNN` | Replicate number | `R01`, `R02`, `R03`, etc. |

**Examples:**
- `0115_W48_Open_R01` - January 15, 48 °C, standard head, door open, replicate 1
- `0122_W11_Open_R03` - January 22, 11 °C, standard head, door open, replicate 3
- `0224_W52_Pepco_Wide_Open_R01` - February 24, 52 °C, Pepco head, wide spray, door open
- `0309_W40_Pepco_Narrow_Open_R01` - March 9, 40 °C, Pepco head, narrow spray
- `0311_W40_Pepco_Narrow_Mannequin_Open_R01` - March 11, mannequin present

**Test Parameter Timeline (Standard shower head):**
- W48 (48 °C): From experiment start (2026-01-14)
- W11 (11 °C): Starting 2026-01-22 14:00
- W25 (25 °C): Starting 2026-02-02 17:00
- W30 (30 °C): Starting 2026-02-05 10:00
- W37 (37 °C): Starting 2026-02-09 10:00
- W23 (23 °C): Starting 2026-02-11 08:00 *(data excluded; kept for documentation)*
- W22 (22 °C): Starting 2026-02-13 11:00
- W43 (43 °C): Starting 2026-02-16 11:00
- W14 (14 °C): Starting 2026-02-18 10:23
- W53 (53 °C): Starting 2026-02-20 08:00

**Test Parameter Timeline (Pepco shower head, installed 2026-02-24):**
- W52 (52 °C), Wide spray: Starting 2026-02-24 08:00
- W49 (49 °C), Wide spray: Starting 2026-02-26 10:23
- W40 (40 °C), Narrow spray: Starting 2026-03-02 09:23
- W40 (40 °C), Wide spray: Starting 2026-03-04 10:23
- W40 (40 °C), Mid spray: Starting 2026-03-06 08:47
- W40 (40 °C), Narrow spray: Starting 2026-03-09 08:47
- W40 (40 °C), Narrow spray + Mannequin: Starting 2026-03-11 09:47
- W38 (38 °C), Narrow spray: Starting 2026-03-12 10:47
- W38 (38 °C), Narrow spray + Mannequin: Starting 2026-03-13 08:45

## Data Analysis Modules

### CO2 Air Change Rate Analysis

Run the CO2 decay analysis to calculate air change rates (λ):

```bash
# Basic analysis (plots enabled by default)
python scripts/co2_decay_analysis.py

# Custom source fractions
python scripts/co2_decay_analysis.py --alpha 0.6 --beta 0.4  # default is 0.5/0.5

# Skip plots, write to a custom directory
python scripts/co2_decay_analysis.py --no-plot --output-dir /path/to/output
```

**Methodology:**
- Combines three Aranet4 sensor files: Bedroom, Entry, and Outside
- Applies 6-minute rolling average to reduce sensor noise
- Fixed 2-hour decay analysis window starting at :50 (10 minutes before the shower hour)
- Three source concentration methods for uncertainty: λ_average (mean of outside + entry), λ_outside, λ_entry
- Calculates λ via linear regression of the log-transformed decay:
  y = −ln[(C(t) − C_avg) / (C₀ − C_avg)], λ = slope of y vs. time

**Output Files:**
- `output/co2_lambda_summary.csv` - Per-event lambda results
- `output/co2_lambda_overall_summary.csv` - Overall statistics
- `output/plots/event_figures/co2_decay/event_NN-testname_co2_decay.png` - Individual event decay plots with shower ON/OFF markers
- `output/plots/lambda_summary.png` - Summary bar chart of λ values
- `output/plots/air_change_rate_boxplot.png` - Box-and-whisker of λ by water temperature

### Particle Decay & Emission Analysis

Run the particle decay analysis to calculate penetration factors, deposition rates, and emission rates across **12 OPC-N3 size bins (0.35–10.0 µm)**:

```bash
# Run particle analysis (requires CO2 analysis results; plots enabled by default)
python scripts/particle_decay_analysis.py

# Skip plots
python scripts/particle_decay_analysis.py --no-plot

# Write to a custom directory without significant figure rounding
python scripts/particle_decay_analysis.py --output-dir /path/to/output --no-sig-figs
```

**Methodology:**

The mass balance equation for indoor particle concentration:

```
V·dC/dt = p·Q·C_out - Q·C - β·C·V + E
dC/dt = p·λ·C_out - λ·C - β·C + E/V
```

1. **Penetration factor (p):** Ratio of indoor to outdoor concentration averaged over before/after windows relative to the shower event; zeros excluded; capped at 1.
   - *QA:* Minimum 10 valid points per window (`MIN_POINTS_PENETRATION`); windows where this is not met are skipped.
   - Night event windows: before = 9 pm (day before) → 2 am; after = 9 am → 2 pm
   - Day event windows: before = 9 am → 2 pm; after = 9 pm → 2 am (next day)
2. **Air change rate (λ):** Loaded from CO2 decay analysis results (h⁻¹).
3. **Other process rate (β, numerical):** Step-by-step solution during the 2-hour (`DEPOSITION_WINDOW_HOURS`) post-shower decay window; beta solved at each time step from the discrete mass balance:
   ```
   β = 1/dt - λ - C_{t+1}/(C_t·dt) + p·λ·(C_out,t / C_t)
   ```
   All estimates ≤ `MAX_OTHER_PROCESS_RATE` (15 h⁻¹) are collected (no lower bound, to avoid upward bias). A **5th–95th percentile trim** removes extreme outliers symmetrically. Beta is then selected via an **R²-based four-step procedure** (threshold R² = 0.80): (a) use unclamped trimmed mean — if R² ≥ 0.80 keep it; (b) clamp to ≥ 0 — if R² ≥ 0.80 keep it; (c) set β = 0 — if R² ≥ 0.80 keep it; (d) if β = 0 also fails → β = NaN (bin invalid, no Ct prediction plotted).
   - *QA:* Minimum 10 valid points in the decay window and after peak (`MIN_POINTS_OTHER_PROCESS`); `MAX_OTHER_PROCESS_RATE` = 15 h⁻¹ upper cap (above this, values are noise spikes for sub-3 µm particles).
4. **Emission rate (E):** Mass balance rearranged and solved numerically at each time step during the shower-on-to-peak window; mean and std reported from positive values.
   - *QA:* Minimum 3 valid consecutive steps (`MIN_POINTS_EMISSION`); E_total clips negative per-step contributions to 0 before trapezoid integration.
5. **Predicted concentration (Ct):** Single continuous forward Euler simulation from shower ON to 2 hours after shower OFF. E = E_mean before peak_time, E = 0 after. Returned as two connected segments forming one continuous predicted Ct curve on figures. Decay phase starts from the *predicted* peak (end of emission simulation), not the measured peak.
   - Emission R² (E_r_squared): R² of the emission-phase simulation vs. measured indoor concentration from shower ON to peak.
6. **Total emission (E_total):** Trapezoidal rule area under the E_t vs. time curve from shower ON to peak; negative per-step contributions clipped to 0.

**Output Files:**
- `output/particle_analysis_summary.xlsx` - Multi-sheet Excel workbook:
  - `all_results` — Full results table (all metrics per event and bin)
  - `p_penetration` — Penetration factors per event and bin (includes test_name)
  - `beta_other` — Other process rates per event and bin (includes test_name)
  - `beta_r_squared` — R² of forward Euler decay fit per event and bin (includes test_name)
  - `E_emission` — Emission rates per event and bin (includes test_name)
  - `E_total_particles` — Total emitted particle counts (E_total) per bin (includes test_name)
  - `E_r_squared` — R² of forward Euler emission-phase simulation per bin (includes test_name)
  - `peak_comparison` — Measured vs. predicted concentration at peak_time and deposition_end (wide format, one row per event)
- `output/plots/event_figures/pm_decay/event_NN-testname_pm_decay.png` - Individual event decay curves (four-panel)
  - *Top panel:* Measured concentration (solid) and continuous predicted Ct (dashed); shower ON/OFF dotted markers; optional outdoor PM overlay for events in `OUTDOOR_PM_EVENTS`
  - *Emission panels (×3):* Bins 0–2, 3–6, and 7–11; per-step E_t as faint lines; E_mean as dashed horizontal line; emission R² annotation; x-axes match concentration panel
- `output/plots/event_figures/excluded_events/` - Figures for duration-excluded (water-temperature-testing) events

- `output/plots/penetration_summary.png` - Bar chart summary of penetration factors by size
- `output/plots/deposition_summary.png` - Bar chart summary of other process rates by size
- `output/plots/emission_summary.png` - Bar chart summary of emission rates by size
- `output/plots/emission_etotal_boxplot_{bin0-2,bin3-6,bin7-11}.png` - E_total by water temperature (fixed temp axis, one figure per bin group)
- `output/plots/other_process_rate_boxplot_{bin0-2,bin3-6,bin7-11}.png` - β by water temperature (fixed temp axis)
- `output/plots/emission_rate_boxplot_{bin0-2,bin3-6,bin7-11}.png` - E_mean by water temperature (fixed temp axis)
- `output/plots/penetration_factor_boxplot_{bin0-2,bin3-6,bin7-11}.png` - p by water temperature (fixed temp axis)
- `output/plots/emission_etotal_by_bedroom_rh_boxplot_{bin0-2,bin3-6,bin7-11}.png` - E_total vs. bedroom RH (metric axis)
- `output/plots/emission_etotal_by_bedroom_temp_boxplot_{bin0-2,bin3-6,bin7-11}.png` - E_total vs. bedroom temperature (metric axis)
- `output/plots/emission_etotal_by_acr_boxplot_{bin0-2,bin3-6,bin7-11}.png` - E_total vs. air change rate (metric axis)
- `output/plots/emission_etotal_by_beta_boxplot_{bin0-2,bin3-6,bin7-11}.png` - E_total vs. other process rate (metric axis)
- `output/plots/emission_etotal_by_p_boxplot_{bin0-2,bin3-6,bin7-11}.png` - E_total vs. penetration factor (metric axis)
- `output/plots/emission_etotal_by_showerhead_boxplot_{bin0-2,bin3-6,bin7-11}.png` - E_total by shower head type (metric axis)

### Relative Humidity, Temperature & Wind Analysis

Run the environmental analysis to characterize conditions before and after shower events:

```bash
# Run RH/temperature/wind analysis (plots enabled by default)
python scripts/rh_temp_other_analysis.py

# Skip plots
python scripts/rh_temp_other_analysis.py --no-plots

# Plot only specific events
python scripts/rh_temp_other_analysis.py --events 1,3,5

# Limit time series plots to first 10 events
python scripts/rh_temp_other_analysis.py --max-plot-events 10
```

**Methodology:**
- Monitors RH and temperature at multiple locations:
  - Entry, Bedroom, Bathroom, Outside (Aranet4)
  - Bedroom, Bathroom, Living Room (DAQ/Vaisala)
  - Bathroom1, Bathroom2, Bath/Bed, Bedroom1, Bedroom2, Bedroom3 (HOBO UX100)
  - Note: QuantAQ `met_rh` / `met_temp` measure inside the flow cell and are excluded from RH/temp analysis
- Analyzes windspeed and direction from outdoor weather station (AIO2)
- Compares pre-shower baseline (30 min) to post-shower response (2 hours)
- Calculates mean, standard deviation, and min-max ranges
- Generates time series and box plot comparisons

**Output Files:**
- `output/rh_temp_wind_summary.xlsx` - Multi-sheet Excel workbook:
  - `RH_Summary`, `Temp_Summary`, `WindSpeed_Summary`, `WindDir_Summary` — per-sensor summary statistics
  - `RH_Details`, `Temp_Details`, `WindSpeed_Details`, `WindDir_Details` — per-event detail statistics
  - `Event_Log` — event metadata (test_name, config_key, shower times, door position, fan status)
  - `Bedroom_Conditions` — per-event bedroom RH and temperature (30-min pre-shower window); used by particle analysis for boxplot annotations; not rounded by sig figs
- `output/plots/event_figures/rh_timeseries/event_NN-testname_rh_timeseries.png` - RH time series per event
- `output/plots/event_figures/temperature_timeseries/event_NN-testname_temperature_timeseries.png` - Temperature per event
- `output/plots/event_figures/wind_timeseries/event_NN-testname_wind_timeseries.png` - Wind data per event
- `output/plots/event_figures/excluded_events/` - Time series figures for duration-excluded events
- `output/plots/rh_pre_post_boxplot.png` - Pre/post RH comparison across all events
- `output/plots/temperature_pre_post_boxplot.png` - Pre/post temperature comparison
- `output/plots/wind_speed_pre_post_boxplot.png` - Pre/post wind speed comparison
- `output/plots/wind_direction_pre_post_boxplot.png` - Pre/post wind direction comparison

## Dependencies

Key packages (see `epa_mh.yml` for complete list):
- pandas, numpy - Data manipulation
- scipy - Numerical integration (trapezoidal rule for E_total calculation)
- matplotlib - Publication-quality figure generation
- bokeh - Interactive visualization
- requests - API communication
- python-dotenv - Environment variable management (.env file loading)
- pyyaml - Configuration management
- openpyxl - Excel file I/O

## References

Instrument manuals, data sheets, and system documentation are located in the [`docs/`](docs/) directory.

### Aranet4 PRO (CO2 Sensor)

- [Aranet4 PRO Data Sheet](docs/Aranet_Datasheet_TDSPC003_Aranet4_PRO_1.pdf)
- [Aranet4 User Manual](docs/aranet4_user_manual_v25_web.pdf)

### Met One AIO 2 (Weather Sensor)

- [AIO 2 Manual (Rev G)](docs/AIO-2-9800-Manual-Rev-G.pdf)
- [AIO 2 Data Sheet](docs/AIO2-1.pdf)

### DAQ System (NI cDAQ-9178)

- [NI cDAQ-9171/9174/9178 User Manual](docs/NI%20cDAQ-9171_9174_9178%20User%20Manual%20-%20National%20Instruments.pdf)
- [NI 9201 Getting Started Guide](docs/ni-9201_getting_started_2-1-2024.pdf)
- [cDAQ Channel Map & Wiring Notes](docs/MH%20cDAQ%20channels%20MAP%20and%20wiring%20notes%20123115.xlsx)
- [Weather Station to DAQ Connections Sketch](docs/MH_weather_sta_to_DAQ_connections%20sketch.pdf)

### Setra Model 264 (Differential Pressure)

- [Setra Model 264 Data Sheet](docs/Setra_Model_264_Data_Sheet.pdf)

### Vaisala Humidity & Temperature Probes

- [HMP155 User Guide](docs/Vaisala%20HMP155_User_Guide_in_English.pdf)
- [HMP45A/D User Guide](docs/Vaisala%20HMP45AD-User-Guide-U274EN.pdf)

## Contact

- **PI:** Dustin G. Poppendieck
- **NIST Organizational Unit:** Engineering Laboratory
- **Division:** Building Energy & Environment Division
- **Group:** Indoor Air Quality and Ventilation Group
- **Email:** dustin.poppendieck@nist.gov

## Data Availability

The experimental data associated with this project are not included in this repository. Data can be made available upon request by contacting the PI.

## License

This software is in the public domain and provided "as is" according to the [LICENSE.md](LICENSE.md) file.

## Citation

If you use this software, please cite it as:

```bibtex
@software{nist_epa_legionella,
  author       = {Lima, Nathan M. and Poppendieck, Dustin G.},
  title        = {NIST-EPA Legionella Study Analysis Tools},
  year         = {2026},
  publisher    = {National Institute of Standards and Technology},
  url          = {https://github.com/usnistgov/NIST_EPA_Legionella}
}
```

## License

This software was developed by employees of the National Institute
of Standards and Technology (NIST), an agency of the Federal
Government and is being made available as a public service. Pursuant
to [Title 17 United States Code Section 105][usc-17], works of NIST
employees are not subject to copyright protection in the United
States. This software may be subject to foreign copyright. Permission
in the United States and in foreign countries, to the extent that
NIST may hold copyright, to use, copy, modify, create derivative
works, and distribute this software and its documentation without
fee is hereby granted on a non-exclusive basis, provided that this
notice and disclaimer of warranty appears in all copies.

See [LICENSE.md](LICENSE.md) for the full NIST licensing statement.

<!-- Link definitions -->

[18f-guide]: https://github.com/18F/open-source-guide/blob/18f-pages/pages/making-readmes-readable.md
[cornell-meta]: https://data.research.cornell.edu/content/readme
[gh-cdo]: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners
[gh-mdn]: https://github.github.com/gfm/
[gh-nst]: https://github.com/usnistgov
[gh-odi]: https://odiwiki.nist.gov/ODI/GitHub.html
[gh-osr]: https://github.com/usnistgov/opensource-repo/
[gh-ost]: https://github.com/orgs/usnistgov/teams/opensource-team
[gh-rob]: https://odiwiki.nist.gov/pub/ODI/GitHub/GHROB.pdf
[li-bsd]: https://opensource.org/licenses/bsd-license
[li-gpl]: https://opensource.org/licenses/gpl-license
[li-mit]: https://opensource.org/licenses/mit-license
[nist-code]: https://code.nist.gov
[nist-disclaimer]: https://www.nist.gov/open/license
[nist-s-1801-02]: https://inet.nist.gov/adlp/directives/review-data-intended-publication
[nist-open]: https://www.nist.gov/open/license#software
[usc-17]: https://www.copyright.gov/title17/
[wk-rdm]: https://en.wikipedia.org/wiki/README
