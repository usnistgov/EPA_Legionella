# NIST-EPA Legionella Study

Repository for NIST-EPA Legionella study Python tools. Includes scripts for aerosol particle size distribution analysis, emission-rate calculations, data cleaning, visualization, and statistical modeling of shower-generated aerosols under varying temperature, humidity, and exhaust-fan conditions.

## Project Overview

This project analyzes aerosol characteristics from shower-generated aerosols with a focus on potential Legionella transmission risks in water fixtures. The research investigates:

- Aerosol particle size distribution analysis
- Emission-rate calculations using numerical approaches
- CO2 decay analysis for air change rate determination
- Temperature and relative humidity monitoring
- Data cleaning, visualization, and statistical modeling

## Repository Structure

```
NIST_EPA_Legionella/
├── .vscode/                          # VS Code configuration
│   └── settings.json
├── .env                              # Environment variables (API keys for QuantAQ, gitignored)
├── .gitignore
├── README.md
├── data_config.json                  # Active configuration (gitignored)
├── data_config.template.json         # Configuration template
├── epa_mh.yml                        # Conda environment specification
│
├── src/                              # Core analysis modules
│   ├── __init__.py                   # Package initialization (re-exports data_paths)
│   ├── data_paths.py                 # Portable data access via data_config.json
│   ├── env_data_loader.py            # Unified environmental sensor data loader
│   ├── co2_decay_analysis.py         # CO2 decay & air-change rate (λ) analysis
│   ├── particle_decay_analysis.py    # Particle penetration, deposition & emission analysis
│   ├── rh_temp_other_analysis.py     # RH, temperature & wind analysis
│   └── deprecated/                   # Deprecated/archived code
│       └── co2_decay_analysis.py     # Previous version of CO2 analysis
│
├── scripts/                          # Executable scripts and utilities
│   │
│   │ # Data Download & Processing
│   ├── download_quantaq_data.py      # Download QuantAQ sensor data from API
│   ├── process_quantaq_data.py       # Process raw/final QuantAQ data
│   ├── quantaq_utils.py              # QuantAQ API client utilities
│   │
│   │ # Log Processing
│   ├── process_co2_log.py            # Consolidate daily CO2 injection logs → state-change log
│   ├── process_shower_log.py         # Consolidate daily shower logs → state-change log
│   ├── fix_log_files.py              # Repair corrupted/empty log files
│   ├── analyze_log_files.py          # Diagnose log file issues (unmatched events, etc.)
│   │
│   │ # Event Management
│   ├── event_manager.py              # Event filtering, naming (MMDD_Temp_ToD_RNN), logging
│   ├── event_matching.py             # Match CO2 injections to shower events by timing
│   ├── event_registry.py             # Unified event registry with synthetic event creation
│   │
│   │ # Visualization
│   ├── plot_style.py                 # Consistent matplotlib styling constants
│   ├── plot_co2.py                   # CO2 decay visualization functions
│   ├── plot_particle.py              # Particle decay visualization functions
│   ├── plot_environmental.py         # RH, temperature, wind visualization
│   └── plot_utils.py                 # Central plotting re-exports (imports all plot_* modules)
│
├── testing/                          # Testing and exploratory scripts
│   └── co2 plot.py                   # Aranet4 CO2 plotting script
│
└── docs/                             # Documentation
    ├── Aranet_Datasheet_TDSPC003_Aranet4_PRO_1.pdf
    ├── aranet4_user_manual_v25_web.pdf
    ├── Data Analysis.docx            # Data analysis planning notes
    ├── data_analysis_checklist.xlsx   # Task tracking for analysis milestones
    ├── IAQMH_instruments             # Instrument reference information
    └── python_script_style_prompt.txt # Coding style guidelines
```

## Installation

### Prerequisites

- Python 3.14+
- Conda (recommended for environment management)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/NathanL-CodeBase/NIST_EPA_Legionella.git
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

The repository includes standalone plotting modules for generating publication-quality figures:

- **`scripts/plot_co2.py`** - Visualize CO2 decay curves and air change rate analysis
- **`scripts/plot_particle.py`** - Plot particle concentration decay and emission rates
- **`scripts/plot_environmental.py`** - Generate RH, temperature, and wind time series
- **`scripts/plot_style.py`** - Consistent matplotlib styling across all figures
- **`scripts/plot_utils.py`** - Shared plotting utilities and helper functions

These can be imported and used by analysis modules or run independently for custom visualizations.

## Configured Instruments

| Instrument | Model | Purpose | Data Location |
|------------|-------|---------|---------------|
| AIO2 | Met One AIO 2 | Outdoor weather (10m tower) | MH DAQ/weather_station |
| Aranet4 | Aranet4 PRO | CO2 monitoring | CO2 |
| QuantAQ | MODULAIR-PM | PM monitoring (indoor/outdoor) | QuantAQ |
| Setra_264 | Setra 264 | Differential pressure | MH DAQ/indoor_daq |
| Vaisala_HMP155 | Vaisala HMP155 | RH/Temperature | MH DAQ/indoor_daq |
| Vaisala_HMP45A | Vaisala HMP45A | RH/Temperature | MH DAQ/indoor_daq |

### DAQ System

- **Chassis:** NI cDAQ-9178 (8-slot USB)
- **Modules:** 5x NI 9201 (8-channel analog input, +/-10V)

## Analysis Workflow

The complete data analysis pipeline follows this sequence:

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
   These scripts reduce ~86,400 records/day down to ~4-10 state-change records per day.

4. **Event Management**: Run the event manager to match, name, and register events
   ```bash
   python scripts/event_registry.py --force
   ```
   This must be run **before** the analysis scripts so that events have consistent names
   (see [Event Naming Convention](#event-naming-convention) below). Produces `event_log.csv` and `event_registry.csv`.

5. **CO2 Analysis**: Determine air change rates (λ) from CO2 decay
   ```bash
   python src/co2_decay_analysis.py
   python src/co2_decay_analysis.py --plot    # with visualization
   ```

6. **Environmental Analysis**: Characterize RH, temperature, and wind conditions
   ```bash
   python src/rh_temp_other_analysis.py
   python src/rh_temp_other_analysis.py --plot
   ```

7. **Particle Analysis**: Calculate penetration, deposition, and emission rates (requires step 5 results)
   ```bash
   python src/particle_decay_analysis.py
   python src/particle_decay_analysis.py --plot
   ```

Each analysis module produces CSV/Excel summaries and optional visualizations in the `output/` directory.

## Event Naming Convention

Events are named using a structured format generated by `scripts/event_manager.py`. This naming convention must be applied (step 4 in the workflow above) **before** running the analysis scripts.

**Format:** `MMDD_TempCode_TimeOfDay_RNN`

| Component | Description | Values |
|-----------|-------------|--------|
| `MMDD` | Month and day of the event | e.g., `0114` for January 14 |
| `TempCode` | Water temperature setting | `HW` (Hot Water), `CW` (Cold Water), `Mixed` |
| `TimeOfDay` | Time classification | `Day` (5 AM - 5 PM), `Night` (5 PM - 5 AM) |
| `RNN` | Replicate number | `R01`, `R02`, `R03`, etc. |

**Examples:**
- `0114_HW_Day_R01` - January 14, hot water, daytime, replicate 1
- `0122_CW_Night_R03` - January 22, cold water, nighttime, replicate 3

**Test Parameter Timeline:**
- Hot water: From experiment start (2026-01-14)
- Cold water: Starting 2026-01-22 14:00
- Mixed: Starting 2026-02-02 17:00

## Data Analysis Modules

### CO2 Air-Change Rate Analysis

Run the CO2 decay analysis to calculate air-change rates (λ):

```bash
# Basic analysis
python src/co2_decay_analysis.py

# With plots
python src/co2_decay_analysis.py --plot

# Custom parameters
python src/co2_decay_analysis.py --alpha 0.6 --beta 0.4 --plot
```

**Methodology:**
- Combines three Aranet4 sensor files: Bedroom, Entry, and Outside
- Applies 6-minute rolling average to reduce sensor noise
- Accounts for time-varying CO2 concentrations at all locations
- Dynamically determines decay end when C_bedroom within 200 ppm of C_outside (max 2 hours)
- Calculates λ using numerical differentiation: λ = -dC/dt / (C_source - C_bedroom)
- Starts decay calculation 10 minutes prior to the top of the hour

**Output Files:**
- `output/co2_lambda_summary.csv` - Per-event lambda results
- `output/co2_lambda_overall_summary.csv` - Overall statistics
- `output/plots/event_XX_decay.png` - Individual event plots with fitted decay curves
- `output/plots/lambda_summary.png` - Summary bar chart of all events

### Particle Decay & Emission Analysis

Run the particle decay analysis to calculate penetration factors, deposition rates, and emission rates:

```bash
# Run particle analysis (requires CO2 analysis results)
python src/particle_decay_analysis.py

# With custom plotting options
python src/particle_decay_analysis.py --plot
```

**Methodology:**
- Analyzes 7 particle size bins: 0.35-0.46, 0.46-0.66, 0.66-1.0, 1.0-1.3, 1.3-1.7, 1.7-2.3, 2.3-3.0 µm
- Combines indoor and outdoor QuantAQ MODULAIR-PM sensor data
- Calculates penetration factor (p) from 1-hour pre-shower window
- Uses air change rate (λ) from CO2 decay analysis results
- Determines deposition loss rate (β) from 2-hour post-shower decay
- Calculates emission rates (E) during 10-minute shower periods
- Numerical solution of time-dependent mass balance equation

**Output Files:**
- `output/particle_summary.csv` - Per-event, per-bin results (p, β, E)
- `output/particle_overall_summary.csv` - Statistical summaries across all events
- `output/plots/event_XX_decay.png` - Decay curves for all bins per event
  - Solid lines indicate valid analysis (complete p, β, E data)
  - Dashed lines indicate invalid/incomplete analysis
  - Markers show penetration window start, shower ON/OFF, deposition window end

### Relative Humidity, Temperature & Wind Analysis

Run the environmental analysis to characterize conditions before and after shower events:

```bash
# Run RH/temperature/wind analysis
python src/rh_temp_other_analysis.py

# With visualization
python src/rh_temp_other_analysis.py --plot
```

**Methodology:**
- Monitors RH and temperature at multiple locations:
  - Entry, Bedroom, Outside (Aranet4)
  - Bedroom, Outside (QuantAQ)
  - Bedroom, Bathroom, Family/Living, Outside (DAQ/Vaisala)
- Analyzes windspeed and direction from outdoor weather station (AIO2)
- Compares pre-shower baseline (30 min) to post-shower response (2 hours)
- Calculates mean, standard deviation, and min-max ranges
- Generates time series and box plot comparisons

**Output Files:**
- `output/rh_temp_wind_summary.xlsx` - Multi-sheet Excel workbook with all statistics
- `output/plots/event_XX_rh_timeseries.png` - RH time series per event
- `output/plots/event_XX_temp_timeseries.png` - Temperature time series per event
- `output/plots/event_XX_wind_timeseries.png` - Wind data time series per event
- `output/plots/rh_pre_post_boxplot.png` - Pre/post RH comparison across all events
- `output/plots/temp_pre_post_boxplot.png` - Pre/post temperature comparison
- `output/plots/wind_pre_post_boxplot.png` - Pre/post wind comparison

## Dependencies

Key packages (see `epa_mh.yml` for complete list):
- pandas, numpy - Data manipulation
- scipy - Statistical analysis (linear regression for decay fitting)
- matplotlib - Publication-quality figure generation
- bokeh - Interactive visualization
- requests - API communication
- python-dotenv - Environment variable management (.env file loading)
- pyyaml - Configuration management
- openpyxl - Excel file I/O
