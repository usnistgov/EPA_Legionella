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
├── src/                              # Source modules
│   ├── __init__.py                   # Package initialization
│   ├── data_paths.py                 # Core data access utilities
│   ├── env_data_loader.py            # Environmental data loader
│   ├── co2_decay_analysis.py         # CO2 decay & air-change rate analysis
│   ├── particle_decay_analysis.py    # Particle penetration & emission analysis
│   ├── rh_temp_other_analysis.py     # RH, temperature & wind analysis
│   └── deprecated/                   # Deprecated/archived code
│       └── co2_decay_analysis.py     # Previous version of CO2 analysis
│
├── scripts/                          # Executable scripts
│   ├── download_quantaq_data.py      # Download QuantAQ sensor data
│   ├── process_quantaq_data.py       # Process raw/final QuantAQ data
│   ├── process_co2_log.py            # Process CO2 injection log
│   ├── process_shower_log.py         # Process shower event log
│   ├── fix_log_files.py              # Utility to fix log file formatting
│   ├── quantaq_utils.py              # QuantAQ API utilities
│   ├── plot_co2.py                   # CO2 decay visualization
│   ├── plot_particle.py              # Particle decay visualization
│   ├── plot_environmental.py         # RH, temperature, wind visualization
│   ├── plot_style.py                 # Consistent plot styling
│   ├── plot_utils.py                 # Plotting utilities
│   └── example_data_access.py        # Example usage of data utilities
│
├── testing/                          # Testing and exploratory scripts
│   └── co2 plot.py                   # Aranet4 CO2 plotting script
│
└── docs/                             # Documentation
    ├── Aranet_Datasheet_TDSPC003_Aranet4_PRO_1.pdf
    ├── aranet4_user_manual_v25_web.pdf
    ├── Data Analysis.docx            # Data analysis planning notes
    ├── data_analysis_checklist.xlsx  # Task tracking for analysis milestones
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

See `scripts/example_data_access.py` for more usage examples.

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

1. **Data Collection**: Download sensor data using [scripts/download_quantaq_data.py](scripts/download_quantaq_data.py)
2. **Data Processing**: Process and merge raw sensor data using [scripts/process_quantaq_data.py](scripts/process_quantaq_data.py)
3. **Event Logging**: Ensure CO2 injection and shower event logs are processed
4. **CO2 Analysis**: Run [src/co2_decay_analysis.py](src/co2_decay_analysis.py) to determine air change rates (λ)
5. **Particle Analysis**: Run [src/particle_decay_analysis.py](src/particle_decay_analysis.py) to calculate penetration, deposition, and emission
6. **Environmental Analysis**: Run [src/rh_temp_other_analysis.py](src/rh_temp_other_analysis.py) to characterize RH, temperature, and wind conditions

Each analysis module produces CSV summaries and optional visualizations in the `output/` directory.

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
- matplotlib - Publication-quality figure generation
- bokeh - Interactive visualization
- requests - API communication
- pyyaml - Configuration management
- openpyxl - Excel file I/O
