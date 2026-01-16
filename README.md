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
├── epa_mh.yaml                       # Conda environment specification
│
├── src/                              # Source modules
│   ├── __init__.py                   # Package initialization
│   └── data_paths.py                 # Core data access utilities
│
├── scripts/                          # Executable scripts
│   ├── download_quantaq_data.py      # Download QuantAQ sensor data
│   ├── process_quantaq_data.py       # Process raw/final QuantAQ data
│   ├── quantaq_utils.py              # QuantAQ API utilities
│   └── example_data_access.py        # Example usage of data utilities
│
├── testing/                          # Testing and exploratory scripts
│   └── co2 plot.py                   # Aranet4 CO2 plotting script
│
└── docs/                             # Documentation
    ├── Aranet_Datasheet_TDSPC003_Aranet4_PRO_1.pdf
    ├── aranet4_user_manual_v25_web.pdf
    ├── Data Analysis.docx            # Data analysis planning notes
    └── IAQMH_instruments             # Link to instrument catalog
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
   conda env create -f epa_mh.yaml
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

## Data Analysis Plan

### CO2 Analysis

Analyze CO2 decay using a numerical approach to determine air change rates:
- Combine three Aranet data files: Bedroom, Entry, and Outside
- Account for time-dependent CO2 concentrations at all locations
- Calculate average and standard deviation of lambda for 2 hours post-injection
- Start decay calculation 10 minutes prior to the top of the hour

### QuantAQ Particle Analysis

Analyze particle decay in bin sizes (0.35-0.46, 0.46-0.66, 0.66-1.0, 1.0-1.3, 1.3-1.7, 1.7-2.3, 2.3-3 um):
- Combine indoor and outdoor QuantAQ sensor data
- Solve for particle penetration factor (P) from C/Cout ratio one hour prior to shower
- Use air change rate (lambda) from CO2 analysis
- Determine deposition loss rate (beta) when emission is zero
- Calculate emission rates during 10-minute shower periods

### Relative Humidity Analysis

Monitor RH at multiple locations using Aranet, QuantAQ, and DAQ sensors:
- Entry, Bedroom, Outside (Aranet)
- Bedroom, Outside (QuantAQ)
- Bedroom, Bathroom, Family/Living, Outside (DAQ)

Report for each location:
- Initial average and standard deviation (30 min prior to shower)
- Average and standard deviation for 2-hour period after shower
- Min/max difference for post-shower period

### Temperature Analysis

Same locations as RH analysis. Report:
- Initial average and standard deviation (30 min prior to shower)
- Average and standard deviation for 2-hour period after shower

### Other Environmental Data

From DAQ: Windspeed and Direction
- Report average and standard deviation for 2-hour period after shower

## Dependencies

Key packages (see `epa_mh.yaml` for complete list):
- pandas, numpy - Data manipulation
- bokeh - Interactive visualization
- requests - API communication
- pyyaml - Configuration management
- openpyxl - Excel file I/O
