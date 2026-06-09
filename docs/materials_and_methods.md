# Materials and Methods

> **Author note:** Section numbers, table numbers, figure references, and items marked
> `[PLACEHOLDER: ...]` require PI review and confirmation before submission. The analysis
> start date used throughout is January 14, 2026 (first W48 test day); the experiment
> start cutoff enforced in software is January 15, 2026 at 15:00.

---

## 2.1 Test Facility

### 2.1.1 Building Description

The study was conducted in a single-story, double-wide manufactured home located on the campus of the National Institute of Standards and Technology (NIST) in Gaithersburg, Maryland. The home has a total interior floor area of approximately 140 m² (1,510 ft²) and contains three bedrooms, two full bathrooms, a family room, living room, kitchen, dining area, morning room, utility room, and an attached garage (Fig. 2-1). The home is supplied with NIST campus municipal water and heated by a standard residential electric hot water heater [PLACEHOLDER: confirm heater type, capacity, and recovery rate]. The building envelope is light-wood-frame construction typical of HUD-code manufactured housing. [PLACEHOLDER: describe insulation levels, window construction, and any known infiltration characteristics if relevant to study context.]

The home is served by a forced-air HVAC system controlled by an Ecobee smart thermostat. [PLACEHOLDER: describe HVAC operating mode during testing — setpoint(s), blower mode (auto vs. continuous), and whether the system was modified or disabled during any test periods. Note whether supply registers in Bedroom #1 or Bathroom #1 were open or closed.]

### 2.1.2 Test Rooms

The test area comprised two adjacent rooms: Bathroom #1 and Bedroom #1 (Fig. 2-1).

**Bathroom #1** has a floor area of 7.09 m² (76.33 ft²) and is located in the north-central section of the home. It contains a [PLACEHOLDER: describe the shower fixture — e.g., fiberglass tub/shower combination or dedicated shower stall; approximate interior shower dimensions]. The bathroom is equipped with a ceiling-mounted exhaust fan [PLACEHOLDER: confirm fan manufacturer, model, and rated airflow in CFM/L·s⁻¹]. A linen closet (0.72 m², 7.73 ft²) and a utility room (3.66 m², 39.39 ft²) are adjacent to the bathroom on its west side.

**Bedroom #1** has a floor area of 14.71 m² (158.31 ft²) and a volume of 36.1 m³ as determined from [PLACEHOLDER: confirm source — architectural CAD drawings or direct measurement]. The ceiling height is approximately 2.45 m (8.0 ft), consistent with HUD-code manufactured housing standards. The bedroom is directly adjacent to Bathroom #1, sharing a common wall, and the two rooms are connected by an interior [PLACEHOLDER: solid-core / hollow-core] door nominally [PLACEHOLDER: door width × height, mm]. A separate interior door on the east wall of the bedroom opens to the main hallway. [PLACEHOLDER: Note the presence, size, and state of any HVAC supply or return registers, windows, or other openings in Bedroom #1 during testing.]

A walk-in closet (2.37 m², 25.5 ft²) is attached to the south side of Bedroom #1.

---

## 2.2 Experimental Design

Seven independent experimental variables were systematically varied across the test period (Table 2-1). Variables were changed sequentially rather than in a randomized or full-factorial design; each unique combination held constant for one or more complete shower events is termed a *configuration*. All shower events within a configuration are treated as replicates.

**Table 2-1. Independent experimental variables and levels investigated.**

| Variable | Levels | Values |
|---|---|---|
| Water temperature | 14 | 11, 14, 22, 23, 25, 30, 37, 38, 40, 43, 48, 49, 52, 53 °C |
| Shower head type | 4 | Standard, Pepco, FilterWand, Used |
| Spray pattern | Up to 4 per head | Wide, Narrow, Mid (Pepco); Rainfall, 12-Nozzle, SingleWide1, SingleWide2 (Used) |
| Mannequin presence | 2 | Present, Absent |
| Bathroom door position | 3 | Open, Ajar, Closed |
| Bedroom door position | 2 | Closed, Ajar |
| Bathroom exhaust fan | 2 | Off; On for 12 min from shower onset |

### 2.2.1 Water Temperature

Water temperature was varied across 14 set points ranging from 11 °C (unheated municipal supply) to 53 °C (very hot). Temperature was controlled by adjusting the residential water heater setpoint and, where necessary, [PLACEHOLDER: cold water mixing valve adjustment or faucet blending]. Temperature was verified at the showerhead outlet using [PLACEHOLDER: instrument model, e.g., calibrated thermocouple or digital thermometer] with the shower running for a stabilization period of [PLACEHOLDER: X minutes] before each measurement. Verification was performed at the beginning and end of each configuration change [PLACEHOLDER: confirm whether temperature was also checked before individual events or only at configuration boundaries].

### 2.2.2 Shower Head Types and Spray Patterns

Four shower head types were evaluated:

- **Standard:** A [PLACEHOLDER: brand/model] residential shower head with a single fixed spray pattern.
- **Pepco:** A [PLACEHOLDER: brand/model] shower head with three selectable spray patterns — Wide (broad, diffuse spray), Narrow (concentrated jet), and Mid (intermediate).
- **FilterWand:** A [PLACEHOLDER: brand/model] filtered shower wand. [PLACEHOLDER: describe spray pattern and any filtration specification.]
- **Used:** A [PLACEHOLDER: brand/model] shower head with multiple spray patterns — Rainfall, 12-Nozzle, SingleWide1, and SingleWide2. [PLACEHOLDER: describe the condition and any aging characteristics of the used head.]

[PLACEHOLDER: Provide rated flow rates (L/min at standard pressure) and any flow-restrictor specifications for each head type.]

### 2.2.3 Mannequin

A human-analog mannequin [PLACEHOLDER: describe material, approximate size/weight, posture (seated or standing), and placement within the shower enclosure] was used for selected configurations to evaluate the influence of a body-analog on aerosol generation and transport.

### 2.2.4 Door and Fan Configurations

The bathroom door (between Bathroom #1 and Bedroom #1) was tested in three positions: fully open, fully closed, and ajar [PLACEHOLDER: define "ajar" — approximate angle or gap width at the latch side]. The bedroom door (between Bedroom #1 and the main hallway) was tested in two positions: fully closed and ajar [PLACEHOLDER: define "ajar"].

The bathroom exhaust fan was tested in two states: off (no mechanical exhaust ventilation) and on. When active, the fan was operated for 12 minutes beginning at shower onset and controlled by the automated test system (Section 2.3.1).

### 2.2.5 Configuration Naming Convention

Each configuration is identified by a structured key of the form:

```
W##[_HeadType[_SprayPattern]][_Mannequin]_BathDoorXxx_BdrmDoorXxx_FanXxx[_FlowRateX.XLPM]
```

where `W##` is the water temperature code in °C (e.g., `W40` for 40 °C), `HeadType` and `SprayPattern` identify the shower head type and spray pattern setting, `_Mannequin` is appended when the mannequin was present, `BathDoorXxx` and `BdrmDoorXxx` specify bathroom and bedroom door positions (Open, Closed, or Ajar), and `FanXxx` specifies fan state (On or Off). The suffix `_FlowRateX.XLPM` is appended only when the measured flow rate falls outside the standard analysis range (4.1–5.6 L/min) so that atypical flow rate events can be distinguished within an otherwise identical configuration.

Individual shower events within a configuration are assigned a sequential event number and a structured test name:

```
MMDD_W##[_Head[_Pattern]][_Mannequin]_DoorPos[_Fan]_RNN
```

where `MMDD` is the calendar date, `DoorPos` is an abbreviated door position code, and `RNN` is the replicate index within that configuration (e.g., R01, R02, ...).

### 2.2.6 Test Timeline and Configuration Transitions

The test period began January 14, 2026. Table 2-2 summarizes the major configuration transitions in chronological order. Variables not listed for a given row were unchanged from the preceding entry.

**Table 2-2. Chronological summary of major experimental configuration transitions.**

| Date | Change | Resulting Configuration (abbreviated) |
|---|---|---|
| 2026-01-14 | Experiment begins | W48, Standard, bath door open, bedroom door closed, fan off |
| 2026-01-22 | Water temperature | W11 |
| 2026-02-02 | Water temperature | W25 |
| 2026-02-05 | Water temperature | W30 |
| 2026-02-09 | Water temperature | W37 |
| 2026-02-13 | Water temperature | W22 |
| 2026-02-16 | Water temperature | W43 |
| 2026-02-18 | Water temperature | W14 |
| 2026-02-20 | Water temperature | W53 |
| 2026-02-24 | Shower head changed; water temperature | W52, Pepco, Wide spray |
| 2026-02-26 | Water temperature | W49, Pepco, Wide |
| 2026-03-02 | Water temperature; spray pattern | W40, Pepco, Narrow |
| 2026-03-04 | Spray pattern | W40, Pepco, Wide |
| 2026-03-06 | Spray pattern | W40, Pepco, Mid |
| 2026-03-09 | Spray pattern | W40, Pepco, Narrow |
| 2026-03-11 | Mannequin introduced | W40, Pepco, Narrow, Mannequin |
| 2026-03-12 | Mannequin removed | W40, Pepco, Narrow |
| 2026-03-13 | Mannequin reintroduced | W40, Pepco, Narrow, Mannequin |
| 2026-03-17 | Mannequin removed | W40, Pepco, Narrow |
| 2026-03-18 | Mannequin reintroduced | W40, Pepco, Narrow, Mannequin |
| 2026-03-19 | Spray pattern; mannequin removed | W40, Pepco, Wide |
| 2026-03-22 | Mannequin reintroduced | W40, Pepco, Wide, Mannequin |
| 2026-03-27 | Mannequin removed | W40, Pepco, Wide |
| 2026-03-31 | Exhaust fan activated | W40, Pepco, Wide, Fan on (12 min) |
| 2026-04-07 | Fan deactivated | W40, Pepco, Wide, Fan off |
| 2026-04-08 | Bathroom door closed | W40, Pepco, Wide, bath door closed |
| 2026-04-10 | Shower head changed; bathroom door reopened | W40, FilterWand, bath door open |
| 2026-04-13 | Shower head changed | W40, Used (Rainfall) |
| 2026-04-23 | Shower head reinstalled; mannequin introduced | W40, Pepco, Mannequin |
| 2026-04-29 | Mannequin removed | W40, Pepco |
| 2026-05-01 | Bedroom door ajar (unintended) | W40, Pepco, bedroom door ajar |
| 2026-05-04 | Bathroom door ajar | W40, Pepco, bath door ajar |
| 2026-05-07 | Both doors returned to baseline positions | W40, Pepco, bath door open, bedroom door closed |
| 2026-05-22 | Shower head changed | W40, FilterWand |
| 2026-05-26 | Shower head changed | W40, Used |

Flow rate was measured at [PLACEHOLDER: all configuration changes, or only major changes?] and ranged from approximately 1.4 to 5.7 L/min across all head types and settings. The effective measurement range for summary analysis was 4.1–5.6 L/min (standard configurations); events outside this range are identified in the event registry.

---

## 2.3 Shower Test Protocol

### 2.3.1 Automated Control System

The shower, bathroom exhaust fan, CO₂ tracer gas injection valve, and CO₂ mixing fan were all actuated by a computer-based control system [PLACEHOLDER: describe control hardware and software — e.g., NI cDAQ relay outputs, LabVIEW state machine, Python script with USB relay board]. Automation ensured consistent event timing across all replicate tests and eliminated operator-induced variability in shower start and stop times.

### 2.3.2 Hourly Test Cycle

All shower events began at the top of the hour (:00) and lasted exactly 10.0 minutes, controlled to ±5 seconds by the automated system. Events were conducted once per hour under steady conditions. The full hourly cycle proceeded as follows (times given as minutes past the start of the preceding hour):

| Time (min past hour) | Action |
|---|---|
| :40 | CO₂ tracer gas injection begins (Bedroom #1, northwest corner) |
| :44 (or :46 from 2026-01-22 onward) | CO₂ injection ends |
| :45 | CO₂ mixing fan stops; equilibration period begins |
| :00 (next hour) | Shower turns on; exhaust fan activates (where applicable) |
| :10 | Shower turns off (10-minute shower complete) |
| :12 | Exhaust fan turns off (12-minute run, where applicable) |
| :00 + 2 h 10 min | End of analysis window |

CO₂ injection began 20 minutes before shower onset (:40 past the hour preceding the shower). The initial injection duration was 4 minutes (through January 21, 2026); from January 22, 2026 onward the duration was extended to 6 minutes to increase the initial bedroom CO₂ concentration and improve the signal-to-noise ratio of the decay measurement. In both cases, the mixing fan stopped approximately 13–15 minutes before shower onset [PLACEHOLDER: confirm exact fan stop time for the 6-minute injection protocol], providing a quiet equilibration period to promote a spatially uniform tracer concentration in the bedroom before aerosol transport from the shower began.

The 10-minute shower duration was enforced by the automated control system. Any shower event with a measured duration outside the range 9 min 55 s to 10 min 5 s (i.e., 10.0 min ± 5 s) was treated as a water-temperature verification or control-system test event and excluded from analysis (Section 3.2.4).

### 2.3.3 Water Temperature and Flow Rate Verification

Water temperature and volumetric flow rate were measured manually at the beginning and end of each configuration change to verify the setpoint and to document any drift. Temperature was measured at [PLACEHOLDER: specify measurement location — e.g., the showerhead outlet, with shower running for X minutes to stabilize] using [PLACEHOLDER: instrument model and calibration status]. Flow rate was measured by [PLACEHOLDER: describe method — e.g., timed collection into a calibrated bucket, or inline flow meter]. The measured flow rates are recorded in the event registry and are used to identify events where the flow rate deviates from the standard analysis range (4.1–5.6 L/min).

---

## 2.4 Carbon Dioxide Tracer Gas System

### 2.4.1 Equipment

Air change rate between Bedroom #1 and the outdoor environment was determined using CO₂ as an inert tracer gas. The system consisted of a [PLACEHOLDER: cylinder size and purity — e.g., standard K-cylinder (49 L water capacity), industrial grade CO₂ ≥ 99.5 % purity] compressed CO₂ cylinder connected through a pressure regulator and a computer-controlled mass flow controller [PLACEHOLDER: manufacturer, model number, flow range, and setpoint used]. The supply line terminated in the northwest corner of Bedroom #1 at a height of approximately [PLACEHOLDER: injection height above floor, m]. A [PLACEHOLDER: describe mixing fan — manufacturer, model, blade diameter, mounting location within bedroom] mixing fan was co-located in the bedroom to promote rapid, uniform distribution of the injected CO₂.

### 2.4.2 Injection and Mixing Protocol

CO₂ was injected at :40 past the hour preceding each shower, 20 minutes before shower onset. The injection duration was 4 minutes in the initial test period (through January 21, 2026) and 6 minutes from January 22, 2026 onward. The mixing fan operated concurrently with injection and for approximately 1 minute after injection ceased [PLACEHOLDER: confirm exact fan run time for the extended injection protocol]. The fan stopped at approximately :45, initiating a quiet period of approximately 13–15 minutes that allowed the tracer gas to approach a spatially uniform concentration in the bedroom before shower-induced airflow began.

---

## 2.5 Instrumentation

### 2.5.1 Particle Counters (QuantAQ MODULAIR-PM)

Particle number concentrations were measured using two QuantAQ MODULAIR-PM optical particle counters (QuantAQ, Inc., Somerville, MA): one deployed indoors (serial number MOD-PM-00195, Bedroom #1) and one deployed outdoors (serial number MOD-PM-00785). Each unit contains an Alphasense OPC-N3 optical particle counter that measures particle number concentration in 24 size bins spanning 0.35 µm to 40 µm. This study used the 12 lowest bins (0.35–10.0 µm; Table 2-3), which encompass the aerosol size range most relevant to respiratory deposition in the tracheobronchial and alveolar regions and to Legionella transmission risk. The MODULAIR-PM also measures temperature and relative humidity within the instrument's internal flow cell (reported as `met_temp` and `met_rh`); these are not representative of ambient conditions and are excluded from all environmental analyses.

Data were acquired at approximately 1-minute intervals and retrieved from the QuantAQ cloud API in weekly chunks for local archiving and processing.

**Table 2-3. Alphasense OPC-N3 size bins analyzed in this study.**

| Bin | Lower bound (µm) | Upper bound (µm) | Reporting group |
|---|---|---|---|
| 0 | 0.35 | 0.46 | Fine (bins 0–2) |
| 1 | 0.46 | 0.66 | Fine (bins 0–2) |
| 2 | 0.66 | 1.0 | Fine (bins 0–2) |
| 3 | 1.0 | 1.3 | Accumulation/Coarse (bins 3–6) |
| 4 | 1.3 | 1.7 | Accumulation/Coarse (bins 3–6) |
| 5 | 1.7 | 2.3 | Accumulation/Coarse (bins 3–6) |
| 6 | 2.3 | 3.0 | Accumulation/Coarse (bins 3–6) |
| 7 | 3.0 | 4.0 | Coarse (bins 7–11) |
| 8 | 4.0 | 5.2 | Coarse (bins 7–11) |
| 9 | 5.2 | 6.5 | Coarse (bins 7–11) |
| 10 | 6.5 | 8.0 | Coarse (bins 7–11) |
| 11 | 8.0 | 10.0 | Coarse (bins 7–11) |

The indoor sensor (MOD-PM-00195) was mounted at approximately bed height (nominally ~0.5 m above the floor) near the center of the south wall of Bedroom #1 [PLACEHOLDER: confirm exact mounting hardware and exact height]. The outdoor sensor (MOD-PM-00785) was located [PLACEHOLDER: describe outdoor placement — distance from house, height above grade, proximity to 10-m met tower].

### 2.5.2 Carbon Dioxide, Temperature, and Relative Humidity (Aranet4 PRO)

CO₂ concentration, air temperature, and relative humidity were measured at four locations using Aranet4 PRO wireless sensors (SAF Tehnika, Riga, Latvia):

| Designation | Location | Purpose |
|---|---|---|
| Aranet4 Bedroom | Bedroom #1 | Primary CO₂ decay tracer zone |
| Aranet4 Bathroom | Bathroom #1 | Environmental context (CO₂ not used in λ) |
| Aranet4 Entry | Building entry / hallway | Background CO₂ reference for decay model |
| Aranet4 Outside | Outdoors | Outdoor CO₂ and ambient reference |

Each sensor uses a non-dispersive infrared (NDIR) detector for CO₂ measurement (manufacturer-stated accuracy: ±50 ppm ± 3 % of reading) and capacitive sensors for temperature and RH. Data were recorded at 1-minute intervals and exported from the Aranet PRO base station as Excel files. The Bathroom Aranet4 was installed in March 2026; for earlier events, Bathroom #1 conditions are characterized using the co-located HOBO UX100 and Vaisala sensors (Sections 2.5.3 and 2.5.4).

All Aranet4 sensors were mounted at approximately 1.2 m (4 ft) above the floor [PLACEHOLDER: confirm exact placement and mounting method for each location].

### 2.5.3 Temperature and Relative Humidity (HOBO UX100 Data Loggers)

Temperature and relative humidity were measured continuously using six HOBO UX100 data loggers (Onset Computer Corporation, Bourne, MA) at the locations listed in Table 2-4. Two models were deployed: UX100-011A (two-channel logger with external probe) and UX100-011 (single integrated probe). Raw temperatures were recorded in °F and converted to °C during processing. The factory-calibration offsets listed in Table 2-4 were applied programmatically to all data.

**Table 2-4. HOBO UX100 data logger deployment and factory calibration offsets.**

| Logger ID | Location | Model | Serial Number | Temp. offset (°F) | RH offset (%) |
|---|---|---|---|---|---|
| MB_D (also MB_G†) | Bathroom #1 (primary) | UX100-011A | 21904275 | −0.04 | −0.55 |
| MB_Bath | Bathroom #1 (secondary) | UX100-011 | 10355906 | +0.13 | +0.57 |
| MB_E | Bathroom–Bedroom doorway | UX100-011 | 20244192 | −0.12 | +0.50 |
| MB_Bed | Bedroom #1 (primary) | UX100-011A | 21904272 | −0.23 | −0.47 |
| MB_F | Bedroom #1 (secondary) | UX100-011 | — | +0.24 | −0.11 |
| MB_C | Bedroom #1 (tertiary) | UX100-011 | — | +0.03 | +0.05 |

† MB_G is the early-period filename designation for logger MB_D (same serial number 21904275); records from both designations are merged during data processing.

Loggers recorded at [PLACEHOLDER: confirm HOBO logging interval — e.g., 5-minute or 1-minute] intervals. [PLACEHOLDER: Confirm exact placement height and mounting method within each room.]

### 2.5.4 Temperature and Relative Humidity (Vaisala HMP Series, Continuous DAQ)

Two Vaisala probe series provided continuous temperature and RH data via the NI cDAQ-9178 data acquisition chassis (Section 2.5.7).

**Vaisala HMP155** probes (Vaisala Oyj, Helsinki, Finland) were deployed in Bedroom #1 (designated Bed1) and the living room (designated Liv), with each probe providing one RH channel and one temperature channel. These probes output 0–10 V DC, require 24 V DC excitation, and were factory calibrated on May 11, 2023. Per the floor plan instrument legend, probes were mounted at approximately 1.2 m (4 ft) above the floor.

**Vaisala HMP45A** probes were deployed at multiple locations throughout the home, including Bathroom #1 (designated MBa). These probes output 0–1 V DC and require 7–35 V DC excitation. [PLACEHOLDER: Confirm HMP45A calibration dates.]

Both probe series were sampled continuously by the NI cDAQ-9178 at [PLACEHOLDER: confirm DAQ scan rate — e.g., 1 sample s⁻¹ or 5 s⁻¹]. Analog voltages were converted to engineering units using manufacturer-supplied calibration transfer functions.

### 2.5.5 Outdoor Meteorology (Met One AIO2)

Outdoor wind speed, wind direction, ambient temperature, relative humidity, and barometric pressure were measured by a Met One AIO2 all-in-one weather sensor (Met One Instruments, Inc., Grants Pass, OR) mounted atop a 10-meter meteorological tower located [PLACEHOLDER: describe tower location relative to the manufactured home — cardinal direction and approximate distance]. The AIO2 communicates via RS-232 serial to a dedicated logger operating independently of the NI cDAQ system. Data were recorded at [PLACEHOLDER: confirm AIO2 logging interval — e.g., 1-minute averages].

### 2.5.6 Differential Pressure (Setra 264)

A Setra Model 264 differential pressure transducer array (Setra Systems, Inc., Boxborough, MA) was installed to measure pressure differences across the building envelope and between building zones. Eleven channels (0–5 V DC output, 0–[PLACEHOLDER: Pa range] range) were wired to the NI cDAQ-9178 chassis (NI 9201 analog input modules, slots 1–2). The Setra 264 array was disconnected [PLACEHOLDER: confirm when the array was disconnected and whether any DP data were collected prior to disconnection]. Differential pressure measurements are not reported in this study.

### 2.5.7 Data Acquisition System

Analog sensors (Vaisala HMP155, Vaisala HMP45A, Setra 264) were sampled by a National Instruments CompactDAQ-9178 chassis (National Instruments, Austin, TX). The cDAQ-9178 is an 8-slot USB chassis rated for 9–30 V DC input at up to 15 W. Five NI 9201 analog input modules (8-channel each, ±10 V input range, simultaneous sampling) occupied slots 1–5; slots 6–8 were unoccupied. The NI 9201 modules are specified for ±0.04 % full-scale accuracy at 25 °C. [PLACEHOLDER: Confirm DAQ scan rate and any software averaging applied before writing to file.]

---

## 2.6 Sensor Placement Summary

Table 2-5 summarizes the measurement locations, approximate heights, and deployment periods for all sensors active during the study.

**Table 2-5. Sensor placement summary.**

| Sensor | Location | Approx. height (m) | Notes |
|---|---|---|---|
| QuantAQ MOD-PM-00195 | Bedroom #1, south wall center | ~0.5 (bed height) | Indoor particle counter |
| QuantAQ MOD-PM-00785 | Outdoors | [PLACEHOLDER] | Outdoor particle counter |
| Aranet4 Bedroom | Bedroom #1 | ~1.2 | CO₂ / RH / Temp |
| Aranet4 Bathroom | Bathroom #1 | ~1.2 | From March 2026; CO₂ context only |
| Aranet4 Entry | Building entry / hallway | ~1.2 | CO₂ background reference |
| Aranet4 Outside | Outdoors | ~1.2 | CO₂ / ambient reference |
| HOBO MB_D | Bathroom #1 | [PLACEHOLDER] | RH / Temp |
| HOBO MB_Bath | Bathroom #1 | [PLACEHOLDER] | RH / Temp (secondary) |
| HOBO MB_E | Bathroom–Bedroom doorway | [PLACEHOLDER] | RH / Temp |
| HOBO MB_Bed | Bedroom #1 | [PLACEHOLDER] | RH / Temp |
| HOBO MB_F | Bedroom #1 | [PLACEHOLDER] | RH / Temp (secondary) |
| HOBO MB_C | Bedroom #1 | [PLACEHOLDER] | RH / Temp (tertiary) |
| Vaisala HMP155 Bed1 | Bedroom #1 | ~1.2 | RH / Temp (DAQ) |
| Vaisala HMP155 Liv | Living room | ~1.2 | RH / Temp (DAQ) |
| Vaisala HMP45A MBa | Bathroom #1 | [PLACEHOLDER] | RH / Temp (DAQ) |
| Met One AIO2 | Outdoors, 10-m tower | 10.0 | Wind / Temp / RH / Pressure |
| Setra 264 | Multiple zones | Various | Disconnected; not reported |

---

## 2.7 Instrument Calibration and Applied Corrections

HOBO UX100 loggers were factory-calibrated by Onset Computer Corporation before deployment. The calibration offsets in Table 2-4 (all ≤ 0.24 °F for temperature and ≤ 0.57 % for RH) correct for individual logger bias and were applied programmatically to all raw data during processing.

Vaisala HMP155 probes were factory-calibrated on May 11, 2023 (calibration records on file). [PLACEHOLDER: Provide Vaisala HMP45A calibration dates.] Analog transfer functions for both Vaisala series were applied using the manufacturer's documented voltage-to-engineering-unit relationships and the individual sensor coefficients recorded in the DAQ channel map [PLACEHOLDER: reference the wiring and calibration document].

[PLACEHOLDER: Describe any cross-calibration, co-location zero/span checks, or comparison measurements performed on the QuantAQ MODULAIR-PM sensors, Aranet4 sensors, or CO₂ injection system during the study.]
