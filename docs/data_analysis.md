# Data Analysis

> **Author note:** Equation numbers, section cross-references, and items marked
> `[PLACEHOLDER: ...]` require PI review before submission. Equation numbering uses
> the format (3-N) for Section 3; renumber as needed when assembled into the full report.

---

## 3.1 Overview

Raw sensor data, shower event logs, and CO₂ injection logs were processed through a multi-step analysis pipeline implemented in Python (version 3.14.2; key library versions: NumPy 2.4.1, SciPy 1.17.0, pandas 2.3.3). All analysis scripts and source modules are archived in the companion data and code repository (DOI: 10.18434/mds2-4153). Three analysis domains were treated independently:

1. **Air change rate (CO₂ decay):** The volumetric air exchange rate between Bedroom #1 and the outdoor environment was determined from the exponential decay of CO₂ tracer gas concentration following each injection event.
2. **Particle analysis:** Size-resolved particle number concentrations from the indoor and outdoor QuantAQ MODULAIR-PM sensors were used to estimate the penetration factor, other process rate, and shower aerosol emission rate for each event and each of 12 optical particle counter size bins.
3. **Environmental conditions:** Relative humidity (RH), temperature, and wind data from all co-located sensors were summarized over pre- and post-shower time windows.

All three analyses are driven by a unified event registry (Section 3.2) that assigns a consistent event number, test name, and metadata record to every shower event.

---

## 3.2 Event Identification and Registry

### 3.2.1 Shower Event Detection

Shower start and stop times were recorded by the automated control system in a comma-separated log file (`shower_log_file.csv`). Each row represents a discrete state change — shower on, shower off, bathroom fan on, or bathroom fan off — with a high-resolution timestamp. Consecutive on/off pairs for each device were parsed into individual events, each characterized by start time, stop time, and duration. Only events that began on or after January 15, 2026 at 15:00 (the official experiment start cutoff) were retained.

### 3.2.2 CO₂ Event Matching

CO₂ injection events were independently logged in `CO2_log_file.csv`, which records timestamps for CO₂ valve state changes and mixing fan activations. Each CO₂ injection event was matched to the nearest shower event within a bidirectional tolerance window of ±10 minutes. Matches were made one-to-one; unmatched shower events were assigned a synthetic CO₂ placeholder record with an inferred duration estimated from neighboring matched events, so that the event registry remained complete and consistently numbered regardless of CO₂ data availability.

### 3.2.3 Event Numbering and Naming

Successfully matched shower–CO₂ pairs were assigned sequential integer event numbers beginning at 1. Each event was also assigned a structured test name (Section 2.2.5) based on the active test configuration at the event's start time, as determined by the time-stamped configuration transition tables maintained in the event management module (`src/event_manager.py`). The air change rate (λ) from the CO₂ decay analysis was stored in the registry for each event that had a valid CO₂ match; events relying on synthetic CO₂ records were flagged accordingly and used a λ value carried forward from the nearest valid event of the same configuration [PLACEHOLDER: confirm whether forward-carrying of λ is used or whether such events are excluded from particle analysis].

### 3.2.4 Exclusion Criteria

Events were excluded from analysis using the criteria described below. Excluded events are retained in the registry with an `is_excluded = True` flag, a null event number, and an `exclusion_reason` field. Exclusions were applied before any analysis step; excluded events do not appear in summary outputs.

**Duration-based exclusion.** Because the automated control system enforced a nominal 10.0-minute shower duration, any shower event with a measured duration outside the range 9 min 55 s to 10 min 5 s (10.0 min ± 5 s) was assumed to represent a manual water-temperature verification run or a control-system test rather than an analysis event. Such events were flagged with `exclusion_reason = "Water temperature testing (duration: X.X min)"` and excluded from all analysis.

**Predefined individual event exclusions.** Four specific events were excluded due to documented confounding activities (personnel inside the home):

| Date and time | Reason |
|---|---|
| 2026-01-22 15:00 | Tour in house |
| 2026-01-29 15:00 | People in house |
| 2026-05-13 15:00 | LVP flooring installation |
| 2026-05-21 15:00 | Bathroom flooring removal |

**Predefined date range exclusions.** Four date ranges were excluded due to known data quality or operational issues:

| Excluded range | Reason |
|---|---|
| 2026-02-11 08:00 to 2026-02-13 11:00 | Conflicting log entries during water temperature configuration change |
| 2026-03-08 00:00 to 2026-03-10 00:00 | Daylight saving time clock change; elevated indoor RH of uncertain origin |
| 2026-03-14 00:00 to 2026-03-15 12:00 | CO₂ injection system failure |
| 2026-05-11 00:00 to 2026-05-15 10:00 | CO₂ injection system failure |

**Particle analysis-specific exclusions.** Two additional criteria were applied exclusively within the particle analysis pipeline (CO₂ decay results for these events were retained):

1. Events for which the CO₂ decay regression R² was below 0.75 were excluded from particle analysis. An unreliable air change rate propagates uncertainty into the penetration factor, other process rate, and emission rate calculations; a minimum R² of 0.75 was judged the practical lower bound for a usable λ estimate.

2. Events for which the indoor QuantAQ sensor's RH response peaked outside the first 45 minutes of the deposition analysis window were excluded. A late RH peak indicates that shower aerosol did not reach the bedroom sensor within the expected timeframe, suggesting an atypical airflow path for that event.

---

## 3.3 Air Change Rate Determination (CO₂ Decay Analysis)

### 3.3.1 Tracer Gas Decay Model

Bedroom #1 is treated as a single, well-mixed zone. After the CO₂ mixing fan stops and before the shower begins, no internal CO₂ sources are active, and the mass balance on CO₂ in the bedroom reduces to:

$$\frac{dC}{dt} = \lambda \left( C_{bg} - C \right) \tag{3-1}$$

where $C(t)$ [ppm] is the bedroom CO₂ concentration, $\lambda$ [h⁻¹] is the air change rate, and $C_{bg}$ [ppm] is a time-constant background concentration representing the mixed composition of air entering the bedroom. $C_{bg}$ is approximated as a weighted average of the outdoor and entry zone concentrations:

$$C_{bg} = \alpha \cdot C_{out} + (1 - \alpha) \cdot C_{ent} \tag{3-2}$$

where $C_{out}$ is the outdoor CO₂ concentration (Aranet4 Outside), $C_{ent}$ is the entry zone CO₂ concentration (Aranet4 Entry), and $\alpha$ is the outdoor air fraction (default $\alpha = 0.5$). Both $C_{out}$ and $C_{ent}$ are averaged over the full decay window to produce a single scalar $C_{bg}$ for the regression.

Integrating Eq. (3-1) with the initial condition $C(0) = C_0$:

$$C(t) = C_{bg} + \left(C_0 - C_{bg}\right) e^{-\lambda t} \tag{3-3}$$

Rearranging into a linear form suitable for regression:

$$y(t) \equiv -\ln\!\left[\frac{C(t) - C_{bg}}{C_0 - C_{bg}}\right] = \lambda \cdot t \tag{3-4}$$

where $y(t)$ is a dimensionless decay variable that increases linearly with time at rate $\lambda$. A linear regression of $y$ versus $t$, with the intercept forced through the origin, yields $\lambda$ as the slope.

### 3.3.2 Source Concentration Methods

Three values of $\alpha$ were applied to each event to quantify sensitivity to the assumed source air composition:

| Method label | $\alpha$ | $C_{bg}$ definition |
|---|---|---|
| Average (primary) | 0.5 | $0.5 \cdot C_{out} + 0.5 \cdot C_{ent}$ |
| Outside only | 1.0 | $C_{out}$ |
| Entry only | 0.0 | $C_{ent}$ |

The resulting $\lambda$ values ($\lambda_{avg}$, $\lambda_{out}$, $\lambda_{ent}$) and their regression R² values are all stored in the event registry and reported in the output file `co2_lambda_summary.csv`. The average method value $\lambda_{avg}$ is used as the primary air change rate in all subsequent particle analysis.

### 3.3.3 Data Preprocessing

Aranet4 data from three locations (Bedroom, Entry, Outside) were loaded from manufacturer-exported Excel files. Timestamps were parsed and data were resampled to a regular 1-minute grid by linear interpolation. A 6-minute centered rolling average was then applied to reduce sensor noise before the decay regression. The Bathroom Aranet4 data were included in event concentration time-series plots for spatial context but were not used in any λ calculation.

### 3.3.4 Regression Procedure

For each shower event, the CO₂ decay analysis window was defined as:

- **Start:** shower_on + 10 min (to allow any airflow transient from shower onset to settle before the regression window begins)
- **End:** shower_on + 2 h 10 min (2-hour analysis window)

Within this window, the initial condition $C_0$ was taken as the first data point at window start ($t = 0$), and $C_{bg}$ was computed once as the mean of $C_{out}$ and $C_{ent}$ (or $C_{out}$ alone, or $C_{ent}$ alone, per Section 3.3.2) over the full 2-hour window. The decay variable $y(t)$ was computed at each 1-minute time step via Eq. (3-4), and the air change rate was estimated by ordinary least squares regression forced through the origin:

$$\hat{\lambda} = \frac{\displaystyle\sum_i t_i \cdot y_i}{\displaystyle\sum_i t_i^2} \tag{3-5}$$

The coefficient of determination $R^2$ was computed from the residuals of the fitted versus observed $y$ values. A minimum initial concentration excess of 50 ppm — that is, $C_0 - C_{bg} \geq 50$ ppm — was required for a valid regression; events not meeting this threshold were excluded from the CO₂ decay analysis.

An optional analysis mode (`--entry-stop` flag) truncates the decay window when the bedroom concentration decays to within 100 ppm of the mean of the entry and outside concentrations:

$$C_{bedroom}(t) \leq 100 + \frac{C_{ent}(t) + C_{out}(t)}{2} \tag{3-6}$$

This prevents the regression from being influenced by low-signal data points when the tracer has nearly equilibrated with the background. This flag was not enabled for the primary analysis reported here.

### 3.3.5 Uncertainty Considerations

The principal sources of uncertainty in the reported $\lambda$ are:

1. **Statistical fitting uncertainty:** Standard error of the regression slope (Eq. 3-5), arising from random sensor noise and short-term concentration fluctuations.
2. **Source concentration assumption:** The range of $\lambda_{avg}$, $\lambda_{out}$, and $\lambda_{ent}$ values quantifies sensitivity to the assumed spatial distribution of incoming air. In practice, this spread is the dominant contributor to inter-method uncertainty.
3. **Well-mixed zone assumption:** The single-zone model assumes that the bedroom CO₂ is spatially uniform after the mixing fan stops. Residual spatial gradients in the first few minutes of the window may introduce systematic bias; the 10-minute delay at window start mitigates, but does not fully eliminate, this effect.
4. **CO₂ sensor accuracy:** The Aranet4 PRO accuracy specification is ±50 ppm ± 3 % of reading. At typical bedroom peak concentrations of 1000–3000 ppm, this corresponds to an absolute accuracy of approximately ±80 ppm to ±140 ppm.
5. **Ambient variability:** Outdoor wind speed and direction influence infiltration; the reported $\lambda$ represents the time-average over the 2-hour decay window rather than an instantaneous value.

---

## 3.4 Particle Analysis

### 3.4.1 Particle Size Bins

All particle analyses were performed independently on each of the 12 OPC-N3 size bins defined in Table 2-3 (0.35–10.0 µm; bins indexed 0–11 from smallest to largest). Bins are organized into three reporting groups: fine particles (bins 0–2, 0.35–1.0 µm), accumulation and coarse particles (bins 3–6, 1.0–3.0 µm), and coarse particles (bins 7–11, 3.0–10.0 µm). The coarse group encompasses the size range most directly associated with respiratory aerosol deposition in the upper airways and, for the larger subfractions, gravitational settling timescales relevant to Legionella transmission risk. Particle concentrations from the QuantAQ sensors are in units of particles per cm³ (#/cm³). No rolling average was applied to the particle data prior to analysis; the 1-minute raw values were used directly.

### 3.4.2 Mass Balance Model

The indoor particle concentration in Bedroom #1 for size bin $k$ is governed by a first-order, single-zone mass balance:

$$\frac{dC_{in}}{dt} = p \cdot \lambda \cdot C_{out} - \lambda \cdot C_{in} - \beta \cdot C_{in} + \frac{E}{V} \tag{3-7}$$

where the symbols are defined as:

| Symbol | Description | Units |
|---|---|---|
| $C_{in}(t)$ | Indoor particle concentration (bin $k$) | #/cm³ |
| $C_{out}(t)$ | Outdoor particle concentration (bin $k$) | #/cm³ |
| $p$ | Penetration factor — fraction of outdoor particles penetrating the envelope per air change | — |
| $\lambda$ | Air change rate (from CO₂ decay, Section 3.3) | h⁻¹ |
| $\beta$ | Other process rate — net first-order indoor particle loss rate not attributable to ventilation | h⁻¹ |
| $E$ | Shower aerosol emission rate (from bathroom into bedroom) | #/h |
| $V$ | Bedroom volume | m³ (36.1 m³) |

The term $p \lambda C_{out}$ represents particle infiltration from outdoors; $\lambda C_{in}$ represents particle removal via exfiltration; $\beta C_{in}$ aggregates all additional first-order indoor loss processes (gravitational settling, inertial impaction on surfaces, thermophoresis, electrostatic deposition); and $E/V$ is the volumetric emission source from shower aerosol transported from the bathroom.

Parameters $p$, $\beta$, and $E$ are estimated sequentially using distinct time windows from each event, described in Sections 3.4.3 through 3.4.5.

### 3.4.3 Penetration Factor ($p$)

The penetration factor was estimated from the ratio of indoor to outdoor particle concentration during periods with no shower emissions ($E = 0$) and with the indoor concentration approximately at quasi-steady state with respect to outdoor variability. At steady state, Eq. (3-7) yields $C_{in,ss}/C_{out} = p\lambda / (\lambda + \beta)$. Because $\beta \ll \lambda$ for fine particles and the ratio $\beta/\lambda$ is assumed to be small over the averaging window, $p$ is approximated as:

$$p \approx \frac{\overline{C_{in}}}{\overline{C_{out}}} \bigg|_{\text{background window}} \tag{3-8}$$

Two 6-hour background windows were used — one before and one after the shower event — with boundaries determined by the time of day of the shower:

**Night events** (shower between 21:00 and 05:00):

| Window | Start | End |
|---|---|---|
| Before | 20:00 (previous day) | 02:00 (day of) |
| After | 08:00 (day of) | 14:00 (day of) |

**Day events** (shower between 05:00 and 21:00):

| Window | Start | End |
|---|---|---|
| Before | 08:00 (day of) | 14:00 (day of) |
| After | 20:00 (day of) | 02:00 (next day) |

"Day" was defined as 05:00–17:00 and "Night" as 17:00–05:00. This classification ensures that background windows bracket the shower event without overlapping the shower-on period or the subsequent 2-hour deposition window. Within each 6-hour window, all 1-minute time steps where either $C_{in}$ or $C_{out}$ was zero or negative were excluded. The window-average ratio $\overline{C_{in}} / \overline{C_{out}}$ was computed as the ratio of the window means (not the mean of per-step ratios). The final penetration factor for each event and bin was the arithmetic mean of the before and after window values, capped at 1.0 (passive infiltration cannot exceed unity):

$$p = \min\!\left(1.0,\; \frac{p_{before} + p_{after}}{2}\right) \tag{3-9}$$

A minimum of 10 valid data points per window was required. Events or bins with fewer than 10 points in either window were excluded from penetration factor estimation and from all downstream particle calculations.

### 3.4.4 Other Process Rate ($\beta$)

The other process rate $\beta$ [h⁻¹] was estimated from the particle concentration decay in Bedroom #1 after the shower ended. During this deposition window ($E = 0$), Eq. (3-7) can be solved for $\beta$ at each 1-minute time step using a forward Euler discretization:

$$\beta_t = \frac{C_t - C_{t+1}}{\Delta t \cdot C_t} + \frac{p \cdot \lambda \cdot C_{out,t}}{C_t} - \lambda \tag{3-10}$$

where $\Delta t = 1/60$ h (1-minute time step), and $C_t$ and $C_{t+1}$ are the measured indoor concentrations at consecutive minutes. Per-step $\beta_t$ values were collected over a 2-hour deposition window beginning at shower_off.

**Outlier removal.** Before aggregating the per-step estimates, a two-stage filter was applied:

1. *Upper cap:* Per-step values $\beta_t > 5.0$ h⁻¹ were discarded. A value of 5 h⁻¹ corresponds to a particle half-life of approximately 8 minutes from deposition alone, which is physically unreasonable for particles smaller than ~3 µm; values above this threshold are attributed to noise spikes in the 1-minute concentration data.

2. *Percentile trim:* From the remaining values, only those within the 5th–95th percentile range were retained (symmetric trim to remove asymmetric extreme residuals).

The trimmed mean of the retained $\beta_t$ values was taken as the candidate estimate.

**Four-step acceptance hierarchy.** An R²-based selection procedure then determined the final reported $\beta$. At each step, a forward Euler simulation of the deposition-window concentration time series was generated using the candidate $\beta$ (with $E = 0$), and the R² of the simulated versus measured concentration was computed. A threshold of R² ≥ 0.80 was required for acceptance at each step:

| Step | Candidate $\beta$ | Accept if |
|---|---|---|
| (a) | Unclamped trimmed mean (may be negative) | Forward Euler R² ≥ 0.80 |
| (b) | Clamp to $\beta \geq 0$ | Forward Euler R² ≥ 0.80 |
| (c) | Set $\beta = 0$ (no net deposition) | Forward Euler R² ≥ 0.80 |
| (d) | Neither positive nor zero β achieves R² ≥ 0.80 | $\beta = \text{NaN}$ (bin invalid) |

The hierarchy proceeds in order: a step is accepted if its R² meets the threshold, and subsequent steps are skipped. The rationale for each step follows from physical arguments: step (a) accepts the unconstrained best-fit value when it adequately reproduces the decay; step (b) accepts a non-negative deposition rate when the unconstrained estimate is slightly negative (physically: near-zero net loss) but still fits the data; step (c) accepts the limiting case of no net deposition when neither positive nor negative β is warranted by the data; step (d) declares the bin invalid when no reasonable β value reproduces the measured decay, and the bin is excluded from all downstream reporting.

This hierarchy avoids a systematic visual artifact present in simpler implementations: when $\beta$ is forced to zero, the forward Euler simulation rises toward the outdoor steady-state concentration $p \cdot C_{out}$ once the indoor concentration decays below this level, creating a spurious step-change in the predicted time series. By allowing NaN (step d) rather than forcing $\beta = 0$ in all cases, invalid bins are clearly identified and not misrepresented by a model that does not fit the data.

A minimum of 10 valid time steps in the deposition window was required. Bins with fewer than 10 valid steps were excluded.

### 3.4.5 Emission Rate ($E$)

The shower aerosol emission rate $E$ [#/h] was estimated over the period from shower onset to the time of peak indoor particle concentration in the bedroom. During this emission window, $p$, $\lambda$, and $\beta$ are held constant at the values estimated above, and Eq. (3-7) is rearranged to solve for $E_t$ at each 1-minute time step:

$$E_t = V \left[\frac{C_{t+1} - C_t}{\Delta t} + (\lambda + \beta) \cdot C_t - p \cdot \lambda \cdot C_{out,t}\right] \tag{3-11}$$

Per-step values $E_t$ were computed for all time steps from shower_on to the concentration peak. Negative values of $E_t$ can arise from short-term concentration decreases within the emission window and were retained in the time series for visualization but excluded from the mean and median calculations; only positive values contributed to the reported statistics $E_{mean}$ and $E_{median}$.

The total particle quantity delivered to the bedroom during the emission phase was estimated by trapezoidal integration:

$$E_{total} = \int_{t_{on}}^{t_{peak}} \max(E_t,\, 0)\, dt \approx \sum_i \max(E_{t_i},\, 0) \cdot \Delta t \tag{3-12}$$

where negative per-step values were clipped to zero before summation. A minimum of 3 valid time steps in the emission window was required; bins with fewer than 3 steps were excluded from emission estimation.

### 3.4.6 Predicted Concentration Time Series

A forward Euler simulation of the bedroom particle concentration was generated for each valid event–bin combination, spanning from shower onset through the end of the deposition window (shower_on to shower_off + 2 h). The simulation used a 1-minute time step and was divided into two sequential phases:

**Emission phase** (shower_on to peak_time): $E_t = E_{mean}$ (constant, mean of positive per-step values from Section 3.4.5).

$$C_{t+1} = C_t + \Delta t \left(p \cdot \lambda \cdot C_{out,t} - \lambda \cdot C_t - \beta \cdot C_t + \frac{E_{mean}}{V}\right) \tag{3-13}$$

**Deposition phase** (peak_time to shower_off + 2 h): $E_t = 0$.

$$C_{t+1} = C_t + \Delta t \left(p \cdot \lambda \cdot C_{out,t} - \lambda \cdot C_t - \beta \cdot C_t\right) \tag{3-14}$$

The deposition phase simulation was initialized from the *simulated* (not measured) peak concentration, isolating any mismatch between the emission phase model and the measured peak from the deposition phase comparison. The R² of the emission phase simulation versus measured concentrations from shower_on to peak_time was computed as a diagnostic of emission phase fidelity.

---

## 3.5 Environmental Conditions Analysis

### 3.5.1 Pre- and Post-Shower Analysis Windows

Environmental sensor data (RH, temperature, wind) were summarized over two time windows relative to each shower event:

- **Pre-shower baseline:** 30 minutes immediately before shower onset (shower_on − 30 min to shower_on).
- **Post-shower response:** 2 hours after shower offset (shower_off to shower_off + 2 h).

These window lengths are consistent with the particle deposition analysis window, allowing direct comparison of environmental conditions to aerosol transport behavior. Mean and standard deviation were computed for each sensor over each window.

### 3.5.2 Bedroom Reference Conditions

To characterize the pre-shower indoor environment for use in boxplot annotations and environmental stratification analyses, bedroom RH and temperature for each event were computed from the following five sensor channels:

- Vaisala HMP155 Bed1 (RH and temperature)
- HOBO UX100 MB_Bed / Bedroom1 (RH and temperature)
- HOBO UX100 MB_F / Bedroom2 (RH and temperature)
- HOBO UX100 MB_C / Bedroom3 (RH and temperature)
- Aranet4 Bedroom (RH and temperature)

The QuantAQ MODULAIR-PM `met_rh` and `met_temp` channels were explicitly excluded from all bedroom environmental characterization; these channels reflect conditions within the instrument's internal flow cell and are not representative of ambient room air.

Mean and standard deviation of RH [%] and temperature [°C] across the five sensors were calculated over the 30-minute pre-shower baseline window. Combined uncertainty was estimated as:

$$u_{RH} = \frac{1.96\,\sigma_{RH}}{\sqrt{n}} \tag{3-15}$$

where $\sigma_{RH}$ is the sample standard deviation across the $n$ available sensors and 1.96 is the 97.5th percentile of the standard normal distribution (approximate 95 % confidence interval). An analogous expression applies to temperature. These bedroom conditions are tabulated in the `Bedroom_Conditions` sheet of `rh_temp_wind_summary.xlsx` and are exempt from the significant-figure rounding described in Section 3.7.1.

### 3.5.3 Sensors Excluded from RH Time-Series Figures

For event-level RH time-series figures, sensors with limited spatial relevance to the bathroom–bedroom aerosol pathway were omitted to reduce visual clutter:

- Vaisala MBa RH (Bathroom #1, HMP45A)
- Vaisala Liv RH (Living room, HMP155)
- Aranet4 Entry RH
- Aranet4 Outside RH
- AIO2 outdoor RH (Met One weather station)

These sensors remain available in the full summary tables and in analyses requiring complete spatial coverage of the home.

---

## 3.6 Quality Assurance and Data Exclusions

### 3.6.1 Consolidated Exclusion Criteria

Table 3-1 consolidates all exclusion criteria applied across the three analysis domains, together with the resulting disposition of excluded data.

**Table 3-1. Summary of event and bin-level exclusion criteria.**

| Criterion | Applied at | Disposition of excluded data |
|---|---|---|
| Shower duration outside 10.0 min ± 5 s | All analysis | `is_excluded = True`; `event_number = NaN`; retained in registry |
| Predefined individual event exclusions (4 events) | All analysis | `is_excluded = True`; retained in registry |
| Predefined date range exclusions (4 ranges) | All analysis | `is_excluded = True`; retained in registry |
| Initial CO₂ concentration excess < 50 ppm | CO₂ decay (λ) | Event excluded from λ analysis only |
| CO₂ decay regression R² < 0.75 | Particle analysis (λ input) | CO₂ result retained; event excluded from particle analysis |
| QuantAQ bedroom RH peak outside first 45 min of deposition window | Particle analysis | Event excluded from particle analysis |
| Penetration window: fewer than 10 valid data points | Penetration factor ($p$) | Bin excluded; $p = $ NaN; no downstream analysis |
| Deposition window: fewer than 10 valid data points after upper-cap filter | Other process rate ($\beta$) | Bin excluded; $\beta = $ NaN; no Ct prediction |
| Emission window: fewer than 3 valid positive time steps | Emission rate ($E$) | Bin excluded from $E$ reporting; Ct prediction unaffected |
| $\beta$ = NaN after 4-step selection (all steps fail R² ≥ 0.80) | All particle parameters | Bin invalid; no Ct prediction; rendered faded/dashed in figures |

### 3.6.2 Flow Rate Filter for Summary Figures

For summary boxplots and categorical comparison figures, events with measured water flow rates outside the standard range of 4.1–5.6 L/min were excluded to avoid confounding shower head type or water temperature effects with flow rate variability. This filter does not affect per-event time-series figures or the event registry.

---

## 3.7 Reporting Conventions

### 3.7.1 Significant Figures

Numerical results are reported to three significant figures in all data output files (CSV and Excel workbooks) and to two significant figures in figure annotations (axis tick labels, in-figure text boxes, legend entries). These conventions are applied programmatically through a project-wide significant figures utility module (`src/sig_figs.py`). Full-precision (unrounded) output can be obtained by invoking any analysis script with the `--no-sig-figs` command-line flag. The `Bedroom_Conditions` sheet of `rh_temp_wind_summary.xlsx` is exempt from rounding and is always written at full precision.

### 3.7.2 Summary Statistics

Unless otherwise stated in a figure caption or table header, summary statistics are defined as follows:

- **Mean:** arithmetic mean across replicate events within a configuration, or across configurations within a water temperature group.
- **Standard deviation:** sample standard deviation (divisor $n - 1$).
- **Uncertainty bars in figures:** ±1.96 × standard error of the mean (approximately 95 % confidence interval, assuming approximate normality; reported alongside the mean).
- **Boxplots:** center line = median; box edges = 25th and 75th percentiles (interquartile range, IQR); whiskers extend to the most extreme observation within 1.5 × IQR of the nearest box edge; individual observations beyond the whiskers are plotted as open circles.

Water temperature groupings in boxplots include only events from baseline configurations (standard nominal flow rate, no mannequin, standard door and fan positions) for each water temperature code (W##), unless the figure title or caption states otherwise. Variant runs (e.g., spray pattern or mannequin experiments conducted at W40) are included only in the categorical comparison figures specific to those variables.

### 3.7.3 Output File Structure

Principal output files generated by each analysis script are listed in Table 3-2.

**Table 3-2. Principal analysis output files.**

| Script | Output file | Contents |
|---|---|---|
| `event_registry.py` | `event_log.csv` | Event registry: all events with numbers, test names, λ, exclusion flags |
| `co2_decay_analysis.py` | `co2_lambda_summary.csv` | Per-event λ (all three source methods), R², decay window details |
| `co2_decay_analysis.py` | `co2_lambda_overall_summary.csv` | Aggregated λ statistics by configuration |
| `co2_decay_analysis.py` | `plots/event_figures/co2_decay/event_NN-*.png` | Per-event CO₂ concentration and decay fit plots |
| `co2_decay_analysis.py` | `plots/air_change_rate_boxplot.png` | λ by water temperature (baseline configurations) |
| `particle_decay_analysis.py` | `particle_analysis_summary.xlsx` | Multi-sheet workbook: all_results, penetration, beta, emission, totals, peak comparison |
| `particle_decay_analysis.py` | `plots/event_figures/pm_decay/event_NN-*.png` | Per-event 4-panel particle figures (concentration + 3 emission panels) |
| `particle_decay_analysis.py` | `plots/*_boxplot_{bin0-2,bin3-6,bin7-11}.png` | Categorical boxplots (water temp, RH, ACR, β, p, head type, spray, door, fan, mannequin) |
| `rh_temp_other_analysis.py` | `rh_temp_wind_summary.xlsx` | Multi-sheet summary: RH, Temp, Wind statistics; Bedroom_Conditions; Event_Log |
| `rh_temp_other_analysis.py` | `plots/event_figures/rh_timeseries/event_NN-*.png` | Per-event RH time-series figures |
| `rh_temp_other_analysis.py` | `plots/event_figures/temperature_timeseries/event_NN-*.png` | Per-event temperature time-series figures |
| `rh_temp_other_analysis.py` | `plots/event_figures/wind_timeseries/event_NN-*.png` | Per-event wind speed and direction figures |
| `export_event_timeseries.py` | `output/event_N_timeseries.xlsx` | Single-event predicted $C_t$ and measured $C_{in}$ for all bins |
| `export_config_timeseries.py` | `output/event_config_timeseries.xlsx` | 1-min average time series per configuration group across replicates |
