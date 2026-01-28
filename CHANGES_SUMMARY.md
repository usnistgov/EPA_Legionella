# Event Matching and Management System Updates

## Summary of Changes (January 2026)

This document summarizes the major updates made to the event matching and management system for the EPA Legionella project.

---

## 1. Fixed Matching Logic (CRITICAL FIX)

### Previous Behavior (INCORRECT):
- Assumed CO2 injection occurred **40 minutes AFTER** shower start
- Expected timeline: Shower at 09:00 → CO2 at 09:40
- This was **backwards** from actual protocol

### Current Behavior (CORRECT):
- CO2 injection occurs **20 minutes BEFORE** shower start
- Actual timeline: CO2 at 08:40 → Shower at 09:00
- Experimental protocol:
  1. CO2 injection starts at XX:40 (e.g., 08:40)
  2. CO2 injection ends at XX:44 (4-minute duration)
  3. Mixing fan turns off at XX:45
  4. Shower starts at (XX+1):00 (e.g., 09:00)
  5. Shower runs 5-15 minutes
  6. CO2 decay measurement begins at (XX+1):50

### Files Modified:
- [scripts/event_matching.py](scripts/event_matching.py): Updated matching logic and docstrings
- [src/particle_decay_analysis.py](src/particle_decay_analysis.py): Updated timing comments and parameters

---

## 2. Date Filtering

### Implementation:
- All events before **2026-01-14** are now filtered out
- This is when the actual experiment started
- Pre-experiment data was test/setup data

### Configuration:
```python
# In scripts/event_manager.py
EXPERIMENT_START_DATE = datetime(2026, 1, 14, 0, 0, 0)
```

---

## 3. Test Parameter Naming Convention

### Format:
```
MMDD_TempCode_TimeOfDay_RNN
```

### Components:
- **MMDD**: Month and day (e.g., `0114` for January 14)
- **TempCode**: Water temperature
  - `HW` = Hot Water (before 2026-01-22 14:00)
  - `CW` = Cold Water (after 2026-01-22 14:00)
- **TimeOfDay**: Category based on shower start time
  - `Morning`: 5am - 11am
  - `Afternoon`: 11am - 5pm
  - `Evening`: 5pm - 9pm
  - `Night`: 9pm - 5am
- **RNN**: Replicate number (e.g., `R01`, `R02`, etc.)
  - Sequential numbering for same date/condition

### Examples:
- `0114_HW_Morning_R01` - First hot water morning test on Jan 14
- `0114_HW_Morning_R02` - Second hot water morning test on Jan 14
- `0122_CW_Afternoon_R01` - First cold water afternoon test on Jan 22

### Optional Parameters:
If bath fan runs **during** or **within 2 hours after** the shower, `_Fan` is added:
- `0114_HW_Morning_Fan_R01` - Hot water morning test with fan running

**Note:** Bath fan running **before** shower is for space draw-down and is NOT included in naming.

---

## 4. Missing Event Detection and Synthetic Events

### Behavior:
- System detects when a shower event has no corresponding CO2 injection
- Automatically creates a **synthetic CO2 event** at the expected time
  - Expected time = shower_time - 20 minutes
  - Marked with `is_synthetic: True` flag
- These events are logged in `event_log.csv`
- Synthetic events have timing information but no actual CO2 measurement data
- Lambda values will be `NaN` for synthetic events

### Logging:
All missing events are recorded in the event log with:
- `has_matching_co2: False`
- `co2_is_synthetic: True` (if synthetic event created)

---

## 5. Event Exclusion System

### Predefined Exclusions:
Currently configured exclusions in `scripts/event_manager.py`:

```python
EXCLUDED_EVENTS = {
    datetime(2026, 1, 22, 15, 0, 0): "Tour in house during test",
}
```

### Adding New Exclusions:
To exclude additional events, add entries to the `EXCLUDED_EVENTS` dictionary:

```python
EXCLUDED_EVENTS = {
    datetime(2026, 1, 22, 15, 0, 0): "Tour in house during test",
    datetime(2026, 1, 25, 12, 0, 0): "Equipment malfunction",
    datetime(2026, 1, 27, 9, 0, 0): "Power outage",
}
```

### Behavior:
- Excluded events are skipped in all analyses
- They appear in `event_log.csv` with:
  - `is_excluded: True`
  - `exclusion_reason: "<reason string>"`
- Particle and CO2 analyses will skip these events

---

## 6. Event Log System

### Output File:
`output/event_log.csv`

### Columns:
| Column | Description |
|--------|-------------|
| `event_type` | "shower" or "co2" |
| `event_number` | Sequential event number (1-indexed) |
| `test_name` | Test condition name (e.g., "0114_HW_Morning_R01") |
| `datetime` | Primary event timestamp |
| `shower_on` | Shower start time |
| `shower_off` | Shower end time |
| `co2_injection` | CO2 injection start time |
| `co2_event_number` | Matched CO2 event number |
| `has_matching_co2` | Boolean: does shower have CO2 data? |
| `co2_is_synthetic` | Boolean: is CO2 event synthetic? |
| `is_excluded` | Boolean: is event excluded from analysis? |
| `exclusion_reason` | Reason for exclusion (if applicable) |
| `water_temp` | "HW" or "CW" |
| `time_of_day` | "Morning", "Afternoon", "Evening", or "Night" |
| `fan_during_test` | Boolean: did bath fan run during test? |

### Usage:
This log provides a complete audit trail of:
- All detected events
- Missing events and synthetic fill-ins
- Excluded events with reasons
- Test parameters for each event

---

## 7. Updated Analysis Outputs

### Particle Analysis Results:
The output file `particle_analysis_summary.xlsx` now includes additional columns:

**New Columns:**
- `test_name`: Test condition identifier
- `water_temp`: "HW" or "CW"
- `time_of_day`: Time category
- `fan_during_test`: Boolean for fan status
- `replicate_num`: Replicate number
- `co2_event_idx`: Index of matched CO2 event

**Benefits:**
- Easy filtering by test conditions
- Clear identification of replicates
- Traceability to CO2 measurements

### Plot Filenames:
Plots are now named using test names instead of event numbers:
- **Old:** `event_01_pm_decay.png`
- **New:** `0114_HW_Morning_R01_pm_decay.png`

This makes it much easier to identify which plot corresponds to which test condition.

---

## 8. Log File Issues Identified

### Issues Found:
1. **Pre-experiment data**: Events before 2026-01-14 (now filtered out)
2. **Non-standard timing**: Some shower events at :50 instead of :00
3. **Orphaned bath fan event**: 2026-01-14 12:05:39 with no corresponding shower/CO2

### Recommendations:
- Review log processing scripts for consistency
- Verify timing synchronization between control systems
- Consider adding validation checks in data acquisition

---

## 9. How to Use the New System

### In Particle Analysis:
```python
from scripts.event_manager import process_events_with_management

# After loading raw shower events
events, co2_events, event_log = process_events_with_management(
    raw_shower_events,
    [],  # CO2 events from co2_results DataFrame
    shower_log,
    co2_results_df,
    output_dir,
    create_synthetic=True
)

# Events are now filtered, named, and matched
for event in events:
    print(f"{event['test_name']}: λ={event.get('lambda_ach', 'N/A')} h⁻¹")
```

### In CO2 Analysis:
```python
from scripts.event_manager import filter_events_by_date, is_event_excluded

# Filter events by date
filtered_events = filter_events_by_date(all_events)

# Check if event should be excluded
for event in filtered_events:
    is_excluded, reason = is_event_excluded(event['injection_start'])
    if is_excluded:
        print(f"Skipping: {reason}")
        continue
    # Process event...
```

---

## 10. Configuration Reference

### Key Constants (in `scripts/event_manager.py`):

```python
# Experiment start date
EXPERIMENT_START_DATE = datetime(2026, 1, 14, 0, 0, 0)

# Water temperature transition
HOT_WATER_END_TIME = datetime(2026, 1, 22, 14, 0, 0)

# Time of day boundaries (hour ranges)
TIME_OF_DAY_RANGES = {
    "Morning": (5, 11),
    "Afternoon": (11, 17),
    "Evening": (17, 21),
    "Night": (21, 5),
}

# Event exclusions
EXCLUDED_EVENTS = {
    datetime(2026, 1, 22, 15, 0, 0): "Tour in house during test",
}

# Matching parameters
EXPECTED_CO2_BEFORE_SHOWER = 20  # minutes
```

### Matching Tolerances (in `scripts/event_matching.py`):

```python
# Default search window: expected_time ± tolerance
time_tolerance_before = 10.0  # minutes
time_tolerance_after = 10.0   # minutes

# Expected CO2 time = shower_time - 20 minutes
# Search window = (shower_time - 30 minutes) to (shower_time - 10 minutes)
```

---

## 11. Testing and Validation

### Recommended Tests:
1. **Run particle analysis** and verify:
   - Events before 2026-01-14 are excluded
   - CO2 matching is correct (20 min before shower)
   - Test names appear in output
   - Event log is generated

2. **Check event_log.csv**:
   - Verify all events are present
   - Check for missing events
   - Confirm exclusions are marked

3. **Review plots**:
   - Filenames use test names
   - Plots correspond to correct test conditions

4. **Validate test names**:
   - Replicates numbered correctly
   - Water temp transitions at Jan 22 2pm
   - Time of day categories correct

---

## 12. Future Enhancements

### Planned Features:
- Additional test parameters (shower head type, door positions)
- Automatic detection of equipment malfunctions
- Statistical comparison between test conditions
- Interactive dashboard for event review

### Parameter Placeholders in Naming System:
The naming convention is designed to be extensible. Future parameters could include:
- `_ShowerHead<Type>`: Different shower head models
- `_DoorOpen` / `_DoorClosed`: Door position configurations
- `_Flow<Rate>`: Water flow rate variations

Example: `0125_CW_Morning_Fan_DoorOpen_R01`

---

## Files Modified

### Created:
- `scripts/event_manager.py` - New event management system
- `scripts/analyze_log_files.py` - Log file analysis utility
- `CHANGES_SUMMARY.md` - This document

### Modified:
- `scripts/event_matching.py` - Fixed matching logic, updated docstrings
- `src/particle_decay_analysis.py` - Integrated event manager, added test names to output
- `src/env_data_loader.py` - Added event_number and particle analysis windows to events

### No Changes Required:
- `src/co2_decay_analysis.py` - Can use event_manager functions but not required
- Data processing scripts - Continue to work as before
- Plotting scripts - Will receive additional metadata but remain compatible

---

## Contact and Questions

For questions about these changes:
- Review code comments in modified files
- Check function docstrings for parameter details
- Refer to this document for high-level overview

---

**Last Updated:** January 28, 2026
**Author:** Nathan Lima, NIST
