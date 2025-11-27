# WinNet_output — Python Interface for WinNet Nucleosynthesis Output

`WinNet_output` is a Python helper class designed to conveniently read, parse, and analyze output files from the WinNet nucleosynthesis post-processing code. It supports tracer output, mainout files, final abundance HDF5 files, and optional snapshot abundance data.

This class centralizes file parsing logic and provides common nucleosynthesis diagnostics such as expansion timescales and neutron/proton/alpha-to-seed ratios.

---

## Features

### ✔ Supported Data Types
Depending on the options passed to the constructor, the class can read:

- **Tracer files** (`tracerXXXXX.dat`)
  - Temperature, density, radius, entropy, composition, velocities, etc.
  - Optional: NSE-reached flag
- **mainout files**
  - Global abundance evolution (Yₙ, Yₚ, Yₐ, seeds, temperature, time, …)
- **Seed files** (`seed*.dat` / `seed*.txt`)
- **Final abundance HDF5 outputs** (`WinNet_data*.h5`)
  - `finabsum`: A-summed abundances
  - `finabelem`: element-resolved abundances
  - Optional: time-resolved snapshot abundances
- Utilities for:
  - Identifying temperature crossing points
  - Computing expansion timescales
  - Retrieving Ye and entropy at freeze-out
  - Computing neutron/proton/alpha-to-seed ratios at a chosen temperature

---

## Constructor

```python
WinNet_output(
    output_dir,
    with_NSE_flag=False,
    want_seeds=False,
    want_tracer_info=False,
    want_mainout_info=False,
    want_finabs_info=False,
    has_snapshots=False,
    want_all=False
)
```

### Arguments

| Argument | Type | Description |
|---------|------|-------------|
| `output_dir` | str | Directory containing WinNet output files |
| `with_NSE_flag` | bool | Whether tracer file contains a `reached_NSE` flag in the header |
| `want_seeds` | bool | Load seed file if available |
| `want_tracer_info` | bool | Load tracerXXXXX.dat file |
| `want_mainout_info` | bool | Load mainout.dat or mainout-* file |
| `want_finabs_info` | bool | Load `WinNet_data*.h5` final abundance file |
| `has_snapshots` | bool | Enable reading snapshot abundances (if available in HDF5) |
| `want_all` | bool | Convenience flag that enables finabs,mainout and tracer flag |

If none of the flags are set, the constructor prints a reminder to select the data type.

---

## Loaded Attributes

Depending on the flags:

### Tracer data (`want_tracer_info=True`)
- `self.tracer_data` – dict of arrays (t, T, rho, x, y, v, entropy, etc.)
- `self.tracer_data_units`
- `self.mass` – tracer mass
- `self.mass_unit`
- `self.reached_NSE` (only when `with_NSE_flag=True`)
- `self.tracer_id`

### Mainout data (`want_mainout_info=True`)
- `self.mainout_data`
- `self.mainout_data_units`

### Final abundances (`want_finabs_info=True`)
- `self.finabsum` – {Y, X, A}
- `self.finabelem` – {Y, X, Z}

If `has_snapshots=True`:
- `self.snapshot_times`
- `self.snapshot_composition` (A-distribution vs time)
- `self.snapshot_composition_elem` (Z-distribution vs time)

---

## Methods

### Temperature Crossing Utility

```python
find_last_fall_below(temps, t_crit)
```

Returns index of the **last temperature drop** below a critical temperature, even for non-monotonic temperature histories.

---

### Expansion Timescale

```python
get_expansion_timescale()
```

Computes the expansion timescale  
\[
	au = rac{r}{v_r}
\]  
evaluated at the temperature where `T = 0.5 MeV` (5.1 GK), as defined in Bliss et al.

Requires: `want_tracer_info=True`.

---

### Freeze-out Quantities

```python
get_Ye_at_freezeout()
get_entr_at_freezeout()
```

Return Ye or entropy at the last temperature crossing below 5.8 GK.

Requires: `want_tracer_info=True`.

---

### Seed Ratios

Each of these computes the ratio at the last time `T` falls below `T_eval`:

```python
get_neutron_to_seed_ratio(T_eval)
get_proton_to_seed_ratio(T_eval)
get_alpha_to_seed_ratio(T_eval)
```

Require: `want_mainout_info=True`.

---

## Example Usage

```python
from WinNet_output import WinNet_output

wn = WinNet_output(
    "/path/to/output/",
    want_tracer_info=True,
    want_mainout_info=True
)

print("n/seed:", wn.get_neutron_to_seed_ratio(3.0))
print("Expansion timescale:", wn.get_expansion_timescale())
```

---

## Notes & TODOs

- Full integration with snapshot abundance tracing still in progress.
- May extend with:
  - Seed production metrics
  - More physical diagnostics
  - Automatic plotting helpers

---

## License
This class is part of a scientific analysis workflow and may be reused or extended with attribution.
