"""
04_compute-metrics.py
Compute annual CHDE metrics (frequency, duration, intensity) from the
yearly flag / event-id files produced by 03_exceedance.py.

What's directly computable from 02's output
--------------------------------------------
Frequency & duration come straight from the saved chde_flag / chde_event_id
arrays:
    n_events       : number of distinct CHDE events in the year
    n_days         : total CHDE days in the year
    duration_mean  : n_days / n_events (mean event length, days)
    duration_max   : longest single event that year (days)

For intensity, two metrics are computed:

1. Simple diagnostic:
     hot_excess_mean   : mean(tasmax - tasmax90p) on CHDE days   [degC]
     dry_deficit_mean  : mean(pr10p - pr)          on CHDE days   [mm/day]

2. Event-magnitude index, following the compound-magnitude formulation in
   Zou & Song (2024, Remote Sens. 16(22):4208), https://doi.org/10.3390/rs16224208:
     For a single event (independently, for the hot event and the dry
     event), the daily excess/deficit is min-max normalized *within that
     event's own days* to R in [0.1, 1.0] (Eqs. 3-4 / 6-7 of the paper),
     then summed over the event's days to get that event's magnitude
     (HMI for hot, PMI-style for dry). The compound magnitude for a CHDE 
     event is the PRODUCT of its parent hot event's magnitude and its 
     parent dry event's magnitude.

Output
------
One NetCDF per model/scenario at:
    {main_dir}/annual/chde_metrics/{model}_{scenario}_chde_metrics.nc
with dims (year, lat, lon) and variables:
    chde_n_events, chde_n_days, chde_duration_mean, chde_duration_max,
    hot_excess_mean, dry_deficit_mean,
    HMI_n_events, HMI_sum, HMI_mean, HMI_max,
    PMI_dry_n_events, PMI_dry_sum, PMI_dry_mean, PMI_dry_max,
    CHDMI_n_events, CHDMI_sum, CHDMI_mean, CHDMI_max
"""
import argparse
import gc
import logging
import re
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

YEAR_RE = re.compile(r"(19|20)\d{2}")
TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)

# ---------------------------------------------------------------------------
# Config / file listing (mirrors 03_exceedance.py)
# ---------------------------------------------------------------------------
def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return yaml.safe_load(f)

def list_year_files(var_dir: Path) -> dict:
    var_dir = Path(var_dir)
    files = sorted(var_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {var_dir}")
    year_map = {}
    for f in files:
        m = YEAR_RE.search(f.stem)
        if m:
            year_map[int(m.group())] = f
        else:
            with xr.open_dataset(f, decode_times=TIME_CODER) as ds:
                yrs = np.unique(ds["time"].dt.year.values)
            if len(yrs) == 1:
                year_map[int(yrs[0])] = f
    return year_map


def convert_units(da: xr.DataArray, var: str) -> xr.DataArray:
    if var == "tasmax" and da.attrs.get("units", "K") == "K":
        da = da - 273.15
        da.attrs["units"] = "degC"
    elif var == "pr" and da.attrs.get("units", "kg m-2 s-1") in ("kg m-2 s-1", "kg/m2/s"):
        da = da * 86400.0
        da.attrs["units"] = "mm day-1"
    return da

def save_dataset(ds: xr.Dataset, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding, engine="netcdf4")

# ---------------------------------------------------------------------------
# Frequency + duration: pure event_id bincount, no raw data needed
# ---------------------------------------------------------------------------
def _event_stats_1d(flag: np.ndarray, event_id: np.ndarray):
    flag = np.asarray(flag, dtype=bool)
    event_id = np.asarray(event_id, dtype=np.int64)
    n_days = int(flag.sum())
    if n_days == 0:
        return 0, 0, 0.0, 0
    n_events = int(event_id.max())
    if n_events == 0:
        return 0, n_days, 0.0, 0
    lengths = np.bincount(event_id[event_id > 0], minlength=n_events + 1)[1:]
    lengths = lengths[lengths > 0]
    if lengths.size == 0:
        return 0, n_days, 0.0, 0
    return int(lengths.size), n_days, float(lengths.mean()), int(lengths.max())


def compute_event_stats(flag: xr.DataArray, event_id: xr.DataArray):
    n_events, n_days, dur_mean, dur_max = xr.apply_ufunc(
        _event_stats_1d,
        flag,
        event_id,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        output_dtypes=[np.int32, np.int32, np.float32, np.int32],
    )
    return n_events, n_days, dur_mean, dur_max

# ---------------------------------------------------------------------------
# Intensity, part A: simple diagnostic (mean excess on CHDE days)
# ---------------------------------------------------------------------------
def compute_simple_intensity(
    hot_excess: xr.DataArray, dry_deficit: xr.DataArray, chde_flag: xr.DataArray
):
    mask = chde_flag == 1
    hot_excess_mean = hot_excess.where(mask).mean(dim="time", skipna=True)
    dry_deficit_mean = dry_deficit.where(mask).mean(dim="time", skipna=True)
    return hot_excess_mean, dry_deficit_mean

# ---------------------------------------------------------------------------
# Intensity, part B: event-magnitude index (Zou & Song 2024, Eqs. 3-10)
# ---------------------------------------------------------------------------
def _event_magnitude_1d(event_id: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Broadcast each event's summed, min-max-normalized R-delta (Eqs. 3-5 /
    6-8 of Zou & Song 2024) onto every day of that event. Returns an array
    the same length as the input; 0.0 on days outside any event."""
    event_id = np.asarray(event_id, dtype=np.int64)
    delta = np.asarray(delta, dtype=np.float64)
    out = np.zeros_like(delta)
    n_events = int(event_id.max()) if event_id.size else 0
    for k in range(1, n_events + 1):
        idx = event_id == k
        if not idx.any():
            continue
        d = delta[idx]
        dmin, dmax = d.min(), d.max()
        if dmax > dmin:
            r = 0.9 * (d - dmin) / (dmax - dmin) + 0.1
        else:
            # single-day event, or constant excess: no within-event spread to
            # normalize against -> full weight by convention (avoids 0/0)
            r = np.ones_like(d)
        out[idx] = r.sum()
    return out

def compute_day_mi(event_id: xr.DataArray, delta: xr.DataArray) -> xr.DataArray:
    """Per-day magnitude index: pass hot_event_id + temperature excess for
    HMI, or dry_event_id + precip deficit for the PMI-style dry index."""
    return xr.apply_ufunc(
        _event_magnitude_1d,
        event_id,
        delta,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        output_dtypes=[np.float64],
    )

def _mi_annual_stats_1d(event_id: np.ndarray, day_mi: np.ndarray):
    """Reduce a per-day (broadcast-constant-within-event) MI array to annual
    per-event stats: n_events, sum, mean, max."""
    event_id = np.asarray(event_id, dtype=np.int64)
    n_events = int(event_id.max()) if event_id.size else 0
    if n_events == 0:
        return 0, 0.0, 0.0, 0.0
    values = []
    for k in range(1, n_events + 1):
        idx = event_id == k
        if idx.any():
            values.append(float(day_mi[idx][0]))
    if not values:
        return 0, 0.0, 0.0, 0.0
    values = np.asarray(values)
    return len(values), float(values.sum()), float(values.mean()), float(values.max())

def compute_mi_annual_stats(event_id: xr.DataArray, day_mi: xr.DataArray):
    n_events, mi_sum, mi_mean, mi_max = xr.apply_ufunc(
        _mi_annual_stats_1d,
        event_id,
        day_mi,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        output_dtypes=[np.int32, np.float64, np.float64, np.float64],
    )
    return n_events, mi_sum, mi_mean, mi_max

def _chdmi_stats_1d(chde_event_id: np.ndarray, hot_mi: np.ndarray, dry_mi: np.ndarray):
    """Compound magnitude per CHDE event = HMI x PMI_dry"""
    chde_event_id = np.asarray(chde_event_id, dtype=np.int64)
    n_events = int(chde_event_id.max()) if chde_event_id.size else 0
    if n_events == 0:
        return 0, 0.0, 0.0, 0.0
    values = []
    for k in range(1, n_events + 1):
        idx = chde_event_id == k
        if idx.any():
            values.append(float((hot_mi[idx] * dry_mi[idx]).mean()))
    if not values:
        return 0, 0.0, 0.0, 0.0
    values = np.asarray(values)
    return len(values), float(values.sum()), float(values.mean()), float(values.max())

def compute_chdmi_stats(chde_event_id: xr.DataArray, hot_mi: xr.DataArray, dry_mi: xr.DataArray):
    n_events, chdmi_sum, chdmi_mean, chdmi_max = xr.apply_ufunc(
        _chdmi_stats_1d,
        chde_event_id,
        hot_mi,
        dry_mi,
        input_core_dims=[["time"], ["time"], ["time"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        output_dtypes=[np.int32, np.float64, np.float64, np.float64],
    )
    return n_events, chdmi_sum, chdmi_mean, chdmi_max

def compute_intensity_for_year(
    tasmax_path: Path,
    pr_path: Path,
    thr_tasmax_da: xr.DataArray,
    thr_pr_da: xr.DataArray,
    chde_flag: xr.DataArray,
    hot_event_id: xr.DataArray,
    dry_event_id: xr.DataArray,
    chde_event_id: xr.DataArray,
):
    with xr.open_dataset(tasmax_path, decode_times=TIME_CODER) as ds:
        tasmax_y = convert_units(ds["tasmax"].load(), "tasmax")
    with xr.open_dataset(pr_path, decode_times=TIME_CODER) as ds:
        pr_y = convert_units(ds["pr"].load(), "pr")

    tasmax_y, pr_y, chde_flag, hot_event_id, dry_event_id, chde_event_id = xr.align(
        tasmax_y, pr_y, chde_flag, hot_event_id, dry_event_id, chde_event_id, join="inner"
    )

    hot_excess = (tasmax_y.groupby("time.dayofyear") - thr_tasmax_da).drop_vars(
        "dayofyear", errors="ignore"
    ).clip(min=0)
    dry_deficit = (thr_pr_da - pr_y.groupby("time.dayofyear")).drop_vars(
        "dayofyear", errors="ignore"
    ).clip(min=0)

    # Part A: simple diagnostic, masked to CHDE days only
    hot_excess_mean, dry_deficit_mean = compute_simple_intensity(hot_excess, dry_deficit, chde_flag)

    # Part B: paper-style event-magnitude index, computed over each
    # component's OWN full event span (not just the compound overlap days)
    hot_mi = compute_day_mi(hot_event_id, hot_excess)
    dry_mi = compute_day_mi(dry_event_id, dry_deficit)

    hmi_n, hmi_sum, hmi_mean, hmi_max = compute_mi_annual_stats(hot_event_id, hot_mi)
    pmi_n, pmi_sum, pmi_mean, pmi_max = compute_mi_annual_stats(dry_event_id, dry_mi)
    chdmi_n, chdmi_sum, chdmi_mean, chdmi_max = compute_chdmi_stats(chde_event_id, hot_mi, dry_mi)

    result = {
        "hot_excess_mean": hot_excess_mean,
        "dry_deficit_mean": dry_deficit_mean,
        "HMI_n_events": hmi_n, "HMI_sum": hmi_sum, "HMI_mean": hmi_mean, "HMI_max": hmi_max,
        "PMI_dry_n_events": pmi_n, "PMI_dry_sum": pmi_sum, "PMI_dry_mean": pmi_mean, "PMI_dry_max": pmi_max,
        "CHDMI_n_events": chdmi_n, "CHDMI_sum": chdmi_sum, "CHDMI_mean": chdmi_mean, "CHDMI_max": chdmi_max,
    }

    del tasmax_y, pr_y, hot_excess, dry_deficit, hot_mi, dry_mi
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Per model/scenario driver
# ---------------------------------------------------------------------------
def compute_metrics_for_scenario(
    model: str,
    scenario: str,
    raw_dir: Path,
    chde_dir: Path,
    threshold_dir: Path,
    hp_label: str,
    dp_label: str,
    skip_intensity: bool,
) -> xr.Dataset:
    chde_scenario_dir = chde_dir / model / scenario
    year_files = sorted(chde_scenario_dir.glob(f"chde_{model}_{scenario}_*.nc"))
    if not year_files:
        raise FileNotFoundError(f"No CHDE files found in {chde_scenario_dir}")

    tasmax_year_map, pr_year_map = {}, {}
    if not skip_intensity:
        tasmax_year_map = list_year_files(raw_dir / model / "historical" / "tasmax")
        tasmax_year_map.update(list_year_files(raw_dir / model / scenario / "tasmax"))
        pr_year_map = list_year_files(raw_dir / model / "historical" / "pr")
        pr_year_map.update(list_year_files(raw_dir / model / scenario / "pr"))
        tasmax_thr = xr.open_dataset(
            threshold_dir / f"{model}_tasmax_threshold_1981-2014.nc", decode_times=TIME_CODER)[f"tasmax{hp_label}p"]
        pr_thr = xr.open_dataset(
            threshold_dir / f"{model}_pr_threshold_1981-2014.nc", decode_times=TIME_CODER)[f"pr{dp_label}p"]

    yearly_results = []
    for f in tqdm(year_files, desc=f"{model} | {scenario}", unit="yr", leave=False):
        m = YEAR_RE.search(f.stem)
        year = int(m.group()) if m else None

        with xr.open_dataset(f, decode_times=TIME_CODER) as ds:
            flag = ds[f"chde_p{hp_label}_{dp_label}_flag"].load()
            eid = ds[f"chde_p{hp_label}_{dp_label}_event_id"].load()
            hot_eid = ds[f"hot_p{hp_label}_event_id"].load()
            dry_eid = ds[f"dry_p{dp_label}_event_id"].load()

        n_events, n_days, dur_mean, dur_max = compute_event_stats(flag, eid)

        data_vars = {
            "chde_n_events": n_events,
            "chde_n_days": n_days,
            "chde_duration_mean": dur_mean,
            "chde_duration_max": dur_max,
        }

        if not skip_intensity and year in tasmax_year_map and year in pr_year_map:
            intensity = compute_intensity_for_year(
                tasmax_year_map[year], pr_year_map[year], tasmax_thr, pr_thr,
                flag, hot_eid, dry_eid, eid,
            )
            data_vars.update(intensity)

        year_ds = xr.Dataset(data_vars).expand_dims(year=[year])
        yearly_results.append(year_ds)
        del flag, eid
        gc.collect()

    out_ds = xr.concat(yearly_results, dim="year")
    out_ds.attrs.update({"model": model, "scenario": scenario, "hot_pct": hp_label, "dry_pct": dp_label})

    if not skip_intensity:
        del tasmax_thr, pr_thr
    gc.collect()
    return out_ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute annual CHDE frequency/duration/intensity metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python 03_compute-metrics.py
            python 03_compute-metrics.py --models ACCESS-CM2 MIROC6 --scenarios ssp245
            python 03_compute-metrics.py --skip-intensity   # frequency/duration only
        """,
    )
    parser.add_argument("--config", default=r"D:\Research\CDHE\configs\config.yaml")
    parser.add_argument("--models", nargs="+", metavar="MODEL")
    parser.add_argument("--scenarios", nargs="+", metavar="SCENARIO")
    parser.add_argument("--hot-pct", type=float, default=0.90)
    parser.add_argument("--dry-pct", type=float, default=0.10)
    parser.add_argument("--skip-intensity", action="store_true",
                         help="Skip re-opening raw data; frequency/duration only (much faster).")
    parser.add_argument("--no-skip", action="store_true",
                         help="Recompute even if the output file already exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    main_dir = Path(cfg["output_dir"])
    raw_dir = main_dir / "raw"
    threshold_dir = main_dir / "daily" / "threshold"
    chde_dir = main_dir / "daily" / "chde"
    metrics_dir = main_dir / "daily" / "chde_metrics"

    models = args.models or list(cfg["models"].keys())
    scenarios = args.scenarios or [s for s in cfg["scenarios"].keys() if s != "historical"]

    hp_label = f"{int(round(args.hot_pct * 100))}"
    dp_label = f"{int(round(args.dry_pct * 100))}"

    log.info(f"Models: {models} | Scenarios: {scenarios} | Intensity: {not args.skip_intensity}")

    for model in tqdm(models, desc="Models", unit="model"):
        for scenario in scenarios:
            out_path = metrics_dir / f"{model}_{scenario}_chde_metrics.nc"
            if not args.no_skip and out_path.exists():
                log.info(f"[{model} | {scenario}] metrics exist, skipping: {out_path}")
                continue

            log.info(f"=== {model} | {scenario} ===")
            try:
                metrics_ds = compute_metrics_for_scenario(
                    model, scenario, raw_dir, chde_dir, threshold_dir,
                    hp_label, dp_label, args.skip_intensity,
                )
            except Exception:
                log.exception(f"[{model} | {scenario}] Failed")
                continue

            save_dataset(metrics_ds, out_path)
            del metrics_ds
            gc.collect()

    log.info("Done.")


if __name__ == "__main__":
    main()
