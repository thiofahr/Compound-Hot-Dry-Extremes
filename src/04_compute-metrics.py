"""
04_compute-metrics.py
Compute annual CHDE metrics (frequency, duration, severity) from the
yearly flag / event-id files produced by 03_exceedance.py.

Frequency & duration
---------------------
Straight from the saved chde_flag / chde_event_id arrays:
    chde_n_events      : number of distinct CHDE events in the year
    chde_n_days        : total CHDE days in the year
    chde_duration_mean : n_days / n_events (mean event length, days)
    chde_duration_max  : longest single event that year (days)

Severity (additive z-score compound index)
--------------------------------------------
For each hot event, z_hot = mean over the event's days of
    (ΔT_day - hot_mu) / hot_sigma
where hot_mu/hot_sigma are FIXED baseline (1981-2014) mean/std of ΔT
on hot-exceedance days (see 05_compute-baseline-severity.py).
Same for z_dry using ΔDI, dry_mu, dry_sigma.

For each CHDE (compound) event, severity = z_hot + z_dry is 
averaged over each CHDE event. 

Annual outputs:
    severity_mean : mean severity across all CHDE events in the year
    severity_max  : max severity across all CHDE events in the year

Output
------
One NetCDF per model/scenario at:
    {main_dir}/daily/chde-metrics/{model}_{scenario}_chde_metrics.nc
with dims (year, lat, lon) and variables:
    chde_n_events, chde_n_days, chde_duration_mean, chde_duration_max,
    severity_mean, severity_max
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
# Severity: additive fixed-baseline z-score index (Option 2)
# ---------------------------------------------------------------------------
def _event_zscore_1d(event_id: np.ndarray, delta: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Broadcast each event's mean z-score (measured against the FIXED
    baseline mu/sigma, not the event's own days) onto every day of that
    event. Returns an array the same length as delta; 0.0 outside events."""
    event_id = np.asarray(event_id, dtype=np.int64)
    delta = np.asarray(delta, dtype=np.float64)
    out = np.zeros_like(delta)
    if not np.isfinite(sigma) or sigma <= 0:
        return out
    n_events = int(event_id.max()) if event_id.size else 0
    for k in range(1, n_events + 1):
        idx = event_id == k
        if not idx.any():
            continue
        z = (delta[idx] - mu) / sigma
        out[idx] = z.mean()
    return out


def compute_day_zscore(event_id: xr.DataArray, delta: xr.DataArray, mu: xr.DataArray, sigma: xr.DataArray) -> xr.DataArray:
    """Per-day z-score, constant within each event (mu/sigma are fixed
    per-pixel baseline anchors, dims (lat, lon), no time dimension)."""
    return xr.apply_ufunc(
        _event_zscore_1d,
        event_id,
        delta,
        mu,
        sigma,
        input_core_dims=[["time"], ["time"], [], []],
        output_core_dims=[["time"]],
        vectorize=True,
        output_dtypes=[np.float64],
    )


def _severity_annual_1d(chde_event_id: np.ndarray, hot_z: np.ndarray, dry_z: np.ndarray):
    """Per-CHDE-event severity = mean(hot_z) + mean(dry_z) over that
    event's own days. Annual output = mean and max severity across all
    CHDE events in the year."""
    chde_event_id = np.asarray(chde_event_id, dtype=np.int64)
    n_events = int(chde_event_id.max()) if chde_event_id.size else 0
    if n_events == 0:
        return 0.0, 0.0
    values = []
    for k in range(1, n_events + 1):
        idx = chde_event_id == k
        if idx.any():
            values.append(float(hot_z[idx].mean() + dry_z[idx].mean()))
    if not values:
        return 0.0, 0.0
    values = np.asarray(values)
    return float(values.mean()), float(values.max())


def compute_severity_annual_stats(chde_event_id: xr.DataArray, hot_z: xr.DataArray, dry_z: xr.DataArray):
    severity_mean, severity_max = xr.apply_ufunc(
        _severity_annual_1d,
        chde_event_id,
        hot_z,
        dry_z,
        input_core_dims=[["time"], ["time"], ["time"]],
        output_core_dims=[[], []],
        vectorize=True,
        output_dtypes=[np.float64, np.float64],
    )
    return severity_mean, severity_max


def compute_severity_for_year(
    tasmax_path: Path,
    pr_path: Path,
    thr_tasmax_da: xr.DataArray,
    thr_pr_da: xr.DataArray,
    hot_event_id: xr.DataArray,
    dry_event_id: xr.DataArray,
    chde_event_id: xr.DataArray,
    hot_mu: xr.DataArray,
    hot_sigma: xr.DataArray,
    dry_mu: xr.DataArray,
    dry_sigma: xr.DataArray,
):
    with xr.open_dataset(tasmax_path, decode_times=TIME_CODER) as ds:
        tasmax_y = convert_units(ds["tasmax"].load(), "tasmax")
    with xr.open_dataset(pr_path, decode_times=TIME_CODER) as ds:
        pr_y = convert_units(ds["pr"].load(), "pr")

    tasmax_y, pr_y, hot_event_id, dry_event_id, chde_event_id = xr.align(
        tasmax_y, pr_y, hot_event_id, dry_event_id, chde_event_id, join="inner"
    )

    hot_excess = (tasmax_y.groupby("time.dayofyear") - thr_tasmax_da).drop_vars(
        "dayofyear", errors="ignore"
    ).clip(min=0)
    dry_deficit = (thr_pr_da - pr_y.groupby("time.dayofyear")).drop_vars(
        "dayofyear", errors="ignore"
    ).clip(min=0)

    hot_z = compute_day_zscore(hot_event_id, hot_excess, hot_mu, hot_sigma)
    dry_z = compute_day_zscore(dry_event_id, dry_deficit, dry_mu, dry_sigma)

    severity_mean, severity_max = compute_severity_annual_stats(chde_event_id, hot_z, dry_z)

    result = {"severity_mean": severity_mean, "severity_max": severity_max}

    del tasmax_y, pr_y, hot_excess, dry_deficit, hot_z, dry_z
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
    baseline_dir: Path,
    hp_label: str,
    dp_label: str,
    skip_severity: bool,
) -> xr.Dataset:
    chde_scenario_dir = chde_dir / model / scenario
    year_files = sorted(chde_scenario_dir.glob(f"chde_{model}_{scenario}_*.nc"))
    if not year_files:
        raise FileNotFoundError(f"No CHDE files found in {chde_scenario_dir}")

    tasmax_year_map, pr_year_map = {}, {}
    if not skip_severity:
        tasmax_year_map = list_year_files(raw_dir / model / "historical" / "tasmax")
        tasmax_year_map.update(list_year_files(raw_dir / model / scenario / "tasmax"))
        pr_year_map = list_year_files(raw_dir / model / "historical" / "pr")
        pr_year_map.update(list_year_files(raw_dir / model / scenario / "pr"))

        tasmax_thr = xr.open_dataset(
            threshold_dir / f"{model}_tasmax_threshold_1981-2014.nc", decode_times=TIME_CODER
        )[f"tasmax{hp_label}p"]
        pr_thr = xr.open_dataset(
            threshold_dir / f"{model}_pr_threshold_1981-2014.nc", decode_times=TIME_CODER
        )[f"pr{dp_label}p"]

        baseline_path = baseline_dir / f"{model}_severity_baseline_1981-2014.nc"
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Missing severity baseline anchors for {model}: {baseline_path}. "
                f"Run 05_compute-baseline-severity.py first."
            )
        baseline_ds = xr.open_dataset(baseline_path)
        hot_mu, hot_sigma = baseline_ds["hot_mu"], baseline_ds["hot_sigma"]
        dry_mu, dry_sigma = baseline_ds["dry_mu"], baseline_ds["dry_sigma"]

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

        if not skip_severity and year in tasmax_year_map and year in pr_year_map:
            severity = compute_severity_for_year(
                tasmax_year_map[year], pr_year_map[year], tasmax_thr, pr_thr,
                hot_eid, dry_eid, eid,
                hot_mu, hot_sigma, dry_mu, dry_sigma,
            )
            data_vars.update(severity)

        year_ds = xr.Dataset(data_vars).expand_dims(year=[year])
        yearly_results.append(year_ds)
        del flag, eid, hot_eid, dry_eid
        gc.collect()

    out_ds = xr.concat(yearly_results, dim="year")
    # keep exactly the requested variables, in a fixed order
    keep_vars = ["chde_n_events", "chde_n_days", "chde_duration_mean", "chde_duration_max"]
    if not skip_severity:
        keep_vars += ["severity_mean", "severity_max"]
    out_ds = out_ds[[v for v in keep_vars if v in out_ds.data_vars]]

    out_ds.attrs.update({
        "model": model, "scenario": scenario, "hot_pct": hp_label, "dry_pct": dp_label,
        "severity_method": "additive fixed-baseline z-score (Option 2): "
                            "severity = mean_event(z_hot) + mean_event(z_dry), "
                            "z measured against 1981-2014 pooled exceedance-day mu/sigma.",
    })

    if not skip_severity:
        del tasmax_thr, pr_thr, baseline_ds
    gc.collect()
    return out_ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute annual CHDE frequency/duration/severity metrics (additive z-score severity).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python 04_compute-metrics.py
            python 04_compute-metrics.py --models ACCESS-CM2 MIROC6 --scenarios ssp245
            python 04_compute-metrics.py --skip-severity   # frequency/duration only

            Requires 05_compute-baseline-severity.py to have been run first
            (produces the fixed 1981-2014 mu/sigma anchors used for severity).
        """,
    )
    parser.add_argument("--config", default=r"D:\Research\CDHE\configs\config.yaml")
    parser.add_argument("--models", nargs="+", metavar="MODEL")
    parser.add_argument("--scenarios", nargs="+", metavar="SCENARIO")
    parser.add_argument("--hot-pct", type=float, default=0.90)
    parser.add_argument("--dry-pct", type=float, default=0.10)
    parser.add_argument("--skip-severity", action="store_true",
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
    baseline_dir = main_dir / "daily" / "severity-baseline"
    metrics_dir = main_dir / "daily" / "metrics"

    models = args.models or list(cfg["models"].keys())
    scenarios = args.scenarios or [s for s in cfg["scenarios"].keys() if s != "historical"]
    hp_label = f"{int(round(args.hot_pct * 100))}"
    dp_label = f"{int(round(args.dry_pct * 100))}"

    log.info(f"Models: {models} | Scenarios: {scenarios} | Severity: {not args.skip_severity}")
    for model in tqdm(models, desc="Models", unit="model"):
        for scenario in scenarios:
            out_path = metrics_dir / f"{model}_{scenario}_chde_metrics.nc"
            if not args.no_skip and out_path.exists():
                log.info(f"[{model} | {scenario}] metrics exist, skipping: {out_path}")
                continue
            log.info(f"=== {model} | {scenario} ===")
            try:
                metrics_ds = compute_metrics_for_scenario(
                    model, scenario, raw_dir, chde_dir, threshold_dir, baseline_dir,
                    hp_label, dp_label, args.skip_severity,
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
