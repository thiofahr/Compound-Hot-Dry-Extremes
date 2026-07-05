"""
03_exceedance.py
═══════════════════════════════════════════════════════════════════════════════
Identify the compound hot-dry events with binary flag and event labelling with
year-by-year pipeline.

Event definition
-----------------
* hot day : tasmax > calendar-day p-th percentile threshold (baseline)
* dry day : pr     < calendar-day p-th percentile threshold (baseline)
* an "event" requires >= min_periods consecutive qualifying days; gaps of
  <= max_gap_days are merged. Events are labeled independently per year.
* CHDE = hot_flag AND dry_flag on the same day, then re-labeled its own way.

Leap-day fix
------------
Baseline threshold is computed on the true calendar dayofyear (1-366),
so Feb-29 gets its own real threshold instead of NaN.

USAGE
─────
  # Process with default config
  python 03_exceedance.py

  # Only one model:
  python 03_exceedance.py --models ACCESS-CM2

  # Only one scenario:
  python 03_exceedance.py --scenarios historical

  # Only one variable:
  python 03_exceedance.py --variables pr

  # Combine filters:
  python 03_exceedance.py --models ACCESS-CM2 MIROC6 --scenarios ssp245 ssp585

═══════════════════════════════════════════════════════════════════════════════
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
# Config
# ---------------------------------------------------------------------------
def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# File listing (one .nc file per year)
# ---------------------------------------------------------------------------
def list_year_files(var_dir: Path) -> dict:
    """Map {year(int): file path} for a raw/{model}/{scenario}/{var} dir."""
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
            else:
                log.warning(f"Could not determine year for {f.name}, skipping")
    return year_map


# ---------------------------------------------------------------------------
# Load / convert one year
# ---------------------------------------------------------------------------
def open_year(path: Path, var: str) -> xr.DataArray:
    with xr.open_dataset(path, decode_times=TIME_CODER) as ds:
        da = ds[var].load()
    return da


def convert_units(da: xr.DataArray, var: str) -> xr.DataArray:
    """tasmax: K -> degC | pr: kg m-2 s-1 -> mm day-1"""
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
    # log.info(f"     Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Per-pixel run-length event labeling
# ---------------------------------------------------------------------------
def _label_events_1d(exceed: np.ndarray, min_periods: int, max_gap_days: int):
    exceed = np.asarray(exceed, dtype=bool)
    n = exceed.shape[0]
    flag = np.zeros(n, dtype=np.int8)
    event_id = np.zeros(n, dtype=np.int32)
    if n == 0 or not exceed.any():
        return flag, event_id

    padded = np.concatenate(([False], exceed, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    lengths = ends - starts + 1
    keep = lengths >= min_periods
    starts, ends = starts[keep], ends[keep]
    if starts.size == 0:
        return flag, event_id

    m_starts, m_ends = [int(starts[0])], [int(ends[0])]
    for s, e in zip(starts[1:], ends[1:]):
        if int(s) - m_ends[-1] - 1 <= max_gap_days:
            m_ends[-1] = int(e)
        else:
            m_starts.append(int(s))
            m_ends.append(int(e))

    for i, (s, e) in enumerate(zip(m_starts, m_ends), start=1):
        flag[s : e + 1] = 1
        event_id[s : e + 1] = i
    return flag, event_id


def label_events(exceed: xr.DataArray, min_periods: int, max_gap_days: int):
    flag, eid = xr.apply_ufunc(
        _label_events_1d,
        exceed,
        input_core_dims=[["time"]],
        output_core_dims=[["time"], ["time"]],
        vectorize=True,
        output_dtypes=[np.int8, np.int32],
        kwargs={"min_periods": min_periods, "max_gap_days": max_gap_days},
    )
    return flag.astype("int8"), eid.astype("int32")


# ---------------------------------------------------------------------------
# Step 1: Threshold (calendar-day percentile climatology) - per model, lazy
# ---------------------------------------------------------------------------
def compute_and_save_threshold(
    model: str,
    raw_dir: Path,
    var: str,
    percentiles: list,
    baseline_years: tuple,
    threshold_dir: Path,
    skip_if_exists: bool = True,
) -> xr.Dataset:
    out_path = threshold_dir / f"{model}_{var}_threshold.nc"
    if skip_if_exists and out_path.exists():
        log.info(f"[{model}] {var} threshold exists, loading: {out_path}")
        return xr.load_dataset(out_path)

    var_dir = raw_dir / model / "historical" / var
    year_map = list_year_files(var_dir)
    files = [year_map[y] for y in range(baseline_years[0], baseline_years[1] + 1) if y in year_map]
    missing = [y for y in range(baseline_years[0], baseline_years[1] + 1) if y not in year_map]
    if missing:
        log.warning(f"[{model}] {var}: missing baseline years {missing}")
    if not files:
        raise FileNotFoundError(f"[{model}] No baseline files for {var} in {var_dir}")

    log.info(f"[{model}] Computing {var} threshold from {len(files)} baseline file(s)")
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": -1}, decode_times=TIME_CODER)
    da = convert_units(ds[var], var).chunk({"time": -1})

    out_vars = {}
    for p in percentiles:
        if not 0 < p < 1:
            raise ValueError(f"Percentiles must be in (0, 1), got {p}")
        label = f"{int(round(p * 100))}"
        log.info(f"[{model}] Computing {var}{label}p (dask, lazy)")
        thr = da.groupby("time.dayofyear").quantile(p, dim="time").drop_vars("quantile", errors="ignore")
        thr.name = f"{var}{label}p"
        thr.attrs = {
            "description": f"{label}th percentile of daily {var} over baseline "
                            f"{baseline_years[0]}-{baseline_years[1]} (leap days included)",
            "model": model,
            "baseline": f"{baseline_years[0]}-{baseline_years[1]}",
            "percentile": p,
        }
        out_vars[thr.name] = thr

    thr_ds = xr.Dataset(out_vars, attrs={"model": model, "var": var,
                                          "baseline": f"{baseline_years[0]}-{baseline_years[1]}"})
    thr_ds = thr_ds.load()
    save_dataset(thr_ds, out_path)
    ds.close()
    del ds, da
    gc.collect()
    return thr_ds


# ---------------------------------------------------------------------------
# Step 2: per-year CHDE - compute exceedance, flag, label, save
# ---------------------------------------------------------------------------
def build_year_list(historical_years: tuple, scenario_years: tuple) -> list:
    years = set(range(historical_years[0], historical_years[1] + 1))
    years |= set(range(scenario_years[0], scenario_years[1] + 1))
    return sorted(years)


def process_year(
    year: int,
    tasmax_path: Path,
    pr_path: Path,
    thr_tasmax_da: xr.DataArray,
    thr_pr_da: xr.DataArray,
    hp_label: str,
    dp_label: str,
    min_periods: int,
    max_gap_days: int,
    chde_max_gap_days: int,
) -> xr.Dataset:
    """Compute exceedance -> flag -> label events for a single year."""
    tasmax_y = convert_units(open_year(tasmax_path, "tasmax"), "tasmax")
    pr_y = convert_units(open_year(pr_path, "pr"), "pr")
    tasmax_y, pr_y = xr.align(tasmax_y, pr_y, join="inner")

    # exceedance
    hot_exceed = (tasmax_y.groupby("time.dayofyear") > thr_tasmax_da).drop_vars("dayofyear", errors="ignore")
    dry_exceed = (pr_y.groupby("time.dayofyear") < thr_pr_da).drop_vars("dayofyear", errors="ignore")

    # flag + label (hot / dry, independently within this year)
    hot_flag, hot_eid = label_events(hot_exceed, min_periods, max_gap_days)
    dry_flag, dry_eid = label_events(dry_exceed, min_periods, max_gap_days)

    # CHDE = hot AND dry, then its own event labeling
    chde_exceed = (hot_flag == 1) & (dry_flag == 1)
    chde_flag, chde_eid = label_events(chde_exceed, min_periods=1, max_gap_days=chde_max_gap_days)

    ds = xr.Dataset({
        f"hot_p{hp_label}_flag": hot_flag,
        f"hot_p{hp_label}_event_id": hot_eid,
        f"dry_p{dp_label}_flag": dry_flag,
        f"dry_p{dp_label}_event_id": dry_eid,
        f"chde_p{hp_label}_{dp_label}_flag": chde_flag,
        f"chde_p{hp_label}_{dp_label}_event_id": chde_eid,
    })
    ds.attrs.update({
        "year": year,
        "min_periods": min_periods,
        "max_gap_days": max_gap_days,
        "chde_max_gap_days": chde_max_gap_days,
    })

    del tasmax_y, pr_y, hot_exceed, dry_exceed, chde_exceed
    gc.collect()
    return ds


def compute_chde_for_scenario(
    model: str,
    scenario: str,
    raw_dir: Path,
    chde_dir: Path,
    tasmax_thr: xr.Dataset,
    pr_thr: xr.Dataset,
    hot_pct: list,
    dry_pct: list,
    historical_years: tuple,
    scenario_years: tuple,
    min_periods: int,
    max_gap_days: int,
    chde_max_gap_days: int,
    skip_if_exists: bool,
) -> None:
    years = build_year_list(historical_years, scenario_years)

    tasmax_year_map = list_year_files(raw_dir / model / "historical" / "tasmax")
    tasmax_year_map.update(list_year_files(raw_dir / model / scenario / "tasmax"))
    pr_year_map = list_year_files(raw_dir / model / "historical" / "pr")
    pr_year_map.update(list_year_files(raw_dir / model / scenario / "pr"))

    hp, dp = hot_pct[0], dry_pct[0]
    hp_label, dp_label = f"{int(round(hp * 100))}", f"{int(round(dp * 100))}"
    thr_tasmax_da = tasmax_thr[f"tasmax{hp_label}p"]
    thr_pr_da = pr_thr[f"pr{dp_label}p"]

    out_dir = chde_dir / model / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in tqdm(years, desc=f"{model} | {scenario} years", unit="yr", leave=False):
        out_path = out_dir / f"chde_{model}_{scenario}_{year}.nc"
        if skip_if_exists and out_path.exists():
            continue
        if year not in tasmax_year_map or year not in pr_year_map:
            log.warning(f"[{model} | {scenario}] year {year} missing tasmax or pr file, skipping")
            continue

        try:
            year_ds = process_year(
                year, tasmax_year_map[year], pr_year_map[year],
                thr_tasmax_da, thr_pr_da, hp_label, dp_label,
                min_periods, max_gap_days, chde_max_gap_days,
            )
        except Exception:
            log.exception(f"[{model} | {scenario}] year {year} failed, skipping")
            continue

        save_dataset(year_ds, out_path)
        del year_ds
        gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-year CHDE flags from CMIP6 daily data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python 03_exceedance.py
        python 03_exceedance.py --models ACCESS-CM2 MIROC6
        python 03_exceedance.py --scenarios ssp245 ssp585
        python 03_exceedance.py --no-skip
        """,
    )
    parser.add_argument("--config", default=r"D:\Research\CDHE\configs\config.yaml")
    parser.add_argument("--models", nargs="+", metavar="MODEL")
    parser.add_argument("--scenarios", nargs="+", metavar="SCENARIO")
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--hot-pct", nargs="+", type=float, default=[0.90])
    parser.add_argument("--dry-pct", nargs="+", type=float, default=[0.10])
    parser.add_argument("--min-periods", type=int, default=3)
    parser.add_argument("--max-gap-days", type=int, default=1)
    parser.add_argument("--chde-max-gap-days", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    main_dir = Path(cfg["output_dir"])
    raw_dir = main_dir / "raw"
    threshold_dir = main_dir / "daily" / "threshold"
    chde_dir = main_dir / "daily" / "chde"

    models = args.models or list(cfg["models"].keys())
    scenarios = args.scenarios or list(cfg["scenarios"].keys())
    proj_scenarios = [s for s in scenarios if s != "historical"]
    if not proj_scenarios:
        log.warning("No projection scenarios given. Add ssp245 / ssp585 etc.")
        return

    baseline_years = tuple(cfg["period"]["threshold"])
    historical_years = tuple(cfg["scenarios"]["historical"]["years"])
    skip_if_exists = not args.no_skip

    log.info(f"Models: {models} | Projection scenarios: {proj_scenarios}")

    for model in tqdm(models, desc="Models", unit="model"):
        # ---- Step 1: threshold, once per model, shared across scenarios ----
        tasmax_thr = xr.open_dataset(threshold_dir / f"{model}_tasmax_threshold_1981-2014.nc", decode_times=TIME_CODER)
        pr_thr = xr.open_dataset(threshold_dir / f"{model}_pr_threshold_1981-2014.nc", decode_times=TIME_CODER)

        # ---- Step 2: CHDE per scenario, per year ----
        for scenario in proj_scenarios:
            log.info(f"=== {model} | {scenario} ===")
            scenario_years = tuple(cfg["scenarios"][scenario]["years"])
            try:
                compute_chde_for_scenario(
                    model, scenario, raw_dir, chde_dir,
                    tasmax_thr, pr_thr, args.hot_pct, args.dry_pct,
                    historical_years, scenario_years,
                    args.min_periods, args.max_gap_days, args.chde_max_gap_days,
                    skip_if_exists,
                )
            except Exception:
                log.exception(f"[{model} | {scenario}] Failed")
                continue

        del tasmax_thr, pr_thr
        gc.collect()

    log.info("Done.")

if __name__ == "__main__":
    main()
