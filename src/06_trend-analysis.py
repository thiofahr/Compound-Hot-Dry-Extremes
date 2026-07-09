"""
06_trend-analysis.py
Compute an area-weighted spatial-mean time series of CHDE annual metrics
and run Sen's slope (Theil-Sen) and Mann-Kendall trend tests over a chosen period. 
Also computes a per-year ensemble mean/std/CI time series.

INPUT
-----
NetCDF files produced by 04_compute-metrics.py:
    {main_dir}/daily/chde_metrics/{model}_{scenario}_chde_metrics.nc
    dims (year, lat, lon), variables e.g. chde_n_days, chde_duration_mean,
    hot_excess_mean, dry_deficit_mean, HMI_mean, PMI_dry_mean, CHDMI_mean, ...

OUTPUT
------
One Excel workbook at:
    {main_dir}/daily/trend/chde_trend_{scope}_{period_start}-{period_end}.xlsx
with two sheets:
    "trend"      : one row per (scope group, variable) — Sen's slope + MK test
    "timeseries" : one row per (scope group, variable, year) — mean, std, ci
"""

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk
import xarray as xr
import yaml
import regionmask

from scipy.stats import mstats, t
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
VARIABLES = ['chde_n_days', 'chde_n_events',
             'chde_duration_mean', 'chde_duration_max',
             'CHDMI_sum', 'CHDMI_mean', 'CHDMI_max']

land_mask = xr.open_dataset(r"D:\Research\CDHE\data\land_mask_cmip6.nc")['land_mask']

countries = regionmask.defined_regions.natural_earth_v5_1_2.countries_50
indo_id = countries[countries.names.index('Indonesia')].number

# ---------------------------------------------------------------------------
# Config / IO (mirrors 04_compute-metrics.py)
# ---------------------------------------------------------------------------
def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return yaml.safe_load(f)

def save_excel(dfs: dict[str, pd.DataFrame], path: Path) -> None:
    """Save multiple DataFrames as separate sheets in one .xlsx workbook."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            if df is None or df.empty:
                continue
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def load_metrics_dataset(model: str, scenario: str, period: tuple, metrics_dir: Path, year_dim: str = "year") -> xr.Dataset:
    path = metrics_dir / f"{model}_{scenario}_chde_metrics.nc"
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    ds = xr.open_dataset(path)[VARIABLES].where(land_mask)
    country_mask = countries.mask(ds)
    ds = ds.where(country_mask == indo_id)

    year_coord = ds["year"].values.astype("float64")
    nan_mask = np.isnan(year_coord)
    if nan_mask.any():
        log.warning(
            f"[{model} | {scenario}] Found NaN 'year' coordinate "
            f"at position(s) {np.where(nan_mask)[0].tolist()} -- filling with 2100."
        )
        year_coord[nan_mask] = 2100
        ds = ds.assign_coords(year=year_coord)
    
    if period is not None:
        ds = ds.sel({year_dim: slice(period[0], period[1])})

    return ds.expand_dims(model=[model], scenario=[scenario])

def load_ensemble_metrics(models: list[str], scenarios: list[str], period: tuple, metrics_dir: Path) -> xr.Dataset:
    """Load per-model/scenario metrics files and combine into one Dataset
    with dims (model, scenario, year, lat, lon)."""
    datasets = []
    for model in models:
        for scenario in scenarios:
            try:
                datasets.append(load_metrics_dataset(model, scenario, period, metrics_dir))
            except FileNotFoundError as e:
                log.warning(str(e))
    if not datasets:
        raise FileNotFoundError(f"No metrics files found for models={models}, scenarios={scenarios}")
    combined = xr.combine_by_coords(datasets, combine_attrs="drop")
    for d in datasets:
        d.close()
    return combined
# ---------------------------------------------------------------------------
# Area-weighted spatial mean
# ---------------------------------------------------------------------------
def weighted_spatial_mean(
    da: xr.DataArray,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.DataArray:
    """Cosine-latitude area-weighted spatial mean over (lat, lon); all
    other dims are preserved."""
    weights = np.cos(np.deg2rad(da[lat_dim]))
    weights = weights / weights.sum()
    return da.weighted(weights).mean([lat_dim, lon_dim], skipna=True)
# ---------------------------------------------------------------------------
# Ensemble reduction + period slice
# ---------------------------------------------------------------------------
def reduce_ensemble(
    da: xr.DataArray,
    dims: list[str],
    period: tuple[int, int] | None,
    year_dim: str = "year",
) -> xr.DataArray:
    """Average over whichever of `dims` are present, then optionally slice
    to [period[0], period[1]] years. Pass period=None to keep the full
    record."""
    existing_dims = [d for d in dims if d in da.dims]
    out = da.mean(existing_dims, skipna=True) if existing_dims else da
    if period is not None:
        out = out.sel({year_dim: slice(period[0], period[1])})
    return out
# ---------------------------------------------------------------------------
# Theil-Sen + Mann-Kendall
# ---------------------------------------------------------------------------
def run_trend_tests(ts: xr.DataArray, alpha: float = 0.05, year_dim: str = "year") -> dict | None:
    """Sen's slope (Theil-Sen) + Mann-Kendall test on a 1-D annual time
    series. Returns None if there are too few valid years to test."""
    # ts = ts.squeeze(drop=True)
    values = ts.values.astype(float)
    years = ts[year_dim].values.astype(float)
    valid = ~np.isnan(values)
    values, years = values[valid], years[valid]
    if values.size < 4:
        log.warning(f"Only {values.size} valid year(s); skipping trend test.")
        return None
    slope, intercept, ci_low, ci_high = (float(x) for x in mstats.theilslopes(values, years, alpha=1 - alpha))
    mk_result = mk.original_test(values, alpha=alpha)
    return {
        "n": int(values.size),
        "slope": slope,
        "intercept": intercept,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mk_stat": mk_result.s,
        "mk_p": mk_result.p,
        "mk_trend": mk_result.trend,
        "significant": mk_result.p <= alpha,
        "alpha": alpha,
    }
# ---------------------------------------------------------------------------
# NEW: per-year mean / std / CI across ensemble members
# ---------------------------------------------------------------------------
def compute_yearly_stats(
    ts: xr.DataArray,
    ens_dims: list[str],
    alpha: float = 0.05,
    year_dim: str = "year",
) -> pd.DataFrame:
    """Given an area-weighted spatial-mean time series that still carries
    one or more ensemble dims (model and/or scenario) plus `year`, compute
    the per-year mean, std, and confidence interval across the ensemble
    members. If no ensemble dims are present (e.g. per_model_scenario
    scope, a single member), std/CI collapse to 0 / the raw value.

    Returns a DataFrame with columns:
        year, n, mean, std, ci_low, ci_high, ci
    where `ci` is the +/- half-width of the interval (mean +/- ci).
    """
    existing_dims = [d for d in ens_dims if d in ts.dims]
    if existing_dims:
        stacked = ts.stack(member=existing_dims)
    else:
        stacked = ts.expand_dims(member=[0])

    years = stacked[year_dim].values
    rows = []
    for yr in years:
        vals = stacked.sel({year_dim: yr}).values.astype(float)
        vals = vals[~np.isnan(vals)]
        n = vals.size
        if n == 0:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        if n > 1:
            se = std / np.sqrt(n)
            tval = float(t.ppf(1 - alpha / 2, df=n - 1))
            ci_half = tval * se
        else:
            ci_half = 0.0
        rows.append({
            "year": int(yr),
            "n": int(n),
            "mean": mean,
            "std": std,
            "ci_low": mean - ci_half,
            "ci_high": mean + ci_half,
            "ci": ci_half,
        })
    return pd.DataFrame(rows)
# ---------------------------------------------------------------------------
# Single-variable drivers: ensemble mean -> spatial mean -> trend/stats
# ---------------------------------------------------------------------------
def analyse_variable(
    ds: xr.Dataset,
    variable: str,
    period: tuple[int, int] | None,
    ens_dims: list[str],
    alpha: float,
) -> dict | None:
    da = ds[variable]
    da = reduce_ensemble(da, dims=ens_dims, period=period)
    ts = weighted_spatial_mean(da)
    result = run_trend_tests(ts, alpha=alpha)
    if result is not None:
        result["variable"] = variable
    return result

def analyse_variable(
    ds: xr.Dataset,
    variable: str,
    period: tuple[int, int] | None,
    ens_dims: list[str],
    alpha: float,
) -> dict | None:
    da = ds[variable]
    da = reduce_ensemble(da, dims=ens_dims, period=period)
    ts = weighted_spatial_mean(da).squeeze(drop=True)  # <-- squeeze here too
    result = run_trend_tests(ts, alpha=alpha)
    if result is not None:
        result["variable"] = variable
    return result

def analyse_variable_timeseries(
    ds: xr.Dataset,
    variable: str,
    period: tuple[int, int] | None,
    ens_dims: list[str],
    alpha: float,
    year_dim: str = "year",
) -> pd.DataFrame:
    """NEW: per-year mean/std/CI for one variable, keeping the year axis
    (i.e. no averaging over year like `analyse_variable` does)."""
    da = ds[variable]
    if period is not None:
        da = da.sel({year_dim: slice(period[0], period[1])})
    ts = weighted_spatial_mean(da)
    df = compute_yearly_stats(ts, ens_dims=ens_dims, alpha=alpha, year_dim=year_dim)
    if not df.empty:
        df["variable"] = variable
    return df
# ---------------------------------------------------------------------------
# Batch drivers across scope / models / scenarios / variables
# ---------------------------------------------------------------------------
def _build_groups(ds, models, scenarios, scope):
    """
    scope options
    -------------
    ensemble           : one trend, averaged over all models + scenarios
    per_model          : one trend per model, averaged over scenarios
    per_scenario       : one trend per scenario, averaged over models
    per_model_scenario : one trend per (model, scenario) pair, no averaging
    """
    if scope == "ensemble":
        return [("ensemble", "all", ds, ["model", "scenario"])]
    elif scope == "per_model":
        return [(m, "all", ds.sel(model=[m]), ["scenario"]) for m in models]
    elif scope == "per_scenario":
        return [("ensemble", s, ds.sel(scenario=[s]), ["model"]) for s in scenarios]
    elif scope == "per_model_scenario":
        return [(m, s, ds.sel(model=[m], scenario=[s]), []) for m in models for s in scenarios]
    else:
        raise ValueError(f"Unknown scope: {scope}")
def compute_trends(
    ds: xr.Dataset,
    models: list[str],
    scenarios: list[str],
    variables: list[str],
    period: tuple[int, int] | None,
    scope: str,
    alpha: float,
) -> list[dict]:
    groups = _build_groups(ds, models, scenarios, scope)
    rows = []
    for model_label, scenario_label, sub_ds, ens_dims in tqdm(groups, desc=f"Trend ({scope})", unit="grp"):
        for variable in variables:
            if variable not in sub_ds.data_vars:
                continue
            result = analyse_variable(sub_ds, variable, period, ens_dims, alpha)
            if result is None:
                continue
            result.update({
                "scope": scope,
                "model": model_label,
                "scenario": scenario_label,
                "period_start": period[0] if period else None,
                "period_end": period[1] if period else None,
            })
            rows.append(result)
    return rows
def compute_timeseries_stats(
    ds: xr.Dataset,
    models: list[str],
    scenarios: list[str],
    variables: list[str],
    period: tuple[int, int] | None,
    scope: str,
    alpha: float,
) -> pd.DataFrame:
    """NEW: batch driver producing one row per (scope group, variable, year)
    with mean/std/CI across ensemble members."""
    groups = _build_groups(ds, models, scenarios, scope)
    frames = []
    for model_label, scenario_label, sub_ds, ens_dims in tqdm(groups, desc=f"Timeseries ({scope})", unit="grp"):
        for variable in variables:
            if variable not in sub_ds.data_vars:
                continue
            df = analyse_variable_timeseries(sub_ds, variable, period, ens_dims, alpha)
            if df.empty:
                continue
            df["scope"] = scope
            df["model"] = model_label
            df["scenario"] = scenario_label
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    cols = ["scope", "model", "scenario", "variable", "year",
            "n", "mean", "std", "ci_low", "ci_high", "ci"]
    return out[[c for c in cols if c in out.columns]]
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute area-weighted trend (Sen's slope + Mann-Kendall) and per-year "
                     "mean/std/CI on CHDE annual metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python 06_trend-analysis.py
            python 06_trend-analysis.py --models ACCESS-CM2 MIROC6 --scenarios ssp245 --period 1981 2014
            python 06_trend-analysis.py --scope per_model_scenario --variables CHDMI_mean chde_n_days
        """,
    )
    parser.add_argument("--config", default=r"D:\Research\CDHE\configs\config.yaml")
    parser.add_argument("--models", nargs="+", metavar="MODEL")
    parser.add_argument("--scenarios", nargs="+", metavar="SCENARIO")
    parser.add_argument("--variables", nargs="+", metavar="VAR",
                         help="Metrics variables to test (default: all found in the Dataset).")
    parser.add_argument("--period", nargs=2, type=int, metavar=("START", "END"),
                         help="Year range for the trend test, e.g. --period 1981 2014 (default: full record).")
    parser.add_argument("--scope", choices=["ensemble", "per_model", "per_scenario", "per_model_scenario"],
                         default="ensemble",
                         help="ensemble: average all models+scenarios; per_model: one trend per model "
                              "(averaged over scenarios); per_scenario: one trend per scenario (averaged "
                              "over models); per_model_scenario: one trend per model x scenario pair.")
    parser.add_argument("--alpha", type=float, default=0.05,
                         help="Significance level for the MK test, Sen's CI, and the yearly CI (default 0.05).")
    parser.add_argument("--no-skip", action="store_true",
                         help="Recompute even if the output file already exists.")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    main_dir = Path(cfg["output_dir"])
    metrics_dir = main_dir / "daily" / "chde_metrics"
    trend_dir = main_dir / "daily" / "trend"
    models = args.models or list(cfg["models"].keys())
    scenarios = args.scenarios or [s for s in cfg["scenarios"].keys() if s != "historical"]
    period = tuple(args.period) if args.period else None
    label = f"{period[0]}-{period[1]}" if period else "full"
    out_path = trend_dir / f"chde_trend_{args.scope}_{label}.xlsx"
    if not args.no_skip and out_path.exists():
        log.info(f"Trend results exist, skipping: {out_path}")
        return
    log.info(f"Models: {models} | Scenarios: {scenarios} | Scope: {args.scope} | Period: {period or 'full'}")
    ds = load_ensemble_metrics(models, scenarios, period, metrics_dir)
    variables = args.variables or list(ds.data_vars)

    # Trend (Sen's slope + Mann-Kendall) -> one row per (group, variable)
    trend_rows = compute_trends(ds, models, scenarios, variables, period, args.scope, args.alpha)
    trend_cols = ["scope", "model", "scenario", "variable", "period_start", "period_end",
                  "n", "slope", "intercept", "ci_low", "ci_high",
                  "mk_stat", "mk_p", "mk_trend", "significant", "alpha"]
    df_trend = pd.DataFrame(trend_rows)
    if not df_trend.empty:
        df_trend = df_trend[[c for c in trend_cols if c in df_trend.columns]]

    # NEW: per-year mean/std/CI -> one row per (group, variable, year)
    df_timeseries = compute_timeseries_stats(ds, models, scenarios, variables, period, args.scope, args.alpha)

    ds.close()
    gc.collect()

    if df_trend.empty and df_timeseries.empty:
        log.warning("No trend or timeseries results produced.")
        return

    save_excel({"trend": df_trend, "timeseries": df_timeseries}, out_path)
    log.info(f"Saved {len(df_trend)} trend rows + {len(df_timeseries)} timeseries rows -> {out_path}")
    log.info("Done.")

if __name__ == "__main__":
    main()
