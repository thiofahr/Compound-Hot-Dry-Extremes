"""
05_toe-regression.py
=====================
Compute the time of emergence (ToE) of compound hot-dry event (CHDE)
metrics using the linear-regression signal-threshold method.

METHODOLOGY
-----------
    input: annual CHDE metrics
    method: Signal threshold method following Snover & Salathe (2015) and
            Maraun et al (2013), as applied in Seck et al (2026, Environ.
            Res. Commun.).

    A linear trend is fit to the annual series from `scan_start` (2015)
    onward. The fitted trend is superimposed on the historical baseline
    mean to reconstruct a smoothed projected signal:

        S(t) = mu_hist + beta * (t - scan_start)

    where mu_hist is the baseline-period mean and beta is the OLS slope
    of the annual series over the scan period. ToE is the first year t
    at which S(t) exits an "emergence envelope" built from historical
    baseline percentiles:

        S(t) > P_hi   or   S(t) < P_lo

    Two envelope widths are computed for every variable:
        90% envelope : P5  / P95   (wide  -- tolerant systems)
        60% envelope : P20 / P80   (narrow -- sensitive systems)

    Because the signal is a smoothed linear reconstruction rather than
    raw annual values, no persistence criterion is required (unlike
    05_ks-test.py's KS-based approach) -- a single crossing of the
    envelope by the fitted trend line is sufficient.

CATEGORY -> VARIABLE MAPPING
-----------------------------
Each run targets one or more categories. For every requested category, 
ToE is computed for ALL variables defined under that category below:
    frequency : chde_n_events, chde_n_days
    duration  : chde_duration_mean, chde_duration_max
    magnitude  : magnitude_mean, magnitude_max

Variables within a category are bundled as separate named DataArrays
inside that category's single output file (see OUTPUT STRUCTURE below).

OUTPUT STRUCTURE
----------------
data/
  daily/
    toe_regression/
      {category}/
        toe_regression_{model}_{scenario}_{category}.nc
          variables: toe90_{variable}, toe60_{variable}  for each
                     variable defined under that category, plus
                     slope_{variable} (trend, units/year) and
                     mu_hist_{variable} (baseline mean).

USAGE
-----
  python 06_toe-regression.py
  python 06_toe-regression.py --models ACCESS-CM2 MIROC6
  python 06_toe-regression.py --scenarios ssp245 ssp585
  python 06_toe-regression.py --categories frequency intensity
  python 06_toe-regression.py --no-skip
  python 06_toe-regression.py --config configs/config.yaml
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from scipy.stats import linregress
from tqdm import tqdm

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
BASELINE_START   = 1981
BASELINE_END     = 2014
SCAN_START       = 2015
NO_EMERGE_FILL   = 2100
ENVELOPES        = {
    "90": (5, 95),    # wide  envelope -- tolerant systems
    "60": (20, 80),   # narrow envelope -- sensitive systems
}
ENCODE_OPTS      = dict(zlib=True, complevel=4, dtype="float32")

# -----------------------------------------------------------------------------
# CATEGORY -> VARIABLE MAPPING (drives output folder/filename + what gets run)
# -----------------------------------------------------------------------------
CATEGORY_MAP = {
    "frequency": ["chde_n_events", "chde_n_days"],
    "duration": ["chde_duration_mean", "chde_duration_max"],
    "magnitude": ["magnitude_mean", "magnitude_max"],
}
VAR_TO_CATEGORY = {v: cat for cat, vs in CATEGORY_MAP.items() for v in vs}

# Default: run every defined category, computing ToE for all of that
# category's variables. Restrict with --categories to run fewer.
DEFAULT_CATEGORIES = list(CATEGORY_MAP.keys())


# =============================================================================
# SECTION 1 — CONFIG
# =============================================================================
def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return yaml.safe_load(f)


# =============================================================================
# SECTION 2 — I/O HELPERS
# =============================================================================
def load_chde_metric(
    model: str, scenario: str, variable: str, metrics_dir: Path
) -> xr.DataArray | None:
    """
    Load a single annual CHDE metric variable from 04_compute-metrics.py's
    output file.

    Returns
    -------
    xr.DataArray  dims (year, lat, lon)
    None if the source file or the requested variable is missing.
    """
    src = metrics_dir / f"{model}_{scenario}_chde_metrics.nc"
    if not src.exists():
        log.warning(f"[{model} | {scenario} | {variable}] Source file not found: {src}")
        return None

    with xr.open_dataset(src, engine="netcdf4") as ds:
        if variable not in ds:
            log.warning(
                f"[{model} | {scenario}] Variable '{variable}' not found in {src} "
                f"(available: {list(ds.data_vars)})"
            )
            return None
        da = ds[variable].load()

    # -- Guard 1: fill any NaN 'year' coordinate with 2100. This happens if
    # an upstream file's year couldn't be parsed (e.g. an incomplete
    # final-year file in 04_compute-metrics.py). Filling with 2100 (rather
    # than dropping) keeps the array length consistent and matches
    # NO_EMERGE_FILL, since the missing year is presumably the final one.
    year_coord = da["year"].values.astype("float64")
    nan_mask = np.isnan(year_coord)
    if nan_mask.any():
        log.warning(
            f"[{model} | {scenario} | {variable}] Found NaN 'year' coordinate "
            f"at position(s) {np.where(nan_mask)[0].tolist()} -- filling with 2100."
        )
        year_coord[nan_mask] = 2100
        da = da.assign_coords(year=year_coord)

    vals, counts = np.unique(da["year"].values, return_counts=True)
    dupes = vals[counts > 1]
    if dupes.size:
        log.warning(
            f"[{model} | {scenario} | {variable}] Duplicate year(s) after fill: "
            f"{dupes.tolist()} -- keeping first occurrence only."
        )
        _, first_idx = np.unique(da["year"].values, return_index=True)
        da = da.isel(year=np.sort(first_idx))

    if da.sizes.get("year", 0) == 0:
        log.warning(f"[{model} | {scenario} | {variable}] No valid years remain after cleaning.")
        return None

    da.name = variable
    log.info(
        f"[{model} | {scenario} | {variable}] Loaded  "
        f"{int(da.year.values[0])}-{int(da.year.values[-1])}  "
        f"({da.sizes['year']} years)"
    )
    return da


def save_combined_nc(ds: xr.Dataset, outpath: Path) -> None:
    """Save a Dataset to compressed NetCDF4, preserving per-variable attrs."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: ENCODE_OPTS for v in ds.data_vars}
    ds.to_netcdf(outpath, encoding=encoding, engine="netcdf4")
    log.info(f"Saved -> {outpath}  ({outpath.stat().st_size / 1e6:.1f} MB)")


# =============================================================================
# SECTION 3 — REGRESSION-BASED ToE  (pixel-level)
# =============================================================================
def regression_toe(
    series: np.ndarray,
    years:  np.ndarray,
    baseline_start: int = BASELINE_START,
    baseline_end:   int = BASELINE_END,
    scan_start:     int = SCAN_START,
    no_emerge_fill: int = NO_EMERGE_FILL,
) -> dict:
    """
    Compute Time of Emergence for a single pixel using the linear
    regression signal-threshold method (Snover & Salathe, 2015;
    Maraun et al., 2013; as applied in Seck et al., 2026).

    Steps
    -----
    1. mu_hist = mean of `series` over the baseline period.
    2. Percentile envelope bounds (P5/P95 and P20/P80) computed from the
       SAME baseline period.
    3. beta = OLS slope of `series` vs `years`, fit ONLY over years
       >= scan_start (isolates the future scenario-forced trend from
       historical/transient variability, per the CMIP6 historical/
       scenario split at 2015).
    4. Reconstructed signal S(t) = mu_hist + beta * (t - scan_start)
       for every year t in the scan period.
    5. ToE (per envelope) = first year t where S(t) > P_hi or
       S(t) < P_lo. If S(t) never exits the envelope, ToE = no_emerge_fill.

    No persistence check is applied: the linear-regression signal is
    already smoothed and insensitive to individual anomalous years.

    Parameters
    ----------
    series : 1-D array of the chosen annual CHDE metric, shape (n_years,)
    years  : 1-D array of corresponding years, shape (n_years,)

    Returns
    -------
    dict with keys: toe90, toe60, slope, mu_hist
        toe90 / toe60 : float -- ToE year for the 90%/60% envelope, or
                        no_emerge_fill if the signal never emerges.
        slope         : float -- OLS trend (units of `series` per year).
        mu_hist       : float -- baseline-period mean.
    """
    nan_result = {"toe90": np.nan, "toe60": np.nan, "slope": np.nan, "mu_hist": np.nan}

    years = np.asarray(years, dtype="float64")
    series = np.asarray(series, dtype="float64")
    valid_year = ~np.isnan(years)
    if not valid_year.all():
        years  = years[valid_year]
        series = series[valid_year]
    if years.size == 0:
        return nan_result

    # -- Baseline stats (mean + percentile envelope bounds) ----------------
    base_mask = (years >= baseline_start) & (years <= baseline_end)
    base      = series[base_mask]
    base      = base[~np.isnan(base)]
    if len(base) < 10:
        return nan_result   # ocean / masked pixel

    mu_hist = float(np.mean(base))
    envelope_bounds = {}
    for env_label, (p_lo, p_hi) in ENVELOPES.items():
        envelope_bounds[env_label] = (
            float(np.percentile(base, p_lo)),
            float(np.percentile(base, p_hi)),
        )

    # -- Fit trend over the scan period only --------------------------------
    scan_mask  = years >= scan_start
    scan_years = years[scan_mask]
    scan_vals  = series[scan_mask]
    valid = ~np.isnan(scan_vals)
    scan_years, scan_vals = scan_years[valid], scan_vals[valid]
    if len(scan_years) < 10:
        return nan_result

    reg = linregress(scan_years, scan_vals)
    beta = float(reg.slope)

    # -- Reconstruct smoothed signal and find first envelope crossing -------
    all_scan_years = np.arange(int(scan_years.min()), int(scan_years.max()) + 1)
    signal = mu_hist + beta * (all_scan_years - scan_start)

    result = {"slope": beta, "mu_hist": mu_hist}
    for env_label, (lo, hi) in envelope_bounds.items():
        exceed = np.where((signal > hi) | (signal < lo))[0]
        toe = float(all_scan_years[exceed[0]]) if exceed.size > 0 else float(no_emerge_fill)
        result[f"toe{env_label}"] = toe

    return result


# =============================================================================
# SECTION 4 — GRID-LEVEL ToE MAPS
# =============================================================================
def compute_toe_maps(
    annual:   xr.DataArray,
    model:    str,
    scenario: str,
    variable: str,
) -> dict[str, xr.DataArray]:
    """
    Apply `regression_toe` over every (lat, lon) pixel via xr.apply_ufunc.

    Returns
    -------
    dict of xr.DataArray, each dims (lat, lon):
        {f"toe90_{variable}", f"toe60_{variable}",
         f"slope_{variable}", f"mu_hist_{variable}"}
    """
    years_arr = annual.year.values
    log.info(f"[{model} | {scenario} | {variable}] Computing regression ToE maps ...")

    def _pixel(s):
        r = regression_toe(s, years_arr)
        return r["toe90"], r["toe60"], r["slope"], r["mu_hist"]

    toe90, toe60, slope, mu_hist = xr.apply_ufunc(
        _pixel,
        annual,
        input_core_dims=[["year"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float, float],
    )

    out = {}
    for env_label, da in (("90", toe90), ("60", toe60)):
        da = da.copy()
        var_name = f"toe{env_label}_{variable}"
        da.name = var_name
        da.attrs = {
            "description":      f"Time of Emergence ({env_label}% envelope) of CHDE metric '{variable}'",
            "source_variable":  variable,
            "category":         VAR_TO_CATEGORY.get(variable, "unknown"),
            "method":           "Linear regression signal threshold (Snover & Salathe, 2015)",
            "envelope":         f"{env_label}% (P{ENVELOPES[env_label][0]}/P{ENVELOPES[env_label][1]})",
            "model":            model,
            "scenario":         scenario,
            "baseline_period":  f"{BASELINE_START}-{BASELINE_END}",
            "scan_start":       SCAN_START,
            "no_emerge_fill":   NO_EMERGE_FILL,
            "units":            "year",
        }
        out[var_name] = da

    slope_name = f"slope_{variable}"
    slope = slope.copy()
    slope.name = slope_name
    slope.attrs = {
        "description":     f"OLS trend of '{variable}' over the scan period ({SCAN_START}-end)",
        "source_variable": variable,
        "model":           model,
        "scenario":        scenario,
        "units":           f"{variable} per year",
    }
    out[slope_name] = slope

    mu_name = f"mu_hist_{variable}"
    mu_hist = mu_hist.copy()
    mu_hist.name = mu_name
    mu_hist.attrs = {
        "description":     f"Baseline-period ({BASELINE_START}-{BASELINE_END}) mean of '{variable}'",
        "source_variable": variable,
        "model":           model,
        "scenario":        scenario,
    }
    out[mu_name] = mu_hist

    log.info(f"[{model} | {scenario} | {variable}] Regression ToE maps done")
    return out


# =============================================================================
# SECTION 5 — FULL PIPELINE
# =============================================================================
def build_ensemble(
    models:   list[str],
    scenario: str,
    category: str,
    toe_dir:  Path,
) -> None:
    """
    Stack each model's per-category regression-ToE output for this
    scenario along a new 'model' dimension and save one combined
    ensemble file directly under toe_dir (not the per-category
    subfolder), named:
        toe_regression_ensemble_{scenario}_{category}.nc
    """
    datasets     = []
    valid_models = []
    for model in models:
        src = toe_dir / category / f"toe_regression_{model}_{scenario}_{category}.nc"
        if not src.exists():
            log.warning(f"[ensemble | {scenario} | {category}] Missing {src}, skipping model {model}")
            continue
        with xr.open_dataset(src, engine="netcdf4") as ds:
            datasets.append(ds.load())
        valid_models.append(model)

    if not datasets:
        log.warning(f"[ensemble | {scenario} | {category}] No model outputs found, skipping ensemble.")
        return
    if len(datasets) < len(models):
        missing = sorted(set(models) - set(valid_models))
        log.warning(
            f"[ensemble | {scenario} | {category}] Ensemble built from {len(valid_models)}/{len(models)} "
            f"models -- missing: {missing}"
        )

    ensemble = xr.concat(
        datasets,
        dim=xr.DataArray(valid_models, dims="model", name="model"),
    )
    ensemble.attrs = {
        "title":       f"Time of Emergence (regression) ensemble — CHDE {category}",
        "method":      "Linear regression signal threshold (Snover & Salathe, 2015; "
                       "Maraun et al., 2013; as in Seck et al., 2026)",
        "scenario":    scenario,
        "category":    category,
        "models":      ", ".join(valid_models),
        "n_models":    len(valid_models),
        "conventions": "CF-1.8",
    }

    out_path = toe_dir / f"toe_regression_ensemble_{scenario}_{category}.nc"
    save_combined_nc(ensemble, out_path)

def run_all(
    models:         list[str],
    proj_scenarios: list[str],
    categories:     list[str],
    cfg:            dict,
    skip_existing:  bool = True,
) -> None:
    output_root = Path(cfg["output_dir"])
    metrics_dir = output_root / "daily" / "metrics"
    toe_dir     = output_root / "daily" / "toe_regression"

    unknown_cats = [c for c in categories if c not in CATEGORY_MAP]
    if unknown_cats:
        log.warning(
            f"Ignoring unrecognized categor{'y' if len(unknown_cats)==1 else 'ies'}: "
            f"{unknown_cats}. Known categories: {list(CATEGORY_MAP)}"
        )
    valid_categories = [c for c in categories if c in CATEGORY_MAP]
    if not valid_categories:
        log.warning("No valid categories to process.")
        return

    log.info(f"Processing {len(models)} model(s) x {len(proj_scenarios)} scenario(s)")
    log.info(f"Categories: {valid_categories}")
    for cat in valid_categories:
        log.info(f"  Variables: {CATEGORY_MAP[cat]}")

    for scenario in proj_scenarios:
        for category in tqdm(valid_categories, desc=f"{scenario} categories", unit="cat"):
            vars_in_cat = CATEGORY_MAP[category]

            for model in models:
                print()
                log.info(f"[{model} | {scenario} | {category}] ---- Starting regression ToE pipeline ----")

                out_path = toe_dir / category / f"toe_regression_{model}_{scenario}_{category}.nc"
                if skip_existing and out_path.exists():
                    log.info(f"SKIP — output already exists: {out_path}")
                    continue

                collected: dict[str, xr.DataArray] = {}
                for variable in vars_in_cat:
                    try:
                        annual = load_chde_metric(model, scenario, variable, metrics_dir)
                        if annual is None:
                            continue
                        var_maps = compute_toe_maps(annual, model, scenario, variable)
                        for name, da in var_maps.items():
                            collected[name] = da.compute()
                    except Exception as exc:
                        log.error(
                            f"[{model} | {scenario} | {variable}] FAILED — {exc}", exc_info=True
                        )

                if not collected:
                    log.warning(f"[{model} | {scenario} | {category}] Nothing produced, skipping save.")
                    continue

                combined = xr.Dataset(collected)
                combined.attrs = {
                    "title":            f"Time of Emergence (regression) — CHDE {category}",
                    "method":           "Linear regression signal threshold (Snover & Salathe, 2015; "
                                        "Maraun et al., 2013; as in Seck et al., 2026)",
                    "model":            model,
                    "scenario":         scenario,
                    "category":         category,
                    "source_variables": ", ".join(vars_in_cat),
                    "baseline_period":  f"{BASELINE_START}-{BASELINE_END}",
                    "scan_start":       SCAN_START,
                    "envelopes":        "90% (P5/P95), 60% (P20/P80)",
                    "no_emerge_fill":   NO_EMERGE_FILL,
                    "conventions":      "CF-1.8",
                }
                save_combined_nc(combined, out_path)
                log.info(f"[{model} | {scenario} | {category}] ---- Done ----")

            # -- All models done for this (scenario, category) -> build ensemble
            build_ensemble(models, scenario, category, toe_dir)

# =============================================================================
# SECTION 6 — CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Time of Emergence (ToE) of CHDE metrics (frequency, duration, "
            "magnitude) for all CMIP6 model x scenario pairs, using the linear "
            "regression signal-threshold method following Snover & Salathe (2015)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python 06_toe-regression.py
            python 06_toe-regression.py --models ACCESS-CM2 MIROC6
            python 06_toe-regression.py --scenarios ssp245 ssp585
            python 06_toe-regression.py --categories frequency intensity
            python 06_toe-regression.py --no-skip
            python 06_toe-regression.py --config configs/config.yaml
        """,
    )
    parser.add_argument(
        "--config", default=r"D:\Research\CDHE\configs\config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="Restrict to specific models (default: all models in config)",
    )
    parser.add_argument(
        "--scenarios", nargs="+", metavar="SCENARIO",
        help="Restrict to specific projection scenarios (default: all non-historical)",
    )
    parser.add_argument(
        "--categories", nargs="+", metavar="CATEGORY",
        choices=list(CATEGORY_MAP.keys()), default=DEFAULT_CATEGORIES,
        help=(
            "CHDE metric categor(y/ies) to compute ToE for. For each category, "
            "ToE is computed for every variable defined under it. "
            f"Default: {DEFAULT_CATEGORIES}. "
            f"Mapping: {CATEGORY_MAP}"
        ),
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Recompute even if the output file already exists",
    )
    return parser.parse_args()


# =============================================================================
# SECTION 7 — MAIN
# =============================================================================
def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    models    = args.models    or list(cfg["models"].keys())
    scenarios = args.scenarios or list(cfg["scenarios"].keys())
    proj_scenarios = [s for s in scenarios if s != "historical"]

    if not proj_scenarios:
        log.warning(
            "No projection scenarios found. "
            "Add ssp245 / ssp585 etc. to your config or --scenarios flag."
        )
        return

    run_all(
        models=models,
        proj_scenarios=proj_scenarios,
        categories=args.categories,
        cfg=cfg,
        skip_existing=not args.no_skip,
    )
    log.info("All done")

if __name__ == "__main__":
    main()
