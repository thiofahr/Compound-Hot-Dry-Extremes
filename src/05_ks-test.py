"""
05_ks-test.py
============
Compute the time of emergence (ToE) of compound hot-dry event (CHDE)
metrics using the Kolmogorov-Smirnov test.

METHODOLOGY
-----------
    input: annual CHDE metrics (from 04_compute-metrics.py), i.e.
           {main_dir}/daily/chde_metrics/{model}_{scenario}_chde_metrics.nc
           dims (year, lat, lon)
    method: Two-sample Kolmogorov-Smirnov test following
            Schuhen et al. (2026) and Christine Padalino et al. (2024)

CATEGORY -> VARIABLE MAPPING
-----------------------------
Each run targets one or more categories. For every requested category,
ToE is computed for ALL variables defined under that category below:
    frequency : chde_n_events, chde_n_days
    duration  : chde_duration_mean, chde_duration_max
    intensity : CHDMI_n_events, CHDMI_sum, CHDMI_mean, CHDMI_max

Variables within a category are bundled as separate named DataArrays
inside that category's single output file (see OUTPUT STRUCTURE below).

OUTPUT STRUCTURE
----------------
data/
  daily/
    toe/
      {category}/
        toe_{model}_{scenario}_{category}.nc
          variables: toe_{variable}  for each variable defined under
                     that category (e.g. toe_chde_n_days, toe_CHDMI_sum)

USAGE
-----
  python 05_ks-test.py
  python 05_ks-test.py --models ACCESS-CM2 MIROC6
  python 05_ks-test.py --scenarios ssp245 ssp585
  python 05_ks-test.py --categories frequency intensity
  python 05_ks-test.py --no-skip
  python 05_ks-test.py --config configs/config.yaml
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from scipy.stats import ks_2samp
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
NO_EMERGE_CUTOFF = 2071
NO_EMERGE_FILL   = 2100
WINDOW           = 30
ALPHA            = 0.05
ENCODE_OPTS      = dict(zlib=True, complevel=4, dtype="float32")

# -----------------------------------------------------------------------------
# CATEGORY -> VARIABLE MAPPING (drives output folder/filename + what gets run)
# -----------------------------------------------------------------------------
CATEGORY_MAP = {
    "frequency": ["chde_n_events", "chde_n_days"],
    "duration": ["chde_duration_mean", "chde_duration_max"],
    "intensity": ["CHDMI_n_events", "CHDMI_sum", "CHDMI_mean", "CHDMI_max"],
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
# SECTION 3 — KS-BASED ToE  (pixel-level)
# =============================================================================
def ks_toe(
    series: np.ndarray,
    years:  np.ndarray,
    baseline_start:   int   = BASELINE_START,
    baseline_end:     int   = BASELINE_END,
    window:           int   = WINDOW,
    alpha:            float = ALPHA,
    scan_start:       int   = SCAN_START,
    no_emerge_cutoff: int   = NO_EMERGE_CUTOFF,
) -> float:
    """
    Compute Time of Emergence for a single pixel using the two-sample
    KS test (Schuhen et al., 2026, NHESS).

    Adaptations for 1981-2100 data:
    - Reference period : 1981-2014
    - Scan start       : 2015
    - No-emerge cutoff : 2071
    - No-emerge return : 2100
    - Persistence      : >95% of SUBSEQUENT windows significant
    - ToE label        : midpoint of qualifying window

    Parameters
    ----------
    series : 1-D array of the chosen annual CHDE metric, shape (n_years,)
    years  : 1-D array of corresponding years, shape (n_years,)

    Returns
    -------
    float — ToE year, or NO_EMERGE_FILL (2100) if no emergence detected.
    """
    # -- Defensive cleanup: drop any (year, value) pair where the year itself
    # is NaN. load_chde_metric() already fills these under normal use, but
    # guarding here too means this function never raises on bad input --
    # e.g. int(NaN) in max_year below -- it just treats it as missing.
    years = np.asarray(years, dtype="float64")
    series = np.asarray(series, dtype="float64")
    valid_year = ~np.isnan(years)
    if not valid_year.all():
        years  = years[valid_year]
        series = series[valid_year]
    if years.size == 0:
        return np.nan

    # -- Reference sample --------------------------------------------------
    ref_mask = (years >= baseline_start) & (years <= baseline_end)
    ref      = series[ref_mask]
    ref      = ref[~np.isnan(ref)]
    if len(ref) < 10:
        return np.nan   # ocean / masked pixel

    # -- Per-window KS significance (step = 1 year) -------------------------
    max_year   = int(np.nanmax(years))
    scan_years = np.arange(scan_start, max_year - window + 2)
    sig        = np.full(len(scan_years), np.nan)
    for i, start in enumerate(scan_years):
        win = series[(years >= start) & (years <= start + window - 1)]
        win = win[~np.isnan(win)]
        if len(win) < 10:
            continue
        _, p   = ks_2samp(ref, win)
        sig[i] = float(p < alpha)

    # -- Persistence check (faithful to R code: tail excludes current window)
    for i, start in enumerate(scan_years):
        if start > no_emerge_cutoff:
            return NO_EMERGE_FILL
        if np.isnan(sig[i]) or sig[i] == 0.0:
            continue
        tf_tail = sig[i + 1:]
        if len(tf_tail) == 0:
            continue
        if np.nansum(tf_tail) > 0.95 * len(tf_tail):
            midpoint = start + window // 2
            return float(midpoint)
    return NO_EMERGE_FILL


# =============================================================================
# SECTION 4 — GRID-LEVEL ToE MAP
# =============================================================================
def compute_toe_map(
    annual:   xr.DataArray,
    model:    str,
    scenario: str,
    variable: str,
) -> xr.DataArray:
    """
    Apply `ks_toe` over every (lat, lon) pixel via xr.apply_ufunc.

    Returns
    -------
    xr.DataArray  dims (lat, lon) — ToE year per pixel,
    named `toe_{variable}` so multiple variables can coexist as distinct
    DataArrays within one category's output Dataset.
    """
    years_arr = annual.year.values
    log.info(f"[{model} | {scenario} | {variable}] Computing ToE map ...")
    toe = xr.apply_ufunc(
        lambda s: ks_toe(s, years_arr),
        annual,
        input_core_dims=[["year"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    var_name = f"toe_{variable}"
    toe.name = var_name
    toe.attrs = {
        "description":      f"Time of Emergence of CHDE metric '{variable}'",
        "source_variable":  variable,
        "category":         VAR_TO_CATEGORY.get(variable, "unknown"),
        "method":           "Two-sample KS test (Schuhen et al., 2026) and Christine Padalino et al (2024)",
        "model":            model,
        "scenario":         scenario,
        "reference_period": f"{BASELINE_START}-{BASELINE_END}",
        "scan_start":       SCAN_START,
        "window_years":     WINDOW,
        "alpha":            ALPHA,
        "no_emerge_cutoff": NO_EMERGE_CUTOFF,
        "no_emerge_fill":   NO_EMERGE_FILL,
        "units":            "year",
    }
    log.info(f"[{model} | {scenario} | {variable}] ToE map done  (var: {var_name})")
    return toe


# =============================================================================
# SECTION 5 — FULL PIPELINE
# =============================================================================
def run_all(
    models:         list[str],
    proj_scenarios: list[str],
    categories:     list[str],
    cfg:            dict,
    skip_existing:  bool = True,
) -> None:
    """
    Loop over all model x scenario x category combinations. For each
    category, compute ToE for every variable defined under that category
    in CATEGORY_MAP, and save one combined file per
    (model, scenario, category) under toe_dir/{category}/.
    """
    output_root = Path(cfg["output_dir"])
    metrics_dir = output_root / "daily" / "chde_metrics"
    toe_dir     = output_root / "daily" / "toe"

    # Validate requested categories
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

    pairs = [(m, s) for m in models for s in proj_scenarios]
    log.info(f"Processing {len(pairs)} model x scenario pair(s)")
    log.info(f"Categories: {valid_categories}")
    for cat in valid_categories:
        log.info(f"  Variables: {CATEGORY_MAP[cat]}")

    for model, scenario in tqdm(pairs, desc="Pairs", unit="pair"):
        for category in valid_categories:
            vars_in_cat = CATEGORY_MAP[category]
            print()
            log.info(f"[{model} | {scenario} | {category}] ---- Starting ToE pipeline ----")

            out_path = toe_dir / category / f"toe_{model}_{scenario}_{category}.nc"
            if skip_existing and out_path.exists():
                log.info(f"SKIP — output already exists: {out_path}")
                continue

            collected: dict[str, xr.DataArray] = {}
            for variable in vars_in_cat:
                try:
                    annual = load_chde_metric(model, scenario, variable, metrics_dir)
                    if annual is None:
                        continue
                    toe = compute_toe_map(annual, model, scenario, variable)
                    collected[toe.name] = toe.compute()
                except Exception as exc:
                    log.error(
                        f"[{model} | {scenario} | {variable}] FAILED — {exc}", exc_info=True
                    )

            if not collected:
                log.warning(f"[{model} | {scenario} | {category}] Nothing produced, skipping save.")
                continue

            combined = xr.Dataset(collected)
            combined.attrs = {
                "title":            f"Time of Emergence — CHDE {category}",
                "method":           "Two-sample KS test (Schuhen et al., 2026, NHESS)",
                "model":            model,
                "scenario":         scenario,
                "category":         category,
                "source_variables": ", ".join(vars_in_cat),
                "baseline_period":  f"{BASELINE_START}-{BASELINE_END}",
                "scan_start":       SCAN_START,
                "window_years":     WINDOW,
                "alpha":            ALPHA,
                "no_emerge_fill":   NO_EMERGE_FILL,
                "conventions":      "CF-1.8",
            }
            save_combined_nc(combined, out_path)
            log.info(f"[{model} | {scenario} | {category}] ---- Done ----")


# =============================================================================
# SECTION 6 — CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Time of Emergence (ToE) of CHDE metrics (frequency, duration, "
            "intensity) for all CMIP6 model x scenario pairs, using the KS test "
            "following Schuhen et al. (2026, NHESS)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python 05_ks-test.py
            python 05_ks-test.py --models ACCESS-CM2 MIROC6
            python 05_ks-test.py --scenarios ssp245 ssp585
            python 05_ks-test.py --categories frequency intensity
            python 05_ks-test.py --no-skip
            python 05_ks-test.py --config configs/config.yaml
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
