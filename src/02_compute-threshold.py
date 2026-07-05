"""
02_compute-threshold.py
═══════════════════════════════════════════════════════════════════════════════
Preprocess daily data: compute calendar-day percentile thresholds
(tasmax hot / pr dry) per model from the historical baseline period.

OUTPUT STRUCTURE
────────────────
data/
└── daily /
    └── threshold/
        └── {model}_{variable}_{threshold}_{baseline period}.nc

USAGE
─────

  # Compute threshold :
  python 02_compute-threshold.py

  # Only one model:
  python 02_compute-threshold.py --models ACCESS-CM2

  # Only one scenario:
  python 02_compute-threshold.py --scenarios historical

  # Combine filters:
  python 02_compute-threshold.py --models ACCESS-CM2 MIROC6 --scenarios ssp245 ssp585

  # No-skip (rewrite existing results):
  python 02_compute-threshold.py --dry-run

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

YEAR_RE = re.compile(r"(?:19|20)\d{2}")  # non-capturing -> findall gives full years
TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str = r"D:\Research\CDHE\configs\config.yaml") -> dict:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# File listing (one .nc file per year, per directory layout)
# ---------------------------------------------------------------------------
def list_year_files(var_dir: Path) -> dict[int, Path]:
    """Map {year: file path} for a raw/{model}/{scenario}/{var} dir.

    Assumes one file per year. Tries to pull the year from the filename
    first (fast, no I/O); falls back to opening the file's time coord if
    the filename doesn't contain exactly one 4-digit year.
    """
    var_dir = Path(var_dir)
    files = sorted(var_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {var_dir}")

    year_map = {}
    for f in files:
        matches = YEAR_RE.findall(f.stem)
        if len(matches) == 1:
            year_map[int(matches[0])] = f
            continue

        with xr.open_dataset(f, decode_times=TIME_CODER) as ds:
            years = np.unique(ds["time"].dt.year.values)

        if len(years) == 1:
            year_map[int(years[0])] = f
        else:
            log.warning(
                f"Could not uniquely determine a single year for {f.name} "
                f"(found years {years}); skipping this file in the year map."
            )
    return year_map


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
def convert_units(da: xr.DataArray, var: str) -> xr.DataArray:
    """Convert to standard analysis units:
    tasmax : K -> degC
    pr     : kg m-2 s-1 -> mm day-1
    """
    if var == "tasmax" and da.attrs.get("units", "K") == "K":
        da = da - 273.15
        da.attrs["units"] = "degC"
    elif var == "pr" and da.attrs.get("units", "kg m-2 s-1") in ("kg m-2 s-1", "kg/m2/s"):
        da = da * 86400.0
        da.attrs["units"] = "mm day-1"
    return da


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_dataset(ds: xr.Dataset, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding, engine="netcdf4")
    log.info(f"     Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Step 1: Threshold (calendar-day percentile climatology)
# ---------------------------------------------------------------------------
def compute_and_save_threshold(
    model: str,
    raw_dir: Path,
    var: str,
    percentiles: list[float],
    baseline_years: tuple[int, int],
    threshold_dir: Path,
    skip_if_exists: bool = True,
) -> xr.Dataset:
    """Compute {var}{p}p thresholds for one model from historical baseline
    years only, using dask so raw baseline data is never fully loaded into
    RAM at once. Returns the (small) threshold Dataset, loaded into memory.
    """
    out_path = threshold_dir / f"{model}_{var}_threshold_{baseline_years[0]}-{baseline_years[1]}.nc"
    if skip_if_exists and out_path.exists():
        log.info(f"[{model}] {var} threshold exists, loading from disk: {out_path}")
        return xr.load_dataset(out_path)

    if model in ['CHIRPS']:
        var_dir = raw_dir / model 
    else:
        var_dir = raw_dir / model / "historical" / var
    year_map = list_year_files(var_dir)
    year_range = range(baseline_years[0], baseline_years[1] + 1)

    files = [year_map[y] for y in year_range if y in year_map]
    missing = [y for y in year_range if y not in year_map]
    if missing:
        log.warning(f"[{model}] {var}: missing baseline years {missing}")
    if not files:
        raise FileNotFoundError(f"[{model}] No baseline files found for {var} in {var_dir}")

    log.info(f"[{model}] Computing {var} threshold from {len(files)} baseline file(s)")
    with xr.open_mfdataset(
        files, combine="by_coords", chunks={"time": -1}, decode_times=TIME_CODER
    ) as ds:
        da = convert_units(ds[var], var)
        da = da.chunk({"time": -1})  # merge per-file time chunks into one

        # NOTE: leap days are intentionally kept (see module docstring) so that
        # dayofyear=366 gets its own real threshold instead of NaN.
        out_vars = {}
        for p in percentiles:
            if not 0 < p < 1:
                raise ValueError(f"Percentiles must be in (0, 1), got {p}")

            perc_label = f"{int(round(p * 100))}"
            log.info(f"[{model}] Computing {var}{perc_label}p (dask, lazy)")

            thr = da.groupby("time.dayofyear").quantile(p, dim="time")
            thr = thr.drop_vars("quantile", errors="ignore")
            thr.name = f"{var}{perc_label}p"
            thr.attrs = {
                "description": (
                    f"{perc_label}th percentile of daily {var} over baseline "
                    f"{baseline_years[0]}-{baseline_years[1]} (leap days included)"
                ),
                "model": model,
                "baseline": f"{baseline_years[0]}-{baseline_years[1]}",
                "percentile": p,
            }
            out_vars[thr.name] = thr

        thr_ds = xr.Dataset(out_vars)
        thr_ds.attrs = {"model": model, "var": var, "baseline": f"{baseline_years[0]}-{baseline_years[1]}"}
        thr_ds = thr_ds.load()  # materialize only the small final result

    save_dataset(thr_ds, out_path)
    del da
    gc.collect()
    return thr_ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess CMIP6 daily data into CHDE indices. "
            "Thresholds always derived from historical baseline; "
            "applied continuously from historical through each projection scenario."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python 02_preprocess.py
            python 02_preprocess.py --models ACCESS-CM2 MIROC6
            python 02_preprocess.py --scenarios ssp245 ssp585
            python 02_preprocess.py --no-skip
        """,
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--models", nargs="+", metavar="MODEL", help="Filter to specific models")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        metavar="SCENARIO",
        help="Filter to specific projection scenarios (historical is always pulled in as baseline)",
    )
    parser.add_argument("--no-skip", action="store_true", help="Recompute even if outputs already exist")
    parser.add_argument("--hot-pct", nargs="+", type=float, default=[0.90])
    parser.add_argument("--dry-pct", nargs="+", type=float, default=[0.10])
    parser.add_argument("--min-periods", type=int, default=3)
    parser.add_argument("--max-gap-days", type=int, default=1)
    parser.add_argument("--chde-max-gap-days", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_dir = Path(cfg["output_dir"])
    raw_dir = data_dir / "raw"
    threshold_dir = data_dir / "daily" / "threshold"

    models = args.models or list(cfg["models"].keys())
    scenarios = args.scenarios or list(cfg["scenarios"].keys())
    proj_scenarios = [s for s in scenarios if s != "historical"]
    if not proj_scenarios:
        log.warning("No projection scenarios given. Add ssp245 / ssp585 etc.")
        return

    baseline_years = tuple(cfg["period"]["threshold"])  # e.g. (1981, 2014)
    log.info(f"Models: {models}")

    for model in tqdm(models, desc="Models", unit="model"):
        if model in ['CHIRPS']:
            compute_and_save_threshold(
            model, raw_dir, "precip", args.dry_pct, baseline_years,
            threshold_dir, skip_if_exists=not args.no_skip,
        )
        else:
            # Step 1: threshold, once per model, shared across scenarios
            compute_and_save_threshold(
                model, raw_dir, "tasmax", args.hot_pct, baseline_years,
                threshold_dir, skip_if_exists=not args.no_skip,
            )
            compute_and_save_threshold(
                model, raw_dir, "pr", args.dry_pct, baseline_years,
                threshold_dir, skip_if_exists=not args.no_skip,
            )
        gc.collect()

    log.info("Done.")


if __name__ == "__main__":
    main()
