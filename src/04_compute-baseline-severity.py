"""
04_compute-baseline-severity.py
Compute fixed-baseline severity anchors (mu, sigma) for the additive
z-score compound severity index, from historical (1981-2014)
hot/dry exceedance-day magnitudes.

Population used
----------------
hot_excess (ΔT = tasmax - tx90p) on all days that are part of a hot
event (hot_event_id > 0), pooled across 1981-2014.
dry_deficit (ΔDI = pr10p - pr) on all days that are part of a dry event
(dry_event_id > 0), pooled across 1981-2014.

Output
------
{main_dir}/daily/severity-baseline/{model}_severity_baseline_1981-2014.nc
with variables: hot_mu, hot_sigma, dry_mu, dry_sigma  (dims: lat, lon)
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
BASELINE_YEARS = range(1981, 2015)  # 1981-2014 inclusive
print(YEAR_RE)

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


def compute_baseline_anchors(
    model: str,
    raw_dir: Path,
    chde_dir: Path,
    threshold_dir: Path,
    hp_label: str,
    dp_label: str,
) -> xr.Dataset:
    tasmax_year_map = list_year_files(raw_dir / model / "historical" / "tasmax")
    pr_year_map = list_year_files(raw_dir / model / "historical" / "pr")
    chde_hist_dir = chde_dir / model / "ssp245"
    chde_year_files = {}

    for f in sorted(chde_hist_dir.glob(f"chde_{model}*.nc")):
        m = YEAR_RE.search(f.stem)

        if m is None:
            log.warning(f"Cannot determine year from {f.name}")
            continue

        chde_year_files[int(m.group())] = f

    tasmax_thr = xr.open_dataset(
        threshold_dir / f"{model}_tasmax_threshold_1981-2014.nc", decode_times=TIME_CODER
    )[f"tasmax{hp_label}p"]
    pr_thr = xr.open_dataset(
        threshold_dir / f"{model}_pr_threshold_1981-2014.nc", decode_times=TIME_CODER
    )[f"pr{dp_label}p"]

    # Running accumulators (lat, lon), initialized lazily from first grid seen
    hot_sum = hot_sqsum = hot_count = None
    dry_sum = dry_sqsum = dry_count = None

    years = [y for y in BASELINE_YEARS if y in tasmax_year_map and y in pr_year_map and y in chde_year_files]
    if not years:
        raise FileNotFoundError(f"[{model}] No overlapping baseline years found across raw/chde files.")

    for year in tqdm(years, desc=f"{model} baseline", unit="yr", leave=False):
        with xr.open_dataset(tasmax_year_map[year], decode_times=TIME_CODER) as ds:
            tasmax_y = convert_units(ds["tasmax"].load(), "tasmax")
        with xr.open_dataset(pr_year_map[year], decode_times=TIME_CODER) as ds:
            pr_y = convert_units(ds["pr"].load(), "pr")
        with xr.open_dataset(chde_year_files[year], decode_times=TIME_CODER) as ds:
            hot_eid = ds[f"hot_p{hp_label}_event_id"].load()
            dry_eid = ds[f"dry_p{dp_label}_event_id"].load()

        tasmax_y, pr_y, hot_eid, dry_eid = xr.align(tasmax_y, pr_y, hot_eid, dry_eid, join="inner")

        hot_excess = (
            (tasmax_y.groupby("time.dayofyear") - tasmax_thr)
            .drop_vars("dayofyear", errors="ignore")
            .clip(min=0)
        )
        dry_deficit = (
            (pr_thr - pr_y.groupby("time.dayofyear"))
            .drop_vars("dayofyear", errors="ignore")
            .clip(min=0)
        )

        hot_mask = hot_eid > 0
        dry_mask = dry_eid > 0

        hot_masked = hot_excess.where(hot_mask, 0.0)
        dry_masked = dry_deficit.where(dry_mask, 0.0)

        year_hot_sum = hot_masked.sum(dim="time")
        year_hot_sqsum = (hot_masked ** 2).sum(dim="time")
        year_hot_count = hot_mask.sum(dim="time")

        year_dry_sum = dry_masked.sum(dim="time")
        year_dry_sqsum = (dry_masked ** 2).sum(dim="time")
        year_dry_count = dry_mask.sum(dim="time")

        if hot_sum is None:
            hot_sum, hot_sqsum, hot_count = year_hot_sum, year_hot_sqsum, year_hot_count
            dry_sum, dry_sqsum, dry_count = year_dry_sum, year_dry_sqsum, year_dry_count
        else:
            hot_sum = hot_sum + year_hot_sum
            hot_sqsum = hot_sqsum + year_hot_sqsum
            hot_count = hot_count + year_hot_count
            dry_sum = dry_sum + year_dry_sum
            dry_sqsum = dry_sqsum + year_dry_sqsum
            dry_count = dry_count + year_dry_count

        del tasmax_y, pr_y, hot_eid, dry_eid, hot_excess, dry_deficit, hot_masked, dry_masked
        gc.collect()

    # mu, sigma with safe division (avoid /0 where a pixel never had exceedance days)
    hot_count_safe = hot_count.where(hot_count > 0)
    dry_count_safe = dry_count.where(dry_count > 0)

    hot_mu = (hot_sum / hot_count_safe).rename("hot_mu")
    hot_var = (hot_sqsum / hot_count_safe) - hot_mu ** 2
    hot_sigma = np.sqrt(hot_var.clip(min=0)).rename("hot_sigma")

    dry_mu = (dry_sum / dry_count_safe).rename("dry_mu")
    dry_var = (dry_sqsum / dry_count_safe) - dry_mu ** 2
    dry_sigma = np.sqrt(dry_var.clip(min=0)).rename("dry_sigma")

    out_ds = xr.Dataset({
        "hot_mu": hot_mu, "hot_sigma": hot_sigma,
        "dry_mu": dry_mu, "dry_sigma": dry_sigma,
    })
    out_ds.attrs.update({
        "model": model, "baseline_years": "1981-2014",
        "description": "Fixed-baseline mean/std of hot-exceedance ΔT and dry-exceedance ΔDI, "
                        "pooled over all event days in 1981-2014. Used for additive z-score "
                        "compound severity index.",
    })
    return out_ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute fixed-baseline severity anchors (mu, sigma).")
    parser.add_argument("--config", default=r"D:\Research\CDHE\configs\config.yaml")
    parser.add_argument("--models", nargs="+", metavar="MODEL")
    parser.add_argument("--hot-pct", type=float, default=0.90)
    parser.add_argument("--dry-pct", type=float, default=0.10)
    parser.add_argument("--no-skip", action="store_true", help="Recompute even if output already exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    main_dir = Path(cfg["output_dir"])
    raw_dir = main_dir / "raw"
    threshold_dir = main_dir / "daily" / "threshold"
    chde_dir = main_dir / "daily" / "chde"
    baseline_dir = main_dir / "daily" / "severity-baseline"

    models = args.models or list(cfg["models"].keys())
    
    hp_label = f"{int(round(args.hot_pct * 100))}"
    dp_label = f"{int(round(args.dry_pct * 100))}"

    log.info(f"Models: {models}")
    for model in tqdm(models, desc="Models", unit="model"):
        out_path = baseline_dir / f"{model}_severity_baseline_1981-2014.nc"
        if not args.no_skip and out_path.exists():
            log.info(f"[{model}] baseline anchors exist, skipping: {out_path}")
            continue
        log.info(f"=== {model} ===")
        try:
            anchors_ds = compute_baseline_anchors(
                model, raw_dir, chde_dir, threshold_dir, hp_label, dp_label
            )
        except Exception:
            log.exception(f"[{model}] Failed")
            continue
        save_dataset(anchors_ds, out_path)
        del anchors_ds
        gc.collect()
    log.info("Done.")


if __name__ == "__main__":
    main()
