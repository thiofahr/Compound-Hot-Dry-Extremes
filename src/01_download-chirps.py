"""
02_download-chirps.py
════════════════════════════════════════════════════════════════════════════
Downloads CHIRPS v2.0 global daily precipitation data (0.25-degree) from the
UCSB Climate Hazards Center file server, crops each file to the Indonesia
bounding box locally with xarray, and saves only the cropped result.

SOURCE
------
  https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25/
  Filename pattern: chirps-v2.0.{year}.days_p25.nc

OUTPUT STRUCTURE
----------------
data/
  raw/
    chirps/
      chirps-v2.0.{year}.days_p25_idn.nc

USAGE
-----
  # Download everything defined by --start-year/--end-year (default 1981-2024):
  python 02_download-chirps.py

  # Specific range:
  python 02_download-chirps.py --start-year 1981 --end-year 2023

  # Specific years only:
  python 02_download-chirps.py --years 1990 1991 1992

  # Dry run (show what would be downloaded without downloading):
  python 02_download-chirps.py --dry-run

  # Keep the full global file alongside the Indonesia subset:
  python 02_download-chirps.py --keep-global

  # Use a different config:
  python 02_download-chirps.py --config configs/config.yaml
════════════════════════════════════════════════════════════════════════════
"""
import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
import xarray as xr
import yaml
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("chirps_download_log.txt", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════
CHIRPS_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25"
CHIRPS_FIRST_YEAR_DEFAULT = 1981
CHIRPS_LAST_YEAR_DEFAULT = 2024


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class ChirpsTask:
    """One task = one year of the global CHIRPS file."""
    year:        int
    url:         str
    global_path: Path   # temp full-globe download destination
    final_path:  Path   # cropped Indonesia-subset destination

    @property
    def label(self) -> str:
        return f"CHIRPS/{self.year}"


@dataclass
class ChirpsResult:
    task:    ChirpsTask
    success: bool
    skipped: bool = False
    error:   str = ""
    size_mb: float = 0.0


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIG LOADING
# ══════════════════════════════════════════════════════════════════════════
def load_config(config_path: str = "configs/config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ["output_dir", "bbox"]:
        if key not in cfg:
            raise ValueError(f"Missing required key in config.yaml: '{key}'")
    # 'download' block is optional here -- fall back to sane defaults if absent
    cfg.setdefault("download", {})
    log.info(f"Config loaded. bbox={cfg['bbox']}  output_dir={cfg['output_dir']}")
    return cfg

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — TASK GENERATION
# ══════════════════════════════════════════════════════════════════════════
def generate_tasks(cfg: dict, years: list[int]) -> list[ChirpsTask]:
    """Build one ChirpsTask per requested year."""
    output_root = Path(cfg["output_dir"]) / "raw" / "CHIRPS"
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for year in years:
        filename = f"chirps-v2.0.{year}.days_p25.nc"
        url = f"{CHIRPS_BASE_URL}/{filename}"
        global_path = output_root / filename
        final_path = output_root / f"chirps-v2.0.{year}.days_p25_idn.nc"
        tasks.append(ChirpsTask(
            year=year, url=url, global_path=global_path, final_path=final_path,
        ))
    log.info(f"Generated {len(tasks)} CHIRPS download task(s) for years {years[0]}-{years[-1]}")
    return tasks


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — SUBSETTING
# ══════════════════════════════════════════════════════════════════════════
def subset_to_indonesia(src_path: Path, dest_path: Path, bbox: dict) -> None:
    """
    Crop a global CHIRPS file to the Indonesia bounding box and save
    compressed NetCDF4.

    CHIRPS netcdf files use 'latitude'/'longitude' dims in most releases,
    but some mirrors use 'lat'/'lon' -- detect whichever is present.
    Latitude is typically stored ascending (south -> north) already, but
    we sort defensively so slice(south, north) always works regardless.
    """
    with xr.open_dataset(src_path) as ds:
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"

        ds = ds.sortby(lat_name).sortby(lon_name)
        subset = ds.sel(
            {
                lat_name: slice(bbox["south"], bbox["north"]),
                lon_name: slice(bbox["west"], bbox["east"]),
            }
        ).load()

        if subset.sizes.get(lat_name, 0) == 0 or subset.sizes.get(lon_name, 0) == 0:
            raise ValueError(
                f"Empty subset for bbox={bbox} -- check bbox matches CHIRPS' "
                f"longitude convention (CHIRPS uses -180..180)."
            )

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        encoding = {v: {"zlib": True, "complevel": 4} for v in subset.data_vars}
        subset.to_netcdf(dest_path, encoding=encoding, engine="netcdf4")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — SINGLE FILE DOWNLOAD + SUBSET
# ══════════════════════════════════════════════════════════════════════════
def download_one(task: ChirpsTask, cfg: dict, keep_global: bool) -> ChirpsResult:
    """
    Download one global CHIRPS year file, crop it to Indonesia, and remove
    the global copy (unless keep_global=True). Mirrors the retry/streaming
    logic used for the CMIP6 downloader.
    """
    dl = cfg["download"]
    skip_existing  = dl.get("skip_existing", True)
    timeout        = dl.get("timeout_seconds", 600)
    retry_attempts = dl.get("retry_attempts", 3)
    retry_wait     = dl.get("retry_wait_seconds", 10)
    chunk_size     = dl.get("chunk_size", 8192)

    # -- Skip if the final (subset) file already exists -----------------------
    if skip_existing and task.final_path.exists() and task.final_path.stat().st_size > 0:
        return ChirpsResult(task=task, success=True, skipped=True)

    part_path = task.global_path.with_suffix(".nc.part")
    last_error = ""

    for attempt in range(1, retry_attempts + 1):
        try:
            response = requests.get(task.url, stream=True, timeout=timeout)

            if response.status_code == 404:
                return ChirpsResult(
                    task=task, success=False,
                    error="404 Not Found — check the year exists on the CHIRPS server"
                )
            if response.status_code == 400:
                return ChirpsResult(
                    task=task, success=False, error="400 Bad Request — check URL"
                )
            if response.status_code in (500, 502, 503, 504):
                wait = retry_wait * (2 ** (attempt - 1))
                last_error = (
                    f"{response.status_code} Server Error "
                    f"(attempt {attempt}/{retry_attempts}) — retrying in {wait}s"
                )
                log.warning(f"  {task.label}: {last_error}")
                if attempt < retry_attempts:
                    time.sleep(wait)
                continue

            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "html" in content_type:
                return ChirpsResult(
                    task=task, success=False,
                    error="Server returned HTML instead of NetCDF — check URL"
                )

            bytes_written = 0
            with open(part_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)

            if bytes_written < 1_000_000:  # global CHIRPS files are tens of MB
                part_path.unlink(missing_ok=True)
                return ChirpsResult(
                    task=task, success=False,
                    error=f"File too small ({bytes_written} bytes) — likely a server error"
                )

            part_path.rename(task.global_path)

            # -- Crop to Indonesia, then drop the global copy ------------------
            try:
                subset_to_indonesia(task.global_path, task.final_path, cfg["bbox"])
            except Exception as exc:
                return ChirpsResult(
                    task=task, success=False,
                    error=f"Subsetting failed: {exc}"
                )
            finally:
                if not keep_global:
                    task.global_path.unlink(missing_ok=True)

            size_mb = task.final_path.stat().st_size / (1024 * 1024)
            return ChirpsResult(task=task, success=True, size_mb=size_mb)

        except requests.exceptions.Timeout:
            last_error = f"Timeout after {timeout}s (attempt {attempt}/{retry_attempts})"
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e} (attempt {attempt}/{retry_attempts})"
        except requests.exceptions.HTTPError as e:
            last_error = f"HTTP error: {e} (attempt {attempt}/{retry_attempts})"
        except Exception as e:
            last_error = f"Unexpected error: {e} (attempt {attempt}/{retry_attempts})"

        if attempt < retry_attempts:
            time.sleep(retry_wait)
            log.warning(f"Retrying {task.label} — {last_error}")

    part_path.unlink(missing_ok=True)
    return ChirpsResult(task=task, success=False, error=last_error)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — PARALLEL DOWNLOAD MANAGER
# ══════════════════════════════════════════════════════════════════════════
def download_all(tasks: list[ChirpsTask], cfg: dict, dry_run: bool, keep_global: bool):
    max_workers = cfg["download"].get("max_workers", 4)

    if dry_run:
        print(f"\n{'─'*70}")
        print(f"DRY RUN — {len(tasks)} CHIRPS files would be downloaded + subset:")
        print(f"{'─'*70}")
        for t in tasks:
            print(f"  {t.label}")
            print(f"    URL   : {t.url}")
            print(f"    global: {t.global_path}  {'(kept)' if keep_global else '(deleted after subset)'}")
            print(f"    final : {t.final_path}")
        return

    results = []
    failed = []
    skipped = 0
    total_mb = 0.0

    print(f"\nDownloading {len(tasks)} CHIRPS year(s) with {max_workers} parallel workers …")
    print(f"Output directory : {Path(cfg['output_dir']) / 'raw' / 'chirps'}")
    print(f"Bbox             : {cfg['bbox']}")
    print(f"Keep global file : {keep_global}")
    print()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(download_one, task, cfg, keep_global): task
            for task in tasks
        }
        with tqdm(total=len(tasks), desc="CHIRPS", unit="year", dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
                if result.skipped:
                    skipped += 1
                    pbar.set_postfix_str(f"skip={skipped}", refresh=False)
                elif result.success:
                    total_mb += result.size_mb
                    pbar.set_postfix_str(
                        f"ok={len(results)-len(failed)-skipped} "
                        f"fail={len(failed)} "
                        f"{total_mb:.1f}MB",
                        refresh=False,
                    )
                    log.info(f"✓ {result.task.label} ({result.size_mb:.1f} MB subset)")
                else:
                    failed.append(result)
                    log.error(f"✗ {result.task.label} — {result.error}")
                pbar.update(1)

    n_ok = len(results) - len(failed) - skipped
    print(f"\n{'═'*60}")
    print(f"  Downloaded+subset : {n_ok} year(s)  ({total_mb:.1f} MB)")
    print(f"  Skipped           : {skipped} year(s) (already existed)")
    print(f"  Failed            : {len(failed)} year(s)")
    print(f"{'═'*60}")

    if failed:
        print(f"\nFailed downloads (see chirps_download_log.txt for details):")
        for r in failed:
            print(f"  ✗ {r.task.label} — {r.error}")
        failed_path = Path("failed_chirps_downloads.txt")
        with open(failed_path, "w") as f:
            f.write("# Failed CHIRPS downloads — re-run with --years\n")
            for r in failed:
                f.write(f"{r.task.year}\n")
        print(f"\nFailed year list saved to: {failed_path}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — CLI
# ══════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Download CHIRPS v2.0 global daily precipitation and subset to Indonesia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_download-chirps.py
  python 02_download-chirps.py --start-year 1981 --end-year 2023
  python 02_download-chirps.py --years 1990 1991 1992
  python 02_download-chirps.py --dry-run
  python 02_download-chirps.py --keep-global
        """,
    )
    parser.add_argument(
        "--config", default=r"D:\Research\CDHE\configs\config.yaml",
        help="Path to config YAML file (needs 'output_dir' and 'bbox' keys)",
    )
    parser.add_argument(
        "--start-year", type=int, default=CHIRPS_FIRST_YEAR_DEFAULT,
        help=f"First year to download (default: {CHIRPS_FIRST_YEAR_DEFAULT})",
    )
    parser.add_argument(
        "--end-year", type=int, default=CHIRPS_LAST_YEAR_DEFAULT,
        help=f"Last year to download, inclusive (default: {CHIRPS_LAST_YEAR_DEFAULT})",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, metavar="YEAR",
        help="Specific year(s) to download (overrides --start-year/--end-year)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be downloaded without downloading",
    )
    parser.add_argument(
        "--keep-global", action="store_true",
        help="Keep the full global NetCDF alongside the Indonesia subset (uses much more disk space)",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    cfg = load_config(args.config)

    years = args.years if args.years else list(range(args.start_year, args.end_year + 1))

    tasks = generate_tasks(cfg, years)
    if not tasks:
        log.warning("No tasks generated — check your year range.")
        return

    download_all(tasks, cfg, dry_run=args.dry_run, keep_global=args.keep_global)


if __name__ == "__main__":
    main()
