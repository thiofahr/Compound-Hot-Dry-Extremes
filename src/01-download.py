"""
01_01-download_cmip6.py
═══════════════════════════════════════════════════════════════════════════════
Downloads NEX-GDDP-CMIP6 data for Indonesia from the NASA NCCS THREDDS server.

OUTPUT STRUCTURE
────────────────
data/
└── ACCESS-CM2/
    ├── historical/
    │   ├── pr/
    │   │   ├── pr_day_ACCESS-CM2_historical_r1i1p1f1_gn_1980.nc
    │   │   └── ...
    │   └── tasmax/
    │       └── ...
    ├── ssp245/
    │   └── ...
    └── ssp585/
        └── ...

USAGE
─────
  # Download everything defined in config.yaml:
  python 01-download_cmip6.py

  # Only one model:
  python 01-download_cmip6.py --models ACCESS-CM2

  # Only one scenario:
  python 01-download_cmip6.py --scenarios historical

  # Only one variable:
  python 01-download_cmip6.py --variables pr

  # Combine filters:
  python 01-download_cmip6.py --models ACCESS-CM2 MIROC6 --scenarios ssp245 ssp585

  # Dry run (show what would be downloaded without downloading):
  python 01-download_cmip6.py --dry-run

═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("download_log.txt", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DownloadTask:
    """
    Represents a single file to download.
    One task = one year × one variable × one scenario × one model.
    """
    model:     str
    scenario:  str
    variable:  str
    year:      int
    member:    str
    url:       str
    dest_path: Path

    @property
    def label(self) -> str:
        """Short label for display in progress bars and logs."""
        return f"{self.model}/{self.scenario}/{self.variable}/{self.year}"


@dataclass
class DownloadResult:
    """Outcome of a single download attempt."""
    task:    DownloadTask
    success: bool
    skipped: bool = False   # True if file already existed
    error:   str  = ""
    size_mb: float = 0.0

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIG LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str = r"configs/config.yaml") -> dict:
    """Load and validate the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Basic validation
    required_keys = ["output_dir", "thredds_base", "bbox", "variables",
                     "scenarios", "models", "download"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required key in config.yaml: '{key}'")

    log.info(f"Config loaded: {len(cfg['models'])} models, "
             f"{len(cfg['scenarios'])} scenarios, "
             f"{len(cfg['variables'])} variables")
    return cfg

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — URL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_thredds_url(
    base:     str,
    model:    str,
    scenario: str,
    member:   str,
    variable: str,
    year:     int,
    bbox:     dict,
) -> str:
    """
    Build the THREDDS NetCDF Subset Service (NCSS) URL for one file.

    URL structure:
      {base}/{model}/{scenario}/{member}/{variable}/
        {variable}_day_{model}_{scenario}_{member}_gn_{year}_v2.0.nc
        ?var={variable}
        &north={bbox.north}&south={bbox.south}
        &west={bbox.west}&east={bbox.east}
        &horizStride=1
        &time_start={year}-01-01T12:00:00Z
        &time_end={year}-12-31T12:00:00Z
        &accept=netcdf4-classic

    The server will crop the global grid to your bbox and return only
    those pixels — much smaller than the full global file.

    Args:
        base    : THREDDS base URL from config
        model   : GCM model name (e.g. "ACCESS-CM2")
        scenario: "historical", "ssp245", or "ssp585"
        member  : ensemble member (e.g. "r1i1p1f1")
        variable: "pr" or "tasmax"
        year    : calendar year
        bbox    : dict with keys north, south, west, east

    Returns:
        Full THREDDS NCSS URL string
    """
    filename = f"{variable}_day_{model}_{scenario}_{member}_gn_{year}_v2.0.nc"
    
    path = f"{base}/{model}/{scenario}/{member}/{variable}/{filename}"

    params = (
        f"var={variable}"
        f"&north={bbox['north']}"
        f"&west={bbox['west']}"
        f"&east={bbox['east']}"
        f"&south={bbox['south']}"
        f"&horizStride=1"
        f"&time_start={year}-01-01T12:00:00Z"
        f"&time_end={year}-12-31T12:00:00Z"
        f"&accept=netcdf4-classic"
    )

    return f"{path}?{params}"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TASK GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_tasks(
    cfg:               dict,
    filter_models:     list[str] | None = None,
    filter_scenarios:  list[str] | None = None,
    filter_variables:  list[str] | None = None,
) -> list[DownloadTask]:
    """
    Generate the full list of DownloadTask objects from config,
    optionally filtered by model / scenario / variable.

    The nested loop order is:
      model → scenario → variable → year
    so all files for one model+scenario+variable are consecutive.
    """
    tasks = []
    output_root = Path(cfg["output_dir"])

    gr_models = [
        "EC-Earth3", "EC-Earth3-Veg-LR", "GFDL-CM4", "GFDL-CM4_gr2", "GFDL-ESM4", 
        "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G", "KIOST-ESM"
    ]
    for model, model_cfg in cfg["models"].items():

        # Apply model filter if provided
        if filter_models and model not in filter_models:
            continue

        member = model_cfg["member"]

        for scenario, scen_cfg in cfg["scenarios"].items():

            # Apply scenario filter
            if filter_scenarios and scenario not in filter_scenarios:
                continue

            year_start, year_end = scen_cfg["years"]

            for variable in cfg["variables"]:

                # Apply variable filter
                if filter_variables and variable not in filter_variables:
                    continue

                for year in range(year_start, year_end + 1):

                    url = build_thredds_url(
                        base=cfg["thredds_base"],
                        model=model,
                        scenario=scenario,
                        member=member,
                        variable=variable,
                        year=year,
                        bbox=cfg["bbox"],
                    )

                    # Neat folder structure: data/{model}/{scenario}/{variable}/
                    dest_dir = output_root / "raw" / model / scenario / variable
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    # Filename matches the standard CMIP6 convention (no query params)

                    if model not in gr_models:
                        filename = (
                            f"{variable}_day_{model}_{scenario}_{member}_gn_{year}_v2.0.nc"
                        )
                    else:
                        filename = (
                            f"{variable}_day_{model}_{scenario}_{member}_gr_{year}_v2.0.nc"
                        )
                    dest_path = dest_dir / filename

                    tasks.append(DownloadTask(
                        model=model,
                        scenario=scenario,
                        variable=variable,
                        year=year,
                        member=member,
                        url=url,
                        dest_path=dest_path,
                    ))

    log.info(f"Generated {len(tasks)} download tasks")
    return tasks

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SINGLE FILE DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_one(task: DownloadTask, cfg: dict) -> DownloadResult:
    """
    Download a single file with retry logic and streaming.

    Steps:
      1. Check if file already exists → skip if skip_existing=True
      2. Send HTTP GET with streaming enabled (so we don't load into RAM)
      3. Write to a temporary .part file while downloading
      4. Rename .part → .nc only on success (prevents corrupt files)
      5. Retry on failure up to retry_attempts times

    Args:
        task: the DownloadTask describing what to fetch
        cfg : full config dict (for download settings)

    Returns:
        DownloadResult with success/failure info
    """
    dl = cfg["download"]
    skip_existing    = dl.get("skip_existing", True)
    timeout          = dl.get("timeout_seconds", 300)
    retry_attempts   = dl.get("retry_attempts", 3)
    retry_wait       = dl.get("retry_wait_seconds", 10)
    chunk_size       = dl.get("chunk_size", 8192)

    # ── Skip if already downloaded ────────────────────────────────────────────
    if skip_existing and task.dest_path.exists() and task.dest_path.stat().st_size > 0:
        return DownloadResult(task=task, success=True, skipped=True)

    # Temporary file to write to (renamed on success)
    part_path = task.dest_path.with_suffix(".nc.part")

    last_error = ""
    for attempt in range(1, retry_attempts + 1):
        try:
            # Stream=True: download in chunks to keep RAM usage low
            response = requests.get(task.url, stream=True, timeout=timeout)

            # ── Hard failures: 4xx = bad URL/file missing, no point retrying ─
            if response.status_code == 404:
                return DownloadResult(
                    task=task, success=False,
                    error="404 Not Found — file may not exist for this model/scenario/year"
                )
            if response.status_code == 400:
                return DownloadResult(
                    task=task, success=False,
                    error="400 Bad Request — check URL parameters (bbox, dates, variable)"
                )

            # ── Soft failures: 5xx = server busy, retry with exponential backoff
            # 503 is common on THREDDS when overloaded — always retry these.
            if response.status_code in (500, 502, 503, 504):
                wait = retry_wait * (2 ** (attempt - 1))  # 10s, 20s, 40s ...
                last_error = (
                    f"{response.status_code} Server Error "
                    f"(attempt {attempt}/{retry_attempts}) — retrying in {wait}s"
                )
                log.warning(f"  {task.label}: {last_error}")
                if attempt < retry_attempts:
                    time.sleep(wait)
                continue  # go straight to next attempt, skip wait below

            response.raise_for_status()  # catch any other unexpected HTTP errors

            # Check content type — THREDDS sometimes returns an HTML error page
            content_type = response.headers.get("Content-Type", "")
            if "html" in content_type:
                return DownloadResult(
                    task=task, success=False,
                    error="Server returned HTML instead of NetCDF — check URL"
                )

            # Stream to disk in chunks
            bytes_written = 0
            with open(part_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)

            # Sanity check: reject tiny files (likely an HTML/text error response)
            if bytes_written < 1000:
                part_path.unlink(missing_ok=True)
                return DownloadResult(
                    task=task, success=False,
                    error=f"File too small ({bytes_written} bytes) — likely a server error"
                )

            # Success: rename .part -> .nc atomically
            part_path.rename(task.dest_path)
            size_mb = bytes_written / (1024 * 1024)
            return DownloadResult(task=task, success=True, size_mb=size_mb)

        except requests.exceptions.Timeout:
            last_error = f"Timeout after {timeout}s (attempt {attempt}/{retry_attempts})"
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e} (attempt {attempt}/{retry_attempts})"
        except requests.exceptions.HTTPError as e:
            last_error = f"HTTP error: {e} (attempt {attempt}/{retry_attempts})"
        except Exception as e:
            last_error = f"Unexpected error: {e} (attempt {attempt}/{retry_attempts})"

        # Linear wait before retrying (for non-5xx errors)
        if attempt < retry_attempts:
            time.sleep(retry_wait)
            log.warning(f"Retrying {task.label} — {last_error}")

    # Clean up partial file on failure
    part_path.unlink(missing_ok=True)
    return DownloadResult(task=task, success=False, error=last_error)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PARALLEL DOWNLOAD MANAGER
# ══════════════════════════════════════════════════════════════════════════════

def download_all(tasks: list[DownloadTask], cfg: dict, dry_run: bool = False):
    """
    Download all tasks in parallel using a thread pool.

    Args:
        tasks  : list of DownloadTask objects
        cfg    : full config dict
        dry_run: if True, print URLs without downloading
    """
    max_workers = cfg["download"].get("max_workers", 4)

    if dry_run:
        print(f"\n{'─'*70}")
        print(f"DRY RUN — {len(tasks)} files would be downloaded:")
        print(f"{'─'*70}")
        for t in tasks:
            print(f"  {t.label}")
            print(f"    → {t.dest_path}")
            print(f"    URL: {t.url[:100]}...")
        return

    results = []
    failed  = []
    skipped = 0
    total_mb = 0.0

    print(f"\nDownloading {len(tasks)} files with {max_workers} parallel workers …")
    print(f"Output directory: {cfg['output_dir']}/")
    print(f"Skip existing: {cfg['download'].get('skip_existing', True)}")
    print()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        # Submit all tasks to the thread pool
        future_to_task = {
            executor.submit(download_one, task, cfg): task
            for task in tasks
        }

        # tqdm progress bar over completed futures
        with tqdm(
            total=len(tasks),
            desc="Downloading",
            unit="file",
            dynamic_ncols=True,
        ) as pbar:

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
                        refresh=False
                    )
                    log.info(f"✓ {result.task.label} ({result.size_mb:.1f} MB)")
                else:
                    failed.append(result)
                    log.error(f"✗ {result.task.label} — {result.error}")

                pbar.update(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ok = len(results) - len(failed) - skipped
    print(f"\n{'═'*60}")
    print(f"  Downloaded : {n_ok} files  ({total_mb:.1f} MB)")
    print(f"  Skipped    : {skipped} files (already existed)")
    print(f"  Failed     : {len(failed)} files")
    print(f"{'═'*60}")

    if failed:
        print(f"\nFailed downloads (see download_log.txt for details):")
        for r in failed:
            print(f"  ✗ {r.task.label}")
            print(f"      {r.error}")

        # Write failed tasks to a file so you can re-run just those
        failed_path = Path("failed_downloads.txt")
        with open(failed_path, "w") as f:
            f.write("# Failed downloads — re-run with --retry-failed\n")
            for r in failed:
                f.write(f"{r.task.model},{r.task.scenario},{r.task.variable},{r.task.year}\n")
        print(f"\nFailed list saved to: {failed_path}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CATALOG WRITER
# ══════════════════════════════════════════════════════════════════════════════

def write_catalog(cfg: dict):
    """
    After downloading, write a simple CSV catalog of all local files.


    Output: data/catalog.csv
    columns: model, scenario, variable, year, member, path
    """
    import csv
    output_root = Path(cfg["output_dir"])
    catalog_path = output_root / "catalog.csv"

    rows = []
    # Walk the output directory and collect all .nc files
    for nc_file in sorted(output_root.rglob("*.nc")):
        # Parse path components: data/{model}/{scenario}/{variable}/{filename}
        parts = nc_file.relative_to(output_root).parts
        if len(parts) != 4:
            continue
        model, scenario, variable, filename = parts

        # Extract year from filename
        # Pattern: {var}_day_{model}_{scenario}_{member}_gn_{year}.nc
        try:
            stem_parts = nc_file.stem.split("_")
            year   = int(stem_parts[-1])
            member = stem_parts[-3]
        except (IndexError, ValueError):
            continue

        rows.append({
            "model":    model,
            "scenario": scenario,
            "variable": variable,
            "year":     year,
            "member":   member,
            "path":     str(nc_file),
        })

    with open(catalog_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "scenario", "variable", "year", "member", "path"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Catalog written: {catalog_path} ({len(rows)} files)")


def write_catalog(cfg: dict):
    import csv
    output_root = Path(cfg["output_dir"])
    catalog_path = output_root / "catalog.csv"

    rows = []
    for nc_file in sorted(output_root.rglob("*.nc")):
        parts = nc_file.relative_to(output_root).parts
        if len(parts) != 4:
            continue
        model, scenario, variable, filename = parts

        try:
            stem_parts = nc_file.stem.split("_")
            # stem ends in  …_gn_1980_v2.0  →  [-1]="v2.0", [-2]="1980", [-4]="member"
            year   = int(stem_parts[-2])   # was stem_parts[-1]
            member = stem_parts[-4]        # was stem_parts[-3]
        except (IndexError, ValueError):
            continue

        rows.append({
            "model":    model,
            "scenario": scenario,
            "variable": variable,
            "year":     year,
            "member":   member,
            "path":     str(nc_file),
        })

    with open(catalog_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "scenario", "variable", "year", "member", "path"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Catalog written: {catalog_path} ({len(rows)} files)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download NEX-GDDP-CMIP6 data for Indonesia from THREDDS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python 01-download_cmip6.py
    python 01-download_cmip6.py --models ACCESS-CM2 MIROC6
    python 01-download_cmip6.py --scenarios ssp245
    python 01-download_cmip6.py --variables pr
    python 01-download_cmip6.py --dry-run
    python 01-download_cmip6.py --catalog-only
        """,
    )
    parser.add_argument(
        "--config", default=r"D:\Research\CDHE\configs\config.yaml",
        help="Path to config YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="Filter to specific models (e.g. ACCESS-CM2 MIROC6)"
    )
    parser.add_argument(
        "--scenarios", nargs="+", metavar="SCENARIO",
        help="Filter to specific scenarios (e.g. ssp245 ssp585)"
    )
    parser.add_argument(
        "--variables", nargs="+", metavar="VAR",
        help="Filter to specific variables (e.g. pr tasmax)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be downloaded without downloading"
    )
    parser.add_argument(
        "--catalog-only", action="store_true",
        help="Skip downloading, just regenerate the catalog CSV"
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if args.catalog_only:
        write_catalog(cfg)
        return

    tasks = generate_tasks(
        cfg,
        filter_models=args.models,
        filter_scenarios=args.scenarios,
        filter_variables=args.variables,
    )

    if not tasks:
        log.warning("No tasks generated — check your filters and config.")
        return

    download_all(tasks, cfg, dry_run=args.dry_run)

    # Write catalog after downloading (skip in dry-run)
    if not args.dry_run:
        write_catalog(cfg)


if __name__ == "__main__":
    main()
