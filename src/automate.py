"""
automate.py
============
Run selected pipeline scripts sequentially.

Example
-------
python run_pipeline.py

or modify SCRIPT_ARGS below to pass arguments to individual scripts.
"""

from pathlib import Path
import subprocess
import sys
import time

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SCRIPT_ARGS = [
    ("01_download-chirps.py", []),

    ("01_download-cmip6.py", [
        "--models", "MIROC6", "MPI-ESM1-2-LR", "TaiESM1",
        "--scenarios", "ssp245", "ssp585",
    ]),
]

# ---------------------------------------------------------------------
def run_script(script_path: Path, args: list[str]) -> None:
    """Run one script and stop if it fails."""

    cmd = [sys.executable, str(script_path)] + args

    print("=" * 80)
    print(f"Running: {script_path.name}")
    print("Command:", " ".join(cmd))
    print("=" * 80)

    start = time.time()

    result = subprocess.run(cmd)

    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_path.name} failed with exit code {result.returncode}"
        )

    print(f"Finished in {elapsed:.1f} s\n")


def main():

    src_dir = Path(__file__).resolve().parent

    for script_name, args in SCRIPT_ARGS:

        script = src_dir / script_name

        if not script.exists():
            raise FileNotFoundError(script)

        run_script(script, args)

    print("=" * 80)
    print("Pipeline completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
