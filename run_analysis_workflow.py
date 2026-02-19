"""run_analysis_workflow.py

Convenient entry point to execute the full data analysis pipeline for the NIST EPA Legionella project.

The pipeline follows the steps described in the README's "Analysis Workflow" section and runs each
script in the required order. If any step fails, the script aborts and returns a non‑zero exit code.

Usage:
    python run_analysis_workflow.py
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(command: list[str], cwd: Path | None = None) -> float:
    """Run a command using ``subprocess.run`` within the epa_mh conda environment.

    Parameters
    ----------
    command: list[str]
        The command and its arguments.
    cwd: Path | None, optional
        Working directory for the command. Defaults to the repository root.
    """
    # Prepend conda run to ensure the epa_mh environment is used
    conda_cmd = ["conda", "run", "-n", "epa_mh"] + command
    print(f"Running: {' '.join(conda_cmd)}")
    start = time.time()
    try:
        subprocess.run(conda_cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(
            f"Error: Command {' '.join(conda_cmd)} failed with exit code {e.returncode}"
        )
        sys.exit(e.returncode)
    elapsed = time.time() - start
    print(f"Finished: {' '.join(conda_cmd)} in {elapsed:.2f} seconds")
    return elapsed


def main() -> None:
    repo_root = Path(__file__).resolve().parent

    # Define the pipeline steps as (command list, optional working directory)
    steps = [
        (["python", "scripts/download_quantaq_data.py"], repo_root),
        (["python", "scripts/process_quantaq_data.py"], repo_root),
        (["python", "scripts/process_co2_log.py"], repo_root),
        (["python", "scripts/process_shower_log.py"], repo_root),
        (["python", "scripts/event_registry.py", "--force"], repo_root),
        (["python", "src/co2_decay_analysis.py"], repo_root),
        (["python", "src/rh_temp_other_analysis.py"], repo_root),
        (["python", "src/particle_decay_analysis.py"], repo_root),
    ]

    for cmd, cwd in steps:
        run_command(cmd, cwd=cwd)

    print("All analysis steps completed successfully.")


if __name__ == "__main__":
    main()
