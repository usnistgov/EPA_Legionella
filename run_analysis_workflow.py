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
    """Run a command using ``subprocess.run``.

    This function ensures the command runs inside the ``epa_mh`` conda environment.
    If the current process is already within that environment, the command is executed
    directly. Otherwise, it activates the environment using ``conda activate epa_mh``
    before running the command.

    Parameters
    ----------
    command: list[str]
        The command and its arguments.
    cwd: Path | None, optional
        Working directory for the command. Defaults to the repository root.
    """
    import os
    import shlex

    # Determine if we are already inside the epa_mh environment
    current_env = os.getenv("CONDA_DEFAULT_ENV")
    # Initialize variables with proper types
    exec_cmd: list[str] = []
    exec_str: str | None = None
    if current_env == "epa_mh":
        # Run the command directly
        exec_cmd = command
    else:
        # Build a shell command that activates the environment then runs the command
        # Join the command list into a properly escaped string for the shell
        cmd_str = " ".join(shlex.quote(arg) for arg in command)
        exec_str = f"conda activate epa_mh && {cmd_str}"
        exec_cmd = command

    if exec_str is not None:
        print(f"Running (with conda activate): {exec_str}")
        start = time.time()
        try:
            subprocess.run(
                exec_str,
                cwd=str(cwd) if cwd is not None else None,
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: Command {exec_str} failed with exit code {e.returncode}")
            sys.exit(e.returncode)
    else:
        # exec_cmd is guaranteed to be a non‑empty list here
        print(f"Running: {' '.join(exec_cmd)}")
        start = time.time()
        try:
            subprocess.run(
                exec_cmd, cwd=str(cwd) if cwd is not None else None, check=True
            )
        except subprocess.CalledProcessError as e:
            print(
                f"Error: Command {' '.join(exec_cmd)} failed with exit code {e.returncode}"
            )
            sys.exit(e.returncode)
    elapsed = time.time() - start
    print(f"Finished in {elapsed:.2f} seconds")
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
        (["python", "scripts/co2_decay_analysis.py"], repo_root),
        (["python", "scripts/rh_temp_other_analysis.py"], repo_root),
        (["python", "scripts/particle_decay_analysis.py"], repo_root),
    ]

    for cmd, cwd in steps:
        run_command(cmd, cwd=cwd)

    print("All analysis steps completed successfully.")


if __name__ == "__main__":
    main()
