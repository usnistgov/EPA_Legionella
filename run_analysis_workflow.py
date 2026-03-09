"""run_analysis_workflow.py

Convenient entry point to execute the full data analysis pipeline for the NIST EPA Legionella project.

The pipeline follows the steps described in the README's "Analysis Workflow" section and runs each
script in the required order. If any step fails, the script aborts and returns a non‑zero exit code.

Each script's full stdout+stderr output is saved to its own log file under
{data_root}/output/logs/<script_name>.log (overwritten on each run).
On failure, the last 30 lines of the log are printed to the terminal.

Usage:
    python run_analysis_workflow.py
"""

import subprocess
import sys
import time
from pathlib import Path

# Number of log tail lines to print to terminal on failure
ERROR_TAIL_LINES = 30


def _get_log_dir() -> Path:
    """Return the log directory, creating it if needed."""
    # Import here so the module can still be imported without src on the path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.data_paths import get_data_root

    log_dir = get_data_root() / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _print_log_tail(log_path: Path, n: int = ERROR_TAIL_LINES) -> None:
    """Print the last *n* lines of a log file to the terminal."""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-n:] if len(lines) > n else lines
        print(f"\n--- Last {len(tail)} lines of {log_path.name} ---")
        for line in tail:
            print(line)
        print(f"--- End of {log_path.name} ---\n")
    except OSError:
        print(f"(Could not read log file: {log_path})")


def run_command(command: list[str], log_dir: Path, cwd: Path | None = None) -> float:
    """Run a command, saving all output to a log file.

    The terminal shows only the "Running …" and "Finished in … seconds" lines.
    On failure the last lines of the log are printed so the error is visible.

    Parameters
    ----------
    command: list[str]
        The command and its arguments.
    log_dir: Path
        Directory where per-script log files are written.
    cwd: Path | None, optional
        Working directory for the command. Defaults to the repository root.
    """
    import os
    import shlex

    # Derive log file name from the script argument (e.g. "scripts/co2_decay_analysis.py")
    script_arg = next((a for a in command if a.endswith(".py")), command[-1])
    script_name = Path(script_arg).stem
    log_path = log_dir / f"{script_name}.log"

    # Determine if we are already inside the epa_mh environment
    current_env = os.getenv("CONDA_DEFAULT_ENV")
    exec_str: str | None = None
    if current_env != "epa_mh":
        cmd_str = " ".join(shlex.quote(arg) for arg in command)
        exec_str = f"conda activate epa_mh && {cmd_str}"

    print(f"Running {script_name} ...")

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_fh:
        try:
            if exec_str is not None:
                subprocess.run(
                    exec_str,
                    cwd=str(cwd) if cwd is not None else None,
                    shell=True,
                    check=True,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                )
            else:
                subprocess.run(
                    command,
                    cwd=str(cwd) if cwd is not None else None,
                    check=True,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                )
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            print(
                f"Error: {script_name} failed with exit code {e.returncode} "
                f"after {elapsed:.2f} seconds"
            )
            print(f"Full log: {log_path}")
            _print_log_tail(log_path)
            sys.exit(e.returncode)

    elapsed = time.time() - start
    print(f"Finished in {elapsed:.2f} seconds  (log: {log_path.name})")
    return elapsed


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    log_dir = _get_log_dir()

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
        run_command(cmd, log_dir=log_dir, cwd=cwd)

    print("All analysis steps completed successfully.")


if __name__ == "__main__":
    main()
