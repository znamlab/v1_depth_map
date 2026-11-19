"""Batch-execute the figure notebooks under memory supervision, then compile the stats.

Running `v1_depth_map/figures/*.ipynb` by hand is impractical: several need many GB of
RAM and tens of minutes each, so running them together exhausts system memory. This
script runs them one at a time, waits for enough free RAM before starting each, watches
the peak usage of the whole process tree, and gives up on anything that overruns its
timeout instead of hanging the batch.

What a run does:
  1. Selects the notebooks (`--notebooks` by name, otherwise every .ipynb in
     `v1_depth_map/figures/` except `unused_*.ipynb`, plus
     `v1_depth_map/figure_utils/methods_dataset_tables.ipynb`) and orders them so that
     each runs after the notebooks whose caches it reads (`DEPENDENCIES`), alphabetically
     otherwise.
  2. Executes each with `jupyter nbconvert --execute`, using the Python interpreter that
     runs this script (run it with `uv run python run_figures_pipeline.py`). A copy with
     a `kernelspec` patched in is written to a temp dir first, because several notebooks
     in this repo carry incomplete kernel metadata. With `--reload` (and/or
     `--recompute-deep`) the flags listed in `RELOAD_FLAGS` (`DEEP_FLAGS`) are set to
     True in that copy only; the notebooks on disk are not edited.
  3. Records status, wall time and peak RAM per notebook in a markdown status file
     (`--status-file`, default `<analysis root>/FIGURES_README.md`, see
     `v1_depth_map.paths.get_analysis_root`), updated after every notebook so the table
     is useful while the batch is running.
  4. Compiles the manuscript statistics the notebooks emitted into
     `v1_depth_map/stats/manuscript_stats.tex`. Nothing is written if validation fails.

Note that **the executed notebooks are discarded**: nbconvert writes to a temp dir, so
the source .ipynb files keep their existing outputs and are never modified. The durable
results of a run are whatever the notebooks themselves save - figure SVGs/PDFs, cached
pickles, and the per-notebook `stats_*.yaml` files - not the notebooks.

Usage:
    uv run python run_figures_pipeline.py
    uv run python run_figures_pipeline.py --notebooks figsupp9_v1_depth_map.ipynb --timeout 1800
    uv run python run_figures_pipeline.py --reload            # rebuild every notebook cache
    uv run python run_figures_pipeline.py --reload --dry-run  # show the flag changes only
"""

import argparse
import asyncio
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import psutil

# Configuration
REPO_ROOT = Path(__file__).parent
NOTEBOOKS_DIR = REPO_ROOT / "v1_depth_map" / "figures"
# Notebooks outside NOTEBOOKS_DIR that are part of a default run
EXTRA_NOTEBOOKS = [
    REPO_ROOT / "v1_depth_map" / "figure_utils" / "methods_dataset_tables.ipynb"
]
STATUS_FILE_NAME = "FIGURES_README.md"

# Run order: each notebook runs after the ones it reads caches from (when both are
# selected). Keys and values are notebook file names.
DEPENDENCIES = {
    # rf_all_sig*.pkl, rf_neurons_df_slim*.pkl (rf_gradient_bootstraps*.pkl for figsupp9_rf_distance)
    "figsupp8_multidepth_receptive_fields.ipynb": ["figsupp9_v1_depth_map.ipynb"],
    "figsupp9_rf_distance.ipynb": ["figsupp9_v1_depth_map.ipynb"],
    # fig2/results_all_rsof_closedloop.parquet (only figure2_rsof_integration writes it)
    "figsupp5_rsof.ipynb": ["figure2_rsof_integration.ipynb"],
    "figsupp10_gratings.ipynb": ["figure2_rsof_integration.ipynb"],
    # supp/results_all_rs_supp.pickle
    "figsupp2_speed.ipynb": ["figsupp6_openloop.ipynb"],
    # fig1/neurons_df_all.pickle
    "figsupp3_depth_pop.ipynb": ["figure1_depth_selectivity.ipynb"],
    # fig1/neurons_df_all.pickle, fig2/results_all_rsof_openloop.pickle (written by
    # figure2_openloop only), cache_gratings/grating_tuning_cells.csv
    "methods_dataset_tables.ipynb": [
        "figure1_depth_selectivity.ipynb",
        "figure2_openloop.ipynb",
        "figsupp10_gratings.ipynb",
    ],
}

# Flags set to True by --reload: rebuild the notebook caches under the analysis root
# from the processed data. Each must be assigned exactly once at the top level of a
# code cell as `<flag> = True|False` (the pipeline stops if one is missing).
RELOAD_FLAGS = {
    "figure1_depth_selectivity.ipynb": ["recompute_summary"],
    "figure2_openloop.ipynb": ["regenerate_cached_files"],
    "figure2_rsof_integration.ipynb": ["regenerate_cached_files"],
    "figure3_depth_cells.ipynb": ["regenerate_cached_files"],
    "figure4_receptive_fields.ipynb": ["recompute_rf_distance"],
    "figsupp2_speed.ipynb": ["RELOAD_EYE_CACHE"],
    "figsupp5_rsof.ipynb": ["recompute_summary"],
    "figsupp6_openloop.ipynb": ["regenerate_cached_files"],
    "figsupp8_multidepth_receptive_fields.ipynb": ["recompute_fov_gradients"],
    "figsupp9_v1_depth_map.ipynb": ["recompute_summary"],
    "figsupp10_gratings.ipynb": ["RELOAD"],
}
# Flags set to True by --recompute-deep: recompute per-session results from raw logs /
# dF/F (hours). They only take effect together with the matching reload flag, so
# --recompute-deep is normally combined with --reload. Model fits are never run by the notebooks:
# they are the re-analysis scripts in batch_analysis/ and precompute_data/ (README, level 3).
DEEP_FLAGS = {
    "figure1_depth_selectivity.ipynb": ["recompute_psth"],
    "figsupp6_openloop.ipynb": ["recompute_rs_stats"],
}
MIN_FREE_RAM_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
POLL_INTERVAL = 0.2  # Interval in seconds to poll memory/process status
TIMEOUT_SECONDS = (
    900  # Timeout for each notebook execution (15 minutes), overridable via --timeout
)


def order_notebooks(notebooks):
    """Sort notebooks so that each comes after its selected `DEPENDENCIES`.

    Ties are broken alphabetically by file name. Prerequisites that are not selected
    are only reported: their caches are expected to exist already.
    """
    by_name = {nb.name: nb for nb in notebooks}
    deps = {
        name: [d for d in DEPENDENCIES.get(name, []) if d in by_name]
        for name in by_name
    }
    for name in sorted(by_name):
        missing = [d for d in DEPENDENCIES.get(name, []) if d not in by_name]
        if missing:
            print(f"Note: {name} reads caches of {missing}, not in this run")
    ordered = []
    remaining = set(by_name)
    while remaining:
        ready = sorted(n for n in remaining if all(d in ordered for d in deps[n]))
        if not ready:
            raise ValueError(f"Circular DEPENDENCIES among {sorted(remaining)}")
        ordered.append(ready[0])
        remaining.remove(ready[0])
    return [by_name[n] for n in ordered]


def set_flags(nb, flags):
    """Set each flag to True in the code cells of an nbformat notebook, in place.

    A flag must be assigned exactly once at the top level of a code cell, as
    `<flag> = True` or `<flag> = False` (a trailing comment is kept).

    Args:
        nb (nbformat.NotebookNode): notebook to edit.
        flags (list): flag names.

    Returns:
        list: (flag, old line, new line) for each flag.

    Raises:
        ValueError: if a flag is not assigned exactly once.
    """
    changes = []
    for flag in flags:
        pattern = re.compile(rf"^{re.escape(flag)}(\s*=\s*)(True|False)\b(.*)$")
        hits = []
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            lines = cell.source.split("\n")
            for iline, line in enumerate(lines):
                match = pattern.match(line)
                if match:
                    hits.append((cell, lines, iline, match))
        if len(hits) != 1:
            raise ValueError(
                f"flag `{flag}` must be assigned once as `{flag} = True|False` at the "
                f"top level of a code cell, found {len(hits)} assignments"
            )
        cell, lines, iline, match = hits[0]
        new_line = f"{flag}{match.group(1)}True{match.group(3)}"
        changes.append((flag, lines[iline], new_line))
        lines[iline] = new_line
        cell.source = "\n".join(lines)
    return changes


def flags_for(notebook_name, reload=False, recompute_deep=False):
    """Flags to set to True in a notebook for this run."""
    flags = []
    if reload:
        flags += RELOAD_FLAGS.get(notebook_name, [])
    if recompute_deep:
        flags += DEEP_FLAGS.get(notebook_name, [])
    return flags


def default_status_file():
    """`<analysis root>/FIGURES_README.md`, see `v1_depth_map.paths`."""
    if not os.environ.get("V1_DEPTH_MAP_ANALYSIS_ROOT"):
        import flexiznam as flz

        flm = flz.get_flexilims_session(project_id="hey2_3d-vision_foodres_20220101")
    else:
        flm = None
    from v1_depth_map.paths import get_analysis_root

    return get_analysis_root(flm) / STATUS_FILE_NAME


class ReadmeAgent:
    def __init__(self, readme_path: Path):
        self.readme_path = readme_path
        self.results = {}
        self._load_existing()

    def _load_existing(self):
        if not self.readme_path.exists():
            return
        for line in self.readme_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if (
                not line.startswith("|")
                or line.startswith("| ---")
                or line.startswith("| Notebook")
            ):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) != 4:
                continue
            nb, status_formatted, time_str, ram_str = parts
            status = (
                status_formatted.split(" ", 1)[-1]
                if " " in status_formatted
                else status_formatted
            )
            self.results[nb] = {"status": status, "time": time_str, "ram": ram_str}

    def register_notebooks(self, notebooks):
        for nb in notebooks:
            self.results[nb] = {"status": "Pending", "time": "-", "ram": "-"}
        self.update_readme()

    def update_status(
        self,
        notebook: str,
        status: str,
        elapsed_time: float = None,
        peak_ram_mb: float = None,
    ):
        self.results[notebook]["status"] = status
        if elapsed_time is not None:
            self.results[notebook]["time"] = f"{elapsed_time / 60:.1f}"
        if peak_ram_mb is not None:
            self.results[notebook]["ram"] = f"{peak_ram_mb / 1024:.2f}"
        self.update_readme()

    def update_readme(self):
        self.readme_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Figures Notebooks Execution Status",
            "",
            "This table is dynamically updated by the pipeline agent.",
            "",
            "| Notebook | Status | Execution Time (min) | Peak RAM (GB) |",
            "| --- | --- | --- | --- |",
        ]
        for nb, info in sorted(self.results.items()):
            status_str = info["status"]
            if status_str == "Success":
                status_formatted = "✅ Success"
            elif status_str == "Failed":
                status_formatted = "❌ Failed"
            elif status_str == "Executing":
                status_formatted = "⏳ Executing"
            else:
                status_formatted = "💤 Pending"
            lines.append(
                f"| {nb} | {status_formatted} | {info['time']} | {info['ram']} |"
            )

        self.readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ResourceManager:
    def __init__(self, min_free_ram: int):
        self.min_free_ram = min_free_ram

    async def wait_for_resources(self, notebook_name: str):
        while True:
            free_ram = psutil.virtual_memory().available
            if free_ram >= self.min_free_ram:
                break
            print(
                f"[Resource Manager] Waiting for resources to run {notebook_name}... (Free RAM: {free_ram / (1024**2):.1f} MB, Target: {self.min_free_ram / (1024**2):.1f} MB)"
            )
            await asyncio.sleep(5)


async def monitor_process(pid: int, stop_event: asyncio.Event, results_dict: dict):
    peak_rss = 0
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    while not stop_event.is_set():
        try:
            total_rss = parent.memory_info().rss
            for child in parent.children(recursive=True):
                total_rss += child.memory_info().rss
            if total_rss > peak_rss:
                peak_rss = total_rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        await asyncio.sleep(POLL_INTERVAL)

    results_dict["peak_ram_mb"] = peak_rss / (1024 * 1024)


async def execute_notebook(
    notebook_path: Path,
    readme_agent: ReadmeAgent,
    resource_manager: ResourceManager,
    flags=(),
):
    notebook_name = notebook_path.name
    print(f"[Executor] Starting execution of {notebook_name}")
    readme_agent.update_status(notebook_name, "Executing")

    await resource_manager.wait_for_resources(notebook_name)

    start_time = time.time()

    # Run jupyter nbconvert using temporary file for output
    with tempfile.TemporaryDirectory() as tmpdir:
        import nbformat

        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
            if "kernelspec" not in nb.metadata:
                nb.metadata["kernelspec"] = {}
            if "name" not in nb.metadata["kernelspec"]:
                nb.metadata["kernelspec"]["name"] = "python3"
            if "display_name" not in nb.metadata["kernelspec"]:
                nb.metadata["kernelspec"]["display_name"] = "Python 3"
            for flag, _, new_line in set_flags(nb, flags):
                print(f"[Executor] {notebook_name}: {new_line}")

            fixed_nb_path = Path(tmpdir) / "fixed.ipynb"
            with open(fixed_nb_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
        except Exception as e:
            print(f"[Executor] Error pre-processing metadata for {notebook_name}: {e}")
            readme_agent.update_status(notebook_name, "Failed", 0.0, 0.0)
            return

        output_nb = Path(tmpdir) / "executed.ipynb"

        # Same interpreter (and so environment) as this script
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(fixed_nb_path),
            "--output",
            str(output_nb),
        ]

        try:
            # Start process
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            # Start memory monitoring task
            stop_monitor = asyncio.Event()
            monitor_results = {"peak_ram_mb": 0.0}
            monitor_task = asyncio.create_task(
                monitor_process(proc.pid, stop_monitor, monitor_results)
            )

            # Wait for execution with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=TIMEOUT_SECONDS
                )
                stop_monitor.set()
                await monitor_task

                elapsed = time.time() - start_time
                peak_ram = monitor_results["peak_ram_mb"]

                if proc.returncode == 0:
                    print(
                        f"[Executor] {notebook_name} completed successfully in {elapsed:.1f}s (Peak RAM: {peak_ram:.1f} MB)"
                    )
                    readme_agent.update_status(
                        notebook_name, "Success", elapsed, peak_ram
                    )
                else:
                    print(
                        f"[Executor] {notebook_name} failed with return code {proc.returncode}"
                    )
                    if stderr:
                        print(stderr.decode(errors="replace")[-1000:])
                    readme_agent.update_status(
                        notebook_name, "Failed", elapsed, peak_ram
                    )

            except asyncio.TimeoutError:
                print(
                    f"[Executor] TIMEOUT: {notebook_name} timed out after {TIMEOUT_SECONDS} seconds."
                )
                # Kill process and children
                try:
                    parent = psutil.Process(proc.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except Exception:
                    pass

                stop_monitor.set()
                await monitor_task
                elapsed = time.time() - start_time
                readme_agent.update_status(
                    notebook_name, "Failed", elapsed, monitor_results["peak_ram_mb"]
                )

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[Executor] Error running {notebook_name}: {e}")
            readme_agent.update_status(notebook_name, "Failed", elapsed, 0.0)


async def main():
    global TIMEOUT_SECONDS

    parser = argparse.ArgumentParser(
        description="Run figure notebooks in dependency order and record their status"
    )
    parser.add_argument(
        "--notebooks",
        nargs="+",
        default=None,
        help="Names of specific notebooks (e.g. figsupp9_v1_depth_map.ipynb) to run. "
        "Defaults to every notebook in the figures directory except unused_*.ipynb, "
        "plus figure_utils/methods_dataset_tables.ipynb.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Set every notebook's reload/regenerate flags (RELOAD_FLAGS) to True in "
        "the executed copies, rebuilding the caches under the analysis root.",
    )
    parser.add_argument(
        "--recompute-deep",
        action="store_true",
        help="Also set the deep flags (DEEP_FLAGS: per-session PSTH and RS stats from "
        "raw logs; hours). Use with --reload.",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help=f"Markdown status table to write. Defaults to <analysis root>/"
        f"{STATUS_FILE_NAME}.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Do not compile the manuscript statistics after the run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run order and the flag lines that would be changed, then exit "
        "without executing anything or writing any file.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=TIMEOUT_SECONDS,
        help="Per-notebook execution timeout in seconds.",
    )
    args = parser.parse_args()
    TIMEOUT_SECONDS = args.timeout

    if not NOTEBOOKS_DIR.exists():
        print(f"Error: notebooks directory {NOTEBOOKS_DIR} does not exist.")
        sys.exit(1)

    extra = {p.name: p for p in EXTRA_NOTEBOOKS}
    if args.notebooks:
        notebooks = [extra.get(nb, NOTEBOOKS_DIR / nb) for nb in args.notebooks]
        missing = [nb for nb in notebooks if not nb.exists()]
        if missing:
            print(f"Error: notebook(s) not found: {[str(m) for m in missing]}")
            sys.exit(1)
    else:
        # `unused_*.ipynb` are kept for reference but produce no manuscript figure,
        # so a default batch skips them. Name one with --notebooks to run it anyway.
        notebooks = [
            p for p in NOTEBOOKS_DIR.glob("*.ipynb") if not p.stem.startswith("unused_")
        ] + [p for p in EXTRA_NOTEBOOKS if p.exists()]
    if not notebooks:
        print("No notebooks found.")
        sys.exit(0)
    notebooks = order_notebooks(notebooks)

    # Check every flag before running anything
    import nbformat

    run_flags = {}
    for nb_path in notebooks:
        flags = flags_for(nb_path.name, args.reload, args.recompute_deep)
        nb = nbformat.read(nb_path, as_version=4)
        try:
            changes = set_flags(nb, flags)
        except ValueError as err:
            print(f"Error in {nb_path.name}: {err}")
            sys.exit(1)
        run_flags[nb_path.name] = flags
        if args.dry_run:
            print(f"{nb_path.name}")
            for _, old, new in changes:
                print(f"    {old.strip()}  ->  {new.strip()}")
    if args.dry_run:
        print(f"Dry run: {len(notebooks)} notebooks, nothing executed")
        return

    status_file = args.status_file or default_status_file()
    print(f"Running {len(notebooks)} notebooks (timeout={TIMEOUT_SECONDS}s)")
    print(f"Status file: {status_file}")

    readme_agent = ReadmeAgent(status_file)
    readme_agent.register_notebooks([nb.name for nb in notebooks])

    resource_manager = ResourceManager(MIN_FREE_RAM_BYTES)

    # Run notebooks one-by-one to safely monitor resource usage and avoid system memory exhaustion
    for nb_path in notebooks:
        await execute_notebook(
            nb_path, readme_agent, resource_manager, flags=run_flags[nb_path.name]
        )

    print(f"Pipeline execution finished. Summary report updated in {status_file}")
    if args.skip_stats:
        return
    try:
        from v1_depth_map.stats import compile_manuscript_stats

        # Validates first and writes nothing if a macro is duplicated or a formatted
        # value would leave LaTeX math mode open, so a bad number is caught here
        # rather than in Overleaf.
        if compile_manuscript_stats() != 0:
            print(
                "WARNING: manuscript stats validation failed - "
                "manuscript_stats.tex was NOT updated (see errors above)."
            )
    except Exception as e:
        print(f"Notice: Could not compile manuscript stats: {e}")


if __name__ == "__main__":
    asyncio.run(main())
