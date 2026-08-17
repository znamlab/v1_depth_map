import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path
import psutil

# Configuration
NOTEBOOKS_DIR = Path(__file__).parent / "v1_depth_map" / "figures"
README_PATH = Path(__file__).parent / "FIGURES_README.md"
MIN_FREE_RAM_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
POLL_INTERVAL = 0.2  # Interval in seconds to poll memory/process status
TIMEOUT_SECONDS = (
    900  # Timeout for each notebook execution (15 minutes), overridable via --timeout
)


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
    notebook_path: Path, readme_agent: ReadmeAgent, resource_manager: ResourceManager
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

            fixed_nb_path = Path(tmpdir) / "fixed.ipynb"
            with open(fixed_nb_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
        except Exception as e:
            print(f"[Executor] Error pre-processing metadata for {notebook_name}: {e}")
            readme_agent.update_status(notebook_name, "Failed", 0.0, 0.0)
            return

        output_nb = Path(tmpdir) / "executed.ipynb"

        # Use uv run to ensure we run inside the correct project environment
        cmd = [
            "uv",
            "run",
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
        description="Run figure notebooks and record status in FIGURES_README.md"
    )
    parser.add_argument(
        "--notebooks",
        nargs="+",
        default=None,
        help="Names of specific notebooks (e.g. figure_rf.ipynb) to run. Defaults to all notebooks in the figures directory.",
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

    if args.notebooks:
        notebooks = [NOTEBOOKS_DIR / nb for nb in args.notebooks]
        missing = [nb for nb in notebooks if not nb.exists()]
        if missing:
            print(f"Error: notebook(s) not found: {[str(m) for m in missing]}")
            sys.exit(1)
    else:
        notebooks = sorted([p for p in NOTEBOOKS_DIR.glob("*.ipynb")])
    if not notebooks:
        print("No notebooks found.")
        sys.exit(0)

    print(f"Running {len(notebooks)} notebooks (timeout={TIMEOUT_SECONDS}s)")

    readme_agent = ReadmeAgent(README_PATH)
    readme_agent.register_notebooks([nb.name for nb in notebooks])

    resource_manager = ResourceManager(MIN_FREE_RAM_BYTES)

    # Run notebooks one-by-one to safely monitor resource usage and avoid system memory exhaustion
    for nb_path in notebooks:
        await execute_notebook(nb_path, readme_agent, resource_manager)

    print("Pipeline execution finished. Summary report updated in FIGURES_README.md")


if __name__ == "__main__":
    asyncio.run(main())
