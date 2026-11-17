"""Generate a self-contained Colab notebook using pip and ordinary Python cells."""

import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    paths = [ROOT / "pyproject.toml", ROOT / "README.md"]
    paths.extend(sorted((ROOT / "src").rglob("*.py")))
    files = {str(path.relative_to(ROOT)): path.read_text() for path in paths}
    raw = json.dumps(files, ensure_ascii=False, sort_keys=True).encode()
    payload = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
    fingerprint = hashlib.sha256(raw).hexdigest()
    cells = []

    def add(kind, text):
        cell = {
            "cell_type": kind,
            "metadata": {},
            "id": f"colab-sir-{len(cells)}",
            "source": text.splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add(
        "markdown",
        """# SIR simulation on a GPU in Colab — pip and ordinary cells

1. Select **Runtime → Change runtime type → GPU** (e.g. T4).
2. Run **Source setup** and **Installation with pip**.
3. **Restart the kernel/session**, then continue with diagnostics. Do not rerun the installation after this restart.

Code runs directly in the notebook kernel: `model`, `result`, and `comparison` remain available in later cells. No separate venv or `run_in_env` is used.

The notebook contains a project snapshot. Installation refreshes the `simulation` metadata and selects **NumPy 2.4.6 + Numba 0.65.1**: NumPy retains the alias required by numba-cuda 0.30.4, and Numba satisfies the reported PyTensor 2.38.3 constraint (`numba<=0.65.1`). PyTensor is not removed. Installation changes packages in the current Colab session; check requirements for any additional dependencies.

Requires Python >=3.12. Repeat setup and installation after deleting the runtime. **VS Code:** install the [Google Colab](https://marketplace.visualstudio.com/items?itemName=google.colab) extension, select **Select Kernel → Colab → New Colab Server**, and choose a GPU.""",
    )
    add(
        "code",
        f"""# Source setup — this cell does not install packages.
import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile

if sys.version_info < (3, 12):
    raise RuntimeError("Select a Colab runtime with Python >=3.12.")
SOURCE_PAYLOAD = {payload!r}
SOURCE_SHA256 = {fingerprint!r}
raw = gzip.decompress(base64.b64decode(SOURCE_PAYLOAD))
assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256
project_files = json.loads(raw)
base = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
project_dir = base / "epidemic_sir_colab"
project_dir.mkdir(parents=True, exist_ok=True)
for relative, content in project_files.items():
    destination = (project_dir / relative).resolve()
    if not destination.is_relative_to(project_dir.resolve()):
        raise ValueError("Invalid source path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
print("Source:", SOURCE_SHA256[:12], "|", project_dir)
""",
    )
    add(
        "code",
        """# Install the project and compatible versions together to refresh stale simulation metadata.
%pip install --upgrade --upgrade-strategy only-if-needed -e "{project_dir}[cuda]" "numpy==2.4.6" "numba==0.65.1"
_sir_restart_required = True
print("Restart the kernel/session, then continue with the diagnostics below.")""",
    )
    add(
        "markdown",
        "## Diagnostics after restarting\n\nContinue from this cell. We check installed versions, import the module that uses `row_stack`, check GPU availability, and execute two kernel steps. If installation failed, resolve the error in the pip cell first.",
    )
    add(
        "code",
        """import os
import subprocess
from importlib.metadata import version

if globals().get("_sir_restart_required", False):
    raise RuntimeError("Restart the kernel after installation and continue from this cell.")
if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1":
    raise RuntimeError("Disable NUMBA_ENABLE_CUDASIM before using a physical GPU.")
subprocess.run(["nvidia-smi"], check=True)
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import numba.cuda.np.arrayobj
from simulation import create_simulation, run_timed, benchmark, State

for package in ("simulation", "numpy", "numba", "numba-cuda"):
    print(package, version(package))
assert version("numpy") == "2.4.6", "Run the current installation cell and restart the kernel."
assert version("numba") == "0.65.1", "Run the current installation cell and restart the kernel."
if not cuda.is_available():
    raise RuntimeError("Select a GPU runtime and check the CUDA installation.")
print("GPU:", cuda.get_current_device().name)
smoke = create_simulation((4, 5), backend="cuda", seed=42)
smoke.advance(2)
assert np.isin(smoke.grid, [1, 2, 3]).all()
print("CUDA kernel: OK")""",
    )
    add(
        "markdown",
        "## Computation\n\nChange parameters and initial infections in the cell below. All variables remain available in the notebook. When recording full history, `compute_seconds` includes transfers after every step; presentation is separate.",
    )
    add(
        "code",
        """model = create_simulation(
    (50, 50), initial_infected=0, seed=42, backend="cuda",
    infection_probability=0.06, recover_probability=0.05,
)
model.grid[10, 15] = State.INFECTIOUS
model.grid[30:33, 35:38] = State.INFECTIOUS
measured = run_timed(model, steps=100)
result = measured.result
print(f"Computation and transfers: {measured.compute_seconds:.4f} s")
print(f"History recording: {measured.recording_seconds:.4f} s")
print(f"Total excluding warmup: {measured.total_seconds:.4f} s")""",
    )
    add(
        "markdown",
        "## Display the recorded result\n\nThese cells do not recompute the simulation. Run them independently after computation has finished.",
    )
    add("code", "result.display(interval=100)")
    add(
        "code",
        """figure, axes = result.plot(step=50)
plt.show()""",
    )
    add(
        "markdown",
        "## CPU versus GPU benchmark\n\nHistory recording and plotting are excluded from measurements. Warmup and buffer/RNG initialization take place outside the samples. GPU measurements include one round-trip transfer per batch and synchronization. CPU and GPU use different random generators, so trajectories will differ even with the same seed.",
    )
    add(
        "code",
        """rows = []
for size in (128, 256, 512):
    for backend in ("numba", "cuda"):
        experiment = create_simulation(
            (size, size), initial_infected=size*size//20, seed=42,
            backend=backend, infection_probability=.06, recover_probability=.05,
        )
        report = benchmark(experiment, steps=100, repeats=3)
        rows.append({"size": size, "backend": backend, "ms_per_step": report.milliseconds_per_step})
        print(f"{size}x{size} {backend}: {report.milliseconds_per_step:.4f} ms/step")""",
    )
    add(
        "code",
        """fig, ax = plt.subplots(figsize=(8, 4))
for backend in ("numba", "cuda"):
    selected = [row for row in rows if row["backend"] == backend]
    ax.plot([r["size"]**2 for r in selected], [r["ms_per_step"] for r in selected], marker="o", label=backend)
ax.set(xlabel="Number of cells", ylabel="Median [ms/step]", title="CPU and GPU — no history or plots")
ax.grid(alpha=.25)
ax.legend()
plt.show()""",
    )
    add(
        "markdown",
        "## Further work\n\nYou can add ordinary cells, for example `result.counts`, `model.grid`, or calls to `compare_sir`. Save results before deleting the Colab runtime. After changing local code, run `python tools/build_colab_notebook.py` to refresh the embedded snapshot.\n\nRun `%pip check` to check dependencies across the session. Conflicts involving other preinstalled packages require separate assessment; this configuration resolves the reported NumPy/numba-cuda and Numba/PyTensor conflicts.",
    )
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "colab_gpu.ipynb", "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = ROOT / "notebooks/colab_gpu.ipynb"
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Created {destination.name}: {len(files)} source files, SHA256 {fingerprint[:12]}"
    )


if __name__ == "__main__":
    build()
