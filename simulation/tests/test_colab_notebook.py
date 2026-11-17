"""Check the pip notebook, source snapshot and ordinary shared-state cells."""

import ast
import base64
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class ColabNotebookTests(unittest.TestCase):
    def test_snapshot_pip_and_shared_state_cells(self):
        notebook = json.loads((ROOT / "notebooks/colab_gpu.ipynb").read_text())
        setup = "".join(notebook["cells"][1]["source"])
        constants = {}
        for statement in ast.parse(setup).body:
            if (
                isinstance(statement, ast.Assign)
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id in ("SOURCE_PAYLOAD", "SOURCE_SHA256")
            ):
                constants[statement.targets[0].id] = ast.literal_eval(statement.value)
        raw = gzip.decompress(base64.b64decode(constants["SOURCE_PAYLOAD"]))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), constants["SOURCE_SHA256"])
        files = json.loads(raw)
        expected = {"README.md", "pyproject.toml"}
        expected.update(str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py"))
        self.assertEqual(set(files), expected)
        for path, content in files.items():
            self.assertEqual(content, (ROOT / path).read_text(), path)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                ast.parse(
                    "\n".join(
                        "# " + line if line.startswith("%") else line
                        for line in source.splitlines()
                    )
                )
                self.assertNotIn("run_in_env", source)
                self.assertNotIn("uv venv", source)
        pip_cell = "".join(notebook["cells"][2]["source"])
        self.assertIn("%pip install", pip_cell)
        self.assertIn('"numba==0.65.1"', pip_cell)
        self.assertIn('"numpy==2.4.6"', pip_cell)
        self.assertIn('-e "{project_dir}[cuda]"', pip_cell)
        # No installer is executed during this test.
        with tempfile.TemporaryDirectory() as directory:
            with patch("tempfile.gettempdir", return_value=directory):
                namespace = {}
                exec(compile(setup, "<colab setup>", "exec"), namespace)  # noqa: S102
            self.assertTrue(
                (
                    namespace["project_dir"] / "src/simulation/core/simulation.py"
                ).is_file()
            )
        # Run the actual calculation and plot cells with a small CPU experiment.
        # Only parameters differ; variables must flow directly between cells.
        import matplotlib.pyplot as plt
        import numpy as np

        from simulation import State, create_simulation, run_timed

        ns = {
            "create_simulation": create_simulation,
            "State": State,
            "run_timed": run_timed,
            "plt": plt,
        }
        calculation = "".join(notebook["cells"][6]["source"])
        calculation = calculation.replace('backend="cuda"', 'backend="numba"').replace(
            "steps=100", "steps=3"
        )
        exec(compile(calculation, "<colab calculation>", "exec"), ns)  # noqa: S102
        self.assertEqual(ns["result"].grids.shape, (4, 50, 50))
        np.testing.assert_array_equal(ns["result"].counts.sum(axis=1), 2500)
        plot = "".join(notebook["cells"][9]["source"]).replace("step=50", "step=2")
        with patch.object(plt, "show"):
            exec(compile(plot, "<colab plot>", "exec"), ns)  # noqa: S102
        plt.close(ns["figure"])
