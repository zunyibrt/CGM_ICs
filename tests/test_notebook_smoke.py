import json
import os
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "ipynb"


def _load_notebook(notebook_path: Path):
    return json.loads(notebook_path.read_text())


def _code_cells(notebook_path: Path):
    data = _load_notebook(notebook_path)
    return [
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if cell.get("cell_type") == "code"
    ]


def _import_lines(notebook_path: Path):
    lines = []
    for cell in _code_cells(notebook_path):
        for line in cell.splitlines():
            if line.strip().startswith(("import ", "from ")):
                lines.append(line.strip())
    return lines


def test_notebooks_are_float_native_and_expose_expected_modules():
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    assert notebooks

    forbidden_import = "astro" + "py"
    forbidden_to = "." + "to("
    forbidden_to_value = "." + "to_value("

    for notebook in notebooks:
        raw = notebook.read_text()
        assert forbidden_import not in raw
        assert forbidden_to not in raw
        assert forbidden_to_value not in raw

    seen = {notebook.name: "\n".join(_import_lines(notebook)) for notebook in notebooks}
    assert "import solve_ode as CF" in seen["generate_CGM_ics.ipynb"]
    assert "import HaloPotential_new as Halo" in seen["generate_CGM_ics.ipynb"]
    assert "import cooling_flow as CF" in seen["generate_cooling_table.ipynb"]
    assert "import solve_ode as CF" in seen["m82_cooling_flow_walkthrough.ipynb"]
    assert "import HaloPotential_new as Halo" in seen["m82_cooling_flow_walkthrough.ipynb"]
    assert "import WiersmaCooling as Cool" in seen["m82_cooling_flow_walkthrough.ipynb"]
    assert "import HaloPotential_new as Halo" in seen["steady_state_integration_example.ipynb"]


def test_notebook_code_cells_execute():
    original_cwd = Path.cwd()
    try:
        for notebook in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
            namespace = {"__name__": "__notebook_smoke__"}
            os.chdir(notebook.parent)
            for cell in _code_cells(notebook):
                exec(cell, namespace)
    finally:
        os.chdir(original_cwd)
