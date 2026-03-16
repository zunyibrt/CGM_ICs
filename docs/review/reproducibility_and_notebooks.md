# Reproducibility And Notebooks

## Baseline
- The review was executed against the local workspace version of the repo.
- The physics code now imports `pysrc/numpy_compat.py` explicitly, which stabilizes imports under modern NumPy while keeping the runtime behavior local to the repo modules.
- Smoke-test verification was done with `pytest -q`, not by fully rerunning every long notebook end-to-end.

## What Was Actually Reproduced

### Automated checks
- Module imports for:
  - `solve_ode`
  - `cooling_flow`
  - `WiersmaCooling`
  - `HaloPotential_new`
- A direct solver smoke test using `solve_ode.IntegrateFlowEquations(...)`.
- Legacy wrapper compatibility checks for `cooling_flow.IntegrateFlowEquations(...)`.
- Analytic regression checks for the paper-2, paper-3, and paper-4 helper layer.
- Cooling-table and potential-construction checks.
- Notebook JSON parsing and import-block execution for every notebook under `ipynb/`.

### Not reproduced automatically
- Full notebook execution through all plotting, pickling, and output-generation cells.
- Visual validation of generated figures.
- Numerical regression against stored reference solution files.

## Notebook Dependency Map

| Notebook | Import cell(s) | Solver path | Output / workflow role |
| --- | --- | --- | --- |
| `ipynb/generate_CGM_ics.ipynb` | cell 1 | `solve_ode` + `HaloPotential_new` + `WiersmaCooling` + `ClusterGenerator` | End-to-end IC generation workflow. |
| `ipynb/generate_cooling_flow_low.ipynb` | cell 1 | `solve_ode` + `HaloPotential_new` + `WiersmaCooling` | Low-mass / modified-potential cooling-flow runs. |
| `ipynb/generate_cooling_table.ipynb` | cell 1 | `cooling_flow` + `WiersmaCooling` | Cooling-table export through the legacy API. |
| `ipynb/generate_coolings_flows.ipynb` | cell 1 | `solve_ode` + `HaloPotential_new` + `WiersmaCooling` | Batch transonic solution generation. |
| `ipynb/generate_coolings_flows_alternate.ipynb` | cell 1 | `solve_ode` + `HaloPotential_new` + `WiersmaCooling` | Alternate batch generation workflow. |
| `ipynb/steady_state_integration_example.ipynb` | cell 1 | `solve_ode` + `HaloPotential_new` + `WiersmaCooling` | Fast direct-integration example using the canonical potential module. |

## Recommended Execution Order
1. `ipynb/generate_cooling_table.ipynb`
2. `ipynb/steady_state_integration_example.ipynb`
3. `ipynb/generate_coolings_flows.ipynb`
4. `ipynb/generate_coolings_flows_alternate.ipynb`
5. `ipynb/generate_cooling_flow_low.ipynb`
6. `ipynb/generate_CGM_ics.ipynb`

Reasoning:
- The table-generation and legacy example notebooks are the fastest path to confirming the import stack and basic solver behavior.
- The flow-generation notebooks are the main steady-state production path.
- IC generation should come last because it depends on the upstream flow setup and the auxiliary cluster generator.

## Reproducibility Hazards

### Hidden path state
- Notebooks commonly manipulate `sys.path`, especially `ipynb/steady_state_integration_example.ipynb`.
- This is workable interactively, but it means the notebooks are not hermetic if launched from arbitrary working directories.

### Cell-order dependence
- Configuration objects, warnings filters, and generated solution lists are often defined in earlier cells and reused later.
- Restart-and-run-all is the correct way to validate a notebook, not selective cell execution.

### Local data assumptions
- Cooling tables are expected under `cooling/Wiersma09_CoolingTables/`.
- The paper PDFs under `papers/` are local review inputs and not generated artifacts.
- Some notebook workflows write pickles or derived data under notebook-local output locations rather than a single central build directory.

### State serialized outside tests
- Some workflows save lists of `CGMSolution` objects or derived quantities via pickle.
- Those files are sensitive to API drift and should be treated as ephemeral workflow outputs, not durable interchange formats.

### Warning suppression and reload patterns
- Several notebooks import `warnings` and `importlib` for iterative interactive use.
- That is normal for exploratory work, but it weakens strict reproducibility because code can be reloaded in partially mutated kernels.

## Current Reproducibility Status
- Import-level reproducibility: good.
- Canonical solver-path reproducibility: good for the tested smoke cases.
- Full notebook reproducibility: partial, because the notebooks are still interactive workflows with manual execution order and side-effectful output cells.

## Recommended Next Step If Full Reproducibility Becomes Required
- Convert the current notebook import/configuration blocks into a small scriptable pipeline under `pysrc/` or `scripts/`.
- Keep notebooks as analysis front ends, but move parameter definitions and output generation into importable functions.
- Add one or two end-to-end regression tests around saved solution summaries rather than only import-block smoke tests.
