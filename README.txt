# CGM Cooling-Flow ICs

This repo builds circumgalactic-medium initial conditions from steady radiatively cooling inflow solutions and related notebook workflows.

## Canonical Solver Stack

- `pysrc/solve_ode.py`
  - Canonical steady-state cooling-flow solver.
  - Owns the `Cooling` and `Potential` interfaces, the `CGMSolution` object, direct ODE integration, sonic-point shooting, and `R_circ` shooting.
- `pysrc/cooling_flow.py`
  - Import-path alias for `solve_ode.py`.
  - Exposes the same float-native solver symbols without maintaining a second API surface.
- `pysrc/solve_ode_changed.py`
  - Import alias re-exporting the canonical solver.
- `pysrc/WiersmaCooling.py`
  - Float-native cooling-table interfaces for Wiersma et al. (2009) and the auxiliary CIE tables in `cooling/`.
- `pysrc/HaloPotential_new.py`
  - Canonical halo + galaxy + outer-halo potential layer.
  - Also carries the simple analytic potential families used by the example notebook.

## Canonical Units

- radius: `kpc`
- mass: `Msun`
- mass flow rate: `Msun/yr`
- velocity: `km/s`
- potential / specific energy: `km^2/s^2`
- density: `g/cm^3`
- number density: `cm^-3`
- temperature: `K`
- time: `Gyr`
- cooling function: `erg cm^3 / s`

## Physics Scope

The repo is now organized around four paper-backed model layers:

- Paper 1: spherical steady cooling-flow solutions for the CGM.
- Paper 2: hot-mode accretion limits and galaxy-scale interpretation.
- Paper 3: slow-rotation diagnostics built on top of the 1D inflow solution.
- Paper 4: analytic turbulence diagnostics built on top of the 1D inflow solution.

The rotation and turbulence additions in this repo are analytic diagnostic layers. They are not full 2D or 3D simulation solvers.

## Main Entry Points

- `ipynb/m82_cooling_flow_walkthrough.ipynb`
  - Main end-to-end tutorial notebook.
  - Builds an M82-motivated halo, sweeps `R_sonic`, compares the full solution family, and then layers on rotation and turbulence diagnostics.
- `ipynb/steady_state_integration_example.ipynb`
  - Fastest example for direct solver use and both sonic-point / `R_circ` shooting.
- `ipynb/generate_coolings_flows.ipynb`
  - Main batch cooling-flow generation notebook.
- `ipynb/generate_CGM_ics.ipynb`
  - Float-native IC field assembly example built on the combined halo potential and solver outputs.
- `ipynb/generate_cooling_table.ipynb`
  - Cooling-table sampling notebook that exercises the import alias and float-native cooling interface.

## Documentation

- `docs/papers/`
  - Summaries of the four physics papers used as the source of truth during the review.
- `docs/review/physics_code_crosswalk.md`
  - Maps the paper assumptions and equations onto code paths and notebook cells.
- `docs/review/code_audit.md`
  - Audit of numerical, API, and implementation issues found during the review.
- `docs/review/reproducibility_and_notebooks.md`
  - Notebook dependency map, smoke-test baseline, and reproducibility notes.

## Tests

Run:

```bash
pytest -q
```

The test suite covers:
- solver smoke behavior,
- explicit float-native public fields,
- analytic rotation / turbulence diagnostics,
- cooling-table and potential construction,
- unit-system and cosmology helpers,
- notebook parsing and execution.

## Notes

- Cooling tables are expected under `cooling/Wiersma09_CoolingTables/`.
- Notebook workflows are still interactive and order-dependent; see `docs/review/reproducibility_and_notebooks.md` for the recommended run order.
- New feature work should target `pysrc/solve_ode.py`, `pysrc/WiersmaCooling.py`, `pysrc/HaloPotential_new.py`, and the explicit analytic/config layers.
