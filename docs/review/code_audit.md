# Code Audit

## Review Standard
- Every Python module under `pysrc/` was inspected with emphasis on numerical stability, units, interpolation behavior, event handling, API consistency, duplicate logic, hidden state, and notebook compatibility.
- The four papers in `papers/` were treated as the physics reference for the cooling-flow model and its intended extensions.

## Major Corrections Applied

### 1. Solver stack canonicalization
- `pysrc/solve_ode.py` is now the single source of truth for steady-state integration and shooting.
- `pysrc/solve_ode_changed.py` was reduced to a re-export shim because it had diverged without clear validation.
- `pysrc/cooling_flow.py` was reduced to an import-path alias so the repo no longer maintains a parallel solver API.

### 2. Physics-facing solver repairs
- `pysrc/solve_ode.py:201` now stores stop reasons correctly.
- `pysrc/solve_ode.py:306` now computes `t_ffs` from `self.Rs` rather than a nonexistent `self.radii`.
- `pysrc/solve_ode.py:311` now computes cumulative gas mass from the actual radius grid.
- `pysrc/solve_ode.py:317` now treats cooling luminosity density as a property rather than calling a property like a function.
- `pysrc/solve_ode.py:327` is now a regular method `R_cool(time)` instead of an invalid property with a required argument.
- `pysrc/solve_ode.py:69` reconciles the cooling abstract interface with the existing concrete cooling classes.

### 3. Paper-3 and paper-4 extension layer
- Added `pysrc/analytic_models.py`.
- Rotation support now lives in explicit types:
  - `RotationConfig`
  - `RotationDiagnostics`
  - `CGMSolution.rotation_diagnostics(...)`
- Turbulence support now lives in explicit types:
  - `TurbulenceConfig`
  - `TurbulenceDiagnostics`
  - `CGMSolution.turbulence_diagnostics(...)`
- Paper-2 hot-mode accretion scaling is now an explicit helper: `maximum_hot_mode_accretion_rate(...)`.

### 4. Cooling-table robustness
- `pysrc/WiersmaCooling.py` no longer depends on the current working directory to find cooling tables.
- Redshift table selection is now numeric and deterministic.
- Cooling-derivative interpolation is hardened against non-positive table values by clipping before taking `log10`.
- `from_config(...)` constructors were added so notebooks and scripts can use documented parameter objects.

### 5. Potential-layer cleanup
- `pysrc/HaloPotential_new.py` now exposes `from_config(...)` constructors for the combined potentials used by the notebooks.
- The simple analytic families and cosmology-backed `NFW` helper were folded into `pysrc/HaloPotential_new.py`.
- `pysrc/HaloPotential.py` was removed so the repo now has a single potential implementation path.

### 6. Python / NumPy compatibility
- Added `pysrc/numpy_compat.py` and imported it explicitly in the physics modules.
- This resolves import failures from NumPy 2 / older Astropy interactions without relying on ambient interpreter state.

## Subsystem Findings

### Core solver: `pysrc/solve_ode.py`
Status: materially improved and suitable as the canonical solver.

Strengths:
- Clean separation between interfaces, solution object, shooting routines, and low-level ODE helpers.
- The canonical solver now uses a repo-local float unit system with explicit unit-bearing field names.
- The solver exposes the diagnostics needed by the paper summaries and notebooks.

Risks that remain:
- `_calc_rho_from_tflow2tcool` still uses a very broad density bracket (`1e-7` to `1e10 cm^-3`). That is physically serviceable for the current use case, but expensive or brittle cooling models could still fail at the bracket stage.
- The ODE system still assumes the reduced quasi-spherical treatment of rotation through the factor `1 - (R_circ / R)^2`; it is not a general rotating hydro solver.
- Event ordering is inherited from `solve_ivp` event sequencing. The current stop-reason logic is correct for the configured events, but if new terminal events are added later the mapping must be updated with care.

### Cooling wrappers: `pysrc/WiersmaCooling.py`
Status: improved and now reproducible across working directories.

Strengths:
- Local data discovery is now repo-relative.
- `Wiersma_Cooling`, `Kartick_Cooling`, and `DopitaSutherland_CIE` all conform to the canonical cooling interface.
- Gradient interpolation is finite for the tested table points.

Risks that remain:
- The Wiersma tables are still interpolated directly on net cooling values. Regions close to zero net cooling remain sensitive to table behavior and interpolation noise even after clipping.
- `Constant_Cooling` still returns scalar zeros for derivatives; that is acceptable for testing but not a realistic production model.

### Potentials: `pysrc/HaloPotential_new.py`
Status: canonical.

Strengths:
- `HaloPotential_new.py` is now the only target for notebook and script workflows.
- Config-based constructors make the notebook halo setup more explicit and less error-prone.
- The same module now covers combined halo models, simple analytic families, and the cosmology-backed `NFW` helper.

Risks that remain:
- `Polynom.phi_kms2(...)` still uses a numerical quadrature on demand, so it remains slower and less battle-tested than the closed-form potentials.
- The module now spans both production halo models and example-oriented analytic families, so future changes should keep those responsibilities clearly separated.

### Legacy compatibility layer: `pysrc/cooling_flow.py`
Status: intentionally thin import alias.

Strengths:
- There is no longer a second implementation of the flow equations behind the import path.
- Notebook and script imports can use either module name without changing runtime behavior.

Risks that remain:
- Users must now adopt the explicit float-native field names and canonical units.
- Future feature work should land in `solve_ode.py`, not in the alias module.

### Notebooks: `ipynb/*.ipynb`
Status: reviewed for imports, call paths, and execution coupling.

Strengths:
- The notebook family already clusters around the `solve_ode` + `HaloPotential_new` + `WiersmaCooling` stack.
- The one notebook that still imports `cooling_flow` now routes through the canonical solver.

Risks that remain:
- Several notebooks mutate `sys.path`.
- The workflows depend on execution order and hidden in-memory state.
- Pickled solution objects and generated tables are not centrally versioned.
- Import cells and configuration cells are duplicated across notebooks rather than factored into reusable scripts.

### Auxiliary utilities: `pysrc/ClusterGenerator.py`
Status: functional, but not yet brought to the same standard as the solver code.

Findings:
- It repeatedly uses `np.random.seed(...)` (`pysrc/ClusterGenerator.py:84`, `pysrc/ClusterGenerator.py:162`, `pysrc/ClusterGenerator.py:492`), which mutates global RNG state.
- It prints directly from mutating methods (`pysrc/ClusterGenerator.py:661`, `pysrc/ClusterGenerator.py:743`), which makes library use noisier and harder to test.
- It is only indirectly connected to the cooling-flow physics and currently lacks dedicated tests.

Recommendation:
- Keep it as an auxiliary IC tool, but move it toward local RNG objects and optional logging if it becomes a long-term maintained dependency.

## Documentation Gaps Closed In This Review
- Added paper summaries under `docs/papers/`.
- Added this audit and the physics/code crosswalk under `docs/review/`.
- Rewrote `README.txt` so the public repo overview matches the current implementation rather than the original pre-refactor layout.

## Remaining Model Gaps
- No full 2D rotating hot-flow solver.
- No turbulence feedback folded back into the steady ODE system.
- No end-to-end notebook execution harness that regenerates all figures and tables automatically.
- The potential layer still mixes production combined halos with lightweight analytic families in one module.

## Verification
- `pytest -q` passes with the new tests.
- The tests cover solver smoke behavior, explicit float-native fields, analytic rotation/turbulence helpers, cooling-table construction, potential construction, notebook execution, and local unit/cosmology helpers.
