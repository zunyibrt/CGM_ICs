# Physics / Code Crosswalk

## Scope
- Physics authority: `papers/CoolingFlow1.pdf`, `papers/CoolingFlow2.pdf`, `papers/coolingflow3_rotation.pdf`, and `papers/coolingflow_4_turbulence.pdf`.
- Canonical solver stack after this review:
  - `pysrc/solve_ode.py`: canonical steady-state solver.
  - `pysrc/cooling_flow.py`: import-path alias to `solve_ode.py`.
  - `pysrc/solve_ode_changed.py`: import alias re-exporting `solve_ode.py`.

## Notebook To Module Map

| Notebook | Exact cells touching solver stack | Imported stack | Notes |
| --- | --- | --- | --- |
| `ipynb/generate_CGM_ics.ipynb` | import cell; halo setup cell; IC snapshot cell | `solve_ode`, `HaloPotential_new`, `WiersmaCooling` | Float-native IC field assembly example. |
| `ipynb/generate_cooling_flow_low.ipynb` | import cell; modified-plummer halo cell; diagnostics cell | `solve_ode`, `HaloPotential_new`, `WiersmaCooling`, `analytic_models` | Low-mass / alternate-potential float workflow. |
| `ipynb/generate_cooling_table.ipynb` | import cell; cooling-table sampling cell | `cooling_flow`, `WiersmaCooling` | Exercises the import alias and float-native cooling interface. |
| `ipynb/generate_coolings_flows.ipynb` | import cell; combined-halo setup cell; batch integration cell | `solve_ode`, `HaloPotential_new`, `WiersmaCooling` | Batch solution generation with canonical float fields. |
| `ipynb/generate_coolings_flows_alternate.ipynb` | import cell; modified-plummer setup cell; hot-mode comparison cell | `solve_ode`, `HaloPotential_new`, `WiersmaCooling`, `analytic_models` | Alternate batch setup. |
| `ipynb/steady_state_integration_example.ipynb` | import cell; simple-halo setup cell; direct-integration cell | `solve_ode`, `HaloPotential_new`, `WiersmaCooling` | Simple analytic-potential example on the canonical potential module. |

## Canonical API Mapping

| Concept | Canonical implementation | Import-path alias / notebook-facing path |
| --- | --- | --- |
| Cooling interface | `pysrc/solve_ode.py` `Cooling` | notebooks instantiate float-native classes from `pysrc/WiersmaCooling.py` |
| Potential interface | `pysrc/solve_ode.py` `Potential` | notebooks instantiate float-native classes from `pysrc/HaloPotential_new.py` |
| Flow solution object | `pysrc/solve_ode.py` `CGMSolution` | same object via `pysrc/cooling_flow.py` import alias |
| Direct ODE integration | `pysrc/solve_ode.py` `IntegrateFlowEquations` | same function via `pysrc/cooling_flow.py` |
| Sonic-point shooting | `pysrc/solve_ode.py` `shoot_from_R_sonic` | same function via `pysrc/cooling_flow.py` |
| Circularization-radius shooting | `pysrc/solve_ode.py` `shoot_from_R_circ` | same function via `pysrc/cooling_flow.py` |
| Rotation diagnostics | `pysrc/analytic_models.py` | surfaced through `CGMSolution.rotation_diagnostics(...)` |
| Turbulence diagnostics | `pysrc/analytic_models.py` | surfaced through `CGMSolution.turbulence_diagnostics(...)` |

## Paper 1: Cooling Flow Solutions For The CGM

| Paper item | Code mapping | Notebook usage | Status | Notes |
| --- | --- | --- | --- | --- |
| Steady-state spherical cooling-flow ODEs in `ln T`, `ln rho` | `pysrc/solve_ode.py:668` `_create_ode_system` | All `solve_ode` notebooks | `implemented` | This is the canonical equation set after solver consolidation. |
| Cooling time, flow time, free-fall time diagnostics | `pysrc/solve_ode.py:294`, `pysrc/solve_ode.py:299`, `pysrc/solve_ode.py:306` | Used for plotting and analysis in flow notebooks | `implemented` | `t_ffs` was repaired to use `self.Rs` consistently. |
| Bernoulli integral and bound/unbound logic | `pysrc/solve_ode.py:283`, `pysrc/solve_ode.py:637`, `pysrc/solve_ode.py:759` | Used implicitly in all shooting workflows | `implemented` | Stop-reason propagation was repaired so unbound solutions are labeled correctly. |
| Sonic-point shooting for transonic marginally bound family | `pysrc/solve_ode.py:384`, `pysrc/solve_ode.py:895` | `generate_CGM_ics.ipynb` cells 12, 14, 15; batch-flow notebooks; `steady_state_integration_example.ipynb` cell 11 | `implemented` | Root selection and inward/outward matching remain 1D and steady, matching paper scope. |
| One-parameter family labeled by `R_sonic` / `Mdot` | `pysrc/solve_ode.py:927`, solution `mass_flow_rate`, `R_sonic` property at `pysrc/solve_ode.py:334` | Sonic-radius scan notebooks | `implemented` | Notebook workflows already use `R_sonic` as the control parameter. |
| Derived observables: Mach, entropy, cooling radius, luminosity density | `pysrc/solve_ode.py:273`, `pysrc/solve_ode.py:289`, `pysrc/solve_ode.py:327`, `pysrc/solve_ode.py:317` | Plots and IC post-processing | `implemented` | `R_cool` and `L_cools_per_volume` were repaired during this review. |
| O VI mismatch / missing multiphase outer-halo physics | None in solver core | None | `missing` | This is a physics limitation of the model, not a local bug. |

## Paper 2: Galaxy Growth From Cooling CGM Flows

| Paper item | Code mapping | Notebook usage | Status | Notes |
| --- | --- | --- | --- | --- |
| Hot-mode threshold / maximum accretion scaling | `pysrc/analytic_models.py:133` `maximum_hot_mode_accretion_rate` | Available to notebooks and scripts; not yet wired into old notebooks | `implemented` | Added as an explicit analytic helper tied to the paper scaling. |
| Use of `R_circ` as a galaxy-scale boundary condition | `pysrc/solve_ode.py:479` `shoot_from_R_circ` | `steady_state_integration_example.ipynb` cell 13 | `implemented` | The core numerical entrypoint already existed; crosswalk is now documented. |
| Distinction between hot-mode and cold-mode behavior through thermodynamic ratios | `CGMSolution.t_cools`, `CGMSolution.t_ffs`, `CGMSolution.R_sonic` | Flow notebooks | `approximate` | The code exposes the diagnostics, but classification logic is still notebook-side rather than a single canonical classifier. |
| Paper-level accretion-limit comparison as a first-class diagnostic | `maximum_hot_mode_accretion_rate` and notebook-side comparisons | None by default | `approximate` | Implemented analytically, but not yet surfaced in a dedicated report object. |

## Paper 3: Rotation In Hot CGM Inflows

| Paper item | Code mapping | Notebook usage | Status | Notes |
| --- | --- | --- | --- | --- |
| Circularization / support radius `R_circ` | `pysrc/analytic_models.py:36` `RotationConfig`; `pysrc/solve_ode.py:479` | `steady_state_integration_example.ipynb` cell 13, IC notebooks | `implemented` | `R_circ` is now explicit in both the ODE boundary-condition path and the diagnostic layer. |
| Conserved-angular-momentum spin-up outside `R_circ` | `pysrc/analytic_models.py:46` `rotation_velocity` | Intended for IC construction and analysis | `implemented` | Implemented as a documented analytic diagnostic on top of the 1D inflow. |
| Winding ratio `v_phi / |v_r|` and total radians rotated before accretion | `pysrc/analytic_models.py:71`, `pysrc/analytic_models.py:90`, `pysrc/solve_ode.py:341` | Available to all solver consumers | `implemented` | Matches the slow-rotation diagnostic target from the plan. |
| Full 2D axisymmetric density / temperature structure of the rotating hot flow | None | None | `missing` | Explicitly out of scope for this repo revision. |
| Rotation feeding back on the 1D ODE outside the boundary-condition term `1 - (R_circ/R)^2` | `pysrc/solve_ode.py:701` | `shoot_from_R_circ` workflow | `approximate` | The code captures a reduced quasi-spherical correction, not the full paper-3 spatial structure. |

## Paper 4: Accretion-Driven Turbulence

| Paper item | Code mapping | Notebook usage | Status | Notes |
| --- | --- | --- | --- | --- |
| Hot-inflow and turbulence-dominated regime split | `pysrc/analytic_models.py:173`, `pysrc/analytic_models.py:176` | Available through `CGMSolution.turbulence_diagnostics` | `implemented` | Regime choice is parameterized by `transition_tcool_to_tff`. |
| Turbulent velocity dispersion scaling | `pysrc/analytic_models.py:174`, `pysrc/analytic_models.py:175` | Same | `implemented` | Includes separate hot-inflow and cooled/turbulent branches. |
| Turbulent pressure contribution | `pysrc/analytic_models.py:179` | Same | `implemented` | Exposed as `pressure_turb`. |
| Density-dispersion closure | `pysrc/analytic_models.py:182` | Same | `approximate` | Implemented as a paper-motivated analytic closure, not a calibrated simulation model. |
| Turbulence back-reaction on the steady ODE solution | None | None | `missing` | Deliberately not folded back into the 1D solver in this revision. |
| Full 3D turbulence simulation | None | None | `missing` | Out of scope by design. |

## Resolved Inconsistencies During This Review

| Previous inconsistency | Resolution |
| --- | --- |
| `solve_ode.py` vs `solve_ode_changed.py` diverged without a documented truth source | `solve_ode.py` is now the canonical implementation and `solve_ode_changed.py` is a thin shim. |
| `cooling_flow.py` carried an incompatible parallel implementation | Replaced with a pure import-path alias so there is only one solver implementation. |
| Cooling and potential APIs mixed quantity-based and float-based names | The public surface now uses explicit float-native methods such as `lambda_cgs`, `vc_kms`, and `phi_kms2`. |
| Several `CGMSolution` diagnostics referenced missing attributes or wrong call styles | The canonical solution object now exposes explicit float-native diagnostics such as `radius_kpc`, `cooling_time_Gyr`, and `bernoulli_kms2`. |

## Bottom Line
- Paper 1 is now represented directly by the canonical solver.
- Paper 2 is represented partly by the solver and partly by explicit analytic helpers.
- Paper 3 and paper 4 are implemented as documented analytic diagnostic layers attached to the validated 1D inflow solution.
- The remaining gaps are the ones that genuinely require a different model class: full rotating 2D structure and turbulence feedback on the flow equations.
