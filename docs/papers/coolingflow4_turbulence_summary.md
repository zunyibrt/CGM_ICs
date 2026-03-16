# Cooling Flow 4 Summary

## Reference
- Roy Goldner et al., "Accretion-Driven Turbulence in the Circumgalactic Medium" (2025 preprint).

## Scientific Goal
- Build a baseline model for CGM turbulence driven by accretion alone.
- Combine cooling-flow style inflow theory with Robertson-Goldreich style turbulence amplification.
- Predict when the CGM is subsonically turbulent versus turbulence-dominated and supersonic.

## Governing Assumptions
- Quasi-steady inflow.
- Turbulent velocity `sigma_t` grows through compressive / adiabatic heating during contraction.
- Turbulent energy dissipates on roughly the outer-eddy turnover time.
- In the cool regime, thermal pressure can become subdominant to turbulent support.

## Key Equations / Results Used Here
- Turbulent amplification plus dissipation:
  - `d sigma_t / dt = sigma_t (-u_r - eta sigma_t) / r`
- If adiabatic heating dominates:
  - `sigma_t ~ r^-1`
- Hot-inflow regime:
  - `sigma_t,hot ~= |u_r| / (2 eta) ~ r^-1/2`
  - turbulence Mach number remains subsonic.
- Cool / turbulence-dominated regime:
  - `sigma_t,td ~= sqrt(3/2) V_c`
  - `u_r,td ~= -sqrt(3/2) eta V_c`
  - `rho ~ r^-2`
- Density distribution in the turbulent regime is approximately lognormal:
  - `sigma_s = sqrt(ln(1 + b^2 M_t^2))`
  - with compressive driving implying `b ~ 1`.

## Physical Interpretation
- Below roughly Milky-Way halo mass, inner-CGM inflows can become turbulence-dominated rather than thermal-pressure-dominated.
- In that regime the relevant support term is turbulent pressure, not a hot quasi-static atmosphere.
- The paper predicts wide ionization/density distributions and large velocity dispersions in UV absorption.

## Implementation Implications For This Repo
- A full 3D turbulence solver is out of scope.
- The useful code target is a parameterized analytic closure layered on top of the inflow solution:
  - turbulent velocity profile,
  - turbulent pressure estimate,
  - turbulent Mach number,
  - density-dispersion estimate,
  - regime classification (`hot-inflow` vs `turbulence-dominated`).

## Code Mapping
- New analytic layer added during this review:
  - `pysrc/analytic_models.py::TurbulenceConfig`
  - `pysrc/analytic_models.py::infer_turbulence_diagnostics`
- Solution entrypoint:
  - `pysrc/solve_ode.py::CGMSolution.turbulence_diagnostics`

## Limits / Caveats
- The implemented closure is intentionally analytic and 1D-informed.
- Turbulent support is not inserted back into the main ODE integrator in this revision.
- The paper’s VSF and density-PDF predictions are represented only through summary diagnostics, not synthetic 3D realizations.
