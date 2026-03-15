# Cooling Flow 2 Summary

## Reference
- Jonathan Stern et al., "The maximum accretion rate of hot gas in dark matter halos" (2020).

## Scientific Goal
- Recast hot-mode versus cold-mode accretion in terms of steady cooling-flow structure.
- Show that the key condition is set near the galaxy scale, not the virial radius.
- Derive a maximum hot-mode accretion rate `Mdot_crit`.

## Governing Assumptions
- Same cooling-flow framework as Paper 1.
- Angular momentum matters through the circularization radius `R_circ`.
- Hot-mode accretion is possible only if gas remains virialized and roughly hydrostatic down to `R_circ`.

## Key Equations / Results
- Hot mode requires the sonic point to lie inside the radius of rotational support:
  - `R_sonic < R_circ`.
- Equivalent criterion near the galaxy scale:
  - `t_cool / t_ff >= ~0.7` at `R_circ`.
- Maximum hot-mode accretion rate:
  - `Mdot_crit ~= 0.7 (v_c/100 km s^-1)^5.4 (R_circ/10 kpc) (Z/Z_sun)^-0.9 Msun/yr`.
- Above `Mdot_crit`, the volume-filling phase can be hot in the outer halo but cool and free-falling near the galaxy.

## Physical Interpretation
- The onset of hot-mode accretion is controlled by conditions at `R_circ`, not only by halo-mass-scale shock stability.
- Lower halo gas fractions move the hot-mode threshold to lower halo masses.
- Near threshold, an "inverted" configuration is possible:
  - outer halo hot and pressure-supported,
  - inner halo cool and transonic/free-falling.

## Implementation Implications For This Repo
- `R_circ` must be a first-class model input, not just a notebook convention.
- The solver stack must support both:
  - transonic solutions shot from `R_sonic`,
  - subsonic/stalling solutions shot from `R_circ`.
- The repo should expose explicit diagnostics for:
  - `R_sonic`,
  - `R_circ`,
  - `t_cool / t_ff`,
  - hot-mode admissibility,
  - `Mdot_crit`.

## Code Mapping
- Canonical solver:
  - `pysrc/solve_ode.py::shoot_from_R_sonic`
  - `pysrc/solve_ode.py::shoot_from_R_circ`
- Analytic helper added during this review:
  - `pysrc/analytic_models.py::maximum_hot_mode_accretion_rate`

## Limits / Caveats
- This is still effectively a 1D inflow model.
- Rotation is represented through a support radius and derived diagnostics, not a full multidimensional solution.
