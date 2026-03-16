# Cooling Flow 3 Summary

## Reference
- Jonathan Stern et al., "Accretion onto disk galaxies via hot and rotating CGM inflows" (2024).

## Scientific Goal
- Extend hot CGM inflow theory to include rotation in the slow-rotation limit.
- Show that hot inflows can remain hot, subsonic, and quasi-spherical until reaching the radius of angular-momentum support.
- Connect CGM inflow structure to disk feeding and observable hot-gas rotation.

## Governing Assumptions
- Axisymmetric, steady inflow.
- Slow-rotation expansion: rotation is dynamically weak at large radius and grows inward.
- Angular momentum is approximately conserved along flowlines outside the support radius.
- The target regime is the hot, pressure-supported inflow, not a cold free-fall solution.

## Key Results Used Here
- There is a support radius `R_c,max` where the hot flow becomes rotationally supported.
- Outside `R_c,max`, the hot inflow spins up roughly via angular-momentum conservation:
  - `v_phi ~ r^-1` along the equatorial flow.
- The inflow remains hot until the disk-halo interface, then cools rapidly from `~10^6 K` to `~10^4 K`.
- The amount of rotation before accretion is set by local cooling-to-free-fall ratio:
  - total rotation `~ 2 * (t_cool / t_ff)` evaluated near `R_c,max`.
- In hot-mode systems the inflow can rotate many radians before cooling; in cold free-fall systems it rotates only `~1` radian.

## Physical Interpretation
- `t_cool / t_ff` near the galaxy is not only a thermal diagnostic; it also controls how much the inflow winds up.
- The hot CGM should show a characteristic rotational velocity profile that is shallower than pre-inflow halo expectations.
- The model is primarily about a hot quasi-spherical inflow transitioning to a disk-like geometry at the support radius.

## Implementation Implications For This Repo
- A full 2D solver is out of scope for this repo revision.
- The useful implementation target is a documented diagnostic layer:
  - `R_circ` as the disk-interface/support radius,
  - specific angular momentum at `R_circ`,
  - diagnostic `v_phi(r, theta)`,
  - winding ratio `v_phi / |v_r|`,
  - total radians rotated before accretion.
- Those diagnostics should sit on top of the validated 1D inflow solution rather than replacing it.

## Code Mapping
- New analytic layer added during this review:
  - `pysrc/analytic_models.py::RotationConfig`
  - `pysrc/analytic_models.py::infer_rotation_diagnostics`
- Solution entrypoint:
  - `pysrc/solve_ode.py::CGMSolution.rotation_diagnostics`

## Limits / Caveats
- The implemented extension is a slow-rotation diagnostic model, not the paper’s full 2D analytic solution.
- No attempt is made here to solve for the paper’s angular dependence of density and temperature.
