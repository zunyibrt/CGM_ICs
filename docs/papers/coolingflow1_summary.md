# Cooling Flow 1 Summary

## Reference
- Jonathan Stern et al., "Cooling flow solutions for the circumgalactic medium" (2019).

## Scientific Goal
- Derive steady-state, spherical, radiatively cooling CGM inflow solutions across halo masses.
- Show that hydrostatic initial halos converge onto a one-parameter family of cooling-flow solutions.
- Test whether those solutions explain Milky Way and cluster observables when feedback heating is weak.

## Governing Assumptions
- Steady state and spherical symmetry.
- Ideal gas with `gamma = 5/3`.
- Optically thin radiative cooling with `n_H^2 Lambda(T, n_H, Z)`.
- External gravitational potential fixed by halo + galaxy + outer-halo mass model.
- Negligible ongoing feedback heating inside the modeled region.

## Key Equations / Results
- Cooling time:
  - `t_cool = P / [(gamma - 1) n_H^2 Lambda]`.
- Free-fall time:
  - `t_ff = sqrt(2) r / v_c`.
- Bernoulli parameter:
  - `B = v^2/2 + c_s^2/(gamma-1) + Phi`.
- Subsonic self-similar solution for roughly flat `Lambda`:
  - `T ~ T_c ~ v_c^2`.
  - `t_flow ~ t_cool`.
  - For `v_c ~ r^m`: `T ~ r^(2m)`, `n_H ~ r^(-2+m)`, `M ~ r^(-2-2m)`.
- The transonic marginally bound solution is a one-parameter family, naturally labeled by `R_sonic`, `Mdot`, or total gas mass.
- In the hot, subsonic part of the flow:
  - `t_cool / t_ff` is weakly radius dependent.
  - For Milky-Way-like solutions it is of order `~ 10`.
- The model reproduces O VII / O VIII constraints better than O VI:
  - inner hot halo works reasonably well;
  - outer-halo O VI is underpredicted unless additional physics is added.

## Observational / Physical Implications
- Cooling flows are a benchmark state for halos between feedback episodes.
- A Milky-Way inflow with `Mdot ~ SFR` gives `t_cool/t_ff ~ few to 10`, consistent with several feedback-regulation arguments.
- Relaxed cluster entropy profiles can resemble cooling-flow expectations in outer regions.

## Implementation Implications For This Repo
- The canonical solver must expose:
  - steady-state integration,
  - sonic-point shooting,
  - Bernoulli-based bound/unbound logic,
  - `t_cool`, `t_ff`, Mach, entropy, and sonic-radius diagnostics.
- The most direct code mapping is now:
  - `pysrc/solve_ode.py`
  - `pysrc/cooling_flow.py` as an import-path alias to the canonical solver
  - `pysrc/WiersmaCooling.py`
  - `pysrc/HaloPotential_new.py`
- The main numerical expectations from the paper that the code should preserve are:
  - `T ~ T_c`,
  - `t_flow ~ t_cool`,
  - one-parameter-family behavior under sonic-point shooting,
  - physically meaningful Bernoulli termination.

## Limits / Caveats
- The paper is spherical and non-rotating.
- The steady-state solution is not expected to remain valid where the flow becomes strongly supersonic and `t_cool < t_ff`.
- O VI tension is a physics gap, not a small numerical detail.
