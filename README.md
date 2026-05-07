# CGM_ICs

Initial conditions for hydrodynamic simulations of the ISM + CGM, based on
1D steady-state cooling-flow solutions from
[Stern et al. 2019, MNRAS 488, 2549](https://doi.org/10.1093/mnras/stz1859).

Rotation is added to the spherical solution as

```
v_phi = v_c · (R / R_circ) · sin(theta)
```

where `v_c` is the circular velocity, `R` the cylindrical radius, `theta`
the polar angle, and `R_circ` a free parameter (≈ 10–20 kpc for
Milky-Way-like halos).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The `-e` install lets the notebooks `import cgm_ics` directly without any
`sys.path` manipulation.

If you need exact pinned versions for reproducibility, use
`pip install -r requirements.txt` instead of (or after) the editable
install.

## Layout

| Path                                | Contents                                                                 |
|-------------------------------------|--------------------------------------------------------------------------|
| `src/cgm_ics/cooling_flow.py`       | Steady-state flow ODE integrator and `Cooling` / `Potential` interfaces. |
| `src/cgm_ics/cooling_functions.py`  | Concrete `Cooling` implementations: `Wiersma_Cooling` (HDF5), `Kartick_Cooling`, `DopitaSutherland_CIE`, `Constant_Cooling`. |
| `src/cgm_ics/HaloPotential.py`      | Analytic potentials: simple (`PowerLaw`, `Polynom`, `NFW`, …) plus the GOTHAM composites (`NFWPotential`, `PlummerPotential`, `OuterHaloPotential`, `CombinedPotential`, …). |
| `src/cgm_ics/ClusterGenerator.py`   | Star-cluster catalog generator (separate pipeline).                      |
| `src/cgm_ics/solve_ode.py`          | Alternate ODE solver used by most production notebooks.                  |
| `cooling_tables/`                   | HDF5 / text cooling tables (Wiersma+09, Dopita & Sutherland, etc.).      |
| `ipynb/`                            | Jupyter notebooks driving the package.                                   |

## Notebooks

- `ipynb/steady_state_integration_example.ipynb` — minimal example of integrating a steady-state flow.
- `ipynb/generate_CGM_ics.ipynb` — produces the actual CGM initial conditions (the headline output).
- `ipynb/generate_coolings_flows.ipynb`, `ipynb/generate_cooling_table.ipynb` — parameter sweeps producing `ipynb/tables/cgm_profiles_*.txt`.
- `ipynb/generate_clusters.ipynb`, `ipynb/catalog_analysis.ipynb` — drive `ClusterGenerator`.

## Cooling-table location

`cooling_functions` resolves `cooling_tables/` relative to the package source, so an
editable install picks it up automatically. To point at a different
directory (e.g. when installing non-editably), set:

```bash
export CGM_ICS_COOLING_DIR=/path/to/cooling
```
