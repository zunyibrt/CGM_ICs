import numpy as np

import cooling_flow
import solve_ode
from analytic_models import RotationConfig, TurbulenceConfig


class DummyCooling:
    def lambda_cgs(self, temperature_K, nH_cm3):
        shape = np.broadcast(np.asarray(temperature_K, dtype=float), np.asarray(nH_cm3, dtype=float)).shape
        return np.full(shape, 1e-22, dtype=float)

    def dln_lambda_dln_T(self, temperature_K, nH_cm3):
        return np.zeros(np.shape(np.asarray(temperature_K, dtype=float)), dtype=float)

    def dln_lambda_dln_rho(self, temperature_K, nH_cm3):
        return np.zeros(np.shape(np.asarray(temperature_K, dtype=float)), dtype=float)


class DummyPotential:
    def vc_kms(self, radius_kpc):
        return 150.0 * np.ones(np.shape(np.asarray(radius_kpc, dtype=float)), dtype=float)

    def phi_kms2(self, radius_kpc):
        return -1.0e6 * np.ones(np.shape(np.asarray(radius_kpc, dtype=float)), dtype=float)

    def dln_vc_dln_r(self, radius_kpc):
        return np.zeros(np.shape(np.asarray(radius_kpc, dtype=float)), dtype=float)


def test_solve_ode_smoke_and_diagnostics():
    result = solve_ode.IntegrateFlowEquations(
        mass_flow_rate_Msun_per_yr=0.01,
        temperature_K=1.0e6,
        density_cgs=1.0e-27,
        potential=DummyPotential(),
        cooling=DummyCooling(),
        direction=1,
        R_min_kpc=10.0,
        R_max_kpc=20.0,
    )

    assert result.stop_reason == solve_ode.StopReason.MAX_RADIUS
    assert len(result.radius_kpc) > 2
    assert np.all(np.isfinite(result.free_fall_time_Gyr))
    assert np.all(result.cooling_luminosity_density_cgs > 0.0)

    rotation = result.rotation_diagnostics(RotationConfig(R_circ_kpc=5.0))
    turbulence = result.turbulence_diagnostics(TurbulenceConfig())

    assert rotation.disk_interface_radius_kpc == 5.0
    assert np.all(rotation.total_rotation_radians > 0.0)
    assert np.all(turbulence.sigma_turb_kms > 0.0)
    assert set(np.unique(turbulence.regime)).issubset({"hot-inflow", "turbulence-dominated"})


def test_solution_exposes_float_native_fields():
    result = solve_ode.IntegrateFlowEquations(
        mass_flow_rate_Msun_per_yr=0.01,
        temperature_K=1.0e6,
        density_cgs=1.0e-27,
        potential=DummyPotential(),
        cooling=DummyCooling(),
        direction=1,
        R_min_kpc=10.0,
        R_max_kpc=20.0,
    )

    assert np.all(result.radius_kpc > 0.0)
    assert np.all(result.temperature_K > 0.0)
    assert np.all(result.density_cgs > 0.0)
    assert np.all(np.isfinite(result.velocity_kms))
    assert np.all(np.isfinite(result.bernoulli_kms2))
    assert result.sonic_radius_kpc is None


def test_cooling_flow_is_an_import_path_alias_for_the_solver():
    assert cooling_flow.IntegrateFlowEquations is solve_ode.IntegrateFlowEquations

    result = cooling_flow.IntegrateFlowEquations(
        mass_flow_rate_Msun_per_yr=0.01,
        temperature_K=1.0e6,
        density_cgs=1.0e-27,
        potential=DummyPotential(),
        cooling=DummyCooling(),
        direction=1,
        R_min_kpc=10.0,
        R_max_kpc=20.0,
    )

    assert result.stop_reason == solve_ode.StopReason.MAX_RADIUS
