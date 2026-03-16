import numpy as np

from analytic_models import (
    RotationConfig,
    TurbulenceConfig,
    infer_rotation_diagnostics,
    infer_turbulence_diagnostics,
    maximum_hot_mode_accretion_rate,
)


class ConstantPotential:
    def vc_kms(self, radius_kpc):
        return 200.0 * np.ones(np.shape(np.asarray(radius_kpc, dtype=float)), dtype=float)


class MockSolution:
    def __init__(self, tcool_to_tff):
        self.radius_kpc = np.array([10.0, 20.0], dtype=float)
        self.velocity_kms = np.array([20.0, 15.0], dtype=float)
        self.sound_speed_kms = np.array([100.0, 100.0], dtype=float)
        self.density_cgs = np.array([1.0e-27, 8.0e-28], dtype=float)
        self.cooling_time_Gyr = np.array(tcool_to_tff, dtype=float)
        self.free_fall_time_Gyr = np.ones(2, dtype=float)
        self.potential = ConstantPotential()


def test_maximum_hot_mode_accretion_rate_increases_with_circular_velocity():
    low = maximum_hot_mode_accretion_rate(100.0, 10.0, 1.0)
    high = maximum_hot_mode_accretion_rate(150.0, 10.0, 1.0)
    assert high > low


def test_rotation_config_matches_expected_piecewise_behavior():
    potential = ConstantPotential()
    config = RotationConfig(R_circ_kpc=10.0)
    radii_kpc = np.array([5.0, 20.0], dtype=float)
    v_phi_kms = config.rotation_velocity_kms(radii_kpc, potential)

    assert np.isclose(v_phi_kms[0], 200.0)
    assert np.isclose(v_phi_kms[1], 100.0)


def test_rotation_diagnostics_use_explicit_float_fields():
    diagnostics = infer_rotation_diagnostics(MockSolution([10.0, 8.0]), RotationConfig(R_circ_kpc=10.0))

    assert diagnostics.disk_interface_radius_kpc == 10.0
    assert np.all(diagnostics.v_phi_kms > 0.0)
    assert diagnostics.specific_angular_momentum_kpc_kms > 0.0
    assert np.all(diagnostics.total_rotation_radians > 0.0)


def test_turbulence_closure_switches_between_hot_and_cool_regimes():
    hot = infer_turbulence_diagnostics(MockSolution([10.0, 8.0]), TurbulenceConfig())
    cool = infer_turbulence_diagnostics(MockSolution([0.5, 0.3]), TurbulenceConfig())

    assert np.all(hot.regime == "hot-inflow")
    assert np.all(cool.regime == "turbulence-dominated")
    assert np.all(cool.sigma_turb_kms > hot.sigma_turb_kms)
