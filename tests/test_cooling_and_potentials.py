import numpy as np

import HaloPotential_new
import WiersmaCooling
from analytic_models import CoolingConfig, HaloConfig
from cosmology import DEFAULT_COSMOLOGY


def test_wiersma_cooling_from_config_returns_finite_values():
    cooling = WiersmaCooling.Wiersma_Cooling.from_config(CoolingConfig(1.0 / 3.0, 0.0))
    value = cooling.lambda_cgs(1.0e6, 1.0e-4)
    grad_t = cooling.dln_lambda_dln_T(1.0e6, 1.0e-4)
    grad_rho = cooling.dln_lambda_dln_rho(1.0e6, 1.0e-4)

    assert np.isfinite(value)
    assert np.isfinite(grad_t)
    assert np.isfinite(grad_rho)


def test_combined_potential_from_config_has_positive_circular_velocity():
    rho_mean = DEFAULT_COSMOLOGY.mean_matter_density_Msun_kpc3(0.0)
    config = HaloConfig(
        M_vir_Msun=1.0e11,
        r_vir_kpc=100.0,
        c_vir=10.0,
        M_gal_Msun=1.0e10,
        a_gal_kpc=3.0,
        b_gal_kpc=0.3,
        rho_mean_Msun_kpc3=rho_mean,
        R200_kpc=120.0,
    )
    potential = HaloPotential_new.CombinedPotential.from_config(config)
    modified = HaloPotential_new.CombinedPotential_using_modified_plummer.from_config(config)
    disk = HaloPotential_new.MiyamotoNagaiPotential(1.0e10, 3.0, 0.3)

    assert potential.vc_kms(10.0) > 0.0
    assert modified.vc_kms(10.0) > 0.0
    assert np.isfinite(potential.phi_kms2(10.0))
    assert np.isfinite(modified.phi_kms2(10.0))
    assert np.isfinite(disk.phi_kms2(8.0, 0.2))


def test_single_canonical_module_exposes_analytic_and_cosmological_halos():
    analytic = HaloPotential_new.PowerLaw(m=-0.1, vc_Rvir_kms=150.0, Rvir_kpc=200.0)
    halo = HaloPotential_new.NFW(1.0e11, 0.0, 10.0)
    sphere = HaloPotential_new.IsothermalSphere(1.0e11, 100.0)

    assert analytic.vc_kms(20.0) > 0.0
    assert np.isfinite(analytic.phi_kms2(20.0))
    assert halo.rvir_kpc() > 0.0
    assert halo.v_vir_kms() > 0.0
    assert halo.r200_kpc() > 0.0
    assert halo.r200m_kpc() > 0.0
    assert halo.t_ff_Gyr(10.0) > 0.0
    assert np.all(sphere.vc_kms(np.array([10.0, 150.0])) > 0.0)
