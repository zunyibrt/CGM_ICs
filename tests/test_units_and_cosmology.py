import numpy as np

from cosmology import DEFAULT_COSMOLOGY
import unit_system as us


def test_unit_conversion_constants_are_self_consistent():
    assert np.isclose(us.MSUN_PER_YR_TO_G_PER_S, us.MSUN_TO_G / us.YR_TO_S)
    assert np.isclose(us.KPC_PER_KM_S_TO_GYR, us.KPC_TO_CM / us.KM_TO_CM / us.GYR_TO_S)
    assert np.isclose(us.KM2_S2_TO_CM2_S2, us.KM_TO_CM**2)


def test_cosmology_helpers_return_finite_physical_values():
    crit0 = DEFAULT_COSMOLOGY.critical_density_Msun_kpc3(0.0)
    crit1 = DEFAULT_COSMOLOGY.critical_density_Msun_kpc3(1.0)
    mean0 = DEFAULT_COSMOLOGY.mean_matter_density_Msun_kpc3(0.0)
    delta0 = DEFAULT_COSMOLOGY.delta_c_vir(0.0)
    rvir = DEFAULT_COSMOLOGY.virial_radius_kpc(1.0e12, z=0.0)

    assert crit0 > 0.0
    assert crit1 > crit0
    assert np.isclose(mean0, DEFAULT_COSMOLOGY.Om0 * crit0)
    assert delta0 > 100.0
    assert rvir > 0.0
