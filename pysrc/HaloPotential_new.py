"""Float-native halo and galaxy potentials for the CGM workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.optimize

import numpy_compat  # noqa: F401
import unit_system as us
from analytic_models import HaloConfig
from cosmology import DEFAULT_COSMOLOGY, Cosmology


def _radius_array(radius_kpc):
    return np.clip(np.asarray(radius_kpc, dtype=float), 1e-12, None)


class PowerLaw:
    def __init__(self, m: float, vc_Rvir_kms: float, Rvir_kpc: float, R_phi0_kpc: float | None = None):
        self.m = float(m)
        self.vc_Rvir_kms = float(vc_Rvir_kms)
        self.Rvir_kpc = float(Rvir_kpc)
        self.R_phi0_kpc = 100.0 * self.Rvir_kpc if R_phi0_kpc is None else float(R_phi0_kpc)

    def vc_kms(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return self.vc_Rvir_kms * (radius / self.Rvir_kpc) ** self.m

    def phi_kms2(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        if self.m != 0.0:
            return -self.vc_Rvir_kms**2 / (2.0 * self.m) * (
                (self.R_phi0_kpc / self.Rvir_kpc) ** (2.0 * self.m) - (radius / self.Rvir_kpc) ** (2.0 * self.m)
            )
        return -self.vc_Rvir_kms**2 * np.log(self.R_phi0_kpc / radius)

    def dln_vc_dln_r(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return np.zeros_like(radius) + self.m


class Polynom:
    def __init__(self, coeffs, Rvir_kpc: float, R_phi0_kpc: float | None = None):
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.Rvir_kpc = float(Rvir_kpc)
        self.R_phi0_kpc = 100.0 * self.Rvir_kpc if R_phi0_kpc is None else float(R_phi0_kpc)

    def vc_kms(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        log_ratio = np.log10(radius / self.Rvir_kpc)
        powers = np.array([coefficient * log_ratio**index for index, coefficient in enumerate(self.coeffs)])
        return 10.0 ** powers.sum(axis=0)

    def phi_kms2(self, radius_kpc):
        radii = _radius_array(radius_kpc)
        scalar_input = radii.shape == ()
        radii_1d = np.atleast_1d(radii)
        phi = np.empty_like(radii_1d, dtype=float)
        for index, radius in enumerate(radii_1d):
            grid = np.geomspace(radius, self.R_phi0_kpc, 512)
            integrand = self.vc_kms(grid) ** 2 / grid
            phi[index] = -np.trapezoid(integrand, grid)
        if scalar_input:
            return float(phi[0])
        return phi

    def dln_vc_dln_r(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        log_ratio = np.log10(radius / self.Rvir_kpc)
        terms = [
            index * coefficient * log_ratio ** (index - 1)
            for index, coefficient in enumerate(self.coeffs)
            if index > 0
        ]
        if not terms:
            return np.zeros_like(radius)
        return np.sum(np.asarray(terms), axis=0)


class PowerLaw_with_AngularMomentum(PowerLaw):
    def __init__(self, m: float, vc_Rvir_kms: float, Rvir_kpc: float, Rcirc_kpc: float, R_phi0_kpc: float | None = None):
        super().__init__(m=m, vc_Rvir_kms=vc_Rvir_kms, Rvir_kpc=Rvir_kpc, R_phi0_kpc=R_phi0_kpc)
        self.Rcirc_kpc = float(Rcirc_kpc)

    def vc_kms(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        base = super().vc_kms(radius)
        return base * np.sqrt(np.clip(1.0 - (self.Rcirc_kpc / radius) ** 2, 0.0, None))

    def dln_vc_dln_r(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        correction = np.where(
            radius > self.Rcirc_kpc,
            1.0 / ((radius / self.Rcirc_kpc) ** 2 - 1.0),
            0.0,
        )
        return super().dln_vc_dln_r(radius) + correction


@dataclass(frozen=True)
class _NFWProfile:
    M_vir_Msun: float
    r_vir_kpc: float
    c_vir: float

    @property
    def r_s_kpc(self) -> float:
        return self.r_vir_kpc / self.c_vir

    @property
    def norm(self) -> float:
        return np.log(1.0 + self.c_vir) - self.c_vir / (1.0 + self.c_vir)

    @property
    def rho_s_Msun_kpc3(self) -> float:
        return self.M_vir_Msun / (4.0 * np.pi * self.r_s_kpc**3 * self.norm)

    def enclosed_mass_Msun(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        x = radius / self.r_s_kpc
        return 4.0 * np.pi * self.rho_s_Msun_kpc3 * self.r_s_kpc**3 * (
            np.log(1.0 + x) - x / (1.0 + x)
        )

    def phi_kms2(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        x = radius / self.r_s_kpc
        return (
            -4.0
            * np.pi
            * us.G_KPC_KM2_S2_PER_MSUN
            * self.rho_s_Msun_kpc3
            * self.r_s_kpc**2
            * np.log(1.0 + x)
            / x
        )

    def dM_dr_Msun_per_kpc(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        x = radius / self.r_s_kpc
        return 4.0 * np.pi * self.rho_s_Msun_kpc3 * self.r_s_kpc**2 * x / (1.0 + x) ** 2


class NFWPotential:
    def __init__(self, M_vir_Msun: float, r_vir_kpc: float, c_vir: float):
        self.M_vir_Msun = float(M_vir_Msun)
        self.r_vir_kpc = float(r_vir_kpc)
        self.c_vir = float(c_vir)
        self._profile = _NFWProfile(self.M_vir_Msun, self.r_vir_kpc, self.c_vir)

    def enclosed_mass_Msun(self, radius_kpc):
        return self._profile.enclosed_mass_Msun(radius_kpc)

    def vc_kms(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return np.sqrt(us.G_KPC_KM2_S2_PER_MSUN * self.enclosed_mass_Msun(radius) / radius)

    def phi_kms2(self, radius_kpc):
        return self._profile.phi_kms2(radius_kpc)

    def dln_vc_dln_r(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        x = radius / self._profile.r_s_kpc
        denominator = (1.0 + x) ** 2 * (np.log(1.0 + x) - x / (1.0 + x))
        return 0.5 * (x**2 / denominator - 1.0)


class NFW:
    mu = 0.6
    X = 0.75

    def __init__(self, Mvir_Msun: float, z: float, cvir: float, cosmology: Cosmology = DEFAULT_COSMOLOGY):
        self.Mvir_Msun = float(Mvir_Msun)
        self.z = float(z)
        self.cvir = float(cvir)
        self.cosmology = cosmology
        self._rvir_kpc = cosmology.virial_radius_kpc(self.Mvir_Msun, z=self.z)
        self._rscale_kpc = self._rvir_kpc / self.cvir
        self._norm = np.log(1.0 + self.cvir) - self.cvir / (1.0 + self.cvir)
        self.rho_scale_Msun_kpc3 = self.Mvir_Msun / (4.0 * np.pi * self._rscale_kpc**3 * self._norm)
        self.rho_scale_cgs = self.rho_scale_Msun_kpc3 * us.MSUN_TO_G / us.KPC_TO_CM**3

    def delta_c_vir(self):
        return self.cosmology.delta_c_vir(self.z)

    def rvir_kpc(self):
        return self._rvir_kpc

    def r_ta_kpc(self, use200m: bool = False):
        return 2.0 * (self.r200m_kpc() if use200m else self.rvir_kpc())

    def r_scale_kpc(self):
        return self._rscale_kpc

    def rho2rho_scale(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        x = radius / self._rscale_kpc
        return 1.0 / (x * (1.0 + x) ** 2)

    def rho_cgs(self, radius_kpc):
        return self.rho_scale_cgs * self.rho2rho_scale(radius_kpc)

    def enclosedMass_Msun(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        x = radius / self._rscale_kpc
        return 4.0 * np.pi * self.rho_scale_Msun_kpc3 * self._rscale_kpc**3 * (
            np.log(1.0 + x) - x / (1.0 + x)
        )

    def v_vir_kms(self):
        return float(np.sqrt(us.G_KPC_KM2_S2_PER_MSUN * self.Mvir_Msun / self._rvir_kpc))

    def v_ff_kms(self, radius_kpc, rdrop_kpc=None):
        if rdrop_kpc is None:
            rdrop_kpc = 2.0 * self.r200m_kpc()
        return np.sqrt(2.0 * (self.phi_kms2(rdrop_kpc) - self.phi_kms2(radius_kpc)))

    def vc_kms(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return np.sqrt(us.G_KPC_KM2_S2_PER_MSUN * self.enclosedMass_Msun(radius) / radius)

    def dln_vc_dln_r(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        x = radius / self._rscale_kpc
        denominator = (1.0 + x) ** 2 * (np.log(1.0 + x) - x / (1.0 + x))
        return 0.5 * (x**2 / denominator - 1.0)

    def mean_enclosed_rho_over_rhocrit(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        density = self.enclosedMass_Msun(radius) / ((4.0 / 3.0) * np.pi * radius**3)
        return density / self.cosmology.critical_density_Msun_kpc3(self.z)

    def _radius_at_overdensity(self, target_overdensity: float):
        def residual(log_radius):
            radius = 10.0**log_radius
            return self.mean_enclosed_rho_over_rhocrit(radius) - target_overdensity

        lower = np.log10(self._rscale_kpc * 1e-4)
        upper = np.log10(self._rvir_kpc * 20.0)
        grid = np.linspace(lower, upper, 256)
        values = residual(grid)
        sign_change = np.where(np.sign(values[:-1]) != np.sign(values[1:]))[0]
        if len(sign_change) == 0:
            return float(self._rvir_kpc)
        idx = int(sign_change[0])
        root = scipy.optimize.brentq(residual, grid[idx], grid[idx + 1])
        return float(10.0**root)

    def r200_kpc(self, delta: float = 200.0):
        return self._radius_at_overdensity(delta)

    def r200m_kpc(self, delta: float = 200.0):
        return self._radius_at_overdensity(delta * self.cosmology.Om_z(self.z))

    def M200_Msun(self, delta: float = 200.0):
        return float(self.enclosedMass_Msun(self.r200_kpc(delta)))

    def M200m_Msun(self, delta: float = 200.0):
        return float(self.enclosedMass_Msun(self.r200m_kpc(delta)))

    def phi_kms2(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        x = radius / self._rscale_kpc
        return (
            -4.0
            * np.pi
            * us.G_KPC_KM2_S2_PER_MSUN
            * self.rho_scale_Msun_kpc3
            * self._rscale_kpc**2
            * np.log(1.0 + x)
            / x
        )

    def t_ff_Gyr(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return np.sqrt(2.0) * us.flow_time_Gyr(radius, self.vc_kms(radius))


class PlummerPotential:
    def __init__(self, M_Msun: float, a_kpc: float):
        self.M_Msun = float(M_Msun)
        self.a_kpc = float(a_kpc)

    def enclosed_mass_Msun(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return self.M_Msun * radius**3 / np.power(radius**2 + self.a_kpc**2, 1.5)

    def vc_kms(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return np.sqrt(us.G_KPC_KM2_S2_PER_MSUN * self.enclosed_mass_Msun(radius) / radius)

    def phi_kms2(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return -us.G_KPC_KM2_S2_PER_MSUN * self.M_Msun / np.sqrt(radius**2 + self.a_kpc**2)

    def dln_vc_dln_r(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return 0.5 * (3.0 * self.a_kpc**2 / (radius**2 + self.a_kpc**2) - 1.0)

    def dM_dr_Msun_per_kpc(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return self.M_Msun * 3.0 * radius**2 * self.a_kpc**2 / np.power(radius**2 + self.a_kpc**2, 2.5)


class ModifiedPlummerPotential:
    """Spherical softening used by the IC notebooks."""

    def __init__(self, M_Msun: float, a_kpc: float, b_kpc: float):
        self.M_Msun = float(M_Msun)
        self.a_kpc = float(a_kpc)
        self.b_kpc = float(b_kpc)

    def enclosed_mass_Msun(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        s = np.sqrt(radius**2 + self.a_kpc**2)
        return self.M_Msun * radius**3 / (s * (s + self.b_kpc) ** 2)

    def dM_dr_Msun_per_kpc(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        s = np.sqrt(radius**2 + self.a_kpc**2)
        numerator = self.a_kpc**2 * (3.0 * s + self.b_kpc) + 2.0 * s**2 * self.b_kpc
        denominator = s**3 * (s + self.b_kpc) ** 3
        return self.M_Msun * radius**2 * numerator / denominator

    def vc_kms(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return np.sqrt(us.G_KPC_KM2_S2_PER_MSUN * self.enclosed_mass_Msun(radius) / radius)

    def phi_kms2(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return -us.G_KPC_KM2_S2_PER_MSUN * self.M_Msun / (np.sqrt(radius**2 + self.a_kpc**2) + self.b_kpc)

    def dln_vc_dln_r(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        mass = self.enclosed_mass_Msun(radius)
        return 0.5 * (self.dM_dr_Msun_per_kpc(radius) * radius / mass - 1.0)


class OuterHaloPotential:
    def __init__(self, rho_mean_Msun_kpc3: float, R200_kpc: float):
        self.rho_mean_Msun_kpc3 = float(rho_mean_Msun_kpc3)
        self.R200_kpc = float(R200_kpc)

    def enclosed_mass_Msun(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        term1 = (5.0 * self.R200_kpc) ** 1.5 * (2.0 / 3.0) * radius**1.5
        term2 = radius**3 / 3.0
        return 4.0 * np.pi * self.rho_mean_Msun_kpc3 * (term1 + term2)

    def dM_dr_Msun_per_kpc(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return 4.0 * np.pi * self.rho_mean_Msun_kpc3 * ((5.0 * self.R200_kpc) ** 1.5 * radius**0.5 + radius**2)

    def vc_kms(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return np.sqrt(us.G_KPC_KM2_S2_PER_MSUN * self.enclosed_mass_Msun(radius) / radius)

    def phi_kms2(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        term1 = (4.0 / 3.0) * (5.0 * self.R200_kpc) ** 1.5 * radius**0.5
        term2 = radius**2 / 6.0
        return 4.0 * np.pi * us.G_KPC_KM2_S2_PER_MSUN * self.rho_mean_Msun_kpc3 * (term1 + term2)

    def dln_vc_dln_r(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        x = radius / (5.0 * self.R200_kpc)
        return 0.5 * (x**-1.5 + 2.0) / (2.0 * x**-1.5 + 1.0)


class _CombinedPotentialBase:
    def __init__(
        self,
        nfw: NFWPotential,
        baryons,
        outer: OuterHaloPotential,
    ):
        self.nfw = nfw
        self.baryons = baryons
        self.outer = outer

        self.M_vir_Msun = nfw.M_vir_Msun
        self.r_vir_kpc = nfw.r_vir_kpc
        self.c_vir = nfw.c_vir
        self.r_s_kpc = nfw._profile.r_s_kpc
        self.rho_s_Msun_kpc3 = nfw._profile.rho_s_Msun_kpc3

    def enclosed_mass_nfw_Msun(self, radius_kpc):
        return self.nfw.enclosed_mass_Msun(radius_kpc)

    def enclosed_mass_baryons_Msun(self, radius_kpc):
        return self.baryons.enclosed_mass_Msun(radius_kpc)

    def enclosed_mass_outer_Msun(self, radius_kpc):
        return self.outer.enclosed_mass_Msun(radius_kpc)

    def enclosed_mass_Msun(self, radius_kpc):
        return (
            self.enclosed_mass_nfw_Msun(radius_kpc)
            + self.enclosed_mass_baryons_Msun(radius_kpc)
            + self.enclosed_mass_outer_Msun(radius_kpc)
        )

    def vc_kms(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return np.sqrt(us.G_KPC_KM2_S2_PER_MSUN * self.enclosed_mass_Msun(radius) / radius)

    def phi_kms2(self, radius_kpc):
        return (
            self.nfw.phi_kms2(radius_kpc)
            + self.baryons.phi_kms2(radius_kpc)
            + self.outer.phi_kms2(radius_kpc)
        )

    def dln_vc_dln_r(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        total_mass = self.enclosed_mass_Msun(radius)
        total_derivative = (
            self.nfw._profile.dM_dr_Msun_per_kpc(radius)
            + self.baryons.dM_dr_Msun_per_kpc(radius)
            + self.outer.dM_dr_Msun_per_kpc(radius)
        )
        return 0.5 * (total_derivative * radius / total_mass - 1.0)


class CombinedPotential(_CombinedPotentialBase):
    def __init__(
        self,
        M_vir_Msun: float,
        r_vir_kpc: float,
        c_vir: float,
        M_gal_Msun: float,
        a_gal_kpc: float,
        rho_mean_Msun_kpc3: float,
        R200_kpc: float,
    ):
        self.M_gal_Msun = float(M_gal_Msun)
        self.a_gal_kpc = float(a_gal_kpc)
        self.rho_mean_Msun_kpc3 = float(rho_mean_Msun_kpc3)
        self.R200_kpc = float(R200_kpc)
        super().__init__(
            NFWPotential(M_vir_Msun, r_vir_kpc, c_vir),
            PlummerPotential(self.M_gal_Msun, self.a_gal_kpc),
            OuterHaloPotential(self.rho_mean_Msun_kpc3, self.R200_kpc),
        )

    @classmethod
    def from_config(cls, config: HaloConfig):
        rho_mean = config.rho_mean_Msun_kpc3
        R200 = config.R200_kpc
        if rho_mean is None or R200 is None or config.M_gal_Msun is None or config.a_gal_kpc is None:
            raise ValueError("CombinedPotential.from_config requires M_gal_Msun, a_gal_kpc, rho_mean_Msun_kpc3, and R200_kpc")
        return cls(
            config.M_vir_Msun,
            config.r_vir_kpc,
            config.c_vir,
            config.M_gal_Msun,
            config.a_gal_kpc,
            rho_mean,
            R200,
        )


class CombinedPotential_using_modified_plummer(_CombinedPotentialBase):
    def __init__(
        self,
        M_vir_Msun: float,
        r_vir_kpc: float,
        c_vir: float,
        M_gal_Msun: float,
        a_gal_kpc: float,
        b_gal_kpc: float,
        rho_mean_Msun_kpc3: float,
        R200_kpc: float,
    ):
        self.M_gal_Msun = float(M_gal_Msun)
        self.a_gal_kpc = float(a_gal_kpc)
        self.b_gal_kpc = float(b_gal_kpc)
        self.rho_mean_Msun_kpc3 = float(rho_mean_Msun_kpc3)
        self.R200_kpc = float(R200_kpc)
        super().__init__(
            NFWPotential(M_vir_Msun, r_vir_kpc, c_vir),
            ModifiedPlummerPotential(self.M_gal_Msun, self.a_gal_kpc, self.b_gal_kpc),
            OuterHaloPotential(self.rho_mean_Msun_kpc3, self.R200_kpc),
        )

    @classmethod
    def from_config(cls, config: HaloConfig):
        if (
            config.M_gal_Msun is None
            or config.a_gal_kpc is None
            or config.b_gal_kpc is None
            or config.rho_mean_Msun_kpc3 is None
            or config.R200_kpc is None
        ):
            raise ValueError(
                "CombinedPotential_using_modified_plummer.from_config requires "
                "M_gal_Msun, a_gal_kpc, b_gal_kpc, rho_mean_Msun_kpc3, and R200_kpc"
            )
        return cls(
            config.M_vir_Msun,
            config.r_vir_kpc,
            config.c_vir,
            config.M_gal_Msun,
            config.a_gal_kpc,
            config.b_gal_kpc,
            config.rho_mean_Msun_kpc3,
            config.R200_kpc,
        )


class MiyamotoNagaiPotential:
    """Axisymmetric disk potential used by the IC notebook workflow."""

    def __init__(self, M_Msun: float, a_kpc: float, b_kpc: float):
        self.M_Msun = float(M_Msun)
        self.a_kpc = float(a_kpc)
        self.b_kpc = float(b_kpc)

    def phi_kms2(self, R_kpc, z_kpc):
        R = np.asarray(R_kpc, dtype=float)
        z = np.asarray(z_kpc, dtype=float)
        vertical = np.sqrt(z**2 + self.b_kpc**2) + self.a_kpc
        return -us.G_KPC_KM2_S2_PER_MSUN * self.M_Msun / np.sqrt(R**2 + vertical**2)

    def vc_kms(self, R_kpc):
        R = _radius_array(R_kpc)
        scale = self.a_kpc + self.b_kpc
        return np.sqrt(us.G_KPC_KM2_S2_PER_MSUN * self.M_Msun * R**2 / np.power(R**2 + scale**2, 1.5))

    def dln_vc_dln_r(self, R_kpc):
        R = _radius_array(R_kpc)
        scale = self.a_kpc + self.b_kpc
        return 1.0 - 1.5 * R**2 / (R**2 + scale**2)


class IsothermalSphere:
    def __init__(self, Mvir_Msun: float, Rvir_kpc: float):
        self.Mvir_Msun = float(Mvir_Msun)
        self.Rvir_kpc = float(Rvir_kpc)
        self.vvir_kms = float(np.sqrt(us.G_KPC_KM2_S2_PER_MSUN * self.Mvir_Msun / self.Rvir_kpc))

    def vc_kms(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        inner = np.full_like(radius, self.vvir_kms)
        outer = np.sqrt(us.G_KPC_KM2_S2_PER_MSUN * self.Mvir_Msun / radius)
        return np.where(radius < self.Rvir_kpc, inner, outer)

    def phi_kms2(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        outer = -us.G_KPC_KM2_S2_PER_MSUN * self.Mvir_Msun / radius
        inner = -self.vvir_kms**2 * (1.0 + np.log(self.Rvir_kpc / radius))
        return np.where(radius < self.Rvir_kpc, inner, outer) - 100.0**2

    def dln_vc_dln_r(self, radius_kpc):
        radius = _radius_array(radius_kpc)
        return np.where(radius < self.Rvir_kpc, 0.0, -0.5)
