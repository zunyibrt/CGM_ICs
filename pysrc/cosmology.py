"""Repo-local cosmology helpers replacing Astropy cosmology usage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import unit_system as us


@dataclass(frozen=True)
class Cosmology:
    H0_km_s_Mpc: float = 67.66
    Om0: float = 0.30966
    Ob0: float = 0.04897
    Ol0: float | None = None

    @property
    def Ode0(self) -> float:
        return 1.0 - self.Om0 if self.Ol0 is None else self.Ol0

    def E2(self, z: float) -> float:
        zp1 = 1.0 + z
        return self.Om0 * zp1**3 + self.Ode0

    def H_km_s_Mpc(self, z: float) -> float:
        return self.H0_km_s_Mpc * np.sqrt(self.E2(z))

    def Om_z(self, z: float) -> float:
        zp1 = 1.0 + z
        return self.Om0 * zp1**3 / self.E2(z)

    def critical_density_cgs(self, z: float = 0.0) -> float:
        H_s = self.H_km_s_Mpc(z) * us.KM_TO_CM / us.MPC_TO_CM
        return 3.0 * H_s**2 / (8.0 * np.pi * us.G_CGS)

    def critical_density_Msun_kpc3(self, z: float = 0.0) -> float:
        rho_cgs = self.critical_density_cgs(z)
        return rho_cgs * us.KPC_TO_CM**3 / us.MSUN_TO_G

    def mean_matter_density_Msun_kpc3(self, z: float = 0.0) -> float:
        return self.Om_z(z) * self.critical_density_Msun_kpc3(z)

    def delta_c_vir(self, z: float = 0.0) -> float:
        x = self.Om_z(z) - 1.0
        return 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2

    def virial_radius_kpc(self, M_vir_Msun: float, z: float = 0.0) -> float:
        rho_crit = self.critical_density_Msun_kpc3(z)
        return (M_vir_Msun / ((4.0 / 3.0) * np.pi * self.delta_c_vir(z) * rho_crit)) ** (1.0 / 3.0)


DEFAULT_COSMOLOGY = Cosmology()


def critical_density_Msun_kpc3(z: float = 0.0, cosmology: Cosmology = DEFAULT_COSMOLOGY) -> float:
    return cosmology.critical_density_Msun_kpc3(z)


def mean_matter_density_Msun_kpc3(z: float = 0.0, cosmology: Cosmology = DEFAULT_COSMOLOGY) -> float:
    return cosmology.mean_matter_density_Msun_kpc3(z)


def delta_c_vir(z: float = 0.0, cosmology: Cosmology = DEFAULT_COSMOLOGY) -> float:
    return cosmology.delta_c_vir(z)


def virial_radius_kpc(M_vir_Msun: float, z: float = 0.0, cosmology: Cosmology = DEFAULT_COSMOLOGY) -> float:
    return cosmology.virial_radius_kpc(M_vir_Msun, z)
