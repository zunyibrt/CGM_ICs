"""Float-native analytic model helpers tied to the paper summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import numpy_compat  # noqa: F401
import unit_system as us


@dataclass(frozen=True)
class CoolingConfig:
    metallicity_solar: float
    redshift: float


@dataclass(frozen=True)
class HaloConfig:
    M_vir_Msun: float
    r_vir_kpc: float
    c_vir: float
    M_gal_Msun: Optional[float] = None
    a_gal_kpc: Optional[float] = None
    b_gal_kpc: Optional[float] = None
    rho_mean_Msun_kpc3: Optional[float] = None
    R200_kpc: Optional[float] = None


@dataclass(frozen=True)
class RotationConfig:
    R_circ_kpc: float
    polar_angle_rad: float = np.pi / 2.0

    def specific_angular_momentum_kpc_kms(self, potential) -> float:
        return float(potential.vc_kms(self.R_circ_kpc) * self.R_circ_kpc)

    def rotation_velocity_kms(
        self,
        radius_kpc,
        potential,
        polar_angle_rad: Optional[float] = None,
    ) -> np.ndarray:
        theta = self.polar_angle_rad if polar_angle_rad is None else polar_angle_rad
        radius = us.as_array(radius_kpc)
        specific_angular_momentum = self.specific_angular_momentum_kpc_kms(potential)
        outer = specific_angular_momentum * np.sin(theta) / radius
        inner = potential.vc_kms(radius)
        mask = radius >= self.R_circ_kpc
        return np.where(mask, outer, inner)

    def winding_ratio(
        self,
        radius_kpc,
        tcool_to_tff,
        polar_angle_rad: Optional[float] = None,
    ) -> np.ndarray:
        theta = self.polar_angle_rad if polar_angle_rad is None else polar_angle_rad
        radius = us.as_array(radius_kpc)
        return (
            np.sqrt(2.0)
            * us.as_array(tcool_to_tff)
            * (self.R_circ_kpc / radius)
            * np.sin(theta)
        )

    def total_rotation_radians(self, tcool_to_tff_at_R_circ) -> np.ndarray:
        return 2.0 * us.as_array(tcool_to_tff_at_R_circ)


@dataclass(frozen=True)
class TurbulenceConfig:
    eta_subsonic: float = 0.2
    eta_supersonic: float = 0.7
    forcing_b: float = 1.0
    saturation_factor: float = np.sqrt(3.0 / 2.0)
    transition_tcool_to_tff: float = 1.0


@dataclass(frozen=True)
class RotationDiagnostics:
    v_phi_kms: np.ndarray
    specific_angular_momentum_kpc_kms: float
    tcool_to_tff: np.ndarray
    winding_ratio: np.ndarray
    total_rotation_radians: np.ndarray
    disk_interface_radius_kpc: float


@dataclass(frozen=True)
class TurbulenceDiagnostics:
    sigma_turb_kms: np.ndarray
    pressure_turb_dyne_cm2: np.ndarray
    mach_turb: np.ndarray
    density_dispersion: np.ndarray
    regime: np.ndarray


def maximum_hot_mode_accretion_rate(
    vc_kms: float,
    R_circ_kpc: float,
    metallicity_solar: float,
) -> float:
    vc_scale = (float(vc_kms) / 100.0) ** 5.4
    r_scale = float(R_circ_kpc) / 10.0
    z_scale = metallicity_solar ** (-0.9)
    return 0.7 * vc_scale * r_scale * z_scale


def infer_rotation_diagnostics(solution, rotation: RotationConfig) -> RotationDiagnostics:
    tcool_to_tff = solution.cooling_time_Gyr / solution.free_fall_time_Gyr
    v_phi_kms = rotation.rotation_velocity_kms(solution.radius_kpc, solution.potential)
    winding_ratio = rotation.winding_ratio(solution.radius_kpc, tcool_to_tff)
    rcirc_idx = int(np.argmin(np.abs(solution.radius_kpc - rotation.R_circ_kpc)))
    total_rotation = rotation.total_rotation_radians(tcool_to_tff[rcirc_idx])
    total_rotation = np.full(solution.radius_kpc.shape, total_rotation, dtype=float)
    return RotationDiagnostics(
        v_phi_kms=v_phi_kms,
        specific_angular_momentum_kpc_kms=rotation.specific_angular_momentum_kpc_kms(solution.potential),
        tcool_to_tff=tcool_to_tff,
        winding_ratio=winding_ratio,
        total_rotation_radians=total_rotation,
        disk_interface_radius_kpc=rotation.R_circ_kpc,
    )


def infer_turbulence_diagnostics(
    solution,
    config: TurbulenceConfig = TurbulenceConfig(),
) -> TurbulenceDiagnostics:
    tcool_to_tff = solution.cooling_time_Gyr / solution.free_fall_time_Gyr
    sigma_hot_kms = np.abs(solution.velocity_kms) / (2.0 * config.eta_subsonic)
    sigma_cool_kms = config.saturation_factor * solution.potential.vc_kms(solution.radius_kpc)
    is_cool = tcool_to_tff < config.transition_tcool_to_tff
    sigma_turb_kms = np.where(is_cool, sigma_cool_kms, sigma_hot_kms)

    pressure_turb_dyne_cm2 = solution.density_cgs * (sigma_turb_kms * us.KM_TO_CM) ** 2 / 3.0
    mach_turb = sigma_turb_kms / solution.sound_speed_kms

    density_dispersion = np.where(
        is_cool,
        np.sqrt(np.log1p((config.forcing_b * mach_turb) ** 2)),
        0.5 / config.eta_subsonic / np.clip(tcool_to_tff, 1e-12, None),
    )
    regime = np.where(is_cool, "turbulence-dominated", "hot-inflow")

    return TurbulenceDiagnostics(
        sigma_turb_kms=sigma_turb_kms,
        pressure_turb_dyne_cm2=pressure_turb_dyne_cm2,
        mach_turb=mach_turb,
        density_dispersion=density_dispersion,
        regime=regime,
    )
