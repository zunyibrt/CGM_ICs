"""Canonical float-unit helpers and physical constants for the repo."""

from __future__ import annotations

import numpy as np


# Canonical units:
# radius: kpc
# mass: Msun
# mass flow rate: Msun / yr
# velocity: km / s
# energy / potential / sound speed squared: km^2 / s^2
# density: g / cm^3
# number density: cm^-3
# temperature: K
# time: Gyr
# cooling function: erg cm^3 / s

KPC_TO_CM = 3.0856775814913673e21
KM_TO_CM = 1.0e5
PC_TO_CM = 3.0856775814913673e18
MPC_TO_CM = 3.0856775814913673e24

YR_TO_S = 3.15576e7
GYR_TO_S = 1.0e9 * YR_TO_S

MSUN_TO_G = 1.988409870698051e33

G_CGS = 6.67430e-8
G_KPC_KM2_S2_PER_MSUN = 4.300917270036279e-6

M_P_G = 1.67262192369e-24
M_E_G = 9.1093837015e-28
K_B_ERG_PER_K = 1.380649e-16
K_B_KEV_PER_K = 8.617333262145e-8
SIGMA_T_CM2 = 6.6524587321e-25
C_CM_PER_S = 2.99792458e10

KM2_S2_TO_CM2_S2 = KM_TO_CM**2
MSUN_PER_YR_TO_G_PER_S = MSUN_TO_G / YR_TO_S
KPC_PER_KM_S_TO_GYR = KPC_TO_CM / KM_TO_CM / GYR_TO_S


def as_array(value) -> np.ndarray:
    """Return a float ndarray view of a scalar or sequence."""
    return np.asarray(value, dtype=float)


def as_scalar(value) -> float:
    """Return a scalar float, requiring scalar-shaped input."""
    arr = as_array(value)
    if arr.shape != ():
        raise ValueError(f"expected scalar value, got shape {arr.shape}")
    return float(arr)


def density_to_nH_cm3(density_cgs, hydrogen_mass_fraction: float) -> np.ndarray:
    return hydrogen_mass_fraction * as_array(density_cgs) / M_P_G


def sound_speed_squared_kms2(temperature_K, gamma: float, mu: float) -> np.ndarray:
    cs2_cgs = gamma * K_B_ERG_PER_K * as_array(temperature_K) / (mu * M_P_G)
    return cs2_cgs / KM2_S2_TO_CM2_S2


def temperature_from_sound_speed_squared(cs2_kms2, gamma: float, mu: float) -> np.ndarray:
    cs2_cgs = as_array(cs2_kms2) * KM2_S2_TO_CM2_S2
    return mu * M_P_G * cs2_cgs / (gamma * K_B_ERG_PER_K)


def mass_flow_to_velocity_kms(
    mass_flow_rate_Msun_per_yr,
    radius_kpc,
    density_cgs,
) -> np.ndarray:
    mdot = as_array(mass_flow_rate_Msun_per_yr) * MSUN_PER_YR_TO_G_PER_S
    radius_cm = as_array(radius_kpc) * KPC_TO_CM
    density = as_array(density_cgs)
    velocity_cms = mdot / (4.0 * np.pi * radius_cm**2 * density)
    return velocity_cms / KM_TO_CM


def mass_flow_rate_from_density_velocity_radius(
    density_cgs,
    velocity_kms,
    radius_kpc,
) -> np.ndarray:
    density = as_array(density_cgs)
    velocity_cms = as_array(velocity_kms) * KM_TO_CM
    radius_cm = as_array(radius_kpc) * KPC_TO_CM
    mdot_g_per_s = 4.0 * np.pi * radius_cm**2 * density * velocity_cms
    return mdot_g_per_s / MSUN_PER_YR_TO_G_PER_S


def flow_time_Gyr(radius_kpc, velocity_kms) -> np.ndarray:
    return as_array(radius_kpc) / as_array(velocity_kms) * KPC_PER_KM_S_TO_GYR


def cooling_time_Gyr(
    density_cgs,
    cs2_kms2,
    nH_cm3,
    lambda_cgs,
    gamma: float,
    gamma_minus_one: float,
) -> np.ndarray:
    density = as_array(density_cgs)
    cs2_cgs = as_array(cs2_kms2) * KM2_S2_TO_CM2_S2
    nH = as_array(nH_cm3)
    cooling = as_array(lambda_cgs)
    cooling_time_s = density * cs2_cgs / (nH**2 * cooling) / (gamma * gamma_minus_one)
    return cooling_time_s / GYR_TO_S


def pressure_dyne_cm2(nH_cm3, temperature_K, hydrogen_mass_fraction: float, mu: float) -> np.ndarray:
    return as_array(nH_cm3) * K_B_ERG_PER_K * as_array(temperature_K) / (hydrogen_mass_fraction * mu)


def shell_mass_Msun(radius_kpc, density_cgs, dr_kpc) -> np.ndarray:
    radius_cm = as_array(radius_kpc) * KPC_TO_CM
    dr_cm = as_array(dr_kpc) * KPC_TO_CM
    shell_mass_g = 4.0 * np.pi * radius_cm**2 * as_array(density_cgs) * dr_cm
    return shell_mass_g / MSUN_TO_G


def entropy_keV_cm2(temperature_K, electron_density_cm3, gamma_minus_one: float) -> np.ndarray:
    return K_B_KEV_PER_K * as_array(temperature_K) / as_array(electron_density_cm3) ** gamma_minus_one


def compton_y_integrand_cm_inverse(electron_density_cm3, temperature_K) -> np.ndarray:
    coeff = SIGMA_T_CM2 * K_B_ERG_PER_K / (M_E_G * C_CM_PER_S**2)
    return coeff * as_array(electron_density_cm3) * as_array(temperature_K)


def hydrogen_number_density_from_pressure(
    pressure_dyne_cm2_value,
    temperature_K,
    hydrogen_mass_fraction: float,
    mu: float,
) -> np.ndarray:
    return as_array(pressure_dyne_cm2_value) * hydrogen_mass_fraction * mu / (K_B_ERG_PER_K * as_array(temperature_K))
