"""Float-native steady-state cooling-flow solver."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.integrate
import scipy.optimize

import numpy_compat  # noqa: F401
import unit_system as us
from analytic_models import (
    CoolingConfig,
    HaloConfig,
    RotationConfig,
    RotationDiagnostics,
    TurbulenceConfig,
    TurbulenceDiagnostics,
    infer_rotation_diagnostics,
    infer_turbulence_diagnostics,
    maximum_hot_mode_accretion_rate,
)


class GC:
    """Global physical constants for the cooling-flow model."""

    MU = 0.62
    X = 0.75
    GAMMA = 5.0 / 3.0
    GAMMA_M1 = GAMMA - 1.0
    NE2NH = 1.2


class NoValidDensityError(Exception):
    """Raised when no valid density matches the sonic-point cooling constraint."""


class NoTranssonicSolutionError(Exception):
    """Raised when the sonic-point regularity equation has no real roots."""


class StartsUnboundError(Exception):
    """Raised when an integration starts with positive Bernoulli parameter."""


class StopReason(Enum):
    SONIC_POINT = "Sonic point"
    UNBOUND = "Unbound"
    TEMPERATURE_FLOOR = "Hit Temperature floor"
    MAX_RADIUS = "Max R reached"


def printv(message: str, verbose: bool = True, end: str = "\n") -> None:
    if verbose:
        print(message, end=end)


class Cooling(ABC):
    """Float-native cooling interface."""

    @abstractmethod
    def lambda_cgs(self, temperature_K, nH_cm3):
        """Cooling function in erg cm^3 / s."""

    @abstractmethod
    def dln_lambda_dln_T(self, temperature_K, nH_cm3):
        """Logarithmic cooling derivative with respect to temperature."""

    @abstractmethod
    def dln_lambda_dln_rho(self, temperature_K, nH_cm3):
        """Logarithmic cooling derivative with respect to density."""


class Potential(ABC):
    """Float-native potential interface."""

    @abstractmethod
    def vc_kms(self, radius_kpc):
        """Circular velocity in km/s."""

    @abstractmethod
    def phi_kms2(self, radius_kpc):
        """Gravitational potential in km^2/s^2."""

    @abstractmethod
    def dln_vc_dln_r(self, radius_kpc):
        """Logarithmic circular-velocity derivative."""


class CGMSolution:
    """Float-native cooling-flow solution object."""

    def __init__(
        self,
        cooling: Cooling,
        potential: Potential,
        integration_result: Any,
        mass_flow_rate_Msun_per_yr: float,
        stop_reason: StopReason,
        direction: int = 1,
    ) -> None:
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")

        self.cooling = cooling
        self.potential = potential
        self.result = integration_result
        self.mass_flow_rate_Msun_per_yr = float(mass_flow_rate_Msun_per_yr)
        self.direction = int(direction)
        self.inward_solution = None
        self.stop_reason = stop_reason
        if np.any(self.bernoulli_kms2 > 0.0):
            self.stop_reason = StopReason.UNBOUND

    def add_inward_solution(self, inward_solution: "CGMSolution") -> None:
        if self.direction != 1 or inward_solution.direction != -1:
            raise ValueError("inward solution must be appended to an outward solution")
        self.inward_solution = inward_solution.result

    def _combine_series(self, outward: np.ndarray, inward: Optional[np.ndarray] = None) -> np.ndarray:
        if inward is None or self.inward_solution is None:
            return outward
        return np.concatenate([inward, outward])

    @property
    def radius_kpc(self) -> np.ndarray:
        outward = np.exp(self.direction * self.result.t[:: int(self.direction)])
        inward = None
        if self.inward_solution is not None:
            inward = np.exp(-self.inward_solution.t[::-1])
        return self._combine_series(outward, inward)

    @property
    def density_cgs(self) -> np.ndarray:
        outward = np.exp(self.result.y[1, :: int(self.direction)])
        inward = None
        if self.inward_solution is not None:
            inward = np.exp(self.inward_solution.y[1, :][::-1])
        return self._combine_series(outward, inward)

    @property
    def temperature_K(self) -> np.ndarray:
        outward = np.exp(self.result.y[0, :: int(self.direction)])
        inward = None
        if self.inward_solution is not None:
            inward = np.exp(self.inward_solution.y[0, :][::-1])
        return self._combine_series(outward, inward)

    @property
    def velocity_kms(self) -> np.ndarray:
        return us.mass_flow_to_velocity_kms(
            self.mass_flow_rate_Msun_per_yr,
            self.radius_kpc,
            self.density_cgs,
        )

    @property
    def nH_cm3(self) -> np.ndarray:
        return us.density_to_nH_cm3(self.density_cgs, GC.X)

    @property
    def nE_cm3(self) -> np.ndarray:
        return GC.NE2NH * self.nH_cm3

    @property
    def pressure_dyne_cm2(self) -> np.ndarray:
        return us.pressure_dyne_cm2(self.nH_cm3, self.temperature_K, GC.X, GC.MU)

    @property
    def sound_speed_squared_kms2(self) -> np.ndarray:
        return us.sound_speed_squared_kms2(self.temperature_K, GC.GAMMA, GC.MU)

    @property
    def sound_speed_kms(self) -> np.ndarray:
        return np.sqrt(self.sound_speed_squared_kms2)

    @property
    def mach_number(self) -> np.ndarray:
        return self.velocity_kms / self.sound_speed_kms

    @property
    def specific_internal_energy_kms2(self) -> np.ndarray:
        return self.sound_speed_squared_kms2 / (GC.GAMMA * GC.GAMMA_M1)

    @property
    def bernoulli_kms2(self) -> np.ndarray:
        return (
            0.5 * self.velocity_kms**2
            + self.sound_speed_squared_kms2 / GC.GAMMA_M1
            + self.potential.phi_kms2(self.radius_kpc)
        )

    @property
    def entropy_keV_cm2(self) -> np.ndarray:
        return us.entropy_keV_cm2(self.temperature_K, self.nE_cm3, GC.GAMMA_M1)

    @property
    def flow_time_Gyr(self) -> np.ndarray:
        return us.flow_time_Gyr(self.radius_kpc, self.velocity_kms)

    @property
    def cooling_time_Gyr(self) -> np.ndarray:
        return us.cooling_time_Gyr(
            self.density_cgs,
            self.sound_speed_squared_kms2,
            self.nH_cm3,
            self.cooling.lambda_cgs(self.temperature_K, self.nH_cm3),
            GC.GAMMA,
            GC.GAMMA_M1,
        )

    @property
    def free_fall_time_Gyr(self) -> np.ndarray:
        return np.sqrt(2.0) * us.flow_time_Gyr(self.radius_kpc, self.potential.vc_kms(self.radius_kpc))

    @property
    def gas_mass_Msun(self) -> np.ndarray:
        return us.shell_mass_Msun(
            self.radius_kpc,
            self.density_cgs,
            np.gradient(self.radius_kpc),
        ).cumsum()

    @property
    def cooling_luminosity_density_cgs(self) -> np.ndarray:
        return self.nH_cm3**2 * self.cooling.lambda_cgs(self.temperature_K, self.nH_cm3)

    @property
    def compton_y_integrand_cm_inverse(self) -> np.ndarray:
        return us.compton_y_integrand_cm_inverse(self.nE_cm3, self.temperature_K)

    def cooling_radius_kpc(self, time_Gyr: float) -> float:
        return float(
            10.0
            ** np.interp(
                np.log10(time_Gyr),
                np.log10(self.cooling_time_Gyr),
                np.log10(self.radius_kpc),
            )
        )

    @property
    def sonic_radius_kpc(self) -> Optional[float]:
        crossings = np.where(np.diff(np.signbit(self.mach_number - 1.0)))[0]
        if len(crossings) == 0:
            return None
        return float(self.radius_kpc[crossings[0]])

    def rotation_diagnostics(self, config: RotationConfig) -> RotationDiagnostics:
        return infer_rotation_diagnostics(self, config)

    def turbulence_diagnostics(
        self,
        config: TurbulenceConfig = TurbulenceConfig(),
    ) -> TurbulenceDiagnostics:
        return infer_turbulence_diagnostics(self, config)


def shoot_from_R_sonic(
    potential: Potential,
    cooling: Cooling,
    R_sonic_kpc: float,
    R_max_kpc: float,
    R_min_kpc: float,
    tol: float = 1e-6,
    max_step: float = 0.1,
    epsilon: float = 1e-3,
    dlnM_dlnR_init: float = -1.0,
    return_all_results: bool = False,
    terminal_unbound: bool = True,
    verbose: bool = False,
    calc_inward_solution: bool = True,
    min_temperature_K: float = 2e4,
    x_low: float = 1e-5,
    x_high: float = 1.0,
    method: str = "RK45",
) -> Optional[Union[CGMSolution, Dict[float, CGMSolution]]]:
    results: Dict[float, CGMSolution] = {}
    res = None

    while x_high - x_low > tol:
        x = 0.5 * (x_high + x_low)
        printv(f"Integrating with v_c^2/c_s^2 (R_sonic) = {2 * x:f} ...", verbose, end=" ")

        try:
            sonic_conditions = _calculate_sonic_point_conditions(
                x,
                R_sonic_kpc,
                potential,
                cooling,
                dlnM_dlnR_init,
            )
            res = _integrate_from_sonic_point(
                sonic_conditions,
                potential,
                cooling,
                R_sonic_kpc,
                R_max_kpc,
                R_min_kpc,
                epsilon,
                terminal_unbound,
                calc_inward_solution,
                min_temperature_K,
                max_step,
                verbose,
                method,
            )
            results[x] = res
            dlnM_dlnR_init = sonic_conditions["dlnM_dlnR"]

            if res.stop_reason in (StopReason.SONIC_POINT, StopReason.TEMPERATURE_FLOOR):
                x_high = x
            elif res.stop_reason == StopReason.UNBOUND:
                x_low = x
            elif res.stop_reason == StopReason.MAX_RADIUS:
                break
        except NoValidDensityError:
            x_high = x
            printv("Stop reason: No valid density", verbose)
        except NoTranssonicSolutionError:
            x_high = x
            printv("Stop reason: No transsonic solutions", verbose)
        except StartsUnboundError:
            x_low = x
            printv("Stop reason: Starts unbound", verbose)

    if return_all_results:
        return results
    if results and res is not None and res.stop_reason == StopReason.MAX_RADIUS:
        return res
    print("No result reached R_max. Set return_all_results=True to inspect intermediate solutions.")
    return None


def shoot_from_R_circ(
    potential: Potential,
    cooling: Cooling,
    R_circ_kpc: float,
    mass_flow_rate_Msun_per_yr: float,
    R_max_kpc: float,
    v0_kms: float = 1.0,
    epsilon: float = 0.1,
    max_step: float = 0.1,
    tol: float = 1e-6,
    T_low_K: float = 1e4,
    T_high_K: float = 1e5,
    terminal_unbound: bool = True,
    verbose: bool = False,
    return_all_results: bool = False,
    method: str = "RK45",
) -> Optional[Union[CGMSolution, Dict[float, CGMSolution]]]:
    results: Dict[float, CGMSolution] = {}
    res = None

    while np.log10(T_high_K / T_low_K) > tol:
        T0_K = np.sqrt(T_high_K * T_low_K)
        density_cgs = (
            mass_flow_rate_Msun_per_yr
            * us.MSUN_PER_YR_TO_G_PER_S
            / (4.0 * np.pi * (R_circ_kpc * us.KPC_TO_CM) ** 2 * v0_kms * us.KM_TO_CM)
        )
        printv(f"Integrating with log T(R_circ) = {np.log10(T0_K):.2f} ...", verbose, end=" ")

        try:
            res = IntegrateFlowEquations(
                mass_flow_rate_Msun_per_yr,
                T0_K,
                density_cgs,
                potential,
                cooling,
                direction=1,
                R_min_kpc=R_circ_kpc * (1.0 + epsilon),
                R_max_kpc=R_max_kpc,
                R_circ_kpc=R_circ_kpc,
                terminal_unbound=terminal_unbound,
                is_supersonic=False,
                min_temperature_K=T_low_K / 2.0,
                max_step=max_step,
                method=method,
            )
            results[T0_K] = res
            printv(
                f"Stop reason: {res.stop_reason.value} (Maximum r = {int(res.radius_kpc[-1]):d} kpc)",
                verbose,
            )

            if res.stop_reason in (StopReason.SONIC_POINT, StopReason.TEMPERATURE_FLOOR):
                T_low_K = T0_K
            elif res.stop_reason == StopReason.UNBOUND:
                T_high_K = T0_K
            elif res.stop_reason == StopReason.MAX_RADIUS:
                break
        except StartsUnboundError:
            printv("Stop reason: Starts unbound", verbose)
            break

    if return_all_results:
        return results
    if res is not None and res.stop_reason == StopReason.MAX_RADIUS:
        return res
    print("No result reached R_max. Set return_all_results=True to inspect intermediate solutions.")
    return None


def IntegrateFlowEquations(
    mass_flow_rate_Msun_per_yr: float,
    temperature_K: float,
    density_cgs: float,
    potential: Potential,
    cooling: Cooling,
    direction: int,
    R_min_kpc: float,
    R_max_kpc: float,
    R_circ_kpc: float = 0.0,
    max_step: float = 0.1,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    check_unbound: bool = True,
    is_supersonic: bool = False,
    terminal_unbound: bool = True,
    min_temperature_K: float = 2e4,
    method: str = "RK45",
) -> CGMSolution:
    ln_R_range = direction * np.log([R_min_kpc, R_max_kpc][:: int(direction)])
    init_vals = (np.log(temperature_K), np.log(density_cgs))

    if terminal_unbound and check_unbound:
        if _check_if_unbound(ln_R_range[0], init_vals, mass_flow_rate_Msun_per_yr, potential, direction) > 0:
            raise StartsUnboundError("flow starts unbound")

    ode_system = _create_ode_system(
        mass_flow_rate_Msun_per_yr,
        potential,
        cooling,
        R_circ_kpc,
        direction,
    )
    events = _create_event_functions(
        mass_flow_rate_Msun_per_yr,
        potential,
        check_unbound,
        is_supersonic,
        terminal_unbound,
        min_temperature_K,
        direction,
    )
    result = scipy.integrate.solve_ivp(
        ode_system,
        ln_R_range,
        init_vals,
        method=method,
        events=events,
        max_step=max_step,
        atol=atol,
        rtol=rtol,
    )
    stop_reason = _get_stop_reason(result.t_events, check_unbound, is_supersonic)
    return CGMSolution(
        cooling,
        potential,
        result,
        mass_flow_rate_Msun_per_yr,
        stop_reason,
        direction=direction,
    )


def _check_if_unbound(
    ln_R: float,
    y: np.ndarray,
    mass_flow_rate_Msun_per_yr: float,
    potential: Potential,
    direction: int,
) -> float:
    radius_kpc = np.exp(direction * ln_R)
    ln_temperature, ln_density = y
    density_cgs = np.exp(ln_density)
    temperature_K = np.exp(ln_temperature)

    velocity_kms = us.mass_flow_to_velocity_kms(mass_flow_rate_Msun_per_yr, radius_kpc, density_cgs)
    cs2_kms2 = us.sound_speed_squared_kms2(temperature_K, GC.GAMMA, GC.MU)
    bernoulli_kms2 = 0.5 * velocity_kms**2 + cs2_kms2 / GC.GAMMA_M1 + potential.phi_kms2(radius_kpc)
    return float(bernoulli_kms2)


def _create_ode_system(
    mass_flow_rate_Msun_per_yr: float,
    potential: Potential,
    cooling: Cooling,
    R_circ_kpc: float,
    direction: int,
) -> Callable[[float, np.ndarray], np.ndarray]:
    def odes(ln_R, y):
        radius_kpc = np.exp(direction * ln_R)
        ln_temperature, ln_density = y
        density_cgs = np.exp(ln_density)
        temperature_K = np.exp(ln_temperature)

        nH_cm3 = us.density_to_nH_cm3(density_cgs, GC.X)
        velocity_kms = us.mass_flow_to_velocity_kms(mass_flow_rate_Msun_per_yr, radius_kpc, density_cgs)
        cs2_kms2 = us.sound_speed_squared_kms2(temperature_K, GC.GAMMA, GC.MU)
        mach_number = velocity_kms / np.sqrt(cs2_kms2)

        vc2_kms2 = potential.vc_kms(radius_kpc) ** 2 * (1.0 - (R_circ_kpc / radius_kpc) ** 2)
        v_ratio = vc2_kms2 / cs2_kms2

        t_flow_Gyr = us.flow_time_Gyr(radius_kpc, velocity_kms)
        lambda_cgs = cooling.lambda_cgs(temperature_K, nH_cm3)
        t_cool_Gyr = us.cooling_time_Gyr(
            density_cgs,
            cs2_kms2,
            nH_cm3,
            lambda_cgs,
            GC.GAMMA,
            GC.GAMMA_M1,
        )
        t_ratio = t_flow_Gyr / t_cool_Gyr

        dln_density_dln_R = (-t_ratio / GC.GAMMA - v_ratio + 2.0 * mach_number**2) / (1.0 - mach_number**2)
        dln_temperature_dln_R = t_ratio + dln_density_dln_R * GC.GAMMA_M1
        return np.array(
            [direction * dln_temperature_dln_R, direction * dln_density_dln_R],
            dtype=float,
        )

    return odes


def _create_event_functions(
    mass_flow_rate_Msun_per_yr: float,
    potential: Potential,
    check_unbound: bool,
    is_supersonic: bool,
    terminal_unbound: bool,
    min_temperature_K: float,
    direction: int,
) -> Tuple[Callable[..., float], ...]:
    def sonic_point(ln_R, y):
        radius_kpc = np.exp(direction * ln_R)
        ln_temperature, ln_density = y
        density_cgs = np.exp(ln_density)
        temperature_K = np.exp(ln_temperature)
        velocity_kms = us.mass_flow_to_velocity_kms(mass_flow_rate_Msun_per_yr, radius_kpc, density_cgs)
        cs2_kms2 = us.sound_speed_squared_kms2(temperature_K, GC.GAMMA, GC.MU)
        mach_number = velocity_kms / np.sqrt(cs2_kms2)
        return float(mach_number - 1.0)

    def low_temperature(ln_R, y):
        return float(np.exp(y[0]) - min_temperature_K)

    def unbound(ln_R, y):
        return _check_if_unbound(ln_R, y, mass_flow_rate_Msun_per_yr, potential, direction)

    sonic_point.terminal = True
    low_temperature.terminal = True
    unbound.terminal = terminal_unbound

    event_functions = [sonic_point]
    if not is_supersonic:
        event_functions.append(low_temperature)
    if check_unbound:
        event_functions.append(unbound)
    return tuple(event_functions)


def _get_stop_reason(t_events, check_unbound: bool, is_supersonic: bool) -> StopReason:
    stop_reasons = [StopReason.SONIC_POINT]
    if not is_supersonic:
        stop_reasons.append(StopReason.TEMPERATURE_FLOOR)
    if check_unbound:
        stop_reasons.append(StopReason.UNBOUND)
    for events, reason in zip(t_events, stop_reasons):
        if len(events) > 0:
            return reason
    return StopReason.MAX_RADIUS


def _integrate_from_sonic_point(
    sonic_conditions,
    potential: Potential,
    cooling: Cooling,
    R_sonic_kpc: float,
    R_max_kpc: float,
    R_min_kpc: float,
    epsilon: float,
    terminal_unbound: bool,
    calc_inward_solution: bool,
    min_temperature_K: float,
    max_step: float,
    verbose: bool,
    method: str,
) -> CGMSolution:
    return _integrate_flow_sonic(
        sonic_conditions["mass_flow_rate_Msun_per_yr"],
        sonic_conditions["temperature_K"],
        sonic_conditions["density_cgs"],
        R_sonic_kpc,
        potential,
        cooling,
        sonic_conditions["dlnT_dlnR"],
        sonic_conditions["dlnrho_dlnR"],
        sonic_conditions["dlnM_dlnR"],
        epsilon,
        R_max_kpc,
        R_min_kpc,
        terminal_unbound,
        calc_inward_solution,
        min_temperature_K,
        max_step,
        verbose,
        method,
    )


def _integrate_flow_sonic(
    mass_flow_rate_Msun_per_yr: float,
    temperature_sonic_K: float,
    density_sonic_cgs: float,
    R_sonic_kpc: float,
    potential: Potential,
    cooling: Cooling,
    dlnT_dlnR: float,
    dlnrho_dlnR: float,
    dlnM_dlnR: float,
    epsilon: float,
    R_max_kpc: float,
    R_min_kpc: float,
    terminal_unbound: bool,
    calc_inward_solution: bool,
    min_temperature_K: float,
    max_step: float,
    verbose: bool,
    method: str,
) -> CGMSolution:
    res = None
    integration_directions = [1, -1] if calc_inward_solution else [1]

    for direction in integration_directions:
        is_supersonic = (direction == -1 and dlnM_dlnR < 0.0) or (direction == 1 and dlnM_dlnR >= 0.0)
        temperature0_K = temperature_sonic_K * (1.0 + direction * epsilon * dlnT_dlnR)
        density0_cgs = density_sonic_cgs * (1.0 + direction * epsilon * dlnrho_dlnR)
        radius0_kpc = R_sonic_kpc * (1.0 + direction * epsilon)

        if direction == 1:
            res = IntegrateFlowEquations(
                mass_flow_rate_Msun_per_yr,
                temperature0_K,
                density0_cgs,
                potential,
                cooling,
                direction=1,
                R_min_kpc=radius0_kpc,
                R_max_kpc=R_max_kpc,
                terminal_unbound=terminal_unbound,
                check_unbound=True,
                is_supersonic=is_supersonic,
                min_temperature_K=min_temperature_K,
                max_step=max_step,
                method=method,
            )
            printv(
                f"Stop reason: {res.stop_reason.value} (Maximum r = {int(res.radius_kpc[-1]):d} kpc)",
                verbose,
            )
            if res.stop_reason in (StopReason.SONIC_POINT, StopReason.TEMPERATURE_FLOOR, StopReason.UNBOUND):
                return res
        else:
            res_inward = IntegrateFlowEquations(
                mass_flow_rate_Msun_per_yr,
                temperature0_K,
                density0_cgs,
                potential,
                cooling,
                direction=-1,
                R_min_kpc=R_min_kpc,
                R_max_kpc=radius0_kpc,
                terminal_unbound=terminal_unbound,
                check_unbound=False,
                is_supersonic=is_supersonic,
                min_temperature_K=min_temperature_K,
                max_step=max_step,
                method=method,
            )
            res.add_inward_solution(res_inward)
            printv(
                f"Inward integration of supersonic part reached r = {res_inward.radius_kpc.min():.3f} kpc",
                verbose,
            )

    if res is None:
        raise RuntimeError("sonic integration produced no solution")
    return res


def _calculate_sonic_point_conditions(
    x: float,
    R_sonic_kpc: float,
    potential: Potential,
    cooling: Cooling,
    dlnM_dlnR_init: float,
) -> Dict[str, float]:
    cs2_kms2, velocity_kms, temperature_K, tflow_to_tcool, density_cgs = _get_ics(
        x,
        R_sonic_kpc,
        potential,
        cooling,
    )
    dlnT_dlnR_1, dlnT_dlnR_2 = _calc_dlnT_dlnR_at_sonic_point(
        R_sonic_kpc,
        x,
        density_cgs,
        temperature_K,
        cooling,
        potential,
    )
    dlnT_dlnR, dlnM_dlnR = _choose_root(dlnT_dlnR_1, dlnT_dlnR_2, dlnM_dlnR_init, x)
    dlnv_dlnR = (1.0 - x) * 2.0 * GC.GAMMA / GC.GAMMA_M1 - 2.0 - dlnT_dlnR / GC.GAMMA_M1
    dlnrho_dlnR = -dlnv_dlnR - 2.0
    return {
        "cs2_kms2": cs2_kms2,
        "velocity_kms": velocity_kms,
        "temperature_K": temperature_K,
        "density_cgs": density_cgs,
        "mass_flow_rate_Msun_per_yr": float(
            us.mass_flow_rate_from_density_velocity_radius(density_cgs, velocity_kms, R_sonic_kpc)
        ),
        "dlnT_dlnR": dlnT_dlnR,
        "dlnrho_dlnR": dlnrho_dlnR,
        "dlnM_dlnR": dlnM_dlnR,
        "tflow_to_tcool": tflow_to_tcool,
    }


def _get_ics(
    x: float,
    R_sonic_kpc: float,
    potential: Potential,
    cooling: Cooling,
) -> Tuple[float, float, float, float, float]:
    cs2_kms2 = float(potential.vc_kms(R_sonic_kpc) ** 2 / (2.0 * x))
    velocity_kms = float(np.sqrt(cs2_kms2))
    temperature_K = float(us.temperature_from_sound_speed_squared(cs2_kms2, GC.GAMMA, GC.MU))
    tflow_to_tcool = 2.0 * GC.GAMMA * (1.0 - x)
    density_cgs = _calc_rho_from_tflow_to_tcool(
        velocity_kms,
        tflow_to_tcool,
        temperature_K,
        R_sonic_kpc,
        cooling,
    )
    return cs2_kms2, velocity_kms, temperature_K, tflow_to_tcool, density_cgs


def _calc_rho_from_tflow_to_tcool(
    velocity_kms: float,
    tflow_to_tcool: float,
    temperature_K: float,
    radius_kpc: float,
    cooling: Cooling,
) -> float:
    cs2_kms2 = float(us.sound_speed_squared_kms2(temperature_K, GC.GAMMA, GC.MU))

    def velocity_residual(nH_cm3):
        density_cgs = nH_cm3 * us.M_P_G / GC.X
        cooling_time_Gyr = us.cooling_time_Gyr(
            density_cgs,
            cs2_kms2,
            nH_cm3,
            cooling.lambda_cgs(temperature_K, nH_cm3),
            GC.GAMMA,
            GC.GAMMA_M1,
        )
        implied_velocity_kms = radius_kpc * us.KPC_PER_KM_S_TO_GYR / (cooling_time_Gyr * tflow_to_tcool)
        return implied_velocity_kms - velocity_kms

    try:
        nH_cm3 = scipy.optimize.brentq(velocity_residual, 1e-7, 1e10)
    except ValueError as exc:
        raise NoValidDensityError from exc
    return float(nH_cm3 * us.M_P_G / GC.X)


def _calc_dlnT_dlnR_at_sonic_point(
    R_sonic_kpc: float,
    x: float,
    density_sonic_cgs: float,
    temperature_sonic_K: float,
    cooling: Cooling,
    potential: Potential,
) -> Tuple[float, float]:
    nH_sonic_cm3 = float(us.density_to_nH_cm3(density_sonic_cgs, GC.X))
    dln_lambda_dln_T = float(cooling.dln_lambda_dln_T(temperature_sonic_K, nH_sonic_cm3))
    dln_lambda_dln_rho = float(cooling.dln_lambda_dln_rho(temperature_sonic_K, nH_sonic_cm3))
    dln_vc_dln_r = float(potential.dln_vc_dln_r(R_sonic_kpc))

    a = (GC.GAMMA + 1.0) / GC.GAMMA_M1**2
    b = (
        2.0
        * (1.0 - x)
        * (dln_lambda_dln_T + (2.0 + dln_lambda_dln_rho) / GC.GAMMA_M1)
        - 2.0
        - 2.0 * ((1.0 - x) * (GC.GAMMA / GC.GAMMA_M1) - 1.0) * ((GC.GAMMA + 3.0) / GC.GAMMA_M1)
    )
    c = (
        8.0 * ((1.0 - x) * (GC.GAMMA / GC.GAMMA_M1) - 1.0) ** 2
        - 4.0 * (1.0 - x) ** 2 * (2.0 + dln_lambda_dln_rho) * (GC.GAMMA / GC.GAMMA_M1)
        + 6.0 * (1.0 - x)
        + 4.0 * x * dln_vc_dln_r
    )
    discriminant = b**2 - 4.0 * a * c
    if discriminant < 0.0:
        raise NoTranssonicSolutionError
    sqrt_discriminant = np.sqrt(discriminant)
    solution1 = (-b + sqrt_discriminant) / (2.0 * a)
    solution2 = (-b - sqrt_discriminant) / (2.0 * a)
    return float(solution1), float(solution2)


def _choose_root(
    dlnT_dlnR_1: float,
    dlnT_dlnR_2: float,
    dlnM_dlnR_ref: float,
    x: float,
) -> Tuple[float, float]:
    dlnM_dlnR_1 = (1.0 - x) * 2.0 * GC.GAMMA / GC.GAMMA_M1 - 2.0 - (0.5 + 1.0 / GC.GAMMA_M1) * dlnT_dlnR_1
    dlnM_dlnR_2 = (1.0 - x) * 2.0 * GC.GAMMA / GC.GAMMA_M1 - 2.0 - (0.5 + 1.0 / GC.GAMMA_M1) * dlnT_dlnR_2
    if abs(dlnM_dlnR_1 - dlnM_dlnR_ref) < abs(dlnM_dlnR_2 - dlnM_dlnR_ref):
        return float(dlnT_dlnR_1), float(dlnM_dlnR_1)
    return float(dlnT_dlnR_2), float(dlnM_dlnR_2)
