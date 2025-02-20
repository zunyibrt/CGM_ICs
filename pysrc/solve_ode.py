"""
Module for deriving steady-state cooling flow solutions.
"""
import numpy as np
import scipy.integrate, scipy.optimize
from astropy import units as un, constants as cons
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from enum import Enum

# Global Constants
class GC:
    """Constants used throughout the module."""
    MU = 0.62      # mean molecular weight
    X = 0.75       # hydrogen mass fraction
    GAMMA = 5/3    # adiabatic index
    GAMMA_M1 = 2/3 # adiabatic index-1
    NE2NH = 1.2    # ratio of electron to hydrogen density

class NoValidDensityError(Exception):
    """Raised when no valid density can be calculated"""
    pass

class NoTranssonicSolutionError(Exception):
    """Raised when no transsonic solution exists"""
    pass

class StartsUnboundError(Exception):
    """Raised when flow starts unbound"""
    pass

class StopReason(Enum):
    """Enumeration of possible reasons for integration termination."""
    SONIC_POINT = "Sonic point"
    UNBOUND = "Unbound"
    TEMPERATURE_FLOOR = "Hit Temperature floor"
    MAX_RADIUS = "Max R reached"
    
def printv(string: str, verbose: bool = True, end: str = '\n') -> None:
    """
    Helper function for printing outputs if a verbose flag is set.
    
    Args:
        string: String to be printed
        verbose: Flag for determining whether to print
        end: String appended after the last character of string, defaults to '\n'

    Returns:
        None
    """
    if verbose:
        print(string, end=end)
    return None
    
########## User Defined Classes. ##########
class Cooling(ABC):
    """Abstract interface for cooling function."""
    @abstractmethod
    def LAMBDA(self, T: un.Quantity, nH: un.Quantity) -> un.Quantity:
        """
        Calculate cooling function value.
        
        Args:
            T: Temperature
            nH: Hydrogen number density
            
        Returns:
            Cooling function value with appropriate units
        """
        pass
        
    @abstractmethod
    def dlnLambda_dlnT(self, T: un.Quantity, nH: un.Quantity) -> float:
        """
        Calculate logarithmic derivative of cooling function with respect to T.
        
        Args:
            T: Temperature
            nH: Hydrogen number density
            
        Returns:
            Logarithmic derivative (dimensionless)
        """
        pass
    
    @abstractmethod
    def dlnLambda_dlnrho(self, T: un.Quantity, nH: un.Quantity) -> float:
        """
        Calculate logarithmic derivative of cooling function with respect to density.
        
        Args:
            T: Temperature
            nH: Hydrogen number density
            
        Returns:
            Logarithmic derivative (dimensionless)
        """
        pass
        
class Potential(ABC): 
    """Interface for gravitational potential."""
    @abstractmethod
    def vc(self, r: un.Quantity) -> un.Quantity: 
        """
        Calculate circular velocity at given radius.
        
        Args:
            r: Radial distance
            
        Returns:
            Circular velocity with velocity units
        """
        pass
        
    @abstractmethod
    def Phi(self, r: un.Quantity) -> un.Quantity: 
        """
        Calculate gravitational potential at given radius.
        
        Args:
            r: Radial distance
            
        Returns:
            Gravitational potential with appropriate units
        """
        pass
        
    @abstractmethod
    def dlnvc_dlnR(self, r: un.Quantity) -> float: 
        """
        Calculate logarithmic derivative of circular velocity with respect to radius.
        
        Args:
            r: Radial distance
            
        Returns:
            Logarithmic derivative (dimensionless)
        """
        pass

########## Solution Class. ##########
class CGMSolution:
    """
    Class representing a cooling gas flow solution in a gravitational potential.
    
    Handles computation and storage of various physical properties of the flow.
    Generated from the integration result.
    """
    def __init__(
        self,
        cooling: Cooling,
        potential: Potential,
        integration_result: Any,
        mass_flow_rate: un.Quantity,
        stop_reason: StopReason,
        direction: int = 1
    ):
        """
        Initialize CGM solution.
        
        Args:
            cooling: Cooling function object
            potential: Gravitational potential object
            integration_result: Integration result object
            mass_flow_rate: Mass flow rate
            direction: Integration direction (1 for outward, -1 for inward)
            
        Raises:
            ValueError: If invalid direction provided
        """
        if direction not in (-1, 1):
            raise ValueError("Direction must be -1 (inward) or 1 (outward)")
            
        self.cooling = cooling
        self.potential = potential
        self.result = integration_result
        self.mass_flow_rate = mass_flow_rate.to('Msun/yr')
        self.direction = direction
        self.inward_solution = None
        self.stopReason = stop_reason
        if np.any(self.Bernoulli > 0): # Check if unbound
            stop_reason = StopReason.UNBOUND
            
    def add_inward_solution(self, inward_solution):
        """Add an inward solution to the existing result."""
        assert(self.direction == 1 and inward_solution.direction == -1)
        self.inward_solution = inward_solution.result
    
    @property
    def Rs(self) -> un.Quantity:
        """Get solution radii."""
        Rs = np.exp(self.direction * self.result.t[::int(self.direction)])
        if self.inward_solution is not None:
            Rs_inward = np.exp(-self.inward_solution.t[::-1])
            Rs = np.concatenate([Rs_inward, Rs])
        return Rs * un.kpc

    @property
    def rhos(self) -> un.Quantity:
        """Get solution densities."""
        rhos = np.exp(self.result.y[1, ::int(self.direction)])
        if self.inward_solution is not None:
            rhos_inward = np.exp(self.inward_solution.y[1, :][::-1])
            rhos = np.concatenate([rhos_inward, rhos])
        return rhos * un.g / un.cm**3

    @property
    def Ts(self) -> un.Quantity:
        """Get solution temperatures."""
        Ts = np.exp(self.result.y[0, ::int(self.direction)])
        if self.inward_solution is not None:
            Ts_inward = np.exp(self.inward_solution.y[0, :][::-1])
            Ts = np.concatenate([Ts_inward, Ts])
        return Ts * un.K

    @property
    def vs(self) -> un.Quantity:
        """Inflow velocity of the solution at all radii."""
        return (self.mass_flow_rate / (4 * np.pi * self.Rs**2 * self.rhos)).to('km/s')
        
    @property
    def nHs(self):
        """Hydrogen densities of the solution at all radii."""
        return (GC.X * self.rhos / cons.m_p).to('cm**-3')

    @property
    def nEs(self):
        """Electron number densities of the solution at all radii."""
        return GC.NE2NH * self.nHs 

    @property
    def pressures(self) -> un.Quantity:
        """Get gas pressures."""
        return (self.nHs * self.Ts * cons.k_B / (GC.X * GC.MU)).to('dyne/cm2')

    @property
    def cs(self) -> un.Quantity:
        """Get adiabatic sound speeds."""
        return ((GC.GAMMA * cons.k_B * self.Ts / (GC.MU * cons.m_p))**0.5).to('km/s')

    @property
    def Ms(self) -> un.Quantity:
        """Get Mach numbers."""
        return (self.vs / self.cs).to('')

    @property
    def internal_energies(self) -> un.Quantity:
        """Get specific internal energies."""
        return (GC.GAMMA * GC.GAMMA_M1)**-1 * self.cs**2

    @property
    def Bernoulli(self) -> un.Quantity:
        """Energy integral of the solution at all radii."""
        return (self.vs**2 / 2. + 
                self.cs**2 / GC.GAMMA_M1 + 
                self.potential.Phi(self.Rs)).to('km**2/s**2')
    @property
    def Ks(self) -> un.Quantity:
        """Entropy of the solution at all radii."""
        return (cons.k_B * self.Ts / self.nEs**GC.GAMMA_M1).to('keV*cm**2')

    @property
    def t_flows(self) -> un.Quantity:
        """Flow times (r/v) of the solution at all radii."""
        return (self.Rs / self.vs).to('Gyr')

    @property
    def t_cools(self) -> un.Quantity:
        """Cooling times of the solution at all radii."""
        return ((GC.GAMMA * GC.GAMMA_M1)**-1. * 
                self.rhos * self.cs**2 / 
                (self.nHs**2 *self.cooling.LAMBDA(self.Ts, self.nHs))).to('Gyr')

    @property
    def t_ffs(self) -> un.Quantity:
        """Free fall time of the solution at all radii."""
        return (2**0.5 * self.Rs / self.potential.vc(self.radii)).to('Gyr')

    @property
    def Mgas(self) -> un.Quantity:
        """Calculate cumulative gas mass."""
        dr = np.gradient(self.radii)
        return (4 * np.pi * self.Rs**2 * self.rhos * dr).cumsum().to('Msun')

    @property
    def L_cools_per_volume(self) -> un.Quantity:
        """Cooling luminosity per volume of the solution at all radii."""
        return self.nHs()**2 *self.cooling.LAMBDA(self.Ts, self.nHs)

    @property
    def y_integrand(self):
        """Integrand for Compton y-parameter calculation."""
        A = cons.sigma_T / (cons.m_e * cons.c**2) * cons.k_B
        return (A * self.nEs * self.Ts).to('cm**-1')

    @property
    def R_cool(self, time: un.Quantity) -> un.Quantity:
        """Cooling radius for a given time (e.g., Hubble time)."""
        return 10.**np.interp(np.log10(time.value), 
                              np.log10(self.t_cools.value),
                              np.log10(self.Rs.value)) * un.kpc

    @property
    def R_sonic(self) -> Optional[un.Quantity]:
        """Find the sonic radius if it exists in the solution."""
        mach_crossings = np.where(np.diff(np.signbit(self.Ms - 1)))[0]
        if len(mach_crossings) > 0:
            return self.Rs[mach_crossings[0]]
        return None
        
########## Top Level Shooting Functions. ##########
def shoot_from_R_sonic(
    potential: Potential, 
    cooling: Cooling,
    R_sonic: un.Quantity, 
    R_max: un.Quantity, 
    R_min: un.Quantity, 
    tol: float = 1e-6, 
    max_step: float = 0.1, 
    epsilon: float = 1e-3, 
    dlnMdlnRInit: float = -1, 
    return_all_results: bool = False, 
    terminalUnbound: bool = True, 
    verbose: bool = False, 
    calcInwardSolution: bool = True, 
    minT: float = 2e4, 
    x_low: float = 1e-5, 
    x_high: float = 1.0
) -> Optional[Union[CGMSolution, Dict[float, CGMSolution]]]:
    """
    Find transonic marginally-bound solution using a shooting method.
    
    Integration proceeds from the sonic point outward, adjusting the temperature
    at the sonic radius between consecutive integrations. After finding
    the outer subsonic region, the solution is integrated inward to derive the
    supersonic part of the solution.

    Args:
        potential: Object defining the gravitational potential.
        cooling: Object defining the cooling function.
        R_sonic: Radius of the sonic point.
        R_max: Maximum radius for the outward integration of subsonic region.
        R_min: Minimum radius for the inward integration of supersonic region.
        tol: Tolerance for stopping integration when consecutive x values differ by less than this.
        max_step: Maximum step size in logarithmic radius during integration.
        epsilon: Small distance from sonic point where integration starts, as R_sonic * (1 +/- epsilon).
        dlnMdlnRInit: Initial guess for d ln Mach / d ln R at sonic point.
        return_all_results: If True, return dictionary of all integration results keyed by x.
        terminalUnbound: If True, integration stops when Bernoulli parameter becomes positive.
        verbose: If True, print integration progress information.
        calcInwardSolution: If True, calculate the inner supersonic part of the solution.
        minT: Minimum temperature threshold below which integration stops.
        x_low: Lower bound for reparameterized temperature at sonic point, where x = v_c^2 / (2 c_s^2).
        x_high: Upper bound for reparameterized temperature at sonic point.

    Returns:
        If return_all_results is False: Single CGMSolution object where integration reached R_max,
                                        or None if no solution is found.
        If return_all_results is True: Dictionary of all attempted solutions keyed by x values.
    """
    results = {} # Dictionary to store intermediate results

    # Binary search for the right temperature at the sonic radius
    while x_high - x_low > tol:
        x = (x_high + x_low) / 2
        printv(f'Integrating with v_c^2/c_s^2 (R_sonic) = {2 * x :f} ...', verbose, end=' ')

        try:
            # Calculate initial conditions at the sonic point
            sonic_conditions = _calculate_sonic_point_conditions(x, R_sonic, potential, cooling, dlnMdlnRInit)

            # Perform integration
            res = _integrate_from_sonic_point(
                sonic_conditions, potential, cooling, R_sonic, R_max, R_min,
                epsilon, terminalUnbound, calcInwardSolution, minT, max_step, verbose
            )

            results[x] = res
            dlnMdlnRInit = sonic_conditions['dlnMdlnR']

            # Adjust search bounds for x based on current solution
            if res.stopReason in (StopReason.SONIC_POINT, StopReason.TEMPERATURE_FLOOR):
                x_high = x
            elif res.stopReason == StopReason.UNBOUND:
                x_low = x
            elif res.stopReason == StopReason.MAX_RADIUS:
                # Solution Found
                break
                
        except NoValidDensityError:
            x_high = x
            printv('Stop reason: No valid density', verbose)
        except NoTranssonicSolutionError:
            x_high = x
            printv('Stop reason: No transsonic solutions', verbose)
        except StartsUnboundError:
            x_low = x
            printv('Stop reason: Starts unbound', verbose)

    if return_all_results:
        return results
    if results and res.stopReason == StopReason.MAX_RADIUS:
        return res
    print('No result reached R_max. Set return_all_results=True to check intermediate solutions.')
    return None

def shoot_from_R_circ(
    potential: Potential, 
    cooling: Cooling,
    R_circ: un.Quantity, 
    Mdot: un.Quantity,
    R_max: un.Quantity, 
    v0: un.Quantity = 1*un.km/un.s,
    epsilon: float = 0.1,
    max_step: float = 0.1, 
    tol: float = 1e-6, 
    T_low: un.Quantity = 1e4*un.K, 
    T_high: un.Quantity = 1e5*un.K,
    terminalUnbound: bool = True,
    verbose: bool = False,
    return_all_results: bool = False
) -> Optional[Union[CGMSolution, Dict[float, CGMSolution]]]:
    """
    Find marginally-bound solution starting at circularization radius.
    
    Marginally-bound condition is satisfied via a shooting method, 
    adjusting temperature at the circularization radius between consecutive integrations.
    
    Args:
        potential: Object defining the gravitational potential.
        cooling: Object defining the cooling function.
        R_circ: Circularization radius where gravitational and centripetal forces balance.
        Mdot: Mass inflow rate of solution.
        R_max: Maximum radius for integration.
        v0: Initial velocity at R_circ(1+epsilon), should be small but positive.
        epsilon: Initial radius of integration is R_circ(1+epsilon).
        max_step: Maximum step size in logarithmic radius during integration.
        tol: Integrations stop when consecutive integrations differ in log T(R_circ) by less than this.
        T_low: Lower bound for temperature at R_circ to search for solution.
        T_high: Upper bound for temperature at R_circ to search for solution.
        terminalUnbound: If True, integration stops when Bernoulli parameter becomes positive.
        verbose: If True, print integration progress information.
        return_all_results: If True, return dictionary of all integration results keyed by temperature.

    Returns:
        If return_all_results is False: Single CGMSolution where integration reached R_max, 
                                        or None if no solution found.
        If return_all_results is True: Dictionary of all attempted solutions keyed by temperature values.
    """
    results = {}
    
    while np.log10(T_high / T_low) > tol:
        # Get initial values
        T0 = (T_high * T_low)**0.5
        rho0 = (Mdot / (4 * np.pi * R_circ**2 * v0)).to('g*cm**-3')
        printv(f'Integrating with log T(R_circ) = {np.log10(T0.to("K").value):.2f} ...', verbose, end=' ')

        # Perform Integration
        try:
            res = IntegrateFlowEquations(
                    Mdot, T0, rho0, potential, cooling, direction=1, 
                    R_min=R_circ*(1 + epsilon), R_max=R_max, R_circ=R_circ,
                    terminal_unbound=terminalUnbound,
                    is_supersonic=False, 
                    min_T=T_low.value / 2, max_step=max_step
                )
        
            results[T0.value] = res

            printv(f'Stop reason: {res.stopReason.value} (Maximum r = {int(res.Rs[-1].to("kpc").value):d} kpc)', verbose)
                       
            # Adjust search bounds for T based on current solution
            if res.stopReason in (StopReason.SONIC_POINT, StopReason.TEMPERATURE_FLOOR):
                T_low = T0
            elif res.stopReason == StopReason.UNBOUND:
                T_high = T0
            elif res.stopReason == StopReason.MAX_RADIUS:
                # Solution Found
                break

        except StartsUnboundError:
            printv('Stop reason: Starts unbound', verbose)
            break
            
    if return_all_results:
        return results    
    if res.stopReason == StopReason.MAX_RADIUS:
        return res
    else:
        print('No result reached R_max. Set return_all_results=True to check intermediate solutions.')
        return None
        
########## Integration Functions. ##########
def IntegrateFlowEquations(
    Mdot: un.Quantity, 
    T0: un.Quantity, 
    rho0: un.Quantity, 
    potential: Potential, 
    cooling: Cooling, 
    direction: int, 
    R_min: un.Quantity, 
    R_max: un.Quantity, 
    R_circ: un.Quantity = 0*un.kpc, 
    max_step: float = 0.1, 
    atol: float = 1e-6, 
    rtol: float = 1e-6,
    check_unbound: bool = True, 
    is_supersonic: bool = False, 
    terminal_unbound: bool = True, 
    min_T: float = 2e4
) -> CGMSolution:
    """
    Integrates steady-state flow equations from initial conditions.
    
    Args:
        Mdot: Mass flow rate.
        T0: Initial temperature at either R_min or R_max.
        rho0: Initial density at either R_min or R_max.
        potential: Object defining the gravitational potential.
        cooling: Object defining the cooling function.
        direction: Direction of integration (-1 for inward, 1 for outward).
        R_min: Minimum radius in the integration range.
        R_max: Maximum radius in the integration range.
        R_circ: Circularization radius (0 if does not exist).
        max_step: Maximum step size in logarithmic radius during integration.
        atol: Absolute tolerance for integration algorithm.
        rtol: Relative tolerance for integration algorithm.
        check_unbound: If True, checks whether flow becomes unbound during integration.
        is_supersonic: Whether the flow is initially supersonic.
        terminal_unbound: If True, stops integration when Bernoulli parameter becomes positive.
        min_T: Minimum temperature threshold below which integration stops.
    
    Returns:
        CGMSolution object containing the integration results.
    
    Raises:
        StartsUnboundError: If initial conditions result in unbound flow.
    """
    # Define integration range
    ln_R_range = direction*np.log([R_min.value, R_max.value][::int(direction)])

    # Initial conditions
    init_vals = (np.log(T0 / un.K), np.log(rho0 / (un.g * un.cm**-3)))

    # Check initial conditions before integration
    if terminal_unbound and check_unbound:
        if _check_if_unbound(ln_R_range[0], init_vals, Mdot, potential, direction) > 0:
            raise StartsUnboundError("Flow starts unbound")
            
    # Create ODE system
    ode_system = _create_ode_system(Mdot, potential, cooling, R_circ, direction)
    
    # Create event functions
    events = _create_event_functions(Mdot, potential, cooling, check_unbound, 
                                     is_supersonic, terminal_unbound, min_T, direction)

    # Solve the ODEs
    result = scipy.integrate.solve_ivp(
        ode_system, ln_R_range, init_vals, events=events,
        max_step=max_step, atol=atol, rtol=rtol
    )
    stop_reason = _get_stop_reason(result.t_events, check_unbound, is_supersonic)
    return CGMSolution(cooling, potential, result, Mdot, stop_reason, direction=direction)

def _check_if_unbound(
    ln_R: float, 
    y: np.ndarray, 
    Mdot: un.Quantity, 
    potential: Potential, 
    direction: int
) -> float:
    """
    Checks if the flow is unbound based on the Bernoulli parameter.
    
    Args:
        ln_R: Natural logarithm of radius.
        y: State vector (ln_T, ln_rho).
        Mdot: Mass flow rate.
        potential: Object defining the gravitational potential.
        direction: Direction of integration (-1 for inward, 1 for outward).
        
    Returns:
        Bernoulli parameter value (dimensionless). Positive values indicate unbound flow.
    """
    R = np.exp(direction * ln_R) * un.kpc
    ln_T, ln_rho = y
    rho = np.exp(ln_rho) * (un.g / un.cm**3)
    T = np.exp(ln_T) * un.K

    v = (Mdot / (4 * np.pi * R**2 * rho)).to('km/s').value
    cs2 = (GC.GAMMA * cons.k_B * T / (GC.MU * cons.m_p)).to('km**2/s**2').value
    B = 0.5 * v**2 + cs2 / GC.GAMMA_M1 + potential.Phi(R).to('km**2/s**2').value

    return B
    
def _create_ode_system(
    Mdot: un.Quantity, 
    potential: Potential, 
    cooling: Cooling, 
    R_circ: un.Quantity, 
    direction: int
) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Creates the ODE system for integration of flow equations.
    
    Args:
        Mdot: Mass flow rate.
        potential: Object defining the gravitational potential.
        cooling: Object defining the cooling function.
        R_circ: Circularization radius.
        direction: Direction of integration (-1 for inward, 1 for outward).
        
    Returns:
        Function that defines the system of ODEs for integration, taking ln_R and state vector y
        and returning derivatives (dlnT/dlnR, dlnrho/dlnR).
    """
    def odes(ln_R, y):
        """Defines the system of ODEs for integration."""
        R = np.exp(direction * ln_R) * un.kpc
        ln_T, ln_rho = y
        rho = np.exp(ln_rho) * un.g / un.cm**3
        T = np.exp(ln_T) * un.K
        
        nH = (GC.X * rho / cons.m_p).to('cm**-3')
        v = (Mdot / (4 * np.pi * R**2 * rho)).to('km/s')
        cs2 = (GC.GAMMA * cons.k_B * T / (GC.MU * cons.m_p)).to('km**2/s**2')
        M = (v / np.sqrt(cs2)).to('')

        vc2 = potential.vc(R)**2 * (1 - (R_circ / R)**2)
        v_ratio = (vc2/cs2).to('')

        t_flow = (R / v).to('Gyr')
        LAMBDA = cooling.LAMBDA(T, nH)
        t_cool = (rho*cs2 / (nH**2 * LAMBDA) / (GC.GAMMA*GC.GAMMA_M1)).to('Gyr')
        t_ratio = (t_flow/t_cool).to('')

        dln_rho_dln_R = (-t_ratio / GC.GAMMA - v_ratio + 2*M**2) / (1 - M**2)
        dln_T_dln_R = t_ratio + dln_rho_dln_R * GC.GAMMA_M1

        return (direction * dln_T_dln_R, direction * dln_rho_dln_R)
    
    return odes

def _create_event_functions(
    Mdot: un.Quantity, 
    potential: Potential, 
    cooling: Cooling, 
    check_unbound: bool, 
    is_supersonic: bool, 
    terminal_unbound: bool, 
    min_T: float, 
    direction: int
) -> Tuple[Callable, ...]:
    """
    Creates the event functions for integration termination.
    
    Args:
        Mdot: Mass flow rate.
        potential: Object defining the gravitational potential.
        cooling: Object defining the cooling function.
        check_unbound: Whether to check if flow becomes unbound.
        is_supersonic: Whether flow is supersonic.
        terminal_unbound: Whether to terminate when flow becomes unbound.
        min_T: Minimum temperature threshold.
        direction: Direction of integration (-1 for inward, 1 for outward).
        
    Returns:
        Tuple of event functions to be used by scipy.integrate.solve_ivp.
    """
    def sonic_point(ln_R, y):
        """Checks if the flow has reached the sonic point."""
        R = np.exp(direction * ln_R) * un.kpc      
        ln_T, ln_rho = y
        rho = np.exp(ln_rho) * (un.g / un.cm**3)
        T = np.exp(ln_T) * un.K
        
        v = Mdot / (4 * np.pi * R**2 * rho)     
        cs2 = GC.GAMMA * cons.k_B * T / (GC.MU * cons.m_p)
        M = (v / np.sqrt(cs2)).to('')
        return M - 1
    
    def low_temperature(ln_R, y):
        """Checks if the temperature has dropped below the threshold."""
        ln_T = y[0]
        return np.exp(ln_T) - min_T
    
    def unbound(ln_R, y):
        """Checks if the flow becomes unbound based on the Bernoulli parameter."""
        return _check_if_unbound(ln_R, y, Mdot, potential, direction)
    
    # Set event terminal properties
    sonic_point.terminal = True
    low_temperature.terminal = True
    unbound.terminal = terminal_unbound
    
    # Compile event functions based on flags
    event_functions = [sonic_point]
    if not is_supersonic:
        event_functions.append(low_temperature)
    if check_unbound:
        event_functions.append(unbound)
    
    return tuple(event_functions)

def _get_stop_reason(
    t_events: List[np.ndarray], 
    check_unbound: bool, 
    is_supersonic: bool
) -> StopReason:
    """
    Determine the stopping reason for integration based on which event was triggered.
    
    Args:
        t_events: List of arrays of event times from scipy.integrate.solve_ivp.
        check_unbound: Whether checking for unbound flow was enabled.
        is_supersonic: Whether flow is supersonic.
        
    Returns:
        StopReason enum indicating why integration was terminated.
    """
    
    stop_reasons = [StopReason.SONIC_POINT]
    if not is_supersonic:
        stop_reasons.append(StopReason.TEMPERATURE_FLOOR)
    if check_unbound:
        stop_reasons.append(StopReason.UNBOUND)
    
    # Iterate over t_events and corresponding stop reasons
    for events, reason in zip(t_events, stop_reasons):
        if len(events) > 0:
            return reason
    
    return StopReason.MAX_RADIUS  # Default stop reason
    
########## Sonic Point Calculations. ##########
def _integrate_from_sonic_point(sonic_conditions, potential, cooling, 
                                R_sonic, R_max, R_min, 
                                epsilon, terminalUnbound, calcInwardSolution, 
                                minT, max_step, verbose):
    """Perform integration from sonic point conditions."""
    return _integrate_flow_sonic(
        sonic_conditions['Mdot'], sonic_conditions['T_sp'], sonic_conditions['rho_sp'],
        R_sonic, potential, cooling, sonic_conditions['dlnTdlnR'], 
        sonic_conditions['dlnrhodlnR'], sonic_conditions['dlnMdlnR'],
        epsilon, R_max, R_min, terminalUnbound, calcInwardSolution, minT, max_step, verbose
    )

def _integrate_flow_sonic(
    Mdot: un.Quantity, 
    T_sonic_point: un.Quantity, 
    rho_sonic_point: un.Quantity, 
    R_sonic: un.Quantity, 
    potential: Potential, 
    cooling: Cooling, 
    dlnTdlnR: float, 
    dlnrhodlnR: float, 
    dlnMdlnR: float, 
    epsilon: float, 
    R_max: un.Quantity, 
    R_min: un.Quantity, 
    terminalUnbound: bool, 
    calcInwardSolution: bool, 
    minT: float, 
    max_step: float, 
    verbose: bool
) -> CGMSolution:
    """
    Integrate the flow equations outward and optionally inward from the sonic point.
    
    Args:
        Mdot: Mass flow rate.
        T_sonic_point: Temperature at the sonic point.
        rho_sonic_point: Density at the sonic point.
        R_sonic: Radius of the sonic point.
        potential: Object defining the gravitational potential.
        cooling: Object defining the cooling function.
        dlnTdlnR: Logarithmic temperature gradient at the sonic point.
        dlnrhodlnR: Logarithmic density gradient at the sonic point.
        dlnMdlnR: Logarithmic Mach number gradient at the sonic point.
        epsilon: Small distance to offset from sonic point for numerical stability.
        R_max: Maximum radius for outward integration.
        R_min: Minimum radius for inward integration.
        terminalUnbound: If True, integration stops when flow becomes unbound.
        calcInwardSolution: If True, calculate the inner supersonic part of the solution.
        minT: Minimum temperature threshold.
        max_step: Maximum step size in logarithmic radius.
        verbose: If True, print integration progress information.
        
    Returns:
        CGMSolution object representing the complete solution.
    """
    res = None

    integration_directions = [1, -1] if calcInwardSolution else [1]
    for direction in integration_directions:
        isSupersonic = (direction == -1 and dlnMdlnR < 0) or (direction == 1 and dlnMdlnR >= 0)

        # initial conditions
        T0 = T_sonic_point * (1.0 + direction * epsilon * dlnTdlnR)
        rho0 = rho_sonic_point * (1.0 + direction * epsilon * dlnrhodlnR)
        R0 = R_sonic * (1.0 + direction * epsilon)

        if direction == 1: # Integrate outwards
            res = IntegrateFlowEquations(
                Mdot, T0, rho0, potential, cooling, direction=1, R_min=R0, R_max=R_max,
                terminal_unbound=terminalUnbound, check_unbound=True, 
                is_supersonic=isSupersonic, min_T=minT, max_step=max_step
            )
            printv(f'Stop reason: {res.stopReason.value} (Maximum r = {int(res.Rs[-1].to("kpc").value):d} kpc)', verbose)
            if res.stopReason in (StopReason.SONIC_POINT, StopReason.TEMPERATURE_FLOOR, StopReason.UNBOUND):
                return res
        else: # Integrate inwards
            res_inward = IntegrateFlowEquations(
                Mdot, T0, rho0, potential, cooling, direction=-1, R_min=R_min, R_max=R0,
                terminal_unbound=terminalUnbound, check_unbound=False, 
                is_supersonic=isSupersonic, min_T=minT, max_step=max_step
            )
            res.add_inward_solution(res_inward)
            printv(f'Inward integration of supersonic part reached r = {res_inward.Rs.to("kpc").value.min():.3f} kpc', verbose)
            
    return res
    
def _calculate_sonic_point_conditions(
    x: float, 
    R_sonic: un.Quantity, 
    potential: Potential, 
    cooling: Cooling, 
    dlnMdlnRInit: float
) -> Dict[str, Any]:
    """
    Calculate all necessary conditions at the sonic point.
    
    Args:
        x: Parameter representing the ratio v_c^2 / (2 c_s^2) at the sonic point.
        R_sonic: Radius of the sonic point.
        potential: Object defining the gravitational potential.
        cooling: Object defining the cooling function.
        dlnMdlnRInit: Initial guess for d ln Mach / d ln R.
        
    Returns:
        Dictionary containing all sonic point conditions including cs2_sp, v_sp, T_sp, rho_sp,
        Mdot, dlnTdlnR, dlnrhodlnR, and dlnMdlnR.
        
    Raises:
        NoValidDensityError: When no valid density can be calculated.
        NoTranssonicSolutionError: When no transsonic solution exists.
    """
    cs2_sp, v_sp, T_sp, t_fc_sp, rho_sp = _get_ics(x, R_sonic, potential, cooling)
       
    dlnTdlnR1, dlnTdlnR2 = _calc_dlnTdlnR_at_sonic_point(R_sonic, x, rho_sp, T_sp, cooling, potential)
    
    dlnTdlnR, dlnMdlnR = _choose_root(dlnTdlnR1, dlnTdlnR2, dlnMdlnRInit, x)
    dlnvdlnR = (1 - x) * 2 * GC.GAMMA / GC.GAMMA_M1 - 2 - dlnTdlnR / GC.GAMMA_M1
    dlnrhodlnR = -dlnvdlnR - 2
    Mdot = 4 * np.pi * R_sonic**2 * rho_sp * v_sp
    
    return {
        'cs2_sp': cs2_sp,
        'v_sp': v_sp,
        'T_sp': T_sp,
        'rho_sp': rho_sp,
        'Mdot': Mdot,
        'dlnTdlnR': dlnTdlnR,
        'dlnrhodlnR': dlnrhodlnR,
        'dlnMdlnR': dlnMdlnR
    }

def _get_ics(
    x: float, 
    R_sonic: un.Quantity, 
    potential: Potential, 
    cooling: Cooling
) -> Tuple[un.Quantity, un.Quantity, un.Quantity, float, un.Quantity]:
    """
    Calculates initial conditions at the sonic point given x parameter.
    
    Args:
        x: Parameter representing the ratio v_c^2 / (2 c_s^2) at the sonic point.
        R_sonic: Radius of the sonic point.
        potential: Object defining the gravitational potential.
        cooling: Object defining the cooling function.

    Returns:
        Tuple containing:
            cs2: Sound speed squared.
            v: Velocity.
            T: Temperature.
            tflow2tcool: Ratio of flow time to cooling time.
            rho: Density.
            
    Raises:
        NoValidDensityError: When no valid density can be calculated.
    """
    cs2 = potential.vc(R_sonic)**2 / (2 * x)
    v = (cs2**0.5).to('km/s')
    T = (GC.MU * cons.m_p * cs2 / (GC.GAMMA * cons.k_B)).to('K')
    tflow2tcool = 2 * GC.GAMMA * (1 - x)
    rho = _calc_rho_from_tflow2tcool(v, tflow2tcool, T, R_sonic, cooling)
    
    return (cs2, v, T, tflow2tcool, rho)

def _calc_rho_from_tflow2tcool(
    v: un.Quantity, 
    tflow2tcool: float, 
    T: un.Quantity, 
    R: un.Quantity, 
    cooling: Cooling
) -> un.Quantity:
    """
    Calculates the density (rho) via root-finding to match flow time to cooling time ratio.
    
    Args:
        v: Velocity.
        tflow2tcool: Ratio of flow time to cooling time.
        T: Temperature.
        R: Radius.
        cooling: Cooling function object.

    Returns:
        rho: Density with appropriate units.
        
    Raises:
        NoValidDensityError: When no valid density can be calculated.
    """
    def v_from_n(n):
        n = n * un.cm**-3
        P = n * cons.k_B * T / (GC.X * GC.MU)
        tcool =  (P / GC.GAMMA_M1 / (n**2 * cooling.LAMBDA(T, n))).to('Gyr')
        vel = (R / (tcool * tflow2tcool)).to('km/s')
        return vel.value - v.to('km/s').value
    try:
        nH = scipy.optimize.brentq(v_from_n, 1e-7, 1e10) * un.cm**-3
    except:
        raise NoValidDensityError
        
    return (nH * cons.m_p/GC.X).to('g/cm3')
    
def _calc_dlnTdlnR_at_sonic_point(
    R_sonic: un.Quantity, 
    x: float, 
    rho_sonic_point: un.Quantity, 
    T_sonic_point: un.Quantity, 
    cooling: Cooling, 
    potential: Potential
) -> Tuple[float, float]:
    """
    Calculates the logarithmic temperature gradient (dlnT/dlnR) at the sonic point.
    
    Args:
        R_sonic: Radius of the sonic point.
        x: Ratio v_c^2 / (2 c_s^2) at the sonic point.
        rho_sonic_point: Density at the sonic point.
        T_sonic_point: Temperature at the sonic point.
        cooling: Cooling function object.
        potential: Potential object.

    Returns:
        Tuple of two possible solutions (roots) for dlnT/dlnR.
        
    Raises:
        NoTranssonicSolutionError: When no transsonic solution exists.
    """
    nH_sonic_point = GC.X * rho_sonic_point / cons.m_p
    
    # Compute gradients of the cooling function and potential
    dlnLambda_dlnT = cooling.f_dlnLambda_dlnT(T_sonic_point, nH_sonic_point)
    dlnLambda_dlnrho = cooling.f_dlnLambda_dlnrho(T_sonic_point, nH_sonic_point)
    dlnvc_dlnR = potential.dlnvc_dlnR(R_sonic)
    
    # Coefficients for the quadratic equation: a * y^2 + b * y + c = 0 where y = dlnT/dlnR
    a = (GC.GAMMA + 1) / GC.GAMMA_M1**2
    b = (2 * (1-x) * (dlnLambda_dlnT + (2 + dlnLambda_dlnrho) / GC.GAMMA_M1) - 2 -
         2 * ((1 - x) * (GC.GAMMA / GC.GAMMA_M1) - 1) * ((GC.GAMMA + 3) / GC.GAMMA_M1))
    c = (8 * ((1 - x) * (GC.GAMMA / GC.GAMMA_M1) - 1)**2 -
         4 * (1 - x)**2 * (2 + dlnLambda_dlnrho) * (GC.GAMMA / GC.GAMMA_M1) +
         6 * (1 - x) + 4 * x * dlnvc_dlnR)
    
    # Solve the quadratic equation
    if (discriminant := b**2 - 4 * a * c) >= 0:
        sqrt_discriminant = discriminant**0.5
        solution1 = (-b + sqrt_discriminant) / (2 * a)
        solution2 = (-b - sqrt_discriminant) / (2 * a)
        return (solution1, solution2)
    else: 
        # No real solutions exist
        raise NoTranssonicSolutionError

def _choose_root(
    dlnTdlnR1: float, 
    dlnTdlnR2: float, 
    dlnMdlnR_ref: float, 
    x: float
) -> Tuple[float, float]:
    """
    Selects the appropriate root for dlnT/dlnR based on which gives dlnM/dlnR closer to reference value.
    
    Args:
        dlnTdlnR1: First candidate for dlnT/dlnR.
        dlnTdlnR2: Second candidate for dlnT/dlnR.
        dlnMdlnR_ref: Reference value for dlnM/dlnR to compare against.
        x: Parameter representing the ratio v_c^2 / (2 c_s^2) at the sonic point.

    Returns:
        Tuple (dlnTdlnR, dlnMdlnR) containing the selected values.
    """
    # Compute dlnMdlnR from dlnTdlnR
    dlnMdlnR1 = (1 - x) * 2 * GC.GAMMA / GC.GAMMA_M1 - 2 - (0.5 + (1 / GC.GAMMA_M1)) * dlnTdlnR1
    dlnMdlnR2 = (1 - x) * 2 * GC.GAMMA / GC.GAMMA_M1 - 2 - (0.5 + (1 / GC.GAMMA_M1)) * dlnTdlnR2

    # Select values closer to reference value
    if np.abs(dlnMdlnR1 - dlnMdlnR_ref) < np.abs(dlnMdlnR2 - dlnMdlnR_ref):
        return (dlnTdlnR1, dlnMdlnR1)
    else:
        return (dlnTdlnR2, dlnMdlnR2)