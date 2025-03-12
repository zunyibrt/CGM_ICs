"""
Implementation of an NFW + Plummer + Outer Halo Potential
All inputs are exepcted to come with (the right) dimensional units
"""

import numpy as np
from astropy import units as un, constants as cons
import cooling_flow as CF

class NFWPotential(CF.Potential):
    def __init__(self, M_vir, r_vir, c_vir):
        """
        Initialize the NFW potential.

        Parameters:
        M_vir (float): Virial mass of the halo.
        c_vir (float): Concentration parameter.
        """
        self.M_vir = M_vir
        self.c_vir = c_vir

        # Compute the virial radius (in meters)
        self.r_vir = r_vir

        # Scale radius
        self.r_s = self.r_vir / c_vir

        # Characteristic density
        self.rho_s = M_vir / (4 * np.pi * self.r_s**3 * (np.log(1 + c_vir) - c_vir / (1 + c_vir)))

    def enclosed_mass(self, r):
        """
        Calculate the enclosed mass at radius r.

        Parameters:
        r (float or np.ndarray): Radius from the center.

        Returns:
        float or np.ndarray: Enclosed mass (in Solar Masses).
        """
        x = r / self.r_s
        mass = 4 * np.pi * self.rho_s * self.r_s**3 * (np.log(1 + x) - x / (1 + x))
        return mass.to('Msun')

    def vc(self, r):
        """
        Calculate the circular velocity at radius r.

        Parameters:
        r (float or np.ndarray): Radius from the center.

        Returns:
        float or np.ndarray: Circular velocity (in km/s).
        """
        vel = np.sqrt(cons.G * self.enclosed_mass(r) / r)
        return vel.to('km/s')

    def Phi(self, r):
        """
        Calculate the gravitational potential at radius r.

        Parameters:
        r (float or np.ndarray): Radius from the center.

        Returns:
        float or np.ndarray: Gravitational potential (in km^2/s^2).
        """
        x = r / self.r_s
        phi = -4 * np.pi * cons.G * self.rho_s * self.r_s**2 * np.log(1 + x) / x
        return phi.to('km**2/s**2')

    def dlnvc_dlnR(self, r):
        """
        Calculate d(ln(circular velocity))/d(ln(r)) at radius r.

        Parameters:
        r (float or np.ndarray): Radius from the center.

        Returns:
        float or np.ndarray: The logarithmic derivative of circular velocity with respect to radius.
        """
        x = r / self.r_s
        denominator = (1 + x)**2 * (np.log(1 + x) - x / (1 + x))
        return 0.5 * (x**2 / denominator - 1)

class PlummerPotential:
    def __init__(self, M, a):
        """
        Initialize the Plummer potential.

        Parameters:
        M (float): Total mass of the Plummer sphere.
        a (float): Plummer scale radius.
        """
        self.M = M
        self.a = a

    def enclosed_mass(self, r):
        """
        Compute the enclosed mass at radius r.

        Parameters:
        r (float or np.ndarray): Radius.

        Returns:
        float or np.ndarray: Enclosed mass (in Solar Masses).
        """
        mass = self.M * (r**3 / (r**2 + self.a**2)**(3/2))
        return mass.to('Msun')

    def vc(self, r):
        """
        Compute the circular velocity at radius r.

        Parameters:
        r (float or np.ndarray): Radius.

        Returns:
        float or np.ndarray: Circular velocity (in km/s).
        """
        vel = np.sqrt(cons.G * self.enclosed_mass(r) / r)
        return vel.to('km/s')

    def Phi(self, r):
        """
        Compute the gravitational potential at radius r.

        Parameters:
        r (float or np.ndarray): Radius.

        Returns:
        float or np.ndarray: Gravitational potential (in km^2/s^2).
        """
        phi = -cons.G * self.M / np.sqrt(r**2 + self.a**2)
        return phi.to('km**2/s**2')

    def dlnvc_dlnR(self, r):
        """
        Compute d(ln(v_c))/d(ln(r)) at radius r.

        Parameters:
        r (float or np.ndarray): Radius.

        Returns:
        float or np.ndarray: Logarithmic derivative of circular velocity with respect to radius.
        """
        return 0.5 * (3 * self.a**2 / (r**2 + self.a**2) - 1)

class OuterHaloPotential:
    def __init__(self, rho_mean, R200):
        """
        Initialize an outer halo potential from DK14.

        Parameters:
        rho_mean (float): Mean density of the universe.
        R200 (float): Radius corresponding to R200m.
        """
        self.rho_mean = rho_mean
        self.R200 = R200
    
    def enclosed_mass(self, r):
        """
        Compute the enclosed mass at radius r.

        Parameters:
        r (float or np.ndarray): Radius.

        Returns:
        float or np.ndarray: Enclosed mass (in Solar Masses).
        """
        term1 = (5 * self.R200) ** 1.5 * (2 / 3) * r ** 1.5
        term2 = (1 / 3) * r ** 3
        mass =  4 * np.pi * self.rho_mean * (term1 + term2)
        return mass.to('Msun')

    def vc(self, r):
        """
        Compute the circular velocity at radius r.

        Parameters:
        r (float or np.ndarray): Radius.

        Returns:
        float or np.ndarray: Circular velocity (in km/s).
        """
        vel = np.sqrt(cons.G * self.enclosed_mass(r) / r)
        return vel.to('km/s')
    
    def Phi(self, r):
        """
        Compute the gravitational potential at radius r.

        Parameters:
        r (float or np.ndarray): Radius.

        Returns:
        float or np.ndarray: Gravitational potential (in km^2/s^2).
        """
        term1 = (4 / 3) * (5 * self.R200) ** 1.5 * r ** 0.5
        term2 = (1 / 6) * r ** 2
        return 4 * np.pi * cons.G * self.rho_mean * (term1 + term2)
    
    def dlnvc_dlnR(self, r):
        """
        Compute d(ln(v_c))/d(ln(r)) at radius r.

        Parameters:
        r (float or np.ndarray): Radius.

        Returns:
        float or np.ndarray: Logarithmic derivative of circular velocity with respect to radius.
        """
        x = r / (5 * self.R200)
        return 0.5 * (x**-1.5 + 2) / (2 * x**-1.5 + 1)

class CombinedPotential:
    def __init__(self,  M_vir, r_vir, c_vir, M_gal, a_gal, rho_mean, R200):
        """
        Initialize a combined potential with an NFW profile and a Plummer sphere.
        
        Parameters:
        M_nfw (float): Virial mass of the NFW halo.
        c_nfw (float): Concentration parameter of the NFW halo.
        M_plummer (float): Total mass of the Plummer sphere.
        a_plummer (float): Plummer scale radius.
        """
        self.M_vir = M_vir
        self.r_vir = r_vir
        self.c_vir = c_vir
        self.M_gal = M_gal
        self.a_gal = a_gal
        self.rho_mean = rho_mean
        self.R200 = R200
        
        # Scale radius
        self.r_s = self.r_vir / c_vir

        # Characteristic density
        self.rho_s = M_vir / (4 * np.pi * self.r_s**3 * (np.log(1 + c_vir) - c_vir / (1 + c_vir)))

    def enclosed_mass_nfw(self, r):
        """
        Calculate the enclosed mass at radius r corresponding to the NFW profile.

        Parameters:
        r (float or np.ndarray): Radius from the center.

        Returns:
        float or np.ndarray: Enclosed mass (in Solar Masses).
        """
        x = r / self.r_s
        mass = 4 * np.pi * self.rho_s * self.r_s**3 * (np.log(1 + x) - x / (1 + x))
        return mass.to('Msun')
        
    def enclosed_mass_plummer(self, r):
        """
        Compute the enclosed mass at radius r corresponding to the Plummer profile.

        Parameters:
        r (float or np.ndarray): Radius.

        Returns:
        float or np.ndarray: Enclosed mass (in Solar Masses).
        """
        mass = self.M_gal * (r**3 / (r**2 + self.a_gal**2)**(3/2))
        return mass.to('Msun')

    def enclosed_mass_outer(self, r):
        """
        Compute the enclosed mass at radius r corresponding to the Plummer profile.

        Parameters:
        r (float or np.ndarray): Radius.

        Returns:
        float or np.ndarray: Enclosed mass (in Solar Masses).
        """
        term1 = (5 * self.R200) ** 1.5 * (2 / 3) * r ** 1.5
        term2 = (1 / 3) * r ** 3
        mass =  4 * np.pi * self.rho_mean * (term1 + term2)
        return mass.to('Msun')
        
    def enclosed_mass(self, r):
        """
        Compute the total enclosed mass at radius r.

        Parameters:
        r (float or np.ndarray): Radius.

        Returns:
        float or np.ndarray: Enclosed mass (in Solar Masses).
        """
        return self.enclosed_mass_nfw(r) + self.enclosed_mass_plummer(r) + self.enclosed_mass_outer(r)

    def Phi(self, r):
        """
        Calculate the gravitational potential at radius r.

        Parameters:
        r (float or np.ndarray): Radius from the center.

        Returns:
        float or np.ndarray: Gravitational potential (in km^2/s^2).
        """
        x = r / self.r_s
        phi_NFW = -4 * np.pi * cons.G * self.rho_s * self.r_s**2 * np.log(1 + x) / x
        phi_Plummer = -cons.G * self.M_gal / np.sqrt(r**2 + self.a_gal**2)
        term1 = (4 / 3) * (5 * self.R200) ** 1.5 * r ** 0.5
        term2 = (1 / 6) * r ** 2
        phi_Outer = 4 * np.pi * cons.G * self.rho_mean * (term1 + term2)
        phi = phi_NFW + phi_Plummer + phi_Outer
        return phi.to('km**2/s**2')
    
    def vc(self, r):
        """
        Compute the circular velocity at radius r.

        Parameters:
        r (float or np.ndarray): Radius.

        Returns:
        float or np.ndarray: Circular velocity (in km/s).
        """
        vel = np.sqrt(cons.G * self.enclosed_mass(r) / r)
        return vel.to('km/s')
    
    def dlnvc_dlnR(self, r):
        """
        Compute d(ln(v_c))/d(ln(r)) at radius r.

        Parameters:
        r (float or np.ndarray): Radius.

        Returns:
        float or np.ndarray: Logarithmic derivative of circular velocity with respect to radius.
        """
        M_nfw_r = self.enclosed_mass_nfw(r)
        M_plummer_r = self.enclosed_mass_plummer(r)
        M_outer_r = self.enclosed_mass_outer(r)
        
        x = r / self.r_s
        dM_nfw_dr = 4 * np.pi * self.rho_s * self.r_s**2 * (x / (1 + x)**2)
        dM_plummer_dr = self.M_gal * (3 * r**2 * self.a_gal**2 / np.power(r**2 + self.a_gal**2, 2.5))
        dM_outer_dr = 4 * np.pi * self.rho_mean * ((5 * self.R200)**1.5 * r**0.5 + r**2)
        
        total_mass = M_nfw_r + M_plummer_r + M_outer_r
        total_derivative = dM_nfw_dr + dM_plummer_dr + dM_outer_dr
        
        return 0.5 * (total_derivative * r / total_mass - 1)

# Example usage
if __name__ == "__main__":
    M_vir = 1e12 * un.Msun # Virial mass
    c_vir = 10  # Concentration parameter

    nfw = NFWPotential(M_vir, c_vir)

    # Example radius (in meters)
    r = 260 * un.kpc

    print(f"Enclosed Mass at r={r} : {nfw.enclosed_mass(r):.3e}")
    print(f"Circular Velocity at r={r} : {nfw.vc(r):.3e}")
    print(f"Gravitational Potential at r={r} : {nfw.Phi(r):.3e}")
    print(f"d(ln(v_c))/d(ln(r)) at r={r} : {nfw.dlnvc_dlnR(r):.3f}")
    print(f"d(ln(v_c))/d(ln(r)) at r={r} : {nfw.dlnvc_dlnR2(r):.3f}")