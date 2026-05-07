"""
Module providing implementations of simple gravity potentials to the cooling_flow module
"""

import numpy as np
import scipy
from numpy import log as ln, log10 as log, e, pi, arange, zeros
from astropy import units as un, constants as cons
from astropy.cosmology import Planck15 as cosmo
from . import cooling_flow as CF

class PowerLaw(CF.Potential):
    def __init__(self,m,vc_Rvir,Rvir,R_phi0=None):
        """
        R_phi0 is radius where Phi is 0, for calculation of Bernoulli parameter
        """
        self.m = m
        self.vc_Rvir = vc_Rvir
        self.Rvir = Rvir
        if R_phi0==None:
            self.R_phi0 = 100*self.Rvir 
        else:
            self.R_phi0 = R_phi0 
    def vc(self, r):
        return self.vc_Rvir * (r/self.Rvir)**self.m
    def Phi(self, r):
        if self.m!=0:
            return -self.vc_Rvir**2 / (2*self.m) * ((self.R_phi0/self.Rvir)**(2*self.m) - (r/self.Rvir)**(2*self.m))
        else:
            return -self.vc_Rvir**2 * ln(self.R_phi0/r)
    def dlnvc_dlnR(self, r):
        return self.m

class Polynom(CF.Potential):
    def __init__(self,coeffs,Rvir,R_phi0=None):
        """
        coeffs are polynomial fit for log vc/kms vs. log r/Rvir
        R_phi0 is radius where Phi is 0, for calculation of Bernoulli parameter
        """
        self.coeffs = coeffs
        self.Rvir = Rvir
        if R_phi0==None:
            self.R_phi0 = 100*self.Rvir 
        else:
            self.R_phi0 = R_phi0 
    def vc(self, r):
        return 10.**np.sum(np.array([self.coeffs[i] * log(r/self.Rvir)**i for i in range(len(self.coeffs))]),axis=0) * un.km/un.s
    def Phi(self, r): #currently numerical calculation, maybe switch to analytic solution
        rs = 10.**np.arange(log(np.min(r.to('kpc').value)),log(self.R_phi0.to('kpc').value),0.01)*un.kpc
        drs = np.pad((rs[2:]-rs[:-2]).value/2,1,mode='edge')*un.kpc
        gs = self.vc(rs)**2 / rs
        dPhis = (-gs*drs).to('km**2/s**2')
        Phis = dPhis[::-1].cumsum()[::-1]
        return np.interp(log(r.value),log(rs.value),Phis)
    def dlnvc_dlnR(self, r):
        return np.sum(np.array([i*self.coeffs[i] * log(r/self.Rvir)**(i-1) for i in range(len(self.coeffs))]),axis=0)

class PowerLaw_with_AngularMomentum(PowerLaw):
    def __init__(self,m,vc_Rvir,Rvir,Rcirc,R_phi0=None):
        PowerLaw.__init__(self,m,vc_Rvir,Rvir,R_phi0)
        self.Rcirc = Rcirc
    def vc(self, r):
        vc = PowerLaw.vc(self,r)
        return vc * (1-(self.Rcirc/r)**2)**0.5
    def dlnvc_dlnR(self,r):
        dlnvc_dlnR = PowerLaw.dlnvc_dlnR(self,r)
        return dlnvc_dlnR + ((r/self.Rcirc)**2-1)**-1.
    
class NFW(CF.Potential):	
    mu=0.6
    X=0.75
    def __init__(self,Mvir,z,cvir,_fdr = 100.):
        self._fdr = _fdr
        self.Mvir = Mvir
        self.z = z
        self.cvir = cvir
        self.dr = self.r_scale()/self._fdr
        rs = arange(self.dr.value,self.rvir().value,self.dr.value) * un.kpc
        self.rho_scale = (self.Mvir / (4*pi * rs**2 * self.dr * 
                                               self.rho2rho_scale(rs) ).sum() ).to('g/cm**3') 
    def Delta_c(self): #Bryan & Norman 98
        x = cosmo.Om(self.z) - 1
        return 18*pi**2 + 82*x - 39*x**2
    def rvir(self):
        return ((self.Mvir / (4/3.*pi*self.Delta_c()*cosmo.critical_density(self.z)))**(1/3.)).to('kpc')
    def r_ta(self,use200m=False): 
        if not use200m:
            return 2*self.rvir()
        else:
            return 2*self.r200m()
    def r_scale(self):
        return self.rvir() / self.cvir
    def rho2rho_scale(self,r): 
        return 4. / ( (r/self.r_scale()) * (1+r/self.r_scale())**2 ) 
    def rho(self,r):
        return self.rho_scale * self.rho2rho_scale(r)
    def enclosedMass(self,r):
        return (16*pi*self.rho_scale * self.r_scale()**3 * 
                        (ln(1+r/self.r_scale()) - (self.r_scale()/r + 1.)**-1.)).to('Msun')
    def v_vir(self):
        return ((cons.G*self.Mvir / self.rvir())**0.5).to('km/s')
    def v_ff(self,r,rdrop=None):
        if rdrop==None: rdrop = 2*self.r200m()
        return ((2*(self.Phi(rdrop) - self.Phi(r)))**0.5).to('km/s')
    def vc(self,r):
        Ms = self.enclosedMass(r)
        return ((cons.G*Ms / r)**0.5).to('km/s')
    def dlnvc_dlnR(self, r):
        return (ln(1+r/self.r_scale())*(self.r_scale()/r + 1.)**2 - (self.r_scale()/r + 1.))**-1.
    def mean_enclosed_rho2rhocrit(self,r):
        Ms = self.enclosedMass(r)
        return Ms / (4/3.*pi*r**3) / cosmo.critical_density(self.z)
    def r200(self,delta=200.):
        rs = arange(self.dr.value,2*self.rvir().value,self.dr.value)*un.kpc
        mean_rho2rhocrit = self.mean_enclosed_rho2rhocrit(rs)
        return rs[searchsorted(-mean_rho2rhocrit,-delta)]
    def r200m(self,delta=200.):
        rs = arange(self.dr.value,2*self.rvir().value,self.dr.value)*un.kpc
        mean_rho2rhocrit = self.mean_enclosed_rho2rhocrit(rs)
        return rs[searchsorted(-mean_rho2rhocrit,-delta*cosmo.Om(self.z))]		
    def M200(self,delta=200.):
        return self.enclosedMass(self.r200(delta))
    def M200m(self,delta=200.):
        return self.enclosedMass(self.r200m(delta))
    def Phi(self,r):
        return -(16*pi*cons.G*self.rho_scale*self.r_scale()**3 / r * ln(1+r/self.r_scale())).to('km**2/s**2')
    def g(self,r):
        Ms = self.enclosedMass(r)
        return cons.G*Ms / r**2
    def t_ff(self,r):
        return 2**0.5 * r / self.vc(r)

class IsothermalSphere(CF.Potential):
    def __init__(self,Mvir,Rvir):
        """
        M(<R) / R = const at r<Rvir
        M(<R) = Mvir at r>Rvir
        """
        self.Mvir = Mvir
        self.Rvir = Rvir
        self.vvir = ((cons.G*self.Mvir/self.Rvir)**0.5).to('km/s')
    def vc(self, r):
        return self.vvir * (r<self.Rvir) + ((cons.G*self.Mvir/r)**0.5).to('km/s') * (r>self.Rvir)
    def Phi(self, r):
        return ( -(cons.G*self.Mvir/r).to('km**2/s**2') * (r>self.Rvir) +
                 -(self.vvir**2 * (1 + ln(self.Rvir/r))) * (r<self.Rvir) ) - (100*un.km/un.s)**2
    def dlnvc_dlnR(self, r):
        return -0.5 * (r>self.Rvir)


# ---------------------------------------------------------------------------
# Potentials added for the GOTHAM simulations: NFW + Plummer (or modified
# Plummer) galaxy + DK14 outer halo, plus the combined potential that sums
# them. Inputs are expected to come with the right astropy units.
# ---------------------------------------------------------------------------

class NFWPotential(CF.Potential):
    def __init__(self, M_vir, r_vir, c_vir):
        """
        NFW potential parameterized directly by (M_vir, r_vir, c_vir).
        """
        self.M_vir = M_vir
        self.c_vir = c_vir
        self.r_vir = r_vir
        self.r_s = self.r_vir / c_vir
        self.rho_s = M_vir / (4 * pi * self.r_s**3 * (ln(1 + c_vir) - c_vir / (1 + c_vir)))

    def enclosed_mass(self, r):
        x = r / self.r_s
        mass = 4 * pi * self.rho_s * self.r_s**3 * (ln(1 + x) - x / (1 + x))
        return mass.to('Msun')

    def vc(self, r):
        return np.sqrt(cons.G * self.enclosed_mass(r) / r).to('km/s')

    def Phi(self, r):
        x = r / self.r_s
        phi = -4 * pi * cons.G * self.rho_s * self.r_s**2 * ln(1 + x) / x
        return phi.to('km**2/s**2')

    def dlnvc_dlnR(self, r):
        x = r / self.r_s
        denominator = (1 + x)**2 * (ln(1 + x) - x / (1 + x))
        return 0.5 * (x**2 / denominator - 1)


class PlummerPotential(CF.Potential):
    def __init__(self, M, a):
        self.M = M
        self.a = a

    def enclosed_mass(self, r):
        return (self.M * r**3 / (r**2 + self.a**2)**(3/2)).to('Msun')

    def vc(self, r):
        return np.sqrt(cons.G * self.enclosed_mass(r) / r).to('km/s')

    def Phi(self, r):
        return (-cons.G * self.M / np.sqrt(r**2 + self.a**2)).to('km**2/s**2')

    def dlnvc_dlnR(self, r):
        return 0.5 * (3 * self.a**2 / (r**2 + self.a**2) - 1)


class OuterHaloPotential(CF.Potential):
    """Outer halo term from Diemer & Kravtsov 2014."""
    def __init__(self, rho_mean, R200):
        self.rho_mean = rho_mean
        self.R200 = R200

    def enclosed_mass(self, r):
        term1 = (5 * self.R200) ** 1.5 * (2 / 3) * r ** 1.5
        term2 = (1 / 3) * r ** 3
        mass = 4 * pi * self.rho_mean * (term1 + term2)
        return mass.to('Msun')

    def vc(self, r):
        return np.sqrt(cons.G * self.enclosed_mass(r) / r).to('km/s')

    def Phi(self, r):
        term1 = (4 / 3) * (5 * self.R200) ** 1.5 * r ** 0.5
        term2 = (1 / 6) * r ** 2
        return 4 * pi * cons.G * self.rho_mean * (term1 + term2)

    def dlnvc_dlnR(self, r):
        x = r / (5 * self.R200)
        return 0.5 * (x**-1.5 + 2) / (2 * x**-1.5 + 1)


class MiyamotoNagaiPotential:
    """Cylindrical Miyamoto-Nagai disk potential (does not implement the
    spherical CF.Potential interface — used only for its 2D Phi(R, z))."""
    def __init__(self, M, a, b):
        self.M = M
        self.a = a
        self.b = b

    def Phi(self, R, z):
        phi = -cons.G * self.M / np.sqrt(R**2 + (np.sqrt(z**2 + self.b**2) + self.a)**2)
        return phi.to('km**2/s**2')


class CombinedPotential(CF.Potential):
    """NFW halo + Plummer central galaxy + DK14 outer halo."""
    def __init__(self, M_vir, r_vir, c_vir, M_gal, a_gal, rho_mean, R200):
        self.M_vir = M_vir
        self.r_vir = r_vir
        self.c_vir = c_vir
        self.M_gal = M_gal
        self.a_gal = a_gal
        self.rho_mean = rho_mean
        self.R200 = R200
        self.r_s = self.r_vir / c_vir
        self.rho_s = M_vir / (4 * pi * self.r_s**3 * (ln(1 + c_vir) - c_vir / (1 + c_vir)))

    def enclosed_mass_nfw(self, r):
        x = r / self.r_s
        mass = 4 * pi * self.rho_s * self.r_s**3 * (ln(1 + x) - x / (1 + x))
        return mass.to('Msun')

    def enclosed_mass_plummer(self, r):
        return (self.M_gal * r**3 / (r**2 + self.a_gal**2)**(3/2)).to('Msun')

    def enclosed_mass_outer(self, r):
        term1 = (5 * self.R200) ** 1.5 * (2 / 3) * r ** 1.5
        term2 = (1 / 3) * r ** 3
        mass = 4 * pi * self.rho_mean * (term1 + term2)
        return mass.to('Msun')

    def enclosed_mass(self, r):
        return self.enclosed_mass_nfw(r) + self.enclosed_mass_plummer(r) + self.enclosed_mass_outer(r)

    def Phi(self, r):
        x = r / self.r_s
        phi_NFW = -4 * pi * cons.G * self.rho_s * self.r_s**2 * ln(1 + x) / x
        phi_Plummer = -cons.G * self.M_gal / np.sqrt(r**2 + self.a_gal**2)
        term1 = (4 / 3) * (5 * self.R200) ** 1.5 * r ** 0.5
        term2 = (1 / 6) * r ** 2
        phi_Outer = 4 * pi * cons.G * self.rho_mean * (term1 + term2)
        return (phi_NFW + phi_Plummer + phi_Outer).to('km**2/s**2')

    def vc(self, r):
        return np.sqrt(cons.G * self.enclosed_mass(r) / r).to('km/s')

    def dlnvc_dlnR(self, r):
        M_nfw_r = self.enclosed_mass_nfw(r)
        M_plummer_r = self.enclosed_mass_plummer(r)
        M_outer_r = self.enclosed_mass_outer(r)
        x = r / self.r_s
        dM_nfw_dr = 4 * pi * self.rho_s * self.r_s**2 * (x / (1 + x)**2)
        dM_plummer_dr = self.M_gal * (3 * r**2 * self.a_gal**2 / np.power(r**2 + self.a_gal**2, 2.5))
        dM_outer_dr = 4 * pi * self.rho_mean * ((5 * self.R200)**1.5 * r**0.5 + r**2)
        total_mass = M_nfw_r + M_plummer_r + M_outer_r
        total_derivative = dM_nfw_dr + dM_plummer_dr + dM_outer_dr
        return 0.5 * (total_derivative * r / total_mass - 1)


class CombinedPotential_using_modified_plummer(CF.Potential):
    """NFW halo + modified-Plummer galaxy (Phi = -GM/(sqrt(r^2+a^2)+b)) +
    DK14 outer halo."""
    def __init__(self, M_vir, r_vir, c_vir, M_gal, a_gal, b_gal, rho_mean, R200):
        self.M_vir = M_vir
        self.r_vir = r_vir
        self.c_vir = c_vir
        self.M_gal = M_gal
        self.a_gal = a_gal
        self.b_gal = b_gal
        self.rho_mean = rho_mean
        self.R200 = R200
        self.r_s = self.r_vir / c_vir
        self.rho_s = M_vir / (4 * pi * self.r_s**3 * (ln(1 + c_vir) - c_vir / (1 + c_vir)))

    def enclosed_mass_nfw(self, r):
        x = r / self.r_s
        mass = 4 * pi * self.rho_s * self.r_s**3 * (ln(1 + x) - x / (1 + x))
        return mass.to('Msun')

    def enclosed_mass_plummer(self, r):
        # M_enc derived from Newton's shell theorem: M r^2/G * dPhi/dr.
        s = np.sqrt(r**2 + self.a_gal**2)
        mass = self.M_gal * r**3 / (s * (s + self.b_gal)**2)
        return mass.to('Msun')

    def _dM_plummer_dr(self, r):
        s = np.sqrt(r**2 + self.a_gal**2)
        numerator = self.a_gal**2 * (3 * s + self.b_gal) + 2 * s**2 * self.b_gal
        denominator = s**3 * (s + self.b_gal)**3
        return self.M_gal * r**2 * numerator / denominator

    def enclosed_mass_outer(self, r):
        term1 = (5 * self.R200) ** 1.5 * (2 / 3) * r ** 1.5
        term2 = (1 / 3) * r ** 3
        mass = 4 * pi * self.rho_mean * (term1 + term2)
        return mass.to('Msun')

    def enclosed_mass(self, r):
        return self.enclosed_mass_nfw(r) + self.enclosed_mass_plummer(r) + self.enclosed_mass_outer(r)

    def Phi(self, r):
        x = r / self.r_s
        phi_NFW = -4 * pi * cons.G * self.rho_s * self.r_s**2 * ln(1 + x) / x
        phi_Plummer = -cons.G * self.M_gal / (np.sqrt(r**2 + self.a_gal**2) + self.b_gal)
        term1 = (4 / 3) * (5 * self.R200) ** 1.5 * r ** 0.5
        term2 = (1 / 6) * r ** 2
        phi_Outer = 4 * pi * cons.G * self.rho_mean * (term1 + term2)
        return (phi_NFW + phi_Plummer + phi_Outer).to('km**2/s**2')

    def vc(self, r):
        return np.sqrt(cons.G * self.enclosed_mass(r) / r).to('km/s')

    def dlnvc_dlnR(self, r):
        M_nfw_r = self.enclosed_mass_nfw(r)
        M_plummer_r = self.enclosed_mass_plummer(r)
        M_outer_r = self.enclosed_mass_outer(r)
        x = r / self.r_s
        dM_nfw_dr = 4 * pi * self.rho_s * self.r_s**2 * (x / (1 + x)**2)
        dM_plummer_dr = self._dM_plummer_dr(r)
        dM_outer_dr = 4 * pi * self.rho_mean * ((5 * self.R200)**1.5 * r**0.5 + r**2)
        total_mass = M_nfw_r + M_plummer_r + M_outer_r
        total_derivative = dM_nfw_dr + dM_plummer_dr + dM_outer_dr
        return 0.5 * (total_derivative * r / total_mass - 1)
