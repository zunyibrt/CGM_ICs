"""
Module for providing a galaxy+NFW+external gravity potential to the cooling_flow module
"""

import numpy as np
import scipy
from numpy import log as ln, log10 as log, e, pi, arange, zeros
from astropy import units as un, constants as cons
from astropy.cosmology import Planck15 as cosmo
import sys
sys.path += ['/usr/local/lib/python2.7/dist-packages'] ### in my system this is required to import colossus
import colossus, colossus.cosmology.cosmology
colossus.cosmology.cosmology.setCosmology('planck15')
from colossus.halo import profile_dk14
import cooling_flow as CF

class PowerLaw(CF.Potential):
    def __init__(self,m,vc_Rvir,Rvir,R_phi0):
        self.m = m
        self.vc_Rvir = vc_Rvir
        self.Rvir = Rvir
        self.R_phi0 = R_phi0
    def vc(self, r):
        return self.vc_Rvir * (r/self.Rvir)**self.m
    def Phi(self, r):
        if self.m!=0:
            return -self.vc_Rvir**2 / (2*self.m) * ((self.R_phi0/self.Rvir)**(2*self.m) - (r/self.Rvir)**(2*self.m))
        else:
            return -self.vc_Rvir**2 * log(self.R_phi0/r)
    def dlnvc_dlnR(self, r):
        return self.m

class PowerLaw_with_AngularMomentum(PowerLaw):
    def __init__(self,m,vc_Rvir,Rvir,R_phi0,Rcirc):
        PowerLaw.__init__(self,m,vc_Rvir,Rvir,R_phi0)
        self.Rcirc = Rcirc
    def vc(self, r):
        vc = PowerLaw.vc(self,r)
        return vc * (1-(self.Rcirc/r)**2)**0.5
    def dlnvc_dlnR(self,r):
        dlnvc_dlnR = PowerLaw.dlnvc_dlnR(self,r)
        return dlnvc_dlnR + ((r/self.Rcirc)**2-1)**-1.
    

def Behroozi_params(z, parameter_file='params/smhm_true_med_cen_params.txt'):
    param_file = open(parameter_file, "r")
    param_list = []
    allparams = []
    for line in param_file:
        param_list.append(float((line.split(" "))[1]))
        allparams.append(line.split(" "))
    
    if (len(param_list) != 20):
        print(("Parameter file not correct length.  (Expected 20 lines, got %d)." % len(param_list)))
        quit()
    
    names = "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A M_1_A2 M_1_Z ALPHA ALPHA_A ALPHA_A2 ALPHA_Z BETA BETA_A BETA_Z DELTA GAMMA GAMMA_A GAMMA_Z CHI2".split(" ");
    params = dict(list(zip(names, param_list)))
    
    
    #Print SMHM relation
    a = 1.0/(1.0+z)
    a1 = a - 1.0
    lna = ln(a)
    zparams = {}
    zparams['m_1'] = params['M_1'] + a1*params['M_1_A'] - lna*params['M_1_A2'] + z*params['M_1_Z']
    zparams['sm_0'] = zparams['m_1'] + params['EFF_0'] + a1*params['EFF_0_A'] - lna*params['EFF_0_A2'] + z*params['EFF_0_Z']
    zparams['alpha'] = params['ALPHA'] + a1*params['ALPHA_A'] - lna*params['ALPHA_A2'] + z*params['ALPHA_Z']
    zparams['beta'] = params['BETA'] + a1*params['BETA_A'] + z*params['BETA_Z']
    zparams['delta'] = params['DELTA']
    zparams['gamma'] = 10**(params['GAMMA'] + a1*params['GAMMA_A'] + z*params['GAMMA_Z'])
    
    smhm_max = 14.5-0.35*z
    if (params['CHI2']>200):
        print('#Warning: chi^2 > 200 implies that not all features are well fit.  Comparison with the raw data (in data/smhm/median_raw/) is crucial.')
    ms = 0.05 * np.arange(int(10.5*20),int(smhm_max*20+1),1)
    dms = ms - zparams['m_1'] 
    dm2s = dms/zparams['delta']
    sms = zparams['sm_0'] - log(10**(-zparams['alpha']*dms) + 10**(-zparams['beta']*dms)) + zparams['gamma']*np.e**(-0.5*(dm2s*dm2s))
    return ms,sms

def MgalaxyBehroozi(lMhalo, z, parameter_file='params/smhm_true_med_cen_params.txt'):
    ms,sms = Behroozi_params(z,parameter_file)
    lMstar = scipy.interpolate.interp1d(ms, sms, fill_value='extrapolate')(lMhalo)
    return 10.**lMstar*un.Msun


def c_DuttonMaccio14(lMhalo, z=0):  #table 3 appropriate for Mvir
    c_z0  = lambda lMhalo: 10.**(1.025 - 0.097*(lMhalo-log(0.7**-1*1e12))) 
    c_z05 = lambda lMhalo: 10.**(0.884 - 0.085*(lMhalo-log(0.7**-1*1e12))) 
    c_z1  = lambda lMhalo: 10.**(0.775 - 0.073*(lMhalo-log(0.7**-1*1e12))) 
    c_z2  = lambda lMhalo: 10.**(0.643 - 0.051*(lMhalo-log(0.7**-1*1e12)))
    zs = np.array([0.,0.5,1.,2.])
    cs = np.array([c_func(lMhalo) for c_func in (c_z0,c_z05,c_z1,c_z2)])
    return np.interp(z, zs, cs)

class DK14_with_Galaxy(CF.Potential):    
    mu = 0.62
    def __init__(self,Mgalaxy,half_mass_radius=None,r0_to_Rvir=20,**kwargs):
        self.dk14=colossus.halo.profile_dk14.getDK14ProfileWithOuterTerms(**kwargs)        
        self.Mgalaxy = Mgalaxy
        self.z = kwargs['z']
        if half_mass_radius==None: 
            self.half_mass_radius = 0.015 * self.RDelta('200c') #Kravtsov+13
        else: self.half_mass_radius = half_mass_radius

        self._Rs = 10.**np.arange(-4.,1.5,0.01) * self.rvir().to('kpc').value
        self._Ms = self.dk14.enclosedMass(self._Rs*cosmo.h) / cosmo.h  #only DM mass
        self._Ms += self.enclosedMass_galaxy(self._Rs*un.kpc).to('Msun').value
        drs = (self._Rs[2:]-self._Rs[:-2])/2.
        drs = np.pad(drs,1,mode='edge')
        self._phis = ((-self.g(self._Rs*un.kpc)[::-1].to('km**2*s**-2*kpc**-1').value * drs[::-1]).cumsum())[::-1]        
        self.r0 = r0_to_Rvir * self.rvir()
    def enclosedMass_galaxy(self,r):
        return self.Mgalaxy * r/(r+self.half_mass_radius)
    def enclosedMass(self,r):
        return np.interp(r.to('kpc').value, self._Rs, self._Ms)*un.Msun
    def enclosedMassInner(self,r):
        return self.dk14.enclosedMassInner(r.to('kpc').value*cosmo.h)*un.Msun / cosmo.h  #only DM mass
    def enclosedMassOuter(self,r):
        return self.dk14.enclosedMassOuter(r.to('kpc').value*cosmo.h)*un.Msun / cosmo.h  #only DM mass
    def g(self, r):
        return cons.G*self.enclosedMass(r) / r**2  
    def rvir(self):
        return self.dk14.RDelta(self.z,'vir') * un.kpc/cosmo.h
    def RDelta(self,mdef):
        return self.dk14.RDelta(self.z,mdef) * un.kpc/cosmo.h
    def Tvir(self):
        return (self.mu * cons.m_p * self.vc(self.rvir())**2 / (2*cons.k_B)).to('K')    
    def vc(self,r):
        return ((cons.G*self.enclosedMass(r) / r)**0.5).to('km/s')
    def Mhalo(self):
        return self.dk14.MDelta(self.z,'vir')*un.Msun/cosmo.h
    def Phi(self,rs,r0):
        phis = np.interp(rs.to('kpc').value, self._Rs, self._phis)
        phi0 = np.interp(r0.to('kpc').value, self._Rs, self._phis)        
        return (phis - phi0) * un.km**2/un.s**2
    def rho(self,r):
        return (self.dk14.density(r.to('kpc').value*cosmo.h) * un.Msun * cosmo.h**2 / un.kpc**3).to('g*cm**-3')
    def tff(self,r):
        return (2**0.5 * r/ self.vc(r)).to('Gyr')
    def rho_b(self,r):
        return self.rho(r) * cosmo.Ob0 / cosmo.Om0
    def dlnvc_dlnR(self,r):
        return 0.5 * ((4.*pi*r**3 * self.rho(r) / self.enclosedMass(r)).to('') - 1)

class DK14_NFW(CF.Potential):
    mu=0.6
    X=0.75
    nuDic = {11:0.63,11.5:0.72,11.75:0.78,12.:0.845,12.2:0.845,12.5:1,13.:1.2,14.:1.8454,15.:3.14}
    polydeg=8
    def __init__(self,lMhalo, Mgalaxy,nu=None,z=0.,r0_to_Rvir=20.):
        self.lMhalo = lMhalo
        if nu==None: self.nu = self.nuDic[lMhalo]
        else: self.nu = nu
        self.z = z
        self.Mgalaxy = Mgalaxy
        self.cnfw = c_DuttonMaccio14(lMhalo,z)
        self.dk14 = DK14_with_Galaxy(Mgalaxy=Mgalaxy,z=z,M = 10.**lMhalo*cosmo.h, c = c_DuttonMaccio14(lMhalo,z), mdef = 'vir')
        self.r200m = self.dk14.RDelta('200m')
        self.half_mass_radius = self.dk14.half_mass_radius
        self._Rs = 10.**np.arange(-4.,1.5,0.01) * self.rvir().to('kpc').value
        gs = self.g_fit(self._Rs*un.kpc).to('km**2*s**-2*kpc**-1')
        self.gs_fit = np.polyfit(log(self._Rs), log(gs.value), self.polydeg)
        
        drs = (self._Rs[2:]-self._Rs[:-2])/2.
        drs = np.pad(drs,1,mode='edge')
        self._phis = ((-self.g(self._Rs*un.kpc)[::-1].to('km**2*s**-2*kpc**-1').value * drs[::-1]).cumsum())[::-1]        
        self.r0 = r0_to_Rvir * self.rvir()
    def Mhalo(self):
        return 10**self.lMhalo * un.Msun
    def rvir(self):
        return self.dk14.rvir()
    def rs(self): 
        """scale radius"""
        return self.rvir()/self.cnfw 
    def rho0(self):
        return (self.Mhalo() / (4 * np.pi * self.rs()**3 * ( ln(1.+self.cnfw) - self.cnfw/(1.+self.cnfw) ))).to('g*cm**-3')
    def rt(self):
        return (1.9-0.18*self.nu)*self.r200m 
    def g_fit(self,r):
        """
        1/r**2 d/dr(  r**2 g ) = 4 pi G rho
        g(r) = integral ( 4 pi G rho_NFW r**2 ) / r**2
        g_NFW(r) = integral ( 4 pi G rhos rs**2 x**2/(x*(1+x)**2)) / (rs**2  x**2 )
        g_NFW(r) = 4 pi G rhos (1/(x+1) + log(x+1))|_0**x /x**2
        g_NFW(r) = 4 pi G rhos (log(x+1)-x/(x+1)) /x**2


        rho_DM = rhos / (x * (1+x)**2 ) / (1.0 + (rs/rt)**4  * x**4)**2 
               + rhom * ( (rs/5*rvir)**-1.5 * x**-1.5 + 1. )

        rho_DM = rhos / (x * (1+x)**2 ) / (1.0 + b**4  * x**4)**2 
               + rhom * ( bb**-1.5 * x**-1.5 + 1. )
        """
        rhom = ((3 * cosmo.H0**2 * cosmo.Om0 * (1.+self.z)**3) / (8*np.pi*cons.G)).to('g*cm**-3')
        rhoc = ((3 * cosmo.H0**2) / (8*np.pi*cons.G)).to('g*cm**-3')
        a = 5. * self.cnfw * self.r200m / self.rvir()
        b = self.rs()/self.rt()    
        x = r/self.rs()
       
        g = ((64.*a**1.5*rhom*x**1.5 + 32.*rhom*x**3. + (96.*self.rho0())/((1. + b**4.)**2.*(1. + x)) - 
             (24.*self.rho0()*(-1. + b**4.*(3. + x*(-4. + x*(3. - 2.*x + b**4.*(-1. + 2.*x))))))/((1. + b**4.)**2.*(1. + b**4.*x**4.)) + 
             (12.*b*(-5.*np.sqrt(2.) + b*(18. - 14.*np.sqrt(2.)*b + 12.*np.sqrt(2.)*b**3. - 16.*b**4. + 2.*np.sqrt(2.)*b**5. + np.sqrt(2.)*b**7. - 2.*b**8.))*self.rho0()*
                np.arctan(1. - np.sqrt(2.)*b*x).value)/(1. + b**4.)**3. + 
             (6*self.rho0()*(4.*(1. + b**4.)*(-5. + 3.*b**4.) + 2.*b**2.*(-9 + 8.*b**4. + b**8.)*np.pi - 
                  2.*b*(-5.*np.sqrt(2.) + b*(-18. - 14.*np.sqrt(2.)*b + 12.*np.sqrt(2.)*b**3. + 16.*b**4. + 2.*np.sqrt(2.)*b**5. + np.sqrt(2.)*b**7. + 2.*b**8.))*
                   np.arctan(1. + np.sqrt(2.)*b*x).value + 16.*(1. - 7.*b**4.)*np.log(1. + x) + 4.*(-1. + 7.*b**4.)*np.log(1. + b**4.*x**4.) - 
                  np.sqrt(2.)*b*(-5. + 14.*b**2. + 12.*b**4. - 2.*b**6 + b**8.)*(np.log(1. + b*x*(-np.sqrt(2.) + b*x)) - np.log(1. + b*x*(np.sqrt(2.) + b*x)))))/
              (1. + b**4.)**3.)/(96.*x**2.))
        g *= 4. * np.pi * cons.G * self.rs()

        g += cons.G*self.Mgalaxy/(r*(r+self.half_mass_radius))

        return g    
    def g(self,r):
        return 10.**np.array([self.gs_fit[i] * log(r.to('kpc').value)**(self.polydeg-i) 
                              for i in range(self.polydeg+1)]).sum(axis=0)*un.km**2/un.s**2/un.kpc
    def vc(self,r):
        return np.sqrt(self.g(r)*r)    
    def Phi(self,rs):
        phis = np.interp(rs.to('kpc').value, self._Rs, self._phis)
        phi0 = np.interp(self.r0.to('kpc').value, self._Rs, self._phis)        
        return (phis - phi0) * un.km**2/un.s**2
    def v_ff(self,rs,r0=None):
        if r0==None: r0 = 2.*self.RDelta('200m')
        return ((-2*self.Phi(rs,r0))**0.5).to('km/s')
    def Tvir(self):
        return (self.mu * cons.m_p * self.vc(self.rvir())**2 / (2*cons.k_B)).to('K')    
    def tff(self,r):
        return (2**0.5 * r/ self.vc(r)).to('Gyr')
    def RDelta(self,mdef):
        return self.dk14.RDelta(mdef) 
    def rho(self,r):
        return (self.dk14.dk14.density(r.to('kpc').value*cosmo.h) * un.Msun * cosmo.h**2 / un.kpc**3).to('g*cm**-3')
    def enclosedMass(self,r):
        return (self.g(r)*r**2 / cons.G).to('Msun')
    def dlnvc_dlnR(self,r):
        return 0.5 * ((4*pi*r**3 * self.rho(r) / self.enclosedMass(r)).to('') - 1)





