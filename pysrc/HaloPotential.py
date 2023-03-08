"""
Module for providing a galaxy+NFW+external gravity potential to the cooling_flow module
"""

import numpy as np
import scipy
from numpy import log as ln, log10 as log, e, pi, arange, zeros
from astropy import units as un, constants as cons
from astropy.cosmology import Planck15 as cosmo
import cooling_flow as CF


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
    
    
class PotentialFromMakeDiscWithHalo(CF.Potential):
    def __init__(self, makeDisk_output_filename,R0,polydeg=10,debug=False):
        import sys, h5py, pylab as pl
        sys.path.append('/home/jonathan/research/separate/gadget-snapshot-reader-master/')
        import gsr
        
        snap = gsr.Snapshot(makeDisk_output_filename)
        masses = np.concatenate([snap.SnapshotData['mass'][iPartType] for iPartType in range(6)])*1e10
        coords = np.concatenate([snap.SnapshotData['pos'][iPartType] for iPartType in range(6)])        
        rs = ((coords**2).sum(axis=1))**0.5
        
        masses = masses[rs!=0]
        rs = rs[rs!=0]
        sorted_rs = np.sort(rs)
        Npad = len(masses)
        sorted_rs = np.concatenate([sorted_rs,np.linspace(sorted_rs[-1],R0,Npad+1)[:-1]])
        enclosedMasses = masses[rs.argsort()].cumsum()
        enclosedMasses = np.pad(enclosedMasses,(0,Npad),'edge')
        
        enclosedMasses = enclosedMasses*un.Msun
        sorted_rs = sorted_rs*un.kpc
        # fit circular velocity
        vcs = ((cons.G * enclosedMasses / sorted_rs)**0.5).to('km/s')
        coeffs = np.polyfit(log(sorted_rs.value),log(vcs.value), polydeg) 
        self.poly_vc = np.poly1d(coeffs)
        derivative_coeffs = coeffs[:-1] * np.arange(polydeg,0,-1)
        self.poly_dvc = np.poly1d(derivative_coeffs)

        # fit gravitational potential
        gs = vcs**2/sorted_rs
        drs = sorted_rs[1:] - sorted_rs[:-1]
        Phis = (gs[:-1]*drs).value.cumsum()
        self.Phi0 = np.interp(R0,sorted_rs[:-1].value,Phis).value
        
        #vescs = 
        Phis_coeffs = np.polyfit(log(sorted_rs[:-1].value),log(Phis),polydeg)
        self.poly_Phi = np.poly1d(Phis_coeffs)
        
        if debug:
            pl.figure()
            _rs = 10.**np.arange(-1,log(R0),.01)*un.kpc
            for i in range(2):
                pl.subplot(1,2,i+1)
                if i==0:                               
                    pl.plot(sorted_rs, vcs,'.',c='.8')
                    pl.plot(_rs ,self.vc(_rs),label=r'$v_{\rm c}$',c='b')
                    pl.plot(sorted_rs[:-1], (self.Phi0-Phis)**0.5,'.',c='.8')
                    pl.plot(_rs,(-self.Phi(_rs))**0.5,label=r'$v_{\rm esc}$',c='r',zorder=100)
                    pl.ylim(0,350)
                if i==1:
                    pl.plot(_rs,self.dlnvc_dlnR(_rs),label=r'$\frac{d\ \log\ v_{\rm c}}{d\ \log\ r}$')
                    pl.ylim(-2,2)                                
                pl.semilogx()
                pl.legend()
                pl.xlim(0.01,1000)
    def vc(self,r): 
        """circular velocity"""
        return 10.**self.poly_vc(log(r.to('kpc').value))*un.km/un.s
    def Phi(self,r):
        """gravitational potential"""
        Phi = 10.**self.poly_Phi(log(r.to('kpc').value))
        return (Phi-self.Phi0) * (un.km/un.s)**2
    def dlnvc_dlnR(self,r):
        """logarithmic derivative of circular velocity"""
        return self.poly_dvc(log(r.to('kpc').value))
class RanitaPotential(CF.Potential):
    """
    Rotation velocity = V_C
    d V_C / dr = DEL_V_C 

V_C = sqrt(r*(DEL_PHI_d+DEL_PHI_h)) where

DEL_V_C = (0.5/sqrt(r))*sqrt(DEL_PHI_d+DEL_PHI_h) + ((0.5*sqrt(r))/(sqrt(DEL_PHI_d+DEL_PHI_h)))*(DEL2_PHI_d + DEL2_PHI_h) where

DEL2_PHI_d = (Grav_const*M_DISC/pow((r*r+(A1+B)*(A1+B)),1.5))*(1.-(3.*r*r/(r*r+(A1+B)*(A1+B))))

DEL2_PHI_h = (Grav_const*M_VIR/FC)*(log(1.+sqrt(r*r+D*D)/R_S)/pow(r*r+D*D,1.5)+
	     ((3.*r*r/R_S)/pow(r*r+D*D,2.))/(1.+sqrt(r*r+D*D)/R_S) -
             3.*r*r*log(1.+sqrt(r*r+D*D)/R_S)/pow(r*r+D*D,2.5) -
	     1./(R_S*(r*r+D*D)*(1.+sqrt(r*r+D*D)/R_S))+
	     pow(r/R_s,2.)/(pow(r*r+D*D,1.5)*pow(1.+sqrt(r*r+D*D)/R_S,2.)))
"""
    def __init__(self,mvir=1e12*un.Msun,mdisc=5e10*un.Msun):
        self.mvir = mvir
        self.mdisc = mdisc
        self.R_S = 21.5*un.kpc
        self.D   = 6*un.kpc
        self.C = 12
        self.FC =  ln(1.0+self.C) - self.C/(1.0+self.C) 
        self.rvir = self.C*self.R_S 
        self.A1 = 4*un.kpc
        self.B = 0.4*un.kpc
    def Phi_h(self,r):
        z = (r**2 + self.D**2)**0.5
        return (-1.0 * (cons.G * self.mvir * ln(1.0+ z/self.R_S)) / (self.FC * z)).to('km**2/s**2')
    def Phi_d(self,r):
        return (-1.0 * (cons.G * self.mdisc) / (r**2 +  (self.A1 + self.B)**2)**0.5).to('km**2/s**2')
    def DEL_PHI_d(self,r):
        return (cons.G*self.mdisc*r/(r**2+ (self.A1+self.B)**2.)**1.5).to('km**2*s**-2*kpc**-1')
    def DEL_PHI_h(self,r):
        z = (r**2+self.D**2)**0.5
        z_to_rs = z / self.R_S
        return (cons.G*self.mvir/self.FC*(r*ln(1.+z_to_rs)/z**3 - (r/self.R_S)/(z**2*(1.+z_to_rs)))).to('km**2*s**-2*kpc**-1')
    def vc(self,r):
        return ((r*(self.DEL_PHI_d(r)+self.DEL_PHI_h(r)))**0.5).to('km/s')
    def dvc_dr(self,r):
        return ( 0.5/r**0.5 * (self.DEL_PHI_d(r)+self.DEL_PHI_h(r))**0.5 + 
                 0.5*r**0.5 / (self.DEL_PHI_d(r)+self.DEL_PHI_h(r))**0.5 * (self.DEL2_PHI_d(r) + self.DEL2_PHI_h(r)) ).to('km*s**-1*kpc**-1')
    def DEL2_PHI_d(self,r):
        return (cons.G * self.mdisc / (r**2+(self.A1+self.B)**2)**1.5 *(1.- (3.*r**2/(r**2+(self.A1+self.B)**2)))).to('km**2*kpc**-2*s**-2')
    def DEL2_PHI_h(self,r):
        z = (r**2+self.D**2)**0.5
        z_to_rs = z / self.R_S
        return (cons.G * self.mvir / self.FC*(ln(1.+z_to_rs)/z**3+
                     ((3.*r**2/self.R_S)/z**4)/(1.+z_to_rs) -
                     3.*r**2*ln(1.+z_to_rs)/z**5 -
                     1./(self.R_S*z**2*(1.+z_to_rs))+
                     (r/self.R_S)**2/(z**3*(1.+z_to_rs)**2.))).to('km**2*kpc**-2*s**-2')
        #z = (r**2 + self.D**2)**0.5
        #g = self.g(r)
        #vs = self.vc(r)
        #return 0.5*r/vc**2*(2*g-3*g*r**2/z**2+cons.G*self.mvir*r**3/(z**3*self.R_S**2*(1+z/self.R_S)))

    def Phi(self,r):
        return  (self.Phi_d(r) + self.Phi_h(r)).to('km**2/s**2')
    def dlnvc_dlnR(self,r):
        return (self.dvc_dr(r) * r / self.vc(r)).to('')
    def save(self,fn):
        rs = 10.**np.arange(-0.3,3.3) * un.kpc
        np.savez(fn, 
                 rs = rs.value,
                 dlnvc_dlnR = self.dlnvc_dlnR(rs).value, 
                 Phi = self.Phi(rs).value,
                 DEL2_PHI_h = self.DEL2_PHI_h(rs).value,
                 DEL2_PHI_d = self.DEL2_PHI_d(rs).value,
                 vc = self.vc(rs).value,
                 dvc_dr = self.dvc_dr(rs).value,
                 DEL_PHI_h = self.DEL_PHI_h(rs).value,
                 DEL_PHI_d = self.DEL_PHI_d(rs).value,
                 PHI_h = self.Phi_h(rs).value,
                 PHI_d = self.Phi_d(rs).value,
                 )
        
    