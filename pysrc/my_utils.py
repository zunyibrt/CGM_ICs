from IPython.core.display import clear_output
import sys
from numpy import *
import scipy
import scipy.stats
import time
import itertools
import math
from scipy.special import wofz
import numpy.random
#from yt import YTArray
from astropy import units as un, constants as cons
from astropy.cosmology import Planck15 as cosmo
import matplotlib, matplotlib.ticker
import numpy as np
import pylab as pl
import pickle
from functools import reduce

un.rydberg = 13.605698065894 * un.eV

ln = log
log = log10
concat = concatenate
Fwhm2Sigma = 2.35482

ndigits = lambda f,maxdigits: ([(f*10**x)%1==0 for x in range(maxdigits)] + [True]).index(True)
intandfloatFunc = lambda x,pos=0: ('%.'+str(ndigits(x,2))+'f')%x
intandfloatFormatter = matplotlib.ticker.FuncFormatter(intandfloatFunc)
singleGroup = noFilter = lambda k: True
dummyFunc = lambda x: x
roman = [None,'I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII','XIII','XIV','XV','XVI','XVII','XVIII','XIV','XX','XXI','XXII','XXIII','XXIV','XXV','XXVI','XXVII','XXVIII','XXIX','XXX']

def lineFunc(x1,y1,x2,y2,pr=False):
    a = (y2-y1) * 1. / (x2-x1)
    b = y1
    if pr: print("y=%.2f x + %.2f"%(a, b-a*x1))
    return lambda x: a*(x-x1)+b

def searchAndInterpolate(arr_xs,val,arr_ys): # beyond edges takes edge values
    ind = searchsorted(arr_xs,val)
    if ind==0: return arr_ys[0]
    if ind==len(arr_xs): return arr_ys[-1]
    if arr_xs[ind]==val: return arr_ys[ind]
    return lineFunc(arr_xs[ind-1], arr_ys[ind-1], arr_xs[ind], arr_ys[ind])(val)

def iround(f,modulo=1):
    if isinstance(f,np.ndarray):
        if modulo >= 1: return np.round(f/modulo).astype(int)*modulo
        else: return np.round(f/modulo).astype(int) / (1/modulo)  #patch for floating point bug
    if modulo >= 1: return int(round(f/modulo))*modulo
    else: return int(round(f/modulo)) / (1/modulo)  #patch for floating point bug

def sizeformat(x,pos=None,units=1.):
    if x*units<=3e20: return r'$%s$ pc'%(nSignificantDigits(x*units/3e18, sigdig=2,retstr=True))
    if 3e20<x*units<=3e23: return r'$%s$ kpc'%(nSignificantDigits(x*units/3e21, sigdig=2,retstr=True))
    if 3e23<x*units<=3e26: return r'$%s$ Mpc'%(nSignificantDigits(x*units/3e24, sigdig=2,retstr=True))
    if 3e26<x*units: return r'$%s$ Gpc'%(nSignificantDigits(x*units/3e27, sigdig=2,retstr=True))
def arilogformat(x,pos=None,dollarsign=True,logbase=1,with1000=True,showHalfs=False):
    if x==0: return r'$0$' #for symlog
    if not (showHalfs and str(x).strip('0.')[0]=='3'):
        if x==0 or iround(log(abs(x)))%logbase!=0: return ''
    if True in [abs(abs(x)-a)/a<1e-6 for a in [1,1.,3.,10,30.,100]+([],[1000])[with1000]]:
        s = '$%d$'%x
    elif True in [abs(abs(x)-a)/a<1e-6 for a in (0.03,0.01,0.3,0.1)]:
        s = '$%s$'%intandfloatFunc(x)
    else:
        s = r'$%s$'%matplotlib.ticker.LogFormatterMathtext()(x)[13:-1]
    if dollarsign: 	return s
    else: return s[1:-1]

def arilogformat2(x,pos=None,dollarsign=True,nDigs=1,maxNormalFont=1000,minNormalFont=0.01):
    if 1<=abs(x)<=maxNormalFont:
        s = '$%s$'%nSignificantDigits(x,max(nDigs+1,1),retstr=True)
    elif minNormalFont<=abs(x)<=1:
        s = '$%s$'%intandfloatFunc(iround(x,1e-4))
    else:
        base = 10**int(log(x))
        sbase = matplotlib.ticker.LogFormatterMathtext()(base)
        if nDigs>=0: scoeff = ('%.'+str(nDigs)+'f')%(x*1./base) +r' \cdot '
        else: scoeff = ''
        s = sbase[0] + scoeff +sbase[1:]
    if dollarsign: 	return s
    else: return s[1:-1]

def myScientificFormat(x,pos=None,base=None,halfs=False,afterDotDigits=1):
    if x==0: return r'$0$'
    if base==None: base = 10**iround(log(x))
    if not halfs and round(x/base)!=x/base: return ''
    digitstr = (r'%.'+str(afterDotDigits)+'f')%(x/base)
    expstr = r'10^{%d}'%(log(float(base)))		
    if x==base: return r'$%s$'%expstr
    return r'$%s\cdot%s$'%(digitstr,expstr)


def myFuncFormatter(formatter,**kwargs):
    return matplotlib.ticker.FuncFormatter(lambda x,pos: formatter(x,pos,**kwargs))

arilogformatter = matplotlib.ticker.FuncFormatter(arilogformat)
arilogformatter2 = matplotlib.ticker.FuncFormatter(arilogformat2)
sizeformatter = matplotlib.ticker.FuncFormatter(sizeformat)
mylogformatter = matplotlib.ticker.FuncFormatter(lambda x,pos: '%d'%log(x))
mypowerformatter = matplotlib.ticker.FuncFormatter(lambda x,pos: arilogformat(10**x,pos))

def to_roman(i):
    return ['I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII','XIII','XIV','XV','XVI','XVII','XVIII','XIV','XX','XXI','XXII','XXIII','XXIV','XXV','XXVI','XXVII','XXVIII','XXIX','XXX','XXXI','XXXII','XXXIII','XXXIV','XXXV','XXXVI','XXXVII','XXXVIII','XXXIX','XL'][i]
fig_width = 6.973848
def c_DuttonMaccio14(lMhalo, z=0):  #table 3 appropriate for Mvir
    c_z0  = lambda lMhalo: 10.**(1.025 - 0.097*(lMhalo-log(0.7**-1*1e12))) 
    c_z05 = lambda lMhalo: 10.**(0.884 - 0.085*(lMhalo-log(0.7**-1*1e12))) 
    c_z1  = lambda lMhalo: 10.**(0.775 - 0.073*(lMhalo-log(0.7**-1*1e12))) 
    c_z2  = lambda lMhalo: 10.**(0.643 - 0.051*(lMhalo-log(0.7**-1*1e12)))
    zs = np.array([0.,0.5,1.,2.])
    cs = np.array([c_func(lMhalo) for c_func in (c_z0,c_z05,c_z1,c_z2)])
    return np.interp(z, zs, cs)
c_nfw_dic = {11:11.5,11.5:11,12:10,12.5:9,13:8,13.5:7,14:6,14.5:5,15:4,15.5:4}

def c_Klypin16(lMhalo,z,fn='/home/jonathan/Dropbox/jonathanmain/CGM/rapidCoolingCGM/data/Klypin16_tableA3_top.dat'): 
    """ top of table A3 in Klypin+16
    appropriate for Mvir and Bolshoi-Planck
    """
    arr = np.genfromtxt(fn)
    arr[:,3] = log(arr[:,3])
    C0, gamma, logM_0 = [scipy.interpolate.interp1d(log(1+arr[:,0]), arr[:,i+1],bounds_error=False,fill_value=(np.nan,arr[-1,i+1]))(log(1+z)) for i in range(3)]    
    M = 10**lMhalo / (1e12*cosmo.h**-1)
    return C0 * M**-gamma * (1+(M/(10**logM_0))**0.4)
class NFW:	
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
        return 4. / ( (r/self.r_scale()) * (1+r/self.r_scale())**2 ) #eq. 1 in Dutton&Maccio14
    def rho(self,r):
        return self.rho_scale * self.rho2rho_scale(r)
    def enclosedMass(self,rs):
        return (16*pi*self.rho_scale * self.r_scale()**3 * 
                        (ln(1+rs/self.r_scale()) - (self.r_scale()/rs + 1.)**-1.)).to('Msun')
        #rs = YTArray(arange(self.dr,r,self.dr),'kpc')
        #dMs = 4*pi*rs**2*self.rho(rs)*self.dr
        #if not cum:
            #return dMs.sum().to('Msun')
        #else:
            #return rs, dMs.cumsum().to('Msun')
    def v_vir(self):
        return ((cons.G*self.Mvir / self.rvir())**0.5).to('km/s')
    def v_ff(self,rs,rdrop=None):
        if rdrop==None: rdrop = 2*self.r200m()
        return ((2*(self.phi(rdrop) - self.phi(rs)))**0.5).to('km/s')
    def v_ff_HubbleExpansion(self,rmax=None):
        if rmax == None: rmax = self.r_ta()
        rs, Ms = self.enclosedMass(rmax,True)
        gs = cons.G * Ms / rs**2
        vs =  YTArray(zeros(len(rs)),'km/s')
        vs[-1]=10.*un.km/un.s
        for i in u.Progress(list(range(len(rs)-2,-1,-1))):
            dv = (vs[i+1]**-1 * gs[i+1]  - cosmo.H(self.z))* self.dr
            vs[i] = vs[i+1] + dv
        return rs, vs.to('km/s')

    def v_circ(self,rs):
        Ms = self.enclosedMass(rs)
        return ((cons.G*Ms / rs)**0.5).to('km/s')
    def T_vir(self):
        try:
            return (self.mu * un.mp * self.v_circ(self.rvir())**2 / (2*un.kb)).to('K')
        except:
            return (self.mu * cons.m_p * self.v_circ(self.rvir())**2 / (2*cons.k_B)).to('K')
    def mean_enclosed_rho2rhocrit(self,rs):
        Ms = self.enclosedMass(rs)
        return Ms / (4/3.*pi*rs**3) / cosmo.critical_density(self.z)
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
    def phi(self,r):
        return -(16*pi*cons.G*self.rho_scale*self.r_scale()**3 / r * ln(1+r/self.r_scale())).to('km**2/s**2')
    def g(self,rs):
        Ms = self.enclosedMass(rs)
        return cons.G*Ms / rs**2
    def t_ff(self,rs):
        return 2**0.5 * rs / self.v_circ(rs)
class NFW_and_Galaxy(NFW):
    def __init__(self, Mvir, z, cvir, Mgalaxy, Re, MBH):
        NFW.__init__(self, Mvir, z, cvir)
        self.Mgalaxy = Mgalaxy
        self.Re = Re
        self.MBH = MBH
    def rho_galaxy(self,r): #eq. 11 in Guo & Mathews 14
        a = self.Re / 1.8153
        return self.Mgalaxy * a / (2*np.pi*r) * (r+a)**-3
    def M_galaxy(self,r):
        a = self.Re / 1.8153
        return self.Mgalaxy * (r/(r+a))**2
    def phi_galaxy(self,r): #eq. 12 in Guo & Mathews 14
        a = self.Re / 1.8153
        return -cons.G * self.Mgalaxy / (r+a)
    def phi_BH(self,r):
        r_g = 2*cons.G * self.MBH / cons.c**2
        return -cons.G * self.MBH / (r-r_g)
    def v_circ(self, rs):
        Ms = self.enclosedMass(rs) #only DM mass
        Ms += self.MBH+self.M_galaxy(rs)
        return ((cons.G*Ms / rs)**0.5).to('km/s')
    def g(self, rs):
        Ms = self.enclosedMass(rs)  #only DM mass
        Ms += self.MBH+self.M_galaxy(rs)
        return cons.G*Ms / rs**2    
    def phi(self, r):
        phi = NFW.phi(self, r) #only DM mass
        phi += self.phi_galaxy(r) + self.phi_BH(r)
        return phi

def stellar_half_mass_Rvir_fraction(z):
    if z in (0,0.1): return 0.018
    if z in (2.,2.25): return 0.022

class DK14_with_Galaxy:    
    mu=0.6
    X=0.75    
    def __init__(self,Mgalaxy,half_mass_radius=None,dk14=None,useSomerville=False,Rcirc=None,**kwargs):
        if dk14!=None: 
            self.dk14=dk14
        else:
            self.dk14=colossus.halo.profile_dk14.getDK14ProfileWithOuterTerms(**kwargs)        
        self.Mgalaxy = Mgalaxy
        self.z = kwargs['z']
        if half_mass_radius==None: 
            if not useSomerville:
                self.half_mass_radius = 0.015 * self.RDelta('200c') #Kravtsov+13
            else:
                self.half_mass_radius = stellar_half_mass_Rvir_fraction(self.z) * self.rvir() #Somerville+18
        else: self.half_mass_radius = half_mass_radius

        self._Rs = 10.**np.arange(-4.,1.5,0.01) * self.rvir().to('kpc').value
        self._Ms = self.dk14.enclosedMass(self._Rs*cosmo.h) / cosmo.h  #only DM mass
        self._Ms += self.enclosedMass_galaxy(self._Rs*un.kpc).to('Msun').value
        drs = (self._Rs[2:]-self._Rs[:-2])/2.
        drs = np.pad(drs,1,mode='edge')
        self._phis = ((-self.g(self._Rs*un.kpc)[::-1].to('km**2*s**-2*kpc**-1').value * drs[::-1]).cumsum())[::-1]        
        self.Rcirc = Rcirc
        
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
        vc = ((cons.G*self.enclosedMass(r) / r)**0.5).to('km/s')
        if self.Rcirc!=None:
            vc *= (1-(self.Rcirc/r)**2)**0.5
        return vc
    def Mhalo(self):
        return self.dk14.MDelta(self.z,'vir')*un.Msun/cosmo.h
    def phi(self,rs,r0):
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
        dlnvc_dlnR =  0.5 * ((4.*pi*r**3 * self.rho(r) / self.enclosedMass(r)).to('') - 1)
        if self.Rcirc!=None: dlnvc_dlnR += ((r/self.Rcirc)**2-1)**-1.
        return dlnvc_dlnR
    
class DK14_NFW:
    mu=0.6
    X=0.75
    nuDic = {11:0.63,11.5:0.72,11.75:0.78,12.:0.845,13.:1.2,14.:1.8454,15.:3.14}
    polydeg=8
    def __init__(self,lMhalo, Mgalaxy,nu=None,z=0.):
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
        gs = self.gDrummond(self._Rs*un.kpc).to('km**2/s**2/kpc')
        self.gs_fit = np.polyfit(log(self._Rs), log(gs.value), self.polydeg)
        
        drs = (self._Rs[2:]-self._Rs[:-2])/2.
        drs = np.pad(drs,1,mode='edge')
        self._phis = ((-self.g(self._Rs*un.kpc)[::-1].to('km**2*s**-2*kpc**-1').value * drs[::-1]).cumsum())[::-1]        
    def Mhalo(self):
        return 10**self.lMhalo * un.Msun
    def rvir(self):
        return self.dk14.rvir()
    def rs(self):
        return self.rvir()/self.cnfw 
    def rho0(self):
        return (self.Mhalo() / (4 * np.pi * self.rs()**3 * ( ln(1.+self.cnfw) - self.cnfw/(1.+self.cnfw) ))).to('g*cm**-3')
    def rt(self):
        return (1.9-0.18*self.nu)*self.r200m 
    def gDrummond(self,r):
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
    def phi(self,rs,r0):
        phis = np.interp(rs.to('kpc').value, self._Rs, self._phis)
        phi0 = np.interp(r0.to('kpc').value, self._Rs, self._phis)        
        return (phis - phi0) * un.km**2/un.s**2
    def v_ff(self,rs,r0=None):
        if r0==None: r0 = 2.*self.RDelta('200m')
        return ((-2*self.phi(rs,r0))**0.5).to('km/s')
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
class GM14_potential(NFW_and_Galaxy):
    def __init__(self, M0, Mvir, rs, Mgalaxy, Re, MBH,cvir):
        self.M0 = M0
        self.rs = rs
        self.Mgalaxy = Mgalaxy
        self.Re = Re
        self.MBH = MBH
        self.cvir = cvir
        self.Mvir = Mvir
    def rho(self,r):
        return self.M0 / (2*np.pi) / (r*(r+self.rs)**2)
    def v_circ(self, r):
        Ms = 2*self.M0 * (ln(1+r/self.rs) - r/(r+self.rs))
        Ms += self.MBH+self.M_galaxy(r)
        return ((cons.G*Ms / r)**0.5).to('km/s')
    def g(self, r):
        Ms = 2*self.M0 * (ln(1+r/self.rs) - r/(r+self.rs))
        Ms += self.MBH+self.M_galaxy(r)
        return cons.G*Ms / r**2    
    def phi(self, r):
        phi = (-2*cons.G*self.M0 / self.rs * ln(1+r/self.rs) / (r/self.rs))#only DM mass
        phi += self.phi_galaxy(r) + self.phi_BH(r)
        return phi.to('km**2/s**2') 
    def rvir(self):
        return self.rs * self.cvir
    
    
        
class NFW_Drummond:
    def __init__(self,Mhalo,c_nfw=10.,z=0,Omega_m=0.27):
        self.Mhalo = Mhalo
        self.c_nfw = c_nfw
        assert(z==0) # R_vir not calculated correctly for z>0
        self.z = z
        self.Omega_m = Omega_m
    def rvir(self): #r200m 
        return 319. * un.kpc * (self.Mhalo / (1e12*un.Msun))**(1/3.)
        #return ((cons.G * self.Mhalo / (100.*self.Omega_m * (1+self.z)**3))**(1./3.)).to('kpc') #319 (M/1e12 Msun)^{1/3} 
    def phi(self,Rs):
        x = Rs/self.rvir()
        return (-cons.G*self.Mhalo/self.rvir() / (ln(1.0+self.c_nfw) - self.c_nfw/(1.0+self.c_nfw)) * ln(1. + self.c_nfw*x)/x).to('km**2/s**2')
    def g(self,Rs):
        x = Rs/self.rvir()
        g = -cons.G*self.Mhalo/self.rvir()**2 / (ln(1.0+self.c_nfw) - self.c_nfw/(1.0+self.c_nfw)) * (
            self.c_nfw / (1. + self.c_nfw*x) / x - ln(1+self.c_nfw*x) / x**2 )
        return g.to('km**2 * s**-2 * kpc**-1')
    def v_ff(self,rs,dr=None,init_vff=0.):
        if dr==None: dr = rs[1]-rs[0]
        vs = (2.*np.concatenate([self.g(rs).value * dr.to('kpc').value,np.array([0.5*init_vff.to('km/s').value**2.])])[::-1].cumsum()[::-1])**0.5
        return vs*un.km/un.s
    def vc2(self,Rs):
        return (self.g(Rs) * Rs).to('km**2/s**2')


class NFW_and_IsothermalSphere: #from Hogan17
    def __init__(self,z, sigma_star, Rs_arc_min_NFW, rho0_NFW, M2500, MBH = 0*un.Msun, mu = 0.59, c_vir = 5):
        self.sigma_star = sigma_star
        self.Rs_NFW = Rs_arc_min_NFW * cosmo.angular_diameter_distance(z) / un.radian.to('arcmin')
        self.coeff_NFW = rho0_NFW / (mu*cons.m_p)
        self._c_vir = c_vir
        self.M2500 = M2500
        self.MBH = MBH
        self.mu = mu
    def GM_NFW(self,r):
        return self.coeff_NFW  * self.Rs_NFW * ( ln(1.+r/self.Rs_NFW)  - (r/self.Rs_NFW) / (1+r/self.Rs_NFW) )
    def GM_ISO(self,r):
        return 2 * self.sigma_star**2 * r
    def rvir(self):
        return (self.Rs_NFW * self._c_vir).to('kpc')
    def Tvir(self):
        return (self.mu * cons.m_p * self.vc(self.rvir())**2 / cons.k_B).to('K')
    def vc(self, r):
        return (((self.GM_NFW(r) + self.GM_ISO(r) + cons.G*self.MBH) / r)**0.5).to('km/s')
    def phi(self, r,maxr=30*un.kpc): #for unbound
        phi_MBH = -cons.G*self.MBH / r
        phi_ISO = minarray(2 * self.sigma_star**2 * ln(r/maxr), 0*un.km**2/un.s**2)
        return -(self.coeff_NFW * ln(1+r/self.Rs_NFW) / (r/self.Rs_NFW)).to('km**2/s**2') + phi_ISO + phi_MBH
    def Mhalo(self):
        return self.M2500 * 5.
    
    
        
        
class Isothermal_Sphere:
    def __init__(self,Mhalo,z):
        self.z = z
        self._Mhalo = Mhalo
    def vc(self,R=None):
        vc = (((Delta_c(self.z)/2.)**0.5*cosmo.H(self.z)*cons.G*self._Mhalo)**(1/3.)).to('km/s')
        if R==None: return vc
        return np.ones(R.shape[0])*vc
    def Mhalo(self):
        return self._Mhalo
    def rvir(self):
        return (self.vc() / ((Delta_c(self.z)/2.)**0.5*cosmo.H(self.z))).to('kpc')
    def Tvir(self):
        return (0.62*cons.m_p*self.vc()**2 / (2*cons.k_B)).to('K')
def NFWFromM200c(M200c,z,cvir):
    enclosedMassFunc = lambda r,rho_s, r_s: (16*pi*rho_s * r_s**3 * 
                                                 (ln(1+r/r_s) - (r_s/r + 1.)**-1.))
    _Mvirs = 10.**arange(11.,13.5,.005)*un.Msun
    _rs = arange(1,1000)*un.kpc
    Mvirs, rs = meshgrid(_Mvirs,_rs)
    r_virs = ((Mvirs / (4/3.*pi*Delta_c(z)*cosmo.critical_density(z)))**(1/3.)).to('kpc') 
    r_scales = r_virs / cvir
    rho2rho_scales = 4. / ( (rs/r_scales) * (1+rs/r_scales)**2 )

    rho_scales = (Mvirs / enclosedMassFunc(r_virs,1.,r_scales)).to('g/cm**3') 	
    enclosedMasses = enclosedMassFunc(rs,rho_scales,r_scales).to('Msun')
    mean_enclosed_rho2rhocrit = enclosedMasses / (4/3.*pi*rs**3) / cosmo.critical_density(z)
    r200cs = zeros(len(_Mvirs)) * un.kpc
    for i in rl(_Mvirs):
        r200cs[i] =  _rs[searchsorted(-mean_enclosed_rho2rhocrit[:,i],-200.)]
    M200cs = (16*pi*rho_scales[0,:] * r_scales[0,:]**3 * 
                  (ln(1+r200cs/r_scales[0,:]) - (r_scales[0,:]/r200cs + 1.)**-1.)).to('Msun')
    Mvir = 10**searchAndInterpolate(log(M200cs/un.Msun), log(M200c/un.Msun),log(_Mvirs/un.Msun))*un.Msun
    return NFW(Mvir,z,cvir)
def Delta_c(z): #Bryan & Norman 98
    x = cosmo.Om(z) - 1
    return 18*pi**2 + 82*x - 39*x**2


def rl(arr,lim=0):
    if lim==0:
        return list(range(len(arr)))
    if lim<0:
        return list(range(len(arr)+lim))
    if lim>0:
        return list(range(lim,len(arr)))


def NFWFromM200m(M200m,z,cvir):
    enclosedMassFunc = lambda r,rho_s, r_s: (16*pi*rho_s * r_s**3 * 
                                                 (ln(1+r/r_s) - (r_s/r + 1.)**-1.))
    _Mvirs = 10.**arange(11.,13.5,.005)*un.Msun
    _rs = arange(1,3000)*un.kpc
    Mvirs, rs = meshgrid(_Mvirs,_rs)
    r_virs = ((Mvirs / (4/3.*pi*Delta_c(z)*cosmo.critical_density(z)))**(1/3.)).to('kpc') 
    r_scales = r_virs / cvir
    rho2rho_scales = 4. / ( (rs/r_scales) * (1+rs/r_scales)**2 )

    rho_scales = (Mvirs / enclosedMassFunc(r_virs,1.,r_scales)).to('g/cm**3') 	
    enclosedMasses = enclosedMassFunc(rs,rho_scales,r_scales).to('Msun')
    mean_enclosed_rho2rhocrit = enclosedMasses / (4/3.*pi*rs**3) / cosmo.critical_density(z)
    r200ms = zeros(len(_Mvirs)) * un.kpc
    for i in rl(_Mvirs):
        r200ms[i] =  _rs[searchsorted(-mean_enclosed_rho2rhocrit[:,i]/cosmo.Om0,-200.)]
    M200ms = (16*pi*rho_scales[0,:] * r_scales[0,:]**3 * 
                  (ln(1+r200ms/r_scales[0,:]) - (r_scales[0,:]/r200ms + 1.)**-1.)).to('Msun')
    Mvir = 10**searchAndInterpolate(log(M200ms/un.Msun), log(M200m/un.Msun),log(_Mvirs/un.Msun))*un.Msun
    return NFW(Mvir,z,cvir)

def mifkad(arr):
    dic={}
    for x in arr:
        if x in dic:
            dic[x] += 1
        else:
            dic[x] = 1
    return dic
def lens(arr):
    return array([len(x) for x in arr])
def labelsize(increase,large=16,small=10):
    for param in 'font.size', 'xtick.labelsize','ytick.labelsize','legend.fontsize':
        pl.rcParams[param]=(small,large)[increase]
    pl.rcParams['axes.labelsize'] = (small,large)[increase] + 4

def nSignificantDigits(N,sigdig,retstr=False):
    if N==0.: 
        if retstr: return '0'
        return 0.
    sn = abs(N)/N
    N = abs(N)
    maxpower = int(log(N))
    if log(N)<0 and maxpower!=log(N): maxpower-=1
    afterpoint = maxpower+1-sigdig
    val = sn * iround(N, 10**afterpoint)
    if retstr: 
        if afterpoint>=0: return '%d'%val
        else: 
            s = ('%.'+'%d'%(-afterpoint)+'f')%val
            if s[-2:] == '.0': s = s[:-2]  #patch for nSigDig(0.997,1,True)
            return s
    return val

def mylogformat2(x,pos,toshow='2357'):
    s = nSignificantDigits(x,1,True)
    if x<1: firstDigit=s[-1]
    if x>1: firstDigit=s[0]
    return ('',s)[firstDigit in toshow]
mylogformatter2 = matplotlib.ticker.FuncFormatter(mylogformat2)
def send(o, fn='pyobjs/tmp/t',smart=False,useCpickle=False):
    if smart:
        try: 
            if useCpickle: oldO = cPickle.load(file(fn))
            else: oldO = pickle.load(file(fn))
            if oldO == o:
                return
        except IOError:
            pass			
    f = open(fn,'w')
    if useCpickle: cPickle.dump(o,f)
    else: pickle.dump(o,f)
    f.close()

def out(arr,fmt=None):
    for a in arr:
        if fmt==None:
            print(a)
        else:
            print(fmt%a)
def mylegend(frame=False,removeErrorBars=False,color=None,**kwargs):
    l = pl.legend(**kwargs)
    if not frame: l.draw_frame(False)
    if removeErrorBars:
        handles, labels = pl.gca().get_legend_handles_labels()   
        handles = [h[0] for h in handles]
        l = pl.legend(handles, labels, **kwargs)
    if color!=None:
        for text in l.get_texts():
            pl.setp(text, color = color)
    return l
class Progress:
    def __init__(self, iterable,name='progress',frac=None,fmt='%d',dummy=False):
        self.iterable = iterable		
        self.fmt=fmt
        self.dummy=dummy
        if frac!=None: self.frac = frac
        else: 
            if len(self.iterable)>1: self.frac = max(0.01, 1./(len(self.iterable)-1.))
            else: self.frac = 1.
        self.name=name
    def __iter__(self):
        self.origIter = self.iterable.__iter__()
        self.startTime = time.time()
        self.i = 0.
        self.looplen = len(self.iterable)
        self.printEvery = iround(self.looplen*self.frac)
        return self
    def __next__(self):
        if not self.dummy and self.printEvery>0:
            if self.i%self.printEvery==0.: 
                if self.looplen>0:
                    clear_output()					
                    print('%s: '%self.name, self.fmt%(self.i*100./self.looplen) + '%', int(time.time()-self.startTime), 'seconds passed'),
                sys.stdout.flush()
        self.i+=1
        try:
            return next(self.origIter)
        except StopIteration:
            if not self.dummy and self.printEvery>0: 
                print()
                sys.stdout.flush()
            raise StopIteration
def filelines(fn):
    f = file(fn)
    ls = f.readlines()
    f.close()
    return ls
def zeroCrossing(arr):
    return ((np.sign(arr[1:]) * np.sign(arr[:-1])) < 0).nonzero()[0]
def plotLine(a=1,b=0,doLogs=[False,False],throughPoint=None,**kwargs):
    xfunc = [lambda x: x, lambda x: log(x)][doLogs[0]]
    yfunc = [lambda x: x, lambda x: 10**x][doLogs[1]]
    if throughPoint!=None:
        x,y = throughPoint
        b = y-a*x
    return pl.plot(pl.xlim(), yfunc(xfunc(array(pl.xlim()))*a + b),**kwargs)
def searchsortedclosest(arr, val):
    assert(arr[0]!=arr[1])
    if arr[0]<arr[1]:
        ind = np.searchsorted(arr,val)
        ind = minarray(ind, len(arr)-1)
        return maxarray(ind - (val - arr[maxarray(ind-1,0)] < arr[ind] - val),0)        
    else:
        ind = np.searchsorted(-arr,-val)
        ind = minarray(ind, len(arr)-1)
        return maxarray(ind - (-val + arr[maxarray(ind-1,0)] < -arr[ind] + val),0)        
def maxarray(arr, v):
    return arr + (arr<v)*(v-arr)
def minarray(arr, v):
    return arr + (arr>v)*(v-arr)

def unzip(tupslist):
    l = min(lens(tupslist))
    return [[tups[i] for tups in tupslist] for i in range(l)]
def sumlst(lst):
    return reduce(lambda x,y: x+y,lst)
def buildMarker(angles):
    """
    angle 0 is uparrow
    """
    verts = []
    if 0 in angles:
        verts += [(0,1), (-0.4,0.6), (0,1), (0,0), (0,1), (0.4,0.6), (0,1), (0,0)]
    if 90 in angles:
        verts += [(1,0), (0.6,-0.4), (1,0), (0,0), (1,0), (0.6,0.4), (1,0), (0,0)]
    if 180 in angles:
        verts += [(0,-1), (-0.4,-0.6), (0,-1), (0,0), (0,-1), (0.4,-0.6), (0,-1), (0,0)]
    if 270 in angles:
        verts += [(-1,0), (-0.6,-0.4), (-1,0), (0,0), (-1,0), (-0.6,0.4), (-1,0), (0,0)]
    if 315 in angles:
        verts += [(-1,1), (-1,0.5), (-1,1), (0,0), (-1,1), (-0.5,1), (-1,1), (0,0)]
    if len(verts)==0: 
        return [(0,1),(0,0.3),(0.3,0),(0,-0.3),(-0.3,0),(0,0.3)]
    else: 
        return verts
def dlens(dic):
    return dict([(k,len(dic[k])) for k in dic])
def group(func, arr):
    dic = {}
    for a in arr:
        if func(a) in dic:
            dic[func(a)].append(a)
        else:
            dic[func(a)] = [a]
    return dic
def shockJumpConditions(M1,gamma=5/3.):
    rho2_to_rho1 = (gamma+1)*M1**2 / ((gamma+1) + (gamma-1)*(M1**2-1)) #equals v2_to_v1^-1, equals 4 for strong shock
    P2_to_P1     = (gamma+1)+2*gamma*(M1**2-1) / (gamma+1) #equals 1.25 M1^2 for strong shock
    return rho2_to_rho1, P2_to_P1
def M1_from_Tratio(Tratio,gamma=5/3.):
    M1s = np.arange(1e-3,1,1e-3)**-1.
    jumps = shockJumpConditions(M1s,gamma)
    Tratios = jumps[1]/jumps[0]
    ind = np.abs(np.log(Tratios/Tratio)).argmin()
    return M1s[ind]
def smooth(ys,xs,polydeg=10,islog=True,xs2=None):
    try: 
        _ = xs2[0]
    except TypeError:
        xs2 = xs
    if islog:
        goods = ys>0
        return 10.**np.poly1d(np.polyfit(log(xs)[goods],log(ys)[goods],deg=polydeg))(log(xs2))
    else:
        return np.poly1d(np.polyfit(xs,ys,polydeg=polydeg))(xs2)

def running_mean(x, N):
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
def double_PL_fit(xs,ys,zs):
    """
    find {a,b,c} which minimize zs = a * xs^b * ys^c 
    """
    inds = ~np.isnan(log(xs)) & ~np.isnan(log(ys)) & ~np.isnan(log(zs))
    A = np.vstack([log(xs)[inds],log(ys)[inds], np.ones(len(inds.nonzero()[0]))]).T
    return np.linalg.lstsq(A, log(zs)[inds], rcond=None)[0]
def kde_Drummond(xx,yy,bw_method,cut=10):
    inds_kde = ~np.isnan(yy) & ~np.isinf(yy) & ~np.isnan(xx) & ~np.isinf(xx)
    xx = xx[inds_kde]; yy = yy[inds_kde]
    kernel = scipy.stats.gaussian_kde(np.array([xx,yy]),bw_method=bw_method)    
    xmin,xmax = np.min(xx), np.max(xx)
    ymin,ymax = np.min(yy), np.max(yy)
    xxx, yyy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xxx.ravel(), yyy.ravel()])
    f = np.reshape(kernel(positions).T, xxx.shape)
    minus1sig,median,plus1sig = [np.array([np.interp(p,np.cumsum(f[i])/np.sum(f[i]), yyy[i])  for i in range(len(f))])
                                 for p in (0.159,0.5,0.841)]
    i = np.argmin(np.abs(xxx[:,0] - np.sort(xx)[cut]))
    j = np.argmin(np.abs(xxx[:,0] - np.sort(xx)[-cut]))    
    return xxx[i:j,0], median[i:j], minus1sig[i:j],plus1sig[i:j]
