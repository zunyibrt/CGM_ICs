# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from astropy import constants as cons, units as un
import cooling_flow as CF, HaloPotential

vc = 206
Mdot = 6.25


class PowerLawLambda:
    def __init__ (self,Lambda,baseT=2e6*un.K,l=-0.5):
        self.Lambda = Lambda
        self.baseT=baseT
        self.l = l
    def LAMBDA(self,T,nH):
        """cooling function"""
        return self.Lambda * (T/self.baseT)**self.l
    
    
class ZeroOrder(CF.CGMsolution): #no ang. mom. solution
    dlogr=0.01
    _Rs = 10.**np.arange(-1,3,dlogr) * un.kpc
    def __init__(self,Mdot,vc,Lambda):
        self.Mdot1 = Mdot/(un.Msun/un.yr)
        self.vc200 = vc/(200*un.km/un.s) 
        self.r10s = self.Rs() / (10*un.kpc)
        self.cooling = PowerLawLambda(Lambda)
        self.potential = HaloPotential.PowerLaw(0,vc,300*un.kpc)    
        self.Lambda22 = Lambda/(10**-22*un.erg/un.s*un.cm**3)
    def Rs(self):
        return self._Rs
    def Ts(self):
        return np.ones(self._Rs.shape) * self.vc200**2*2.01e6 *un.K
    def rhos(self):
        nHs =  0.826e-3*self.r10s**-1.5*self.vc200*self.Mdot1**0.5*self.Lambda22**-0.5*un.cm**-3
        return cons.m_p * nHs / CF.X
    def vrs(self):
        """inflow velocity of the solution at all radii"""
        return (self.Mdot1*un.Msun/un.yr / (4*pi*self.Rs()**2*self.rhos())).to('km/s')
    def vs(self):
        return self.vr()
    def Omegas(self):
        return np.zeros(self._Rs.shape)*un.Gyr**-1

class FirstOrder(ZeroOrder):
    dtheta=0.01
    _thetas = np.arange(0,np.pi+dtheta/2,dtheta)    
    def __init__(self,Mdot,vc,Lambda,Rcirc):
        super(FirstOrder,self).__init__(Mdot, vc, Lambda)
        self.Rcirc=Rcirc
        self.r2Rcirc = self.Rs()/Rcirc
    def Rs(self):
        return np.meshgrid(self._Rs,self._thetas)[0]
    def thetas(self):
        return np.meshgrid(self._Rs,self._thetas)[1]
    def Ts(self):
        return super(FirstOrder,self).Ts() * (1-self.r2Rcirc**-2*(2*np.sin(self.thetas())**2-5/6))
    def rhos(self):
        return super(FirstOrder,self).rhos() * (1+self.r2Rcirc**-2*(11/4*np.sin(self.thetas())**2-35/24))
    def P2ks(self):
        return (CF.mu**-1 / cons.m_p * 
                super(FirstOrder,self).rhos() * 
                super(FirstOrder,self).Ts() * 
                (1+self.r2Rcirc**-2*(3/4*np.sin(self.thetas())**2-5/8)))
    def vrs(self):
        """inflow velocity of the solution"""
        return self.vs()[0,:]
    def vthetas(self):
        """polar velocity of the solution"""
        return self.vs()[1,:]
    def vphis(self):
        """angular velocity of the solution"""
        return self.vs()[2,:]
    def vs(self):
        return np.array([
            super(FirstOrder,self).vrs() * (1-self.r2Rcirc**-2*(23/12*np.sin(self.thetas())**2-65/72)),
            super(FirstOrder,self).vrs() * 5/18.*self.r2Rcirc**-2*np.sin(2*self.thetas()),
            self.Omega()*self.Rs()*np.sin(self.thetas())
            ])            
    def Omegas(self):
        return (self.vc() * self.Rcirc / self.Rs()**2).to('Gyr**-1')

    
    