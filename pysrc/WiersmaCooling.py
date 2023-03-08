"""
Module for providing the Wiersma et al. (2009) cooling functions to the cooling_flow module
"""
dataDir = '../cooling/Wiersma09_CoolingTables/'
import glob,h5py
import scipy, numpy as np
from scipy import integrate, interpolate
from numpy import log as ln, log10 as log, e, pi, arange, zeros
from astropy import units as un, constants as cons
import cooling_flow as CF

class Constant_Cooling(CF.Cooling):
    def __init__(self,LAMBDA):
        self._LAMBDA = LAMBDA        
    def LAMBDA(self,T,nH):
        """cooling function"""
        return self._LAMBDA
    def f_dlnLambda_dlnT(self,T,nH):
        """logarithmic derivative of cooling function with respect to T"""
        return 0
    def f_dlnLambda_dlnrho(self,T,nH):
        """logarithmic derivative of cooling function with respect to rho"""
        return 0
    
class Wiersma_Cooling(CF.Cooling):
    """
    creates Wiersma+09 cooling function for given metallicity and redshift
    """
    def __init__(self,Z2Zsun,z):
        fns = np.array(glob.glob(dataDir+'z_?.???.hdf5'))
        zs = np.array([float(fn[-10:-5]) for fn in fns])
        fn = fns[zs.argsort()][searchsortedclosest(sorted(zs), z)]
        
        f=h5py.File(fn,'r')
        
        He2Habundance = 10**-1.07 * (0.71553 + 0.28447*Z2Zsun) #Asplund+09, Groves+04
        X = (1 - 0.014*Z2Zsun) / (1.+4.*He2Habundance)
        Y = 4.*He2Habundance * X
        iHe = searchsortedclosest(f['Metal_free']['Helium_mass_fraction_bins'][:],Y)
        
        H_He_Cooling  = f['Metal_free']['Net_Cooling'][iHe,...]
        Tbins         = f['Metal_free']['Temperature_bins'][...]
        nHbins        = f['Metal_free']['Hydrogen_density_bins'][...]
        Metal_Cooling = f['Total_Metals']['Net_cooling'][...] * Z2Zsun    
        
        self.f_Cooling = interpolate.RegularGridInterpolator((log(Tbins), log(nHbins)),
                                                        Metal_Cooling+H_He_Cooling, 
                                                        bounds_error=False, fill_value=None)
        #### calculate gradients of cooling function
        X, Y = np.meshgrid(Tbins, nHbins, copy=False)
        dlogT = np.diff(log(Tbins))[0] 
        dlogn = np.diff(log(nHbins))[0] 
        vals = log(self.LAMBDA(X*un.K,Y*un.cm**-3).value)
        dlnLambda_dlnrhoArr, dlnLambda_dlnTArr = np.gradient(vals,dlogn, dlogT)    
        self.dlnLambda_dlnT_interpolation = interpolate.RegularGridInterpolator((log(Tbins), log(nHbins)),dlnLambda_dlnTArr.T, bounds_error=False, fill_value=None)
        self.dlnLambda_dlnrho_interpolation = interpolate.RegularGridInterpolator((log(Tbins), log(nHbins)),dlnLambda_dlnrhoArr.T, bounds_error=False, fill_value=None)                
    def LAMBDA(self, T, nH):
        """cooling function"""
        return self.f_Cooling((log(T.to('K').value), log(nH.to('cm**-3').value))) * un.erg*un.cm**3/un.s
    def tcool(self,T,nH):
        """cooling time"""
        return 3.5 * cons.k_B * T / (nH * self.LAMBDA(T, nH))
    def f_dlnLambda_dlnT(self, T, nH):         
        """logarithmic derivative of cooling function with respect to T"""
        return self.dlnLambda_dlnT_interpolation((log(T.to('K').value), log(nH.to('cm**-3').value)))
    def f_dlnLambda_dlnrho(self, T, nH):
        """logarithmic derivative of cooling function with respect to rho"""
        return self.dlnLambda_dlnrho_interpolation((log(T.to('K').value), log(nH.to('cm**-3').value)))
    



class Kartick_Cooling(CF.Cooling):
    table_path = '../cooling/Kartick_CIE_cooling.table'
    def __init__(self):
        table = np.genfromtxt(self.table_path)
        self.Tbins = table[:,0]
        self.LAMBDAs = table[:,1] * table[:,2] #LAMBDA times n_e in the table        
        #### calculate gradient of cooling function
        dlogT = np.diff(log(self.Tbins))
        vals = log(self.LAMBDA(self.Tbins*un.K).value)
        self.LAMBDA_gradient_Tbins = (log(self.Tbins[1:])+log(self.Tbins[:-1]))/2
        self.LAMBDA_gradient = np.diff(vals)/dlogT
    def LAMBDA(self, T, nH=None):
        """cooling function"""
        return 10.**np.interp(log(T.to('K').value),log(self.Tbins),log(self.LAMBDAs)) * un.erg*un.cm**3/un.s
    def f_dlnLambda_dlnT(self, T, nH=None):         
        """logarithmic derivative of cooling function with respect to T"""
        return np.interp(log(T.to('K').value),self.LAMBDA_gradient_Tbins,self.LAMBDA_gradient)
    def f_dlnLambda_dlnrho(self, T, nH):
        """logarithmic derivative of cooling function with respect to rho"""
        return 0.
    

class DopitaSutherland_CIE(CF.Cooling):
    table_path = '../cooling/DopitaSutherland_CIE.dat'
    def __init__(self,Z2Zsun):
        table = np.genfromtxt(self.table_path)
        self.Tbins = table[:,0]
        if Z2Zsun==1:   self.LAMBDAs = table[:,1] 
        if Z2Zsun==1/3.: self.LAMBDAs = table[:,2] 
        ### convert to definition that n_H^2 Lambda is cooling per unit volume
        n_i_to_n_H = 1.22**-1 / 0.7
        n_e_to_n_H = 1.17**-1 / 0.7
        self.LAMBDAs *= n_i_to_n_H * n_e_to_n_H
        #### calculate gradient of cooling function
        dlogT = np.diff(log(self.Tbins))
        vals = log(self.LAMBDA(self.Tbins*un.K).value)
        self.LAMBDA_gradient_Tbins = (log(self.Tbins[1:])+log(self.Tbins[:-1]))/2
        self.LAMBDA_gradient = np.diff(vals)/dlogT
    def LAMBDA(self, T, nH=None):
        """cooling function"""
        return 10.**np.interp(log(T.to('K').value),log(self.Tbins),log(self.LAMBDAs)) * un.erg*un.cm**3/un.s
    def f_dlnLambda_dlnT(self, T, nH=None):         
        """logarithmic derivative of cooling function with respect to T"""
        return np.interp(log(T.to('K').value),self.LAMBDA_gradient_Tbins,self.LAMBDA_gradient)
    def f_dlnLambda_dlnrho(self, T, nH):
        """logarithmic derivative of cooling function with respect to rho"""
        return 0.
    

        
        
def searchsortedclosest(arr, val):
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

