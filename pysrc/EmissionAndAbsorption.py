import pyatomdb
# this function was  a member in IntegrationResult

def emissivities(self,element,ionization,wlRange=(6.2,24.8)):     
    if not hasattr(self, 'epsilonDic'): self.epsilonDic = {}
    k = (element,ionization,wlRange)
    if k not in self.epsilonDic:
        self.epsilonDic[k] = np.array([emissivity(logT,*k) for logT in log(self.Ts().value)]) * un.cm**3*un.s**-1
    return self.epsilonDic[k]

def emissivity(logT, element,ionization,wlRange=[6.2,24.8]):    
    logTs_base = np.arange(0.05,9.,.1)
    logT_less = logTs_base[logTs_base<=logT][-1]
    logT_more = logTs_base[logTs_base>logT][0]  
    vals = np.zeros(2)
    for i,_logT in enumerate((logT_less, logT_more)):
        ite = pyatomdb.spectrum.get_index( 10**_logT, teunits='K', logscale=False,filename='/home/jonathan/research/separate/atomdb/apec_line.fits')    
        llist = pyatomdb.spectrum.list_lines(wlRange, index=ite,linefile='/home/jonathan/research/separate/atomdb/apec_line.fits')
        vals[i] = sum([x[2] for x in llist if x[4]==element and x[5]==ionization])
    return np.interp(logT,np.array([logT_less, logT_more]),vals)

