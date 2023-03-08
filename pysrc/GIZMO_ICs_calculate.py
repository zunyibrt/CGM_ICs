import sys, pdb
from importlib import reload
from astropy import units as un, constants as cons
from astropy.cosmology import Planck15 as cosmo
import numpy as np
from numpy import log10 as log
sys.path.append('../pysrc/')
import GIZMO_ICs as ics

ics.ICs.R_max=4000*un.kpc
ics.ICs.fn_diskOnly = '/mnt/home/jstern/ceph/ICs/m11_no_halo_%d%s%s%s.ic'
vc = 150*un.km/un.s
Rsonics = np.array([0.3,0.00003])[:]*un.kpc
Z = 0.3
Rcirc = 10*un.kpc
resolution = 2e3*un.Msun #2e4
instances = [None]*len(Rsonics)
for fgas in (0.2,0.3,0.4,0.5)[:1]:
    for i,Rsonic in enumerate(Rsonics):
        instance = instances[i] = ics.ICs(vc,Rcirc,Rsonic,Z,fgas=fgas,resolution = resolution)
        print('v_c=%d, R_sonic=%.1f, Z=%.2f, f_gas=%.1f'%(vc.value,Rsonic.value,Z,fgas))
        instance.calc_CF_solution(pr=True,tol=1e-8)
        if instance.CF_solution!=None:
            log_nH = log(instance.CF_solution.nHs()[instance.CF_solution.Rs()>Rcirc][0].value)
            print(' log nH(10 kpc)=%.2f, Mdot=%.3f'%(log_nH,instance.CF_solution.Mdot.value))
            instance.create_output_files()
        print('')
