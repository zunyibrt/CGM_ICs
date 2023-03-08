import sys, os, h5py, numpy as np
from numpy import log10 as log
from astropy import units as un, constants as cons

import cooling_flow as CF
import HaloPotential as Halo
import WiersmaCooling as Cool
import gsr

if os.getenv('HOME')=='/home/jonathan':
    outdir_data = '/home/jonathan/Dropbox/jonathanmain/CGM/KY_sims/ICs/' 
elif os.getenv('HOME')=='/Users/jonathanstern':
    outdir_data = '/Users/jonathanstern/Dropbox/jonathanmain/CGM/KY_sims/ICs/' 
else:
    outdir_data = '/mnt/home/jstern/ceph/ICs/' #outdir_base

keys_dict = {'pos':'Coordinates', 
             'vel':'Velocities',
             'ids':'ParticleIDs',
             'mass':'Masses',
             'energies':'InternalEnergy'}


MASS_UNITS = 1e10
ORIGIN = np.array([0,0,0])
Zsun = 0.0129
def mifkad(arr):
    dic={}
    for x in arr:
        if x in dic:
            dic[x] += 1
        else:
            dic[x] = 1
    return dic

    
class ICs:
    outdir_base = '../GIZMO_ICs/'
    init_base_filename = outdir_base+'init.c_base'
    analytic_gravity_base_filename = outdir_base+'analytic_gravity.h_base'
    fn_diskOnly = '../MakeDisk_wHalo_m11_lr/ICs/m11_no_halo_%d%s%s%s%s.ic'
    outdir_template = outdir_data + 'vc%d_Rs%d_Mdot%d_Rcirc%d%s_res%s'

    max_step = 0.1                    #lowest resolution of solution in ln(r)
    R_min    = 0.3*un.kpc             #inner radius of supersonic part of solution
    R_max    = 10000.*un.kpc          #outer radius of integration
    z_cooling = 0    
    # these params should be the same as in MakeDisk/main.c
    DiscScale = 2.5*un.kpc
    DiscHeight = 0.2*un.kpc
    Z_disk = 1
    
    def __init__(self,vc=None,Rcirc=None,Rsonic=None,Z_CGM=None,smallGalaxy=False,resolution = 8e4*un.Msun,ics=None,fgas=None,m=0,Rvc=200*un.kpc,
                 Rres2Rcool=2):
        if ics!=None: #copy constructor for changing only Rcirc
            self.vc = ics.vc
            self.Rcirc = ics.Rcirc
            self.Rsonic = ics.Rsonic
            self.Z_CGM = ics.Z_CGM
            self.potential = ics.potential
            self.cooling = ics.cooling
            self.CF_solution = ics.CF_solution
            self.smallGalaxy = ics.smallGalaxy
            self.resolution = ics.resolution
            self.fgas_str = ics.fgas_str
            self.Rres2Rcool=ics.Rres2Rcool
        else:                       
            self.vc = vc
            self.Rcirc = Rcirc
            self.Rsonic = Rsonic
            self.Z_CGM = Z_CGM
            self.potential = Halo.PowerLaw(m=m,vc_Rvir=self.vc,Rvir=Rvc)
            self.cooling = Cool.Wiersma_Cooling(self.Z_CGM,self.z_cooling)
            self.smallGalaxy = smallGalaxy
            self.resolution = resolution
            self.fgas_str = '_fgas' + ('%s'%fgas).replace('.','')
            self.Rres2Rcool = Rres2Rcool
                 
    def calc_CF_solution(self,tol=1e-6,pr=True):
        self.CF_solution = CF.shoot_from_sonic_point(self.potential,
                                        self.cooling,
                                        self.Rsonic,
                                        self.R_max,self.R_min,max_step=self.max_step,tol=tol,
                                        pr=pr)
    def sample(self):
        return CF.sample(self.CF_solution,self.resolution.to('Msun'),self.Rcirc,self.DiscScale,self.DiscHeight,Rres2Rcool=self.Rres2Rcool)
    def outdirname(self):
        res_str = '%.1g'%self.resolution.value
        res_str = res_str[:2]+res_str[-1:]
        return self.outdir_template%(self.vc.value,
                                    self.Rsonic.value,
                                    self.CF_solution.Mdot.value*1000,
                                    self.Rcirc.value,
                                    self.fgas_str,
                                    res_str)
    def makedisk_filename(self):
        res_str = '%.1g'%self.resolution.value
        res_str = res_str[:2]+res_str[-1:]
        Rcirc_str = ('','_Rcirc%d'%self.Rcirc.value)[self.Rcirc.value!=10]
        return self.fn_diskOnly%(self.vc.value,
                                 ('','_small_galaxy')[self.smallGalaxy],
                                 '_res'+res_str,
                                 self.fgas_str,
                                 Rcirc_str)
        
        
    def create_output_files(self):
        outdir = self.outdirname()
        if not os.path.exists(outdir): os.mkdir(outdir)
        print('files saved to: %s'%outdir)
        self.create_ICs_hdf5_file(outdir+'/init_snapshot')
        self.update_GIZMO_files(outdir+'/')
    def create_ICs_hdf5_file(self,fn_out):                
        makeDisk_filename = self.makedisk_filename()
        print(makeDisk_filename)
        snap = gsr.Snapshot(makeDisk_filename)
        gas_masses, gas_coords, gas_vels, gas_internalEnergies = self.sample()
        
        fwrite = h5py.File("%s.hdf5"%fn_out, "w")
        try:        
            whead  = fwrite.create_group("Header")            
            for iPartType in range(6):
                grp = fwrite.create_group("PartType%d"%iPartType)
                for k in snap.SnapshotData.keys():
                    if k in ('header',): continue
                    elif k=='energies': 
                        if iPartType!=0: continue
                        data = snap.SnapshotData[k]
                    else: 
                        data = snap.SnapshotData[k][iPartType]            
                    if iPartType==0:                
                        if k=="pos":
                            number_without_CGM = data.shape[0]
                            data   = np.concatenate([data,gas_coords],axis=0)                     
                        if k=="energies":
                            data   = np.concatenate([data,gas_internalEnergies.value])       
                        if k=="mass":                    
                            Zs = np.concatenate([np.ones((len(data),      11))*self.Z_disk,
                                                 np.ones((len(gas_masses),11))*self.Z_CGM],axis=0) * Zsun
                            data   = np.concatenate([data,gas_masses.value/MASS_UNITS])                    
                            grp.create_dataset('Metallicity', Zs.shape, dtype=Zs.dtype)
                            grp['Metallicity'][:] = Zs                      
                            #print(['%.2g %d'%item for item in mifkad(Zs[:,0]).items()])
                            #print(['%.2g %d'%item for item in mifkad(data).items()])
                            
                        if k=='vel':
                            data   = np.concatenate([data,gas_vels],axis=0)             
                        if k=='ids':
                            data   = np.arange(len(data)+len(gas_masses))
                            print('max gas id:',data.max())
                        
                    if k=='mass': 
                        if data.shape[0]!=0:
                            unique_masses = ['%.2g'%x for x in np.unique(data)*1e10]
                            print('Ms: %s%s'%(' '.join(unique_masses[:3]),('... %s'%unique_masses[-1],'')[len(unique_masses)<=3]))
                        if iPartType in (2,3):      
                            Zs = np.ones((len(data),11))*self.Z_disk * Zsun                     
                            grp.create_dataset('Metallicity', Zs.shape, dtype=Zs.dtype)
                            grp['Metallicity'][:] = Zs
                        
                    if k=='pos': 
                        if data.shape[0]!=0:
                            print('Part%d num=%d'%(iPartType,data.shape[0]),end=',')
                            if iPartType==0: print(' w/o CGM=%d'%number_without_CGM,end=',')                                
                            print(' |<r>|=%.1f'%((np.mean(data,axis=0)**2).sum()**0.5),end=',')
                        data += ORIGIN
                        coords = data                        
                    if k=='vel':
                        if data.shape[0]!=0:
                            print(' |<v>|=%.1f'%((np.mean(data,axis=0)**2).sum()**0.5),end=',')
                            print(' <j>=%s'%(' '.join(['%.1f'%x for x in np.mean(np.cross(coords,data),axis=0)])),end=',')
                    grp.create_dataset(keys_dict[k], data.shape, dtype=data.dtype)
                    grp[keys_dict[k]][:] = data 
            header = snap.get_header()
            header['Npart'][0] += len(gas_masses)
            header['Ntot'] += len(gas_masses)
            whead.attrs['NumPart_Total'] = whead.attrs['NumPart_ThisFile'] = header['Npart']
            whead.attrs['MassTable'] = header['Mpart']
            whead.attrs['NumPart_Total'] = header['Time']
            whead.attrs['NumFilesPerSnapshot'] = 1
            whead.attrs['Flag_Sfr'] = 0
            whead.attrs['Flag_Cooling'] = 0
            whead.attrs['Flag_Feedback'] = 0
            whead.attrs['Flag_StellarAge'] = 0
            whead.attrs['Flag_Metals'] = 0
            whead.attrs['BoxSize'] = whead.attrs['Omega0'] = whead.attrs['OmegaLambda'] = whead.attrs['HubbleParam'] = whead.attrs['Redshift'] = 0
        except:           
            fwrite.close()
            raise
    def update_GIZMO_files(self,outdir):
        f = open(self.init_base_filename)
        s = f.read().replace('Z_CGM_TO_BE_REPLACED','%.3f'%self.Z_CGM)
        f.close()
        f = open(outdir+'init.c','w')
        f.write(s)
        f.close()
        f = open(self.analytic_gravity_base_filename)
        s = f.read().replace('VC_TO_BE_REPLACED','%.0f'%self.vc.value)
        f.close()
        f = open(outdir+'analytic_gravity.h','w')
        f.write(s)
        f.close()   
            
def verify_not_self_gravitating():
    # verify sample and gas is negligible in terms of gravity   
    gas_masses, gas_coords, gas_vels, gas_internalEnergies = sampling
    rs = ((gas_coords**2).sum(axis=1))**0.5
    sampled_Mgas = np.array([gas_masses[rs<R.value].sum() for R in res.Rs()])*un.Msun
    pl.plot(res.Rs(),((cons.G*res.Mgas()/res.Rs())**0.5).to('km/s'),ls='--')
    pl.plot(res.Rs(),((cons.G*sampled_Mgas/res.Rs())**0.5).to('km/s'))
    pl.axhline(res.potential.vc_Rvir.value,c='k',ls=':')
    pl.semilogx()
    
        

def plot_solutions1(solutions):
    for iPanel in range(3):
        ax = pl.subplot(3,1,iPanel+1)
        for res in solutions:
            pl.plot(res.Rs(),(res.vs(),res.Ts(),res.nHs()*res.Rs()**1.5)[iPanel],label='Rsonic=$%.1f$ kpc'%(res.R_sonic().value))
        if iPanel==0: 
            pl.legend()
    #         pl.ylim(0,1)
        if iPanel==1: 
            pl.ylim(0,1e6)
        if iPanel==2:
            ax.set_yscale('log')
            pl.ylim(0.01,1)
        ax.set_xscale('log')
        
def plot_solutions2(solutions):
    fig = pl.figure(figsize=(10,10))
    pl.subplots_adjust(hspace=0.4,wspace=0.5)
    for iPanel in range(7):
        ax = pl.subplot(4,2,iPanel+1)
        for ires,res in enumerate(solutions):
            c= 'br'[ires//2]
            ls = ('-','--')[ires%2]
            label = r'$\dot{M} = %.2f$'%res.Mdot.value
            if iPanel==0: ys = res.Ts()
            if iPanel==1: ys = res.nHs()
            if iPanel==2: ys = res.Ms()
            if iPanel==3: ys = res.t_cools() / res.t_flows()
            if iPanel==4: ys = res.t_cools() 
            if iPanel==5: ys = res.Mgas() 
            if iPanel==6: ys = res.P2ks() / (PB/cons.k_B).to('cm**-3*K')
            pl.loglog(res.Rs(),ys,c=c,ls=ls,label=label)
            ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=10,numticks=10))
            ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=10,numticks=10))
            pl.xlim(1,1000)
            pl.xlabel(r'$r$ [kpc]')
            if iPanel==0:
                pl.ylabel(r'temperature [K]')
                pl.ylim(1e4,1e7)            
            if iPanel==1: 
                pl.ylabel(r'hydrogen density [cm$^{-3}$]')
                pl.ylim(1e-5,1)
                pl.legend(loc='lower left',fontsize=10,handlelength=1.2)
            if iPanel==2: 
                pl.ylabel(r'mach number')
                pl.ylim(0.03,30)
                pl.axhline(1.,c='.5',lw=0.5)
            if iPanel==3: 
                pl.ylabel(r'$v_r / (r/t_{\rm cool})$')
                pl.ylim(0.03,30)
                pl.axhline(1.,c='.5',lw=0.5)
            if iPanel==4: 
                pl.ylabel(r'$t_{\rm cool}\ [{\rm Gyr}]$')
                pl.ylim(1e-2,100)
                pl.axhline(10,c='.5',lw=0.5)
                pl.plot(res.Rs(),res.tff().to('Gyr'),c='.5',lw=0.5)
            if iPanel==5: 
                pl.ylabel(r'$M_{\rm gas}\ [{\rm M}_\odot]$')
                pl.ylim(0,1e12)
                pl.axhline(0.16e12,c='.5',lw=0.5)
            if iPanel==6:
                pl.ylabel(r'plasma $\beta$')
                pl.ylim(0.5,1e6)
                pl.axhline(1.,c='.5',lw=0.5)
                pl.text(0.95,0.95,r'$\log\ B_z=%.2g\ {\rm G}$'%log(B),transform=ax.transAxes,ha='right',va='top')
                Rcool = res.Rcool(10*un.Gyr)
                PB_ideal = PB * (res.Rs()/Rcool)**-4
                pl.plot(res.Rs(), res.P2ks() / (PB_ideal/cons.k_B).to('cm**-3*K'),c='k',ls=':',lw=0.5)    
        
  
    
