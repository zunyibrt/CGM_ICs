Package for deriving CGM ICs for hydrodynamic simulation of ISM+CGM
Example in ipynb/create_ICs.ipynb

CGM ICs are based on 1D steady-state solutions described in 
http://adsabs.harvard.edu/doi/10.1093/mnras/stz1859
to which rotation is added 
v_phi = v_c (R/R_circ) sin(theta)
where v_c is circular velocity, R is cylindrical radius, theta is polar angle and R_circ is a parameter (R_circ~10-20 kpc for Milky-Way-like halos)


Files:
ipynb/steady_state_integration_example.ipynb    Examples how to integrate steady-state solution
pysrc/cooling_flow.py				Main module for integration of flow equations. 
pysrc/WiersmaCooling.py				Wraps the Wiersma et al. (2009) cooling functions
pysrc/HaloPotential.py 				Creates the gravity potential used in the paper. Includes an NFW halo, a central galaxy with mass based on the Behroozi et al. (2018) stellar-mass halo-mass relation, and an external component following Diemer & Kravtsov (2014). 
cooling/					Hdf5 files for the Wiersma+09 cooling tables


