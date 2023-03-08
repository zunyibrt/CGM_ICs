Package for deriving steady-state cooling flow solutions described in 
http://adsabs.harvard.edu/doi/10.1093/mnras/stz1859

Files:
integrate.ipynb		Jupyter Notebook with a couple of examples
cooling_flow.py		Main module for integration of flow equations. 
WiersmaCooling.py	Wraps the Wiersma et al. (2009) cooling functions
HaloPotential.py 	Creates the gravity potential used in the paper. Includes an NFW halo, a central galaxy with mass based on the Behroozi et al. (2018) stellar-mass halo-mass relation, and an external component following Diemer & Kravtsov (2014_. 
params/			Files for calculation of Behroozi+18 SMHM relation
cooling/		Hdf5 files for the Wiersma+09 cooling tables


