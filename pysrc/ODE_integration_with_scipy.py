######### steady-state equations integration
def IntegrateFlowEquations(Mdot,T0,rho0,potential,cooling,metallicity,isInward,R_min,R_max,R_circ=None,max_step=0.1,
                           atol=1e-6,rtol=1e-6,checkUnbound=True,issupersonic=False,terminalUnbound=True,minT=2e4):

    """
    Function for integrating steady-state flow equations. Called by shoot_from_R_circ() and shoot_from_sonic_point()    
    Accepts:
    Mdot, T0, rho0: hydrodynamic variables at initial radius (either R_min or R_max, depending on direction of integration)
    potential: Potential object
    cooling: Cooling object
    isInward: direction of integration (outward for subsonic part, inward for supersonic part)
    R_min, R_max: range of integration 
    max_step: minimum resolution of solution in ln(r)
    terminalUnbound, checkUnbound: if terminalUnbound=True, integration stops when Bernoulli>0. if checkUnbound==False, Bernoulli parameter is not calculated during integration
    issupersonic: whether solution is supersonic or subsoni    
    minT: integration stops when T drops below this value. 
    atol,rtol: input for scipy.integrate.solve_ivp    
    Returns:
    IntegrationResult object
    """
    def odes(ln_R, y,Mdot=Mdot,potential=potential,cooling=cooling,metallicity=metallicity,isInward=isInward,R_circ=R_circ):
        if isInward: R = e**-ln_R*un.kpc
        else:        R = e**ln_R*un.kpc
        ln_T,ln_rho = y
        rho,T=e**ln_rho*un.g/un.cm**3, e**ln_T*un.K
        nH = (X*rho/cons.m_p).to('cm**-3')

        v = (Mdot/(4*pi*R**2*rho)).to('km/s')
        cs2 = (gamma*cons.k_B * T / (mu*cons.m_p)).to('km**2/s**2')
        M = (v/cs2**0.5).to('')

        vc2 = potential.vc(R)**2
        Z = metallicity.Z(R)[0]
        
        if R_circ!=None:
            vc2 *= (1-(R_circ/R)**2)
        v_ratio = (vc2/cs2).to('')

        t_flow = (R/v).to('Gyr')
        LAMBDA = cooling.LAMBDA(T,Z,nH)
        t_cool = (rho*cs2 / (nH**2*LAMBDA) / (gamma*(gamma-1))).to('Gyr')
        t_ratio = (t_flow/t_cool).to('')

        dln_rho2dln_R =  (-t_ratio/gamma - v_ratio + 2*M**2)  / (1-M**2)
        dln_T2dln_R = t_ratio + dln_rho2dln_R*(gamma-1)

        if isInward: return -dln_T2dln_R, -dln_rho2dln_R
        else: return dln_T2dln_R, dln_rho2dln_R

    def sonic_point(ln_R, y,Mdot=Mdot,isInward=isInward,issupersonic=issupersonic): 
        if isInward: R = e**-ln_R*un.kpc
        else: R = e**ln_R*un.kpc        
        ln_T,ln_rho = y
        rho, T = e**ln_rho*un.g/un.cm**3, e**ln_T*un.K
        v = Mdot/(4*pi*R**2*rho)        
        cs2 = gamma*cons.k_B * T / (mu*cons.m_p)
        M = (v/cs2**0.5).to('')
        return M - 1
    def lowT(ln_R, y, minT=minT):
        ln_T,ln_rho = y
        T=e**ln_T*un.K
        return T.to('K').value-minT
    def unbound(ln_R, y,potential=potential,Mdot=Mdot,isInward=isInward,R_max=R_max): 
        if isInward: R = e**-ln_R*un.kpc
        else: R = e**ln_R*un.kpc    
        ln_T,ln_rho = y
        rho, T = e**ln_rho*un.g/un.cm**3, e**ln_T*un.K
        v = (Mdot/(4*pi*R**2*rho)).to('km/s').value
        cs2 = (gamma*cons.k_B * T / (mu*cons.m_p)).to('km**2/s**2').value
        B = 0.5*v**2 + cs2/(gamma-1) + potential.Phi(R).to('km**2/s**2').value
        return B
    def dummy(ln_R,y):
        return 1.
    sonic_point.terminal = True
    lowT.terminal = True
    unbound.terminal = terminalUnbound
    events = sonic_point,(dummy,unbound)[checkUnbound],(lowT,dummy)[issupersonic]

    if isInward: 
        Rrange =  R_max, R_min
        lnRrange = -ln(R_max.to('kpc').value),-ln(R_min.to('kpc').value)
    else:        
        Rrange =  R_min, R_max
        lnRrange = ln(R_min.to('kpc').value), ln(R_max.to('kpc').value)

    initVals = ln(T0/un.K), ln(rho0/(un.g*un.cm**-3))


    if not issupersonic and sonic_point(lnRrange[0], initVals)>0: return 'starts supersonic'
    if     issupersonic and sonic_point(lnRrange[0], initVals)<0: return 'starts subsonic'
    if terminalUnbound and checkUnbound and unbound(lnRrange[0], initVals)>0: return 'starts unbound'

    return scipy.integrate.solve_ivp(odes,lnRrange,initVals,events=events,
                                    max_step=max_step,atol=atol,rtol=rtol)

