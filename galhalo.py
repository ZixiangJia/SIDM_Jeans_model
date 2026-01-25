################## Functions for galaxy-halo connection #################

# Arthur Fangzhou Jiang 2019, HUJI
# Arthur Fangzhou Jiang 2020, Caltech
# Zixiang Jia 2025, Peking University
# - added general treatment of adiabatic contraction that works for 
#   different halo profiles, in contra_general
# - added Mo Mao and White (98) recipe for disk 

#########################################################################

import numpy as np
from numba import jit

import config as cfg
import aux
import profiles as pr
import cosmo as co

from lmfit import minimize, Parameters
from scipy.integrate import simpson,quad
from scipy import interpolate
import scipy.optimize as opt

#########################################################################

#---galaxy-size-halo-structure relation   

def Reff(Rv,c2):
    """
    Effective radius (3D half-stellar-mass radius) of a galaxy, given
    the halo virial radius and concentration, using the empirical formula
    of Jiang+19 (MN, 488, 4801) eq.6
    
        R_eff = 0.02 (c/10)^-0.7 R_vir
    
    Syntax:
    
        Reff(Rv,c2)
        
    where
    
        Rv: virial radius [kpc] (float or array)
        c2: halo concentration defined as R_vir / r_-2, where r_-2 is the
            radius at which dln(rho)/dln(r) = -2 (float or array)
    """
    return 0.02 * (c2/10.)**(-0.7) * Rv
    
#---stellar-halo-mass relation

def lgMs_B13(lgMv,z=0.):
    r"""
    Log stellar mass [M_sun] given log halo mass and redshift, using the 
    fitting function by Behroozi+13.
    
    Syntax:
    
        lgMs_B13(lgMv,z)
    
    where 
        lgMv: log virial mass [Msun] (float or array)
        z: redshift (float) (default=0.)
    """
    a = 1./(1.+z)
    v = v_B13(a)
    e0 = -1.777
    ea = -0.006
    ez = 0.000
    ea2 = -0.119
    M0 = 11.514
    Ma = -1.793
    Mz = -0.251
    lge = e0 + (ea*(a-1.)+ez*z)*v + ea2*(a-1.)
    lgM = M0 + (Ma*(a-1.)+Mz*z)*v
    return lge+lgM + f_B13(lgMv-lgM,a) - f_B13(0.,a)
def v_B13(a):
    r"""
    Auxiliary function for lgMs_B13.
    """
    return np.exp(-4.*a**2)
def f_B13(x,a):
    r"""
    Auxiliary function for lgMs_B13.
    """
    a0 = -1.412
    aa = 0.731
    az = 0.0
    d0 = 3.508
    da = 2.608
    dz = -0.043
    g0 = 0.316
    ga = 1.319
    gz = 0.279
    v = v_B13(a)
    z = 1./a-1.
    alpha = a0 + (aa*(a-1.)+az*z)*v
    delta = d0 + (da*(a-1.)+dz*z)*v
    gamma = g0 + (ga*(a-1.)+gz*z)*v
    return delta*(np.log10(1.+np.exp(x)))**gamma/(1.+np.exp(10**(-x)))-\
        np.log10(1.+10**(alpha*x))

def lgMs_RP17(lgMv,z=0.):
    """
    Log stellar mass [M_sun] given log halo mass and redshift, using the 
    fitting function by Rodriguez-Puebla+17.
    
    Syntax:
    
        lgMs_RP17(lgMv,z)
    
    where 
    
        lgMv: log virial mass [M_sun] (float or array)
        z: redshift (float) (default=0.)
    """
    a = 1./(1.+z)
    v = v_RP17(a)
    e0 = -1.758
    ea = 0.110
    ez = -0.061
    ea2 = -0.023
    M0 = 11.548
    Ma = -1.297
    Mz = -0.026
    lge = e0 + (ea*(a-1.)+ez*z)*v + ea2*(a-1.)
    lgM = M0 + (Ma*(a-1.)+Mz*z)*v
    return lge+lgM + f_RP17(lgMv-lgM,a) - f_RP17(0.,a)
def v_RP17(a):
    """
    Auxiliary function for lgMs_RP17.
    """
    return np.exp(-4.*a**2)
def f_RP17(x,a):
    r"""
    Auxiliary function for lgMs_RP17.
    
    Note that RP+17 use 10**( - alpha*x) while B+13 used 10**( +alpha*x).
    """
    a0 = 1.975
    aa = 0.714
    az = 0.042
    d0 = 3.390
    da = -0.472
    dz = -0.931
    g0 = 0.498
    ga = -0.157
    gz = 0.0
    v = v_RP17(a)
    z = 1./a-1.
    alpha = a0 + (aa*(a-1.)+az*z)*v
    delta = d0 + (da*(a-1.)+dz*z)*v
    gamma = g0 + (ga*(a-1.)+gz*z)*v
    return delta*(np.log10(1.+np.exp(x)))**gamma/(1.+np.exp(10**(-x)))-\
        np.log10(1.+10**( - alpha*x))

#---halo-response patterns

def slope(X,choice='NIHAO'):
    """
    Logarithmic halo density slope at 0.01 R_vir, as a function of the 
    stellar-to-halo-mass ratio X, based on simulation results.
    
    Syntax:
    
        slope(X,choice='NIHAO')
        
    where
    
        X: M_star / M_vir (float or array)
        choice: choice of halo response -- 
            'NIHAO' (default, Tollet+16, mimicking strong core formation)
            'APOSTLE' (Bose+19, mimicking no core formation)
    """
    if choice=='NIHAO':
        s0 = X / 8.77e-3
        s1 = X / 9.44e-5
        return np.log10(26.49*(1.+s1)**(-0.85) + s0**1.66) + 0.158
    elif choice=='APOSTLE':
        s0 = X / 8.77e-3
        return np.log10( 20. + s0**1.66 ) + 0.158 

def c2c2DMO(X,choice='NIHAO'):
    """
    The ratio between the baryon-influenced concentration c_-2 and the 
    dark-matter-only c_-2, as a function of the stellar-to-halo-mass
    ratio, based on simulation results. 
    
    Syntax:
    
        c2c2DMO(X,choice='NIHAO')
        
    where
    
        X: M_star / M_vir (float or array)
        choice: choice of halo response -- 
            'NIHAO' (default, Tollet+16, mimicking strong core formation)
            'APOSTLE' (Bose+19, mimicking no core formation)
    """
    if choice=='NIHAO':
        #return 1. + 227.*X**1.45 - 0.567*X**0.131 # <<< Freundlich+20
        return 1.2 + 227.*X**1.45 - X**0.131 # <<< test
    elif choice=='APOSTLE':
        return 1. + 227.*X**1.45
        
#---concentration-mass-redshift relations

def c2_Zhao09(Mv,t):
    """
    Halo concentration from the mass assembly history, using the Zhao+09
    relation.
    
    Syntax:
    
        c2_Zhao09(Mv,t)
        
    where
    
        Mv: main-branch virial mass history [M_sun] (array)
        t: the time series of the main-branch mass history (array of the
            same size as Mv)
    
    Note that we need Mv and t in reverse chronological order, i.e., in 
    decreasing order, such that Mv[0] and t[0] is the instantaneous halo
    mass and time.
    
    Note that Mv is the Bryan and Norman 98 M_vir.
    
    Return:
        
        halo concentration R_vir / r_-2 (float)
    """
    idx = aux.FindNearestIndex(Mv,0.04*Mv[0])
    return 4.*(1.+(t[0]/(3.75*t[idx]))**8.4)**0.125
    
def lgc2_DM14(Mv,z=0.):
    r"""
    Halo concentration given virial mass and redshift, using the 
    fitting formula from Dutton & Maccio 14 (eqs.10-11)
    
    Syntax:
    
        lgc2_DM14(Mv,z=0.)
    
    where 
    
        Mv: virial mass, M_200c [M_sun] (float or array)
        z: redshift (float or array of the same size as Mv,default=0.)
        
    Note that this is for M_200c, for the BN98 M_vir, use DM14 eqs.12-13
    instead. 
    
    Note that the parameters are for the Planck(2013) cosmology.
    
    Return:
    
        log of halo concentration c_-2 = R_200c / r_-2 (float or array)
    """
    # <<< concentration from NFW fit
    #a = 0.026*z - 0.101 # 
    #b = 0.520 + (0.905-0.520) * np.exp(-0.617* z**1.21)
    # <<< concentration from Einasto fit
    a = 0.029*z - 0.130
    b = 0.459 + (0.977-0.459) * np.exp(-0.490* z**1.303) 
    return a*np.log10(Mv*cfg.h/10**12.)+b

def c2_DK15(Mv,z=0.,n=-2):
    """
    Halo concentration from Diemer & Kravtsov 15 (eq.9).
    
    Syntax:
    
        c2_DK15(Mv,z)
        
    where
    
        Mv: virial mass, M_200c [M_sun] (float or array)
        z: redshift (float or array of the same size as Mv,default=0.)
        n: logarithmic slope of power spectrum (default=-2 or -2.5 for
            typical values of LCDM, but more accurately, use the power
            spectrum to calculate n)
    
    Note that this is for M_200c.
    Note that the parameters are for the median relation
    
    Return:
        
        halo concentration R_200c / r_-2 (float)
    """
    cmin = 6.58 + 1.37*n
    vmin = 6.82 + 1.42*n
    v = co.nu(Mv,z,**cfg.cosmo)
    fac = v / vmin
    return 0.5*cmin*(fac**(-1.12)+fac**1.69)

#---halo contraction model

@jit(nopython=True,cache=True)
def AC_Hernquist_jit_general(x,mi,fb,rb,A,w,damping=0.5):
    """
    Solves the adiabatic contraction equation iteratively for a Hernquist 
    baryon profile using the Gnedin+04 model with numerical damping.
    
    This function computes the contracted dark matter halo radius by solving
    the adiabatic contraction equation through fixed-point iteration with
    damping for numerical stability.
    
    Syntax:
    
        AC_Hernquist_jit_general(x,mi,fb,rb,A,w,damping)
        
    where
    
        x: dimensionless initial radius normalized by virial radius, r/r_vir 
           (array)
        mi: dimensionless initial total mass enclosed at orbit-averaged radius,
            M_i(<r_average)/M_vir (array, same length as x)
        fb: baryon fraction, M_b/M_vir (float)
        rb: scale radius of the Hernquist profile normalized by virial radius, 
            a/r_vir (float)
        A: coefficient in the orbit-averaged radius relation (float)
        w: power-law index in the orbit-averaged radius relation (float)
        damping: damping factor for iteration stability (0<damping<=1, 
                 default=0.5)
                 
    Return:
    
        y: dimensionless contraction factor, r_final/r_initial (array of the 
           same length as x)
    """
    # Initialization
    y_init = x.copy()
    fdm = 1-fb

    # Fixed-point iteration to solve contraction equation
    while True:
        xyave = A*(x*y_init)**w
        mb = fb*(xyave*(1.+rb)/(xyave+rb))**2.
        y_temp = 1/(fdm+mb/mi)
        y = (1-damping)*y_init + damping*y_temp # damping method
        rela_diff = np.abs(y_init-y)/y_init # relative difference for convergence check
        if np.all(rela_diff < 1e-6): break
        y_init = y

    return y    

def contra_Hernquist_jit_general(r,h,d,A=0.85,w=0.8):
    """
    Returns contracted halo profile given baryon profile and initial halo 
    profile, following the model of Gnedin+04.
    
    Syntax:
    
        contra(r,h,d)
        
    where
    
        r: initial radii at which we evaluate the mass profile [kpc]
            (array)
        h: initial halo profile (object of classes defined in 
            profiles.py)
        d: baryon profile (object of the Hernquist class as defined in
            profiles.py)
        A: coefficient in the relation between the orbit-averaged radius 
            of a particle that is currently in a shell and the instant
            radius of the shell: <r>/r_vir = A (r/r_vir)^w 
            (default=0.85)
        w: power-law index in the relation between the orbit-averaged
            radius and instant radius (default=0.8)
            
    Note that there is halo-to-halo variation in the values of A and w,
    which is discussed in Gnedin+11. Here we ignore the halo-to-halo 
    variation and adopt the fiducial values A=0.85 and w=0.8 as in 
    Gnedin+04.
    
    Note that the input halo object "h" is for the total mass profile,
    which includes an initial baryon mass distribution that is assumed
    to be self-similar to the initial DM profile, i.e.,
    
        M_dm,i = (1-f_b) M_i(r)
        M_b,i = f_b M_i(r)
            
    Return:
    
        contracted radii, r_f [kpc] (array of the same length as r)
        enclosed DM mass at r_f [M_sun] (array of the same length as r) 
    """
    # prepare variables
    Mv = h.Mh
    rv = h.rh
    Mb = d.Mb
    r0 = d.r0
    fb = Mb/Mv
    fdm = 1-fb
    xb = r0/rv
    x = r/rv
    rave = A*x**w*rv
    mi = h.M(rave)/Mv

    # solve contraction equation
    y = AC_Hernquist_jit_general(x,mi,fb,xb,A,w)
    rf = y*r

    return rf, fdm*h.M(r)


@jit(nopython=True,cache=True)
def AC_exp_jit_general(x,mi,fb,rb,A,w,damping=0.5):
    """
    Solves the adiabatic contraction equation iteratively for an exponential 
    disk baryon profile using the Gnedin+04 model with numerical damping.
    
    This function computes the contracted dark matter halo radius by solving
    the adiabatic contraction equation through fixed-point iteration with
    damping for numerical stability.
    
    Syntax:
    
        AC_exp_jit_general(x,mi,fb,rb,A,w,damping)
        
    where
    
        x: dimensionless initial radius normalized by virial radius, r/r_vir 
           (array)
        mi: dimensionless initial total mass enclosed at orbit-averaged radius,
            M_i(<r_average)/M_vir (array, same length as x)
        fb: cosmic baryon fraction, M_b/M_vir (float)
        rb: scale radius of the exponential disk normalized by virial radius, 
            a/r_vir (float)
        A: coefficient in the orbit-averaged radius relation (float)
        w: power-law index in the orbit-averaged radius relation (float)
        damping: damping factor for iteration stability (0<damping<=1, 
                 default=0.5)
                 
    Return:
    
        y: dimensionless contraction factor, r_final/r_initial (array of the 
           same length as x)
    """
    # Initialization
    y_init = x.copy()
    fdm = 1-fb

    # Fixed-point iteration to solve contraction equation
    while True:
        xyaverb = A*(x*y_init)**w/rb
        mb = fb*(1.-(1.+xyaverb)*np.exp(-xyaverb))
        y_temp = 1/(fdm+mb/mi)
        y = (1-damping)*y_init + damping*y_temp # damping method
        rela_diff = np.abs(y_init-y)/y_init # relative difference for convergence check
        if np.all(rela_diff < 1e-6): break
        y_init = y
    return y    

def contra_exp_jit_general(r,h,d,A=0.85,w=0.8):
    """
    Returns contracted halo profile given baryon profile and initial halo 
    profile, following the model of Gnedin+04.
    
    Similar to "contra_Hernquist_jit_general", but here we assume the final baryon
    distribution to be an exponential disk, instead of a spherical 
    Hernquist profile
    
    Syntax:
    
        contra(r,h,d)
        
    where
    
        r: initial radii at which we evaluate the mass profile [kpc]
            (array)
        h: initial halo profile (object of any halo class as defined
            in profiles.py)
        d: baryon profile (object of the exponential class as defined in
            profiles.py)
        A: coefficient in the relation between the orbit-averaged radius 
            of a particle that is currently in a shell and the instant
            radius of the shell: <r>/r_vir = A (r/r_vir)^w 
            (default=0.85)
        w: power-law index in the relation between the orbit-averaged
            radius and instant radius (default=0.8)
            
    Note that there is halo-to-halo variation in the values of A and w,
    which is discussed in Gnedin+11. Here we ignore the halo-to-halo 
    variation and adopt the fiducial values A=0.85 and w=0.8 as in 
    Gnedin+04.
    
    Note that the input halo object "h" is for the total mass profile,
    which includes an initial baryon mass distribution that is assumed
    to be self-similar to the initial DM profile, i.e.,
    
        M_dm,i = (1-f_b) M_i(r)
        M_b,i = f_b M_i(r)
            
    Return:
    
        contracted radii, r_f [kpc] (array of the same length as r)
        enclosed DM mass at r_f [M_sun] (array of the same length as r) 
    """
    # prepare variables
    Mv = h.Mh
    rv = h.rh
    Mb = d.Mb
    rb = d.a
    fb = Mb/Mv
    fdm = 1.-fb
    xb = rb/rv
    x = r/rv
    rave = A*x**w*rv
    mi = h.M(rave)/Mv
    
    # solve contraction equation
    y = AC_exp_jit_general(x,mi,fb,xb,A,w)
    rf = y*r

    return rf, fdm*h.M(r)

def contra_general(r,h,d,A=0.85,w=0.8):
    """
    Returns contracted halo profile given baryon profile and initial halo 
    profile, following the model of Gnedin+04.
    
    Syntax:
    
        contra(r,h,d)
        
    where
    
        r: initial radii at which we evaluate the mass profile [kpc]
            (array)
        h: initial halo profile (object of any halo class as defined
            in profiles.py)
        d: baryon profile (object of the Hernquist class as defined in
            profiles.py)
        A: coefficient in the relation between the orbit-averaged radius 
            of a particle that is currently in a shell and the instant
            radius of the shell: <r>/r_vir = A (r/r_vir)^w 
            (default=0.85)
        w: power-law index in the relation between the orbit-averaged
            radius and instant radius (default=0.8)
            
    Note that there is halo-to-halo variation in the values of A and w,
    which is discussed in Gnedin+11. Here we ignore the halo-to-halo 
    variation and adopt the fiducial values A=0.85 and w=0.8 as in 
    Gnedin+04.
    
    Note that the input halo object "h" is for the total mass profile,
    which includes an initial baryon mass distribution that is assumed
    to be self-similar to the initial DM profile, i.e.,
    
        M_dm,i = (1-f_b) M_i(r)
        M_b,i = f_b M_i(r)
            
    Return:
    
        the contracted DM profile (object of the Dekel class as defined
            in profiles.py) 
        contracted radii, r_f [kpc] (array of the same length as r)
        enclosed DM mass at r_f [M_sun] (array of the same length as r) 
    """
    # contract
    if isinstance(d,pr.Hernquist):
        rf,Mdmf = contra_Hernquist_jit_general(r,h,d,A,w)
    elif isinstance(d,pr.exp):
        rf,Mdmf = contra_exp_jit_general(r,h,d,A,w)

    # fit contracted profile
    params = Parameters()
    params.add('Mv', value=(1.-d.Mb/h.Mh)*h.Mh, vary=False)
    params.add('c', value=h.ch,min=1.,max=100.)
    params.add('a', value=1.,min=-2.,max=2.)
    out = minimize(fobj_Dekel, params, args=(rf,Mdmf,h.Deltah,h.z)) 
    MvD = out.params['Mv'].value
    cD = out.params['c'].value
    aD = out.params['a'].value

    return pr.Dekel(MvD,cD,aD),rf,Mdmf 

def fobj_Dekel(p, xdata, ydata, Delta, z):
    """
    Auxiliary function for "contra" -- objective function for fitting
    a Dekel+ profile to the contracted halo
    """
    h = pr.Dekel(p['Mv'].value,p['c'].value,p['a'].value,Delta=Delta,z=z)
    ymodel = h.M(xdata)
    return (ydata - ymodel) / ydata

def contra_general_Minterp(r,h,d,A=0.85,w=0.8):
    """
    Returns the interpolation of contracted halo mass profile 
    given baryon profile and initial halo profile, following the model of Gnedin+04.
    Notice that the input radii should extend to virial radius, and r[0] should be  
    in the inner region, so that the domain of defination of interpolation function can
    cover 0 <=r<= rv.

    Syntax:
    
        contra(r,h,d)
        
    where
    
        r: initial radii at which we evaluate the mass profile [kpc]
            (array)
        h: initial halo profile (object of any halo class as defined
            in profiles.py)
        d: baryon profile (object of the Hernquist class as defined in
            profiles.py)
        A: coefficient in the relation between the orbit-averaged radius 
            of a particle that is currently in a shell and the instant
            radius of the shell: <r>/r_vir = A (r/r_vir)^w 
            (default=0.85)
        w: power-law index in the relation between the orbit-averaged
            radius and instant radius (default=0.8)
            
    Note that there is halo-to-halo variation in the values of A and w,
    which is discussed in Gnedin+11. Here we ignore the halo-to-halo 
    variation and adopt the fiducial values A=0.85 and w=0.8 as in 
    Gnedin+04.
    
    Note that the input halo object "h" is for the total mass profile,
    which includes an initial baryon mass distribution that is assumed
    to be self-similar to the initial DM profile, i.e.,
    
        M_dm,i = (1-f_b) M_i(r)
        M_b,i = f_b M_i(r)
            
    Return:
    
        The interpolation function for contracted DM mass profile (with input r and output M,
        both are numpy array objects.)
        contracted radii, r_f [kpc] (array of the same length as r)
        enclosed DM mass at r_f [M_sun] (array of the same length as r) 
    """
    # contract
    if isinstance(d,pr.Hernquist):
        rf,Mdmf = contra_Hernquist_jit_general(r,h,d,A,w)
    elif isinstance(d,pr.exp):
        rf,Mdmf = contra_exp_jit_general(r,h,d,A,w)

    # interp contracted profile
    rf_interp,Mdmf_interp = np.insert(rf,0,0),np.insert(Mdmf,0,0) # extend to r=0
    Minterp = interpolate.interp1d(rf_interp,Mdmf_interp,kind='cubic',fill_value='extrapolate')
    return Minterp,rf,Mdmf 

def r1_direct_contra(r1_init,halo_init,disk,dlgr=1e-3):
    """
    Calculate contracted halo properties at a given initial radius.
    
    Syntax:
    
        r1_direct_contra(r1_init, halo_init, disk, dlgr=0.001)
        
    where
    
        r1_init: initial radius [kpc] (scalar)
        halo_init: initial halo mass profile (object of NFW class)
        disk: galactic disk mass profile (object of Hernquist or exp class)
        dlgr: logarithmic radius interval for numerical differentiation (default=0.001)
        
    Return:
    
        r1_f: contracted radius [kpc] (scalar)
        rho_r1: DM density at r1_f after contraction [M_sun/kpc^3] (scalar)
        M_r1: enclosed DM mass within r1_f after contraction [M_sun] (scalar)
    """
    # Create three radii for numerical differentiation
    r_array = np.array([r1_init*10**(-dlgr),r1_init,r1_init*10**dlgr])

    # Apply appropriate contraction model
    if isinstance(disk,pr.Hernquist):
        rf,Mdmf = contra_Hernquist_jit_general(r_array,halo_init,disk)
    elif isinstance(disk,pr.exp):
        rf,Mdmf = contra_exp_jit_general(r_array,halo_init,disk)
    r1_f,M_r1 = rf[1],Mdmf[1]

    # Compute DM density using finite differences
    rho_r1 = (Mdmf[-1]-Mdmf[0]) / (4*np.pi*r1_f**2) / (rf[-1]-rf[0])

    return r1_f,rho_r1,M_r1

def rd_MMW98(halo_init,spin,jd,md,damping=0.5):
    '''
    Compute the size of an exponential disk embedded in a halo 
    using the Mo, Mao & White (1998) model with adiabatic contraction.
    
    This function iteratively solves for the scale radius of an exponential
    disk that forms within a dark matter halo, accounting for angular momentum
    conservation and adiabatic contraction of the halo.
    
    Syntax:
    
        rd_MMW98(halo_init, spin, jd, md, damping=0.5)
        
    where
    
        halo_init: initial halo object defined in profiles.py, representing 
                   the dark matter halo before disk formation 
                   (object)
        spin: dimensionless spin parameter (lambda) of the halo (float)
        jd: fraction of halo angular momentum acquired by the disk, 
             J_disk / J_halo (float)
        md: fraction of halo mass in the disk, M_disk / M_halo (float)
        damping: the coefficient in damping method (default=0.5)
            
    Return:
    
        rd: scale radius of the exponential disk [kpc] (float)
        disk: the exponential disk (object defined in profiles.py)
        halo_contra: the contracted halo (object of Dekel class defined in profiles.py)
    '''
    def E_integ_func(r):
        # integration function for computing total energy
        return halo_init.rho(r)*halo_init.M(r)*r
    
    def integ_func(u,Mcontra_interp):
        # function for integration in the iteration below, u = r/rd
        r = u*rd_init
        Vc_dm = np.sqrt(cfg.G*Mcontra_interp(r)/r)
        Vc_disk = disk.Vcirc(r)
        Vcirc = np.sqrt(Vc_dm**2+Vc_disk**2)
        return Vcirc*u**2*np.exp(-u)
    
    # constant variables
    Mh = halo_init.Mh
    Md = Mh*md
    rh = halo_init.rh
    Vvir = halo_init.Vcirc(rh)
    Etot = 0.5*(-cfg.FourPiG*quad(E_integ_func,0,rh)[0]) 
    Esing = -0.5*cfg.G*Mh**2/rh # total energy of a singular isothermal sphere profile
    fc = Etot/Esing

    # modification of total energy from different profile
    r_full = np.logspace(-3,np.log10(5.*rh),500) # used in adiabatic contraction

    # setting of damping method
    threshold = 1e-4/(1.-damping)

    # variables in iteration
    Niter = 0
    rd_init = rh*0.03 #initial guess
    rd_temp = rd_init

    # iteration to solve for rd
    while Niter < 1 or np.abs(rd_init-rd_temp)/rd_init > threshold:
        Niter += 1
        rd_init = rd_temp

        # define disk and halo
        disk = pr.exp(Md,rd_init)
        Mcontra_interp = contra_general_Minterp(r_full,halo_init,disk)[0]

        # grid used in integration
        r_cut = min(10.*rd_init,rh) # upper limit of integration
        integ = quad(integ_func,0,r_cut/rd_init,args=(Mcontra_interp))[0]

        # iteration
        rd0 = np.sqrt(2)*jd/md*spin*rh/np.sqrt(fc)*Vvir/integ
        rd_temp = (1-damping)*rd0 + damping*rd_init
        #print(Niter,rd_temp,np.abs(rd_init-rd_temp)/rd_init)

    return rd_temp,disk,Mcontra_interp

def rd_MMW98_SIDM(halo_init,spin,jd,md,p,pmerge_interp,rd_guess,damping=0.5,N=50000):
    """
    Calculate scale radius of disk in SIDM halo using iterative method in MMW98 as before.
    
    Syntax:
    
        rd_MMW98_SIDM(halo_init, spin, jd, md, p, pmerge_interp, damping=0.5, N=50000)
        
    where
    
        halo_init: initial halo profile (object of halo class)
        spin: dimensionless spin parameter (scalar)
        jd: disk angular momentum fraction (scalar)
        md: disk mass fraction (scalar)  
        p: age of the system [Gyr] (scalar)
        pmerge_interp: interpolation function for merger time (callable)
        rd_guess: initial position of rd for iteration (scalar)
        damping: damping factor for iteration stability (default=0.5)
        N: number of radial grid points for integration (default=50000)
        
    Return:
    
        rd: disk scale radius [kpc] (scalar)
        sol[0]: core radius of SIDM halo [kpc] (scalar)
    """
    # Extract basic halo properties
    Mh = halo_init.Mh
    Md = halo_init.Mh*md
    rh = halo_init.rh
    Vvir = halo_init.Vcirc(rh)
    Esing = -0.5*cfg.G*Mh**2/rh # total energy of a singular isothermal sphere profile

    # Set up for the iteration
    r_full = np.logspace(-3,np.log10(rh),500)
    rd_init = rd_guess
    rd_temp = rd_init
    threshold = 1e-4/(1.-damping)
    Niter = 0

    # Main iteration loop
    while Niter < 1 or np.abs(rd_init-rd_temp)/rd_init > threshold:
        # iteration
        Niter += 1
        rd_init = rd_temp
        
        # compute pmerge and AC
        pmerge = pmerge_interp(rd_init)
        disk = pr.exp(Md,rd_init)
        Mcontra_interp = contra_general_Minterp(r_full,halo_init,disk)[0]

        # Set up integration grid
        r_cut = min(10.*rd_init,rh) # upper limit of integration
        r_array = np.linspace(0,r_cut,N) # used in integration
        r_array = r_array[1:]

        # Determine SIDM core solution based on system age (p=tage*sigmamx)
        if p < pmerge:
            r1,rhoCDM1,MCDM1 = pr.r1(halo_init,sigmamx=1.,tage=p,disk=disk)
            solid = 0 # flag for low-density solution
        elif p < 2.*pmerge: 
            p1 = 2.*pmerge - p
            r1,rhoCDM1,MCDM1 = pr.r1(halo_init,sigmamx=1.,tage=p1,disk=disk)
            solid = 1 # flag for high-density solution
        else: 
            print('tage is larger than 2*tmerge!!!')
            break

        # Run Jeans model and calculate dark matter circular velocity profile
        if r_cut <= r1: # Single integration domain
            sol1,sol2,merge,find2sol = pr.stitchSIDMcore2_exp(r1,rhoCDM1,MCDM1,halo_init,disk,r_array,N)
            if find2sol: # Both solutions exist, choose based on solid flag
                sol = sol1 if solid == 0 else sol2
            elif merge: # System has merged, both solutions are physically invalid
                print('WARNING: system have merged! using low-density solution')
                sol = sol1
            else: # Only one solution available, which is always given as sol1 in pr.stitchSIDMcore2
                if solid == 0: 
                    sol = sol1
                else:
                    if sol2[-1] > 1e-5: 
                        print('WARNING: delta2_high > 1e-5, using low-density solution')
                        sol = sol1
                    else:
                        sol = sol2
            Vc_dm = sol[-3]
        else: # Two integration domains: core region and outer region
            sol1,sol2,merge,find2sol = pr.stitchSIDMcore2_exp(r1,rhoCDM1,MCDM1,halo_init,disk,r_array[r_array<=r1],N)
            if find2sol: # Both solutions exist, choose based on solid flag
                sol = sol1 if solid == 0 else sol2
            elif merge: # System has merged, both solutions are physically invalid
                print('WARNING: system have merged! using low-density solution')
                sol = sol1
            else: # Only one solution available, which is always given as sol1 in pr.stitchSIDMcore2
                if solid == 0: 
                    sol = sol1
                else:
                    if sol2[-1] > 1e-5: 
                        print('WARNING: delta2_high > 1e-5, using low-density solution')
                        sol = sol1
                    else:
                        sol = sol2
            Vc_dm1 = sol[-3] # DM circular velocity in SIDM inner halo
            Vc_dm2 = np.sqrt(cfg.G*Mcontra_interp(r_array[r_array>r1])/r_array[r_array>r1]) # in CDM outskirt
            Vc_dm = np.concatenate((Vc_dm1,Vc_dm2))
        
        # Calculate total circular velocity (DM + disk)
        Vc_disk = disk.Vcirc(r_array)
        Vcirc = np.sqrt(Vc_dm**2+Vc_disk**2)
        r_array = np.append(0,r_array)
        Vcirc = np.append(0,Vcirc)

        # Calculate energy correction factor fc
        # inner region properties
        r_in,rhodm_in,Vdm_in = sol[4],sol[2],sol[3] 
        r_in,rhodm_in,Vdm_in = np.append(0,r_in),np.append(0,rhodm_in),np.append(0,Vdm_in)
        Mdm_in = Vdm_in**2/cfg.G*r_in
        # outer region properties
        r_out = np.linspace(r1,rh,N)
        Mdm_out = Mcontra_interp(r_out)
        rhodm_out = rho_GivenInterpMass(r_out,Mcontra_interp)
        # total energy
        Etot = -0.5 * cfg.FourPiG * (
            simpson(rhodm_in*Mdm_in*r_in,x=r_in) + 
            simpson(rhodm_out*Mdm_out*r_out,x=r_out)
            )
        fc = Etot/Esing

        # Calculate new disk scale radius
        u_array = r_array/rd_init
        integ_array = Vcirc*u_array**2*np.exp(-u_array) # integrand for angular momentum
        integ = simpson(integ_array,x=u_array) # perform integration
        rd0 = np.sqrt(2.)*jd/md*spin*rh/np.sqrt(fc)*Vvir/integ
        rd_temp = (1.-damping)*rd0 + damping*rd_init # damped update
        print("Niter=%d, rd_temp=%f, rd_temp/rd_init=%f, solid=%d"%(Niter,rd_temp,rd_temp/rd_init,solid))
        print("lgrhodm0_1=%.8f,lgrhodm0_2 = %.8f,delta2_1=%.2e,delta2_2=%.2e"%(np.log10(sol1[0]),np.log10(sol2[0]),sol1[-1],sol2[-1]))
    
    rd = rd_init
    return rd,sol[0]

def rho_GivenInterpMass(r, Minterp, dlgr=1e-3):
    """
    Calculate the density profile at given radii using an interpolated mass profile.
    
    This function computes the density by numerically differentiating the 
    interpolated cumulative mass profile M(r). The density is obtained from 
    the mass derivative using the relation:
        rho(r) = (1/4*pi*r²) * dM/dr
    
    Syntax:
        rho_GivenInterpMass(r, Minterp, dlgr=1e-3)
        
    where
        r: radii at which to evaluate the density [kpc] (float or array)
        Minterp: interpolated mass profile function M(r) that returns 
                 cumulative mass [M_sun] for given radius (callable)
        dlgr: logarithmic step size for numerical differentiation 
              (float, default=1e-3)
    
    Return:
        Density at radii r [M_sun/kpc^3] (float or array)
    """
    # Calculate radii for finite difference: r2 > r > r0
    r2,r0 = r * 10.**(dlgr), r * 10.**(-dlgr)           
    
    # Evaluate the interpolated mass profile at the two radii
    M2,M0 = Minterp(r2), Minterp(r0)    
    
    # Compute density using finite difference approximation
    return (M2 - M0) / (cfg.FourPi * r**2) / (r2 - r0)

def rho_GivenEnclosedMass(r,M):
    """
    Compute density profile given enclosed mass profile.
    This can be used to compute the non-parametric density profile 
    upon input of the non-parametric enclosed mass profile given
    by the adiabatic contraction calculation from e.g.,
        contra_exp_jit_general
    
    Syntax:
    
        rho_GivenEnclosedMass(r,M)
        
    where
    
        r: radii [kpc] (array)
        M: enclosed mass [M_sun] (array)
        
    Return:
        
        rho: density [M_sun/kpc^3] (array of the same length as r)
    """
    rtmp = np.append(0.,r[:-1]) 
    Mtmp = np.append(0.,M[:-1])
    dr = r - rtmp
    dM = M - Mtmp
    rave = (rtmp + r) * 0.5
    return rave, dM / (cfg.FourPi * rave**2 * dr)

def Vc_tot(r,d,EnclosedHaloMass,Mbh):
    """
    Total circular velocity given exponential disk and adiabatically 
    contracted dark-matter halo.

    Syntax:

        Vc_tot(r,d,EnclosedHaloMass,Mbh)

    where

        r: radius at which we evaluate [kpc] (float or array)
        d: exponential disk object (defined in profiles.py)
        EnclosedHaloMass: a function that returns the enclosed mass
            given radius [M_sun] (e.g., a function made by rd_MMW98)
        Mbh: black hole mass [Msun] (float)
    
    Return:
        total circular velocity [kpc/Gyr]
    """
    return np.sqrt( d.Vcirc(r)**2 + cfg.G*EnclosedHaloMass(r)/r + cfg.G*Mbh/r)

def dlnVcdlnr(r,d,EnclosedHaloMass,Mbh):
    """
    Compute the logarithmic slope of the total circular velocity
    profile.

    Syntax:

        dlnVcdlnr(r,d,EnclosedHaloMass,Mbh)

    where

        r: radius at which we evaluate the slope [kpc] (float or array)
        d: exponential disk object (defined in profiles.py)
        EnclosedHaloMass: a function that returns the enclosed mass
            given radius [M_sun] (e.g., a function made by rd_MMW98)
        Mbh: black hole mass [Msun] (float)
    
    Return: 
        the logarithmic slope of the total circular velocity 
        (float or array, same length as r)
    """
    r1 = r * (1.+cfg.eps)
    r2 = r * (1.-cfg.eps)
    Vc1 = Vc_tot(r1,d,EnclosedHaloMass,Mbh)
    Vc2 = Vc_tot(r2,d,EnclosedHaloMass,Mbh)
    return np.log(Vc1/Vc2) / np.log(r1/r2)

def kappa(r,d,EnclosedHaloMass,Mbh):
    """
    Compute the epicyclic frequency

        kappa = 2 Omega^2 [1 + d ln V_c / d ln r] 

    Syntax:

        kappa(r,d,EnclosedHaloMass,Mbh)

    where

        r: radius at which we evaluate the slope [kpc] (float or array)
        d: exponential disk object (defined in profiles.py)
        EnclosedHaloMass: a function that returns the enclosed mass
            given radius [M_sun] (e.g., a function made by rd_MMW98)
        Mbh: black hole mass [Msun] (float)

    Return:

        the epicyclic frequency [1/Gyr]
    """
    Omega = Vc_tot(r,d,EnclosedHaloMass,Mbh) / r
    slope = dlnVcdlnr(r,d,EnclosedHaloMass,Mbh)
    return np.sqrt(2.) * Omega * np.sqrt(1.+slope)

def ToomreQ(r,sigmad,d,EnclosedHaloMass,Mbh):
    """
    Compute the Toomre Q at radius r given 
     - gas disk's velocity dispersion
     - the exponential disk profile of the disk
     - the mass profile of the host dark-matter halo

    Syntax:

        ToomreQ(r,sigmad,d,EnclosedHaloMass,Mbh)

    where

        r: radius at which we evaluate the slope [kpc] (float or array)
        sigmad: 1D velocity dispersion of the gas disk [kpc/Gyr]
            (same type and length as r if considering a temperature profile, 
            or a single float value for isothermal disk)
        d: exponential disk object (defined in profiles.py)
        EnclosedHaloMass: a function that returns the enclosed mass
            given radius [M_sun] (e.g., a function made by rd_MMW98)
        Mbh: black hole mass [Msun] (float)

    Return: Q (same type and length as r)
    """
    return sigmad * kappa(r,d,EnclosedHaloMass,Mbh) / (np.pi * cfg.G * d.Sigma(r))