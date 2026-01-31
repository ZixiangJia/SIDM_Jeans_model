####################### potential well classes ##########################

# Arthur Fangzhou Jiang (2016, HUJI) --- original version

# Arthur Fangzhou Jiang (2019, HUJI and UCSC) --- revisions
# 
# - Dekel+ profile added some time earlier than 2019
# - support of velocity dispersion profile and thus dynamical friction 
#   (DF) for Dekel+ profile 
# - improvement of DF implementation: 
#   1. now compute the common part of the Chandrasekhar formula only once
#      for each profile class (as opposed to three times in each of the 
#      force component .FR, .Fphi, and .Fz);
#   2. removed satelite-profile-dependent CoulombLogChoice, and now 
#      CoulombLogChoices only depend on satellite mass m and host 
#      potential 

# Arthur Fangzhou Jiang 2022, Caltech, Carnegie --- revision:
# - added core-stalling of dynamical friction following Petts+15
#   To this end, 
#   1. add attribute .rhalf to each profile class
#   2. revise the Coulomb log in the function fDF below

# Zixiang Jia 2025, Peking University --- revision:
# - added functions that are used for computing effective SIDM 
#   cross section (Yang+22)
# - added a new halo class DC14 (Di Cintio+14)
# - improved the isotherma Jeans model (Jia+26, Jiang+23)

#########################################################################

import config as cfg # for global variables
import cosmo as co # for cosmology related functions
import galhalo as gh 

from scipy import integrate
from scipy.integrate import quad
import math
import numpy as np
from scipy.optimize import brentq,minimize
from scipy.integrate import quad,odeint
from scipy.special import erf,gamma,gammainc,gammaincc 
from scipy.interpolate import RegularGridInterpolator
import pickle
from scipy.interpolate import interp1d

def gamma_lower(a,x):
    """
    Non-normalized lower incomplete gamma function
    
        integrate t^(a-1) exp(-t) from t=0 to t=x
        
    Syntax:
        
        gamma_lower(a,x)
    """
    return gamma(a)*gammainc(a,x)
def gamma_upper(a,x):
    """
    Non-normalized upper incomplete gamma function
    
        integrate t^(a-1) exp(-t) from t=x to t=infty
        
    Syntax:
        
        gamma_upper(a,x)
    """
    return gamma(a)*gammaincc(a,x)

#########################################################################

#---
class NFW(object):
    """
    Class that implements the Navarro, Frenk, & White (1997) profile:

        rho(R,z) = rho_crit * delta_char / [(r/r_s) * (1+r/r_s)^2]
                 = rho_0 / [(r/r_s) * (1+r/r_s)^2]
    
    in a cylindrical frame (R,phi,z), where 
    
        r = sqrt(R^2 + z^2)
        r_s: scale radius, at which d ln rho(r) / d ln(r) = -2
        rho_crit: critical density of the Universe
        delta_char = Delta_halo / 3 * c^3 / f(c), where c = R_vir / r_s 
            is the concentration parameter
    
    Syntax:
    
        halo = NFW(M,c,Delta=200.,z=0.)
        
    where 
    
        M: halo mass [M_sun], where halo is defined as spherical 
            overdensity of Delta times critical density (float) 
        c: halo concentration (float)
        Delta: average overdensity of the halo, in multiples of the 
            critical density of the Universe (float)
            (default 200.)
        z: redshift (float) (default 0.)
    
    Attributes:
    
        .Mh: halo mass [M_sun]
        .ch: halo concentration
        .Deltah: spherical overdensity wrt instantaneous critical density
        .z: redshift
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: average density of halo [M_sun kpc^-3]
        .rh: halo radius within which density is Delta times rhoc [kpc]
        .rs: scale radius [kpc]
        .rmax: radius at which maximum circular velocity is reached [kpc]
        .Vmax: maximum circular velocity [kpc/Gyr]
        .s001: logarithmic density slope at 0.01 halo radius
        .rhalf: half-mass radius [kpc]
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at radius r=sqrt(R^2+z^2)
        .s(R,z=0.): logarithmic density slope at radius r=sqrt(R^2+z^2)
        .M(R,z=0.): mass [M_sun] enclosed in radius r=sqrt(R^2+z^2)
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2+z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at radius r=sqrt(R^2+z^2)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
        .Vesc(R,z=0.): escape velocity [jpc/Gyr] at radius r=sqrt(R^2+z^2)
        .sigma(R,z=0.): vel. disp. [kpc/Gyr] at radius r=sqrt(R^2+z^2)      
    
    HISTORY: Arthur Fangzhou Jiang (2016-10-24, HUJI)
             Arthur Fangzhou Jiang (2016-10-30, HUJI)
             Arthur Fangzhou Jiang (2019-08-24, HUJI)
    """
    def __init__(self,M,c,Delta=200.,z=0.):
        """
        Initialize NFW profile.
        
        Syntax:
        
            halo = NFW(M,c,Delta=200.,z=0.)
        
        where
        
            M: halo mass [M_sun] (float), 
            c: halo concentration (float),        
            Delta: spherical overdensity with respect to the critical 
                density of the universe (default is 200.)         
            z: redshift (float)
        """
        # input attributes
        self.Mh = M 
        self.ch = c
        self.Deltah = Delta
        self.z = z
        #
        # derived attributes
        self.rhoc = co.rhoc(z,cfg.h,cfg.Om,cfg.OL)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3.*self.Mh / (cfg.FourPi*self.rhoh))**(1./3.)
        self.rs = self.rh / self.ch
        self.rmax = self.rs * 2.163
        self.rho0 = self.rhoc*self.Deltah/3.*self.ch**3./self.f(self.ch)
        self.Phi0 = -cfg.FourPiG*self.rho0*self.rs**2.     
        self.Vmax = self.Vcirc(self.rmax)
        self.s001 = self.s(0.01*self.rh)
        self.rhalf = self.rmax # <<< to be updated, use rmax temporarily
    def f(self,x):
        """
        Auxiliary method for NFW profile: f(x) = ln(1+x) - x/(1+x)
    
        Syntax:
    
            .f(x)
        
        where
        
            x: dimensionless radius r/r_s (float or array)
        """
        return np.log(1.+x) - x/(1.+x) 
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rho(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        return self.rho0 / (x * (1.+x)**2.)
    def s(self,R,z=0.):
        """
        Logarithmic density slope 
            
            - d ln rho / d ln r 
        
        at radius r = sqrt(R^2 + z^2). 
        
        Syntax:
        
            .s(R,z=0.)

        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        return 1. + 2*x / (1.+x)
    def M(self,R,z=0.):
        """
        Mass [M_sun] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)       
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        return cfg.FourPi*self.rho0*self.rs**3. * self.f(x)
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
        
        where 
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)   
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        return 3.*self.rho0 * self.f(x)/x**3.
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).

        Syntax:
        
            .tdyn(R,z=0.)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)     
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z))
    def Phi(self,R,z=0.):
        """
        Potential [(kpc/Gyr)^2] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Phi(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        return self.Phi0 * np.log(1.+x)/x
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs   
        fac = self.Phi0 * (self.f(x)/x) / r**2.
        return fac*R, fac*0., fac*z
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return np.sqrt(r*-self.fgrav(r,0.)[0])
    def Vesc(self,R,z=0.):
        """
        Escape velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vesc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        return np.sqrt(-2*self.Phi(R,z))
    def sigma(self,R,z=0.):
        """
        Velocity dispersion [kpc/Gyr] at radius r = sqrt(R^2 + z^2), 
        assuming isotropic velicity dispersion tensor, and following the 
        Zentner & Bullock (2003) fitting function:
        
            sigma(x) = V_max 1.4393 x^0.345 / (1 + 1.1756 x^0.725)
            
        where x = r/r_s.
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
        return self.Vmax*1.4393*x**0.354/(1.+1.1756*x**0.725)
    def sigma_accurate(self,R,z=0.,beta=0.):
        """
        Velocity dispersion [kpc/Gyr].
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
            beta: anisotropy parameter (default=0., i.e., isotropic)
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
        if isinstance(x,list) or isinstance(x,np.ndarray):
            I = []
            for xx in x:
                II = quad(self.dIdx_sigma, xx, np.inf,args=(beta,))[0]
                I.append(II)
            I = np.array(I)
        else:
            I = quad(self.dIdx_sigma, x, np.inf,args=(beta,))[0]
        f = self.f(x)
        sigmasqr = -self.Phi0 / x**(2.*beta-1) *(1.+x)**2 * I
        return np.sqrt(sigmasqr)
    def dIdx_sigma(self,x,beta):
        """
        Integrand for the integral in the velocity dispersion.
        """
        f = self.f(x)
        return x**(2.*beta-3.) * f / (1.+x)**2
    def dlnsigmasqrdlnr_accurate(self,R,z=0.,beta=0.):
        """
        d ln sigma^2 / d ln r
        """
        r = np.sqrt(R**2.+z**2.)
        r1 = r * (1.+cfg.eps)
        r2 = r * (1.-cfg.eps)
        y1 = np.log(self.sigma_accurate(r1))
        y2 = np.log(self.sigma_accurate(r2))
        return (y1-y2)/(r1-r2)

class Burkert(object):
    """
    Class that implements the Burkert (1995) profile:

        rho(R,z) = rho_0 / [(1+x)(1+x^2)], x = r/r_s
    
    in a cylindrical frame (R,phi,z), where 
    
        r = sqrt(R^2 + z^2)
        r_s: scale radius, at which d ln rho(r) / d ln(r) = -2
        rho_crit: critical density of the Universe
        delta_char = Delta_halo / 3 * c^3 / f(c), where c = R_vir / r_s 
            is the concentration parameter
    
    Syntax:
    
        halo = Burkert(M,c,Delta=200.,z=0.)
        
    where 
    
        M: halo mass [M_sun], where halo is defined as spherical 
            overdensity of Delta times critical density (float) 
        c: Burkert concentration, R_vir/r_s (float)
        Delta: average overdensity of the halo, in multiples of the 
            critical density of the Universe (float)
            (default 200.)
        z: redshift (float) (default 0.)
    
    Attributes:
    
        .Mh: halo mass [M_sun]
        .ch: halo concentration
        .Deltah: spherical overdensity wrt instantaneous critical density
        .z: redshift
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: average density of halo [M_sun kpc^-3]
        .rh: halo radius within which density is Delta times rhoc [kpc]
        .rs: scale radius [kpc]
        .rmax: radius at which maximum circular velocity is reached [kpc]
        .rho0: central density [M_sun kpc^-3]
        .Vmax: maximum circualr velocity [kpc/Gyr]
        .s001: logarithmic density slope at 0.01 halo radius
        .rhalf: half-mass radius [kpc]
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at radius r=sqrt(R^2+z^2)
        .s(R,z=0.): logarithmic density slope at radius r=sqrt(R^2+z^2)
        .M(R,z=0.): mass [M_sun] enclosed in radius r=sqrt(R^2+z^2)
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2+z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at radius r=sqrt(R^2+z^2)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
        .sigma(R,z=0.): vel. disp. [kpc/Gyr] at radius r=sqrt(R^2+z^2)      
    
    HISTORY: Arthur Fangzhou Jiang (2020-07-29, Caltech)
    """
    def __init__(self,M,c,Delta=200.,z=0.):
        """
        Initialize Burkert profile.
        
        Syntax:
        
            halo = Burkert(M,c,Delta=200.,z=0.)
        
        where
        
            M: halo mass [M_sun] (float), 
            c: halo concentration (float),        
            Delta: spherical overdensity with respect to the critical 
                density of the universe (default is 200.)         
            z: redshift (float)
        """
        # input attributes
        self.Mh = M 
        self.ch = c
        self.Deltah = Delta
        self.z = z
        #
        # derived attributes
        self.rhoc = co.rhoc(z,cfg.h,cfg.Om,cfg.OL)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3.*self.Mh / (cfg.FourPi*self.rhoh))**(1./3.)
        self.rs = self.rh / self.ch
        self.rho0=self.Mh/cfg.TwoPi/self.rh**3/self.f(self.ch)*self.ch**3
        self.rmax = 3.24 * self.rs
        self.Vmax = self.Vcirc(self.rmax)    
        self.s001 = self.s(0.01*self.rh)
        self.rhalf = self.rmax # <<< to be updated, use rmax temporarily
    def f(self,x):
        """
        Auxiliary method for NFW profile: 
            
            f(x) = 0.5 ln(1+x^2) + ln(1+x) - arctan(x)
    
        Syntax:
    
            .f(x)
        
        where
        
            x: dimensionless radius r/r_s (float or array)
        """
        return 0.5*np.log(1.+x**2)+np.log(1.+x)-np.arctan(x)
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rho(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        return self.rho0 / ((1.+x) * (1.+x**2))
    def s(self,R,z=0.):
        """
        Logarithmic density slope 
            
            - d ln rho / d ln r 
        
        at radius r = sqrt(R^2 + z^2). 
        
        Syntax:
        
            .s(R,z=0.)

        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        xsqr = x**2
        return x / (1.+x) + 2*xsqr / (1.+xsqr)
    def M(self,R,z=0.):
        """
        Mass [M_sun] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)       
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        return cfg.TwoPi*self.rho0*self.rs**3 * self.f(x)
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
        
        where 
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)   
        """
        r = np.sqrt(R**2.+z**2.)
        return self.M(r)/(cfg.FourPiOverThree*r**3)
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).

        Syntax:
        
            .tdyn(R,z=0.)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)     
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z))
    def Phi(self,R,z=0.):
        """
        Potential [(kpc/Gyr)^2] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Phi(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        return - np.pi*cfg.G*self.rho0*self.rs**2 /x * (-np.pi+\
            2.*(1.+x)*np.arctan(1./x)+2.*(1.+x)*np.log(1.+x)+\
            (1.-x)*np.log(1.+x**2))
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs   
        fac = np.pi*cfg.G*self.rho0*self.rs/x**2 * \
            (np.pi-2.*np.arctan(1./x)-2.*np.log(1.+x)-np.log(1.+x**2))/r
        return fac*R, fac*0., fac*z
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return np.sqrt(r*-self.fgrav(r,0.)[0])
    def Vesc(self,R,z=0.):
        """
        Escape velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vesc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        return np.sqrt(-2*self.Phi(R,z))
    def sigma(self,R,z=0.):
        """
        Velocity dispersion [kpc/Gyr] assuming isotropic velicity 
        dispersion tensor ... 
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
        return self.Vmax * 0.299*np.exp(x**0.17) / (1.+0.286*x**0.797)
    def sigma_accurate(self,R,z=0.,beta=0.):
        """
        Velocity dispersion [kpc/Gyr].
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
            beta: anisotropy parameter (default=0., i.e., isotropic)
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        if isinstance(x,list) or isinstance(x,np.ndarray):
            I = []
            for xx in x:
                II = quad(self.dIdx_sigma, xx, np.inf,args=(beta,))[0]
                I.append(II)
            I = np.array(I)
        else:
            I = quad(self.dIdx_sigma, x, np.inf,args=(beta,))[0]
        sigmasqr = cfg.TwoPiG*self.rho0*(1.+x)*(1.+x**2)*self.rs**2 / \
            x**(2.*beta) * I
        return np.sqrt(sigmasqr)
    def dIdx_sigma(self,x,beta):
        """
        Integrand for the integral in the velocity dispersion of Burkert.
        """
        return (0.5*np.log(1.+x**2)+np.log(1.+x)-np.arctan(x))* \
            x**(2.*beta-2.) / ((1.+x)* (1.+x**2))

class coreNFW(object):
    """
    Class that implements the "coreNFW" profile (Read+2016):

        M(r) = M_NFW(r) g(y)
        rho(r) = rho_NFW(r) g(y) + [1-g(y)^2] M_NFW(r) / (4 pi r^2 r_c)
    
    in a cylindrical frame (R,phi,z), where 
    
        r = sqrt(R^2 + z^2)
        y = r / r_c with r_c a core radius, usually smaller than r_s
        g(y) = tanh(y)
    
    Syntax:
    
        halo = coreNFW(M,c,rc,Delta=200.,z=0.)
        
    where 
    
        M: halo mass [M_sun], where halo is defined as spherical 
            overdensity of Delta times critical density (float) 
        c: NFW halo concentration (float)
        rc: core radius [kpc]
        Delta: average overdensity of the halo, in multiples of the 
            critical density of the Universe (float)
            (default 200.)
        z: redshift (float) (default 0.)
    
    Attributes:
    
        .Mh: halo mass [M_sun]
        .ch: halo concentration
        .Deltah: spherical overdensity wrt instantaneous critical density
        .z: redshift
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: average density of halo [M_sun kpc^-3]
        .rh: halo radius within which density is Delta times rhoc [kpc]
        .rs: scale radius [kpc]
        .rc: core radius [kpc]
        .rmax: radius at which maximum circular velocity is reached [kpc]
        .Vmax: maximum circular velocity [kpc/Gyr]
        .s001: logarithmic density slope at 0.01 halo radius
        .rhalf: half-mass radius [kpc]
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at radius r=sqrt(R^2+z^2)
        .s(R,z=0.): logarithmic density slope at radius r=sqrt(R^2+z^2)
        .M(R,z=0.): mass [M_sun] enclosed in radius r=sqrt(R^2+z^2)
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2+z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at radius r=sqrt(R^2+z^2)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
        .sigma(R,z=0.): vel. disp. [kpc/Gyr] at radius r=sqrt(R^2+z^2)      
    
    HISTORY: Arthur Fangzhou Jiang (2021-03-11, Caltech)
    """
    def __init__(self,M,c,rc,Delta=200.,z=0.):
        """
        Initialize coreNFW profile.
        
        Syntax:
        
            halo = coreNFW(M,c,rc,Delta=200.,z=0.)
        
        where
        
            M: halo mass [M_sun] (float), 
            c: halo concentration (float),  
            rc: core radius [kpc]      
            Delta: spherical overdensity with respect to the critical 
                density of the universe (default is 200.)         
            z: redshift (float)
        """
        # input attributes
        self.Mh = M 
        self.ch = c
        self.rc = rc
        self.Deltah = Delta
        self.z = z
        #
        # derived attributes
        self.rhoc = co.rhoc(z,cfg.h,cfg.Om,cfg.OL)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3.*self.Mh / (cfg.FourPi*self.rhoh))**(1./3.)
        self.rs = self.rh / self.ch
        self.xc = self.rc / self.rs
        self.rmax = self.rs * 2.163 # accurate only if r_c < r_s
        self.rho0 = self.rhoc*self.Deltah/3.*self.ch**3./self.f(self.ch)
        self.Phi0 = -cfg.FourPiG*self.rho0*self.rs**2.     
        self.Vmax = self.Vcirc(self.rmax) # accurate only if r_c < r_s
        self.s001 = self.s(0.01*self.rh)
        self.rhalf = self.rmax # <<< to be updated, use rmax temporarily
    def f(self,x):
        """
        Auxiliary method for NFW profile: f(x) = ln(1+x) - x/(1+x)
    
        Syntax:
    
            .f(x)
        
        where
        
            x: dimensionless radius r/r_s (float or array)
        """
        return np.log(1.+x) - x/(1.+x) 
    def g(self,y):
        """
        Auxiliary method for coreNFW profile: f(y) = tanh(y)
    
        Syntax:
    
            .g(y)
        
        where
        
            y: dimensionless radius r/r_c (float or array)
        """
        return np.tanh(y)
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rho(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        y = r / self.rc
        f = self.f(x)
        g = self.g(y)
        return self.rho0*(g/(x *(1.+x)**2)+(1.-g**2)*f/(self.xc*x**2.))
    def s(self,R,z=0.):
        """
        Logarithmic density slope 
            
            - d ln rho / d ln r 
        
        at radius r = sqrt(R^2 + z^2). 
        
        Syntax:
        
            .s(R,z=0.)

        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        r1 = r*(1.+cfg.eps)
        r2 = r*(1.-cfg.eps)
        rho1 = self.rho(r1)
        rho2 = self.rho(r2)
        return - np.log(rho1/rho2) / np.log(r1/r2)
    def M(self,R,z=0.):
        """
        Mass [M_sun] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)       
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        y = r/self.rc
        return cfg.FourPi*self.rho0*self.rs**3. * self.f(x) * self.g(y)
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
        
        where 
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)   
        """
        r = np.sqrt(R**2.+z**2.)
        return self.M(r)/(cfg.FourPiOverThree*r**3)
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).

        Syntax:
        
            .tdyn(R,z=0.)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)     
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z))
    def Phi_accurate(self,R,z=0.):
        """
        Potential [(kpc/Gyr)^2] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Phi_accurate(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
        Phi1 = - cfg.G * self.M(r)/r
        if isinstance(x,list) or isinstance(x,np.ndarray):
            if len(x.shape)==1: # i.e., if the input R array is 1D
                I = []
                for xx in x:
                    #II = quad(self.dIdx_Phi, xx, self.ch,)[0]
                    II = quad(self.dIdx_Phi, xx, np.inf,)[0]
                    I.append(II)
                I = np.array(I)
            elif len(x.shape)==2: # i.e., if the input R array is 2D
                I = np.empty(x.shape)
                for i,xx in enumerate(x):
                    for j,xxx in enumerate(xx):
                        #II = quad(self.dIdx_Phi, xxx, self.ch,)[0]
                        II = quad(self.dIdx_Phi, xxx, np.inf,)[0]
                        I[i,j] = II
        else:
            I = quad(self.dIdx_Phi, x, self.ch,)[0]
        Phi2 = self.Phi0 * I
        return Phi1 + Phi2
    def dIdx_Phi(self,x):
        """
        Integrand for the second-term of the potential of coreNFW.
        """
        f = self.f(x)
        g = self.g(x/self.xc)
        return g/(1.+x)**2 + (1.-g**2)*f/(x*self.xc)
    def Phi(self,R,z=0.):
        """
        Approximation expression for gravitational potential 
        [(kpc/Gyr)^2] at radius r = sqrt(R^2 + z^2):
        
            Phi(x) ~ [1+s(x)] Phi_core + s(x) Phi_NFW(x) 
        
        where
        
            x = r/r_s
            Phi_core ~ Phi_NFW(0.8 x_c) is the flat potential in the core
            Phi_NFW(x) is the NFW potential
        
        For exact (but slower evaluation of the) potential, use 
        .Phi_accurate 
            
        Syntax:
        
            .Phi(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
        xtrans = 0.8 * self.xc # an empirical transition scale
        s = 0.5 + 0.5*np.tanh((x-xtrans)/xtrans) # transition function
        Phic = self.Phi0 * np.log(1.+xtrans)/xtrans
        PhiNFW = self.Phi0 * np.log(1.+x)/x
        return (1.-s)*Phic + s*PhiNFW
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs   
        y = r / self.rc
        fac = self.Phi0 * (self.g(y)*self.f(x)/x) / r**2.
        return fac*R, fac*0., fac*z
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return np.sqrt(r*-self.fgrav(r,0.)[0])
    def Vesc(self,R,z=0.):
        """
        Escape velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vesc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        return np.sqrt(-2*self.Phi(R,z))
    def rmax_accurate(self):
        """
        Radius [kpc] at which maximum circular velocity is reached, which
        is given by the root of:
        
            g(y)/(1+x)^2 - f(x)g(y)/x^2 + [1-g(y)^2]f(x)/(x x_c) = 0
            
        where
        
            x = r/r_s
            x_c = r_c / r_s
            y = r/r_c = x/x_c
            g(y) = tanh(y)
        """
        xmax = brentq(self.Findxmax, 0.1, 10., args=(),
            xtol=0.001,rtol=1e-5,maxiter=1000)
        return xmax * self.rs
    def Findxmax(self,x):
        """
        The left-hand-side function for finding x_max = r_max / r_s.
        """
        f = self.f(x)
        g = self.g(x/self.xc)
        return g/(1.+x)**2-f*g/x**2+(1.-g**2)*f/x/self.xc
    def sigma_accurate(self,R,z=0.,beta=0.):
        """
        Velocity dispersion [kpc/Gyr].
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
            beta: anisotropy parameter (default=0., i.e., isotropic)
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
        y = r / self.rc
        if isinstance(x,list) or isinstance(x,np.ndarray):
            I = []
            for xx in x:
                II = quad(self.dIdx_sigma, xx, np.inf,args=(beta,))[0]
                I.append(II)
            I = np.array(I)
        else:
            I = quad(self.dIdx_sigma, x, np.inf,args=(beta,))[0]
        f = self.f(x)
        g = self.g(y)
        A = g/(x*(1.+x)**2)+(1.-g**2)*f/(self.xc*x**2)
        sigmasqr = -self.Phi0 / x**(2.*beta) / A * I
        return np.sqrt(sigmasqr)
    def dIdx_sigma(self,x,beta):
        """
        Integrand for the integral in the velocity dispersion of Burkert.
        """
        f = self.f(x)
        g = self.g(x/self.xc)
        return (g/(x*(1.+x)**2)+(1.-g**2)*f/(self.xc*x**2))*\
            f*g*x**(2.*beta-2.)
    def sigma(self,R,z=0.):
        """
        Approximation expression for velocity dispersion [kpc/Gyr] at 
        radius r = sqrt(R^2 + z^2), assuming isotropic velicity.
        
        For exact (but slower evaluation of the) dispersion, use
        .sigma_accurate
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
        xtrans = 0.8 * self.xc # an empirical transition scale
        s = 0.5 + 0.5*np.tanh((x-xtrans)/xtrans) # transition function
        sigmac = self.Vmax*1.4393*xtrans**0.354/(1.+1.1756*xtrans**0.725)
        sigmaNFW = self.Vmax*1.4393*x**0.354/(1.+1.1756*x**0.725)
        return (1.-s)*sigmac + s*sigmaNFW
        
class Dekel(object):
    """
    Class that implements Dekel+ (2016) profile:

        rho(R,z)=rho_0/[(r/r_s)^alpha * (1+(r/r_s)^(1/2))^(2(3.5-alpha))]
        M(R,z) = M_vir * g(x,alpha) / g(c,alpha) 
               = M_vir [chi(x)/chi(c)]^[2(3-alpha)] 
        
    in a cylindrical frame (R,phi,z), where   
        
        r = sqrt(R^2 + z^2)
        c: concentration parameter
        r_s: scale radius, i.e., R_vir / c, where R_vir is the virial
            radius. (Note that unlike NFW or Einasto, where r_s is r_-2,
            here r_s is not r_-2, but 2.25 r_-2 / (2-alpha)^2 ) 
        alpha: shape parameter, the innermost logarithmic density slope
        x = r/r_s
        chi(x) = x^(1/2) / (1+x^(1/2))
        g(x,alpha) = chi(x)^[2(3-alpha)]
        rho_0: normalization density, 
            rho_0 = c^3 (3-alpha) Delta rho_crit / [3 g(c,alpha)]
        M_vir: virial mass, related to rho_0 via 
            M_vir = 4 pi rho_0 r_s^3 g(c,alpha) / (3-alpha)
    
    Syntax:
    
        halo = Dekel(M,c,alpha,Delta=200.,z=0.)
        
    where 
    
        M: halo mass [M_sun], where halo is defined as spherical 
            overdensity of Delta times critical density (float)
        c: concentration (float)
        alpha: shape parameter, the inner most log density slope (float)
            (there are singularities for computing potential and 
            velocity dispersion, at 1+i/4, for i=0,1,2,...,8)
        Delta: multiples of the critical density of the Universe 
            (default 200.)
        z: redshift (float) (default 0.)
    
    Attributes:
    
        .Mh: halo mass [M_sun]
        .ch: halo concentration
        .alphah: halo innermost logarithmic density slope
        .Deltah: spherical overdensity wrt instantaneous critical density
        .z: redshift
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: average density of halo [M_sun kpc^-3]
        .rh: halo radius within which density is Delta times rhoc [kpc]
        .rs: scale radius [kpc]
        .rmax: radius at which maximum circular velocity is reached [kpc]
        .c2: concentration parameter determined by rh/r2
        .Vmax: maximum circular velocity [kpc/Gyr]
        .s001: logarithmic density slope at 0.01 halo radius
        .sh: the old name for s001, kept for compatibility purposes
        .rhalf: half-mass radius [kpc]
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at radius r =sqrt(R^2 + z^2)
        .s(R,z=0.): logarithmic density slope at radius r=sqrt(R^2+z^2)
        .M(R,z=0.): mass [M_sun] enclosed in radius r = sqrt(R^2 + z^2)
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2 + z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at radius r=sqrt(R^2+z^2)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
        .sigma(R,z=0.): vel. disp. [kpc/Gyr] at radius r=sqrt(R^2+z^2)  
    
    HISTORY: Arthur Fangzhou Jiang (2018-03-23, UCSC)
             Arthur Fangzhou Jiang (2019-08-26, UCSC)
    """
    def __init__(self,M,c,alpha,Delta=200.,z=0.):
        """
        Initialize Dekel+ profile.
        
        Syntax:
            
            halo = Dekel(M,c,alpha,Delta=200.,Om=0.3,h=0.7)
        
        where
        
            M: halo mass [M_sun] (float)
            c: halo concentration (float)
            alpha: innermost logarithmic density slope (float)
            Delta: spherical overdensity with respect to the critical 
                density of the universe (default 200.)  
            z: redshift (float) (default 0.)
        """
        # input attributes
        self.Mh = M 
        self.ch = c
        self.alphah = alpha
        self.Deltah = Delta
        self.z = z
        #
        # derived attributes
        self.rhoc = co.rhoc(z,cfg.h,cfg.Om,cfg.OL)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3.*self.Mh / (cfg.FourPi*self.rhoh))**(1./3.)
        self.rs = self.rh / self.ch
        self.rmax = self.rs * (2.-self.alphah)**2.
        self.r2 = self.rmax/2.25
        self.c2 = self.rh/self.r2
        self.rho0 = self.rhoc*self.Deltah * (3.-self.alphah)/3. * \
            self.ch**3./self.g(self.ch)
        self.Phi0 = -cfg.FourPiG*self.rho0*self.rs**2. / \
            ((3.-self.alphah)*(2.-self.alphah)*(2.*(2.-self.alphah)+1))
        self.Vmax = self.Vcirc(self.rmax)
        self.sh = (self.alphah+0.35*self.ch**0.5) / (1.+0.1*self.ch**0.5)
        self.s001 = self.s(0.01*self.rh)
        self.rhalf = self.rmax # <<< to be updated, use rmax temporarily
    def X(self,x):
        """
        Auxiliary function for Dekel+ profile
    
            chi := x^0.5 / 1+x^0.5  
    
        Syntax:
    
            .X(x)
    
        where 
        
            x: dimensionless radius r/r_s (float or array)
        """
        u = x**0.5
        return u/(1.+u)
    def g(self,x):
        """
        Auxiliary function for Dekel+ profile
    
            g(x;alpha):= chi^[2(3-alpha)], with chi := x^0.5 / 1+x^0.5  
    
        Syntax:
    
            .g(x)
    
        where 
    
            x: dimensionless radius r/r_s (float or array)
        """
        return self.X(x)**(2.*(3.-self.alphah))
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .rho(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        return self.rho0 / ( x**self.alphah * \
            (1.+x**0.5)**(2.*(3.5-self.alphah)) )
    def s(self,R,z=0.):
        """
        Logarithmic density slope 
            
            - d ln rho / d ln r 
        
        at radius r = sqrt(R^2 + z^2). 
        
        Syntax:
        
            .s(R,z=0.)

        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        sqrtx = np.sqrt(x) 
        return (self.alphah+3.5*sqrtx) / (1.+sqrtx) 
    def M(self,R,z=0.):
        """
        Mass [M_sun] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        return self.Mh * self.g(x)/self.g(self.ch)
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        r = np.sqrt(R**2.+z**2.)
        return 3./(cfg.FourPi*r**3.) * self.M(R,z) # <<< to be replaced
            # by a simpler analytic expression, but this one is good
            # enough for now. 
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .tdyn(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z))
    def Phi(self,R,z=0.):
        """
        Potential [(kpc/Gyr)^2] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Phi(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        X = self.X(x)
        Vvsqr = self.Vcirc(self.rh)**2.
        u = 2*(2.-self.alphah)
        return -Vvsqr * 2*self.ch / self.g(self.ch) * \
            ((1.-X**u)/u - (1.-X**(u+1))/(u+1))
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs    
        fac = ((2.-self.alphah)*(2.*(2.-self.alphah)+1.)) * \
            self.Phi0 * (self.g(x)/x) / r**2.
        return fac*R, fac*0., fac*z 
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return np.sqrt(r*-self.fgrav(r,0.)[0])
    def Vesc(self,R,z=0.):
        """
        Escape velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vesc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        return np.sqrt(-2*self.Phi(R,z))
    def sigma(self,R,z=0.):
        """
        Velocity dispersion [kpc/Gyr] at radius r = sqrt(R^2 + z^2), 
        assuming isotropic velicity dispersion tensor, following what I 
        derived based on Zhao (1996) eq.19 and eqs.A9-A11:
        
            sigma^2(r) = 2 Vv^2 c/g(c,alpha) x^3.5 / chi^(2(3.5-alpha))
                Sum_{i=0}^{i=8} (-1)^i 8! (1-chi^(4(1-alpha)+i)) / 
                ( i! (8-i)! (4(1-alpha)+i) ).
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        X = self.X(x)
        Vvsqr = self.Vcirc(self.rh)**2.
        u = 4*(1.-self.alphah)
        sigmasqr = 2.*Vvsqr*self.ch/self.g(self.ch) \
            *(x**3.5)/(X**(2.*(3.5-self.alphah))) \
            * ( (1.-X**u)/u - 8.*(1.-X**(u+1.))/(u+1.) \
            + 28.*(1.-X**(u+2.))/(u+2.) - 56.*(1.-X**(u+3.))/(u+3.) \
            + 70.*(1.-X**(u+4.))/(u+4.) - 56.*(1.-X**(u+5.))/(u+5.) \
            + 28.*(1.-X**(u+6.))/(u+6.) - 8.*(1.-X**(u+7.))/(u+7.) \
            + (1.-X**(u+8.))/(u+8.) )
        return np.sqrt(sigmasqr)  
        
class Einasto(object):
    """
    Class that implements Einasto (1969a,b) profile:

        rho(R,z) = rho_s exp{ - d(n) [ (r/r_s)^(1/n) - 1 ] }
        
    in a cylindrical frame (R,phi,z), where
        
        r = sqrt(R^2 + z^2)
        r_s: scale radius, at which d ln rho(r) / d ln(r) = -2
        n: Einasto shape index, the inverse of which, alpha=1/n, is also 
            called the Einasto shape parameter
        d(n): geometric constant which makes that r_s to be a 
            characteristic radius. (We usually use d(n)=2n, 
            which makes r_s = r_-2, i.e., the radius at which 
            d ln rho(r) / d ln(r) = -2.) 
        rho_s: density at r=r_s  (Since r_s=r_-2, rho_s is also denoted 
            as rho_-2.)
    
    See Retana-Montenegro+2012 for details.
    
    Syntax:
    
        halo = Einasto(M,c,alpha,Delta=200.,Om=0.3,h=0.7)

    where 
    
        M: halo mass [M_sun], where halo is defined as spherical 
            overdensity of Delta times critical density (float)
        c: concentration (float)
        alpha: shape (float)
        Delta: multiples of the critical density of the Universe (float)
            (default 200.)
        z: redshift (float) (default 0.)
    
    Attributes:
    
        .Mh: halo mass [M_sun]
        .ch: halo concentration (halo radius / scale radius)
        .alphah: halo shape
        .Deltah: spherical overdensity wrt instantaneous critical density
        .z: redshift
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: halo's average density [M_sun kpc^-3]
        .rh: halo radius [kpc], within which density is Deltah times rhoc
        .rs: scale radius [kpc], at which log density slope is -2
        .nh: inverse of shape paramter (1 / alphah)
        .hh: scale length [kpc], defined as rs / (2/alphah)^(1/alphah)
        .rho0: halo's central density [M_sun kpc^-3]
        .xmax: dimensionless rmax, defined as (rmax/hh)^alphah
        .rmax: radius [kpc] at which maximum circular velocity is reached 
        .Vmax: maximum circular velocity [kpc/Gyr]
        .Mtot: total mass [M_sun] of the Einasto profile integrated to 
            infinity
        .s001: logarithmic density slope at 0.01 halo radius
        .rhalf: half-mass radius [kpc]
        
    Methods:

        .rho(R,z=0.): density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2)
        .M(R,z=0.): mass [M_sun] within radius r = sqrt(R^2 + z^2)
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2 + z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at radius r=sqrt(R^2+z^2)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circular velocity [kpc/Gyr] at radius r
        .sigma(R,z=0.): velocity dispersion [kpc/Gyr] at radius r=R
    
    HISTORY: Arthur Fangzhou Jiang (2016-11-08, HUJI)
             Arthur Fangzhou Jiang (2019-09-10, HUJI)
    """
    def __init__(self,M,c,alpha,Delta=200.,z=0.):
        """
        Initialize Einasto profile.
        
        Syntax:
            
            halo = Einasto(M,c,alpha,Delta=200.,Om=0.3,h=0.7)
        
        where
        
            M: halo mass [M_sun] (float)
            c: halo concentration (float)
            alpha: Einasto shape (float)
            Delta: spherical overdensity with respect to the critical 
                density of the universe (default 200.)  
            z: redshift (float) (default 0.)
        """
        # input attributes
        self.Mh = M 
        self.ch = c
        self.alphah = alpha
        self.Deltah = Delta
        self.z = z
        #
        # derived attributes
        self.rhoc = co.rhoc(z,cfg.h,cfg.Om,cfg.OL)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3.*self.Mh / (cfg.FourPi*self.rhoh))**(1./3.)
        self.rs = self.rh / self.ch
        self.nh = 1./self.alphah
        self.hh = self.rs / (2.*self.nh)**self.nh
        self.xh = (self.rh / self.hh)**self.alphah
        self.rho0 = self.Mh / (cfg.FourPi * self.hh**3. * self.nh * \
            gamma_lower(3.*self.nh,self.xh)) 
        self.rmax = 1.715*self.alphah**(-0.00183) * \
            (self.alphah+0.0817)**(-0.179488) * self.rs
        self.xmax = (self.rmax / self.hh)**self.alphah
        self.Mtot = cfg.FourPi * self.rho0 * self.hh**3. * self.nh \
            * gamma(3.*self.nh)
        self.GMtot = cfg.G*self.Mtot 
        self.Vmax = self.Vcirc(self.rmax)
        self.s001 = self.s(0.01*self.rh)
        self.rhalf = self.rmax # <<< to be updated, use rmax temporarily
    def x(self,r):
        """
        Auxilary method that computes dimensionless radius 
        
            x := (r/h)^alpha 
        
        at radius r = sqrt(R^2+z^2).
            
        Syntax:
        
            .x(r)
            
        where
            
            r = sqrt(R^2 + z^2) [kpc] (float or array)
        """
        return (r / self.hh)**self.alphah 
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .rho(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.)
        return self.rho0 * np.exp(-self.x(r))
    def s(self,R,z=0.):
        """
        Logarithmic density slope 
            
            - d ln rho / d ln r 
        
        at radius r = sqrt(R^2 + z^2). 
        
        Syntax:
        
            .s(R,z=0.)

        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        return self.x(r) / self.nh
    def M(self,R,z=0.):
        """
        Mass [M_sun] within radius r = sqrt(R^2 + z^2).
        
            M(R,z) = M_tot gamma(3n,x)/Gamma(3n)
            
        where x = (r/h)^alpha; h = r_s/(2n)^n; and gamma(a,x)/Gamma(a) 
        together is the normalized lower incomplete gamma function, as 
        can be computed directly by scipy.special.gammainc.    
            
        Syntax:
        
            .M(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return self.Mtot * gammainc(3.*self.nh,self.x(r))
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        r = np.sqrt(R**2.+z**2.)
        return 3./(cfg.FourPi*r**3.) * self.M(R,z)
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .tdyn(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z))
    def Phi(self,R,z=0.):
        """
        Potential [(kpc/Gyr)^2] at radius r = sqrt(R^2 + z^2):
        
            Phi = - G M_tot/[h Gamma(3n)] [gamma(3n,x)/x^n + Gamma(2n,x)]
        
        where x = (r/h)^alpha; h = r_s/(2n)^n; gamma(a,x)/Gamma(a) 
        together is the normalized lower incomplete gamma function; 
        Gamma(a,x) is the non-normalized upper incomplete gamma function;
        and Gamma(a) is the (complete) gamma function.
            
        Syntax:
        
            .Phi(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        r = np.sqrt(R**2.+z**2.)
        x = self.x(r)
        a = 3.*self.nh
        return - self.GMtot/self.hh * ( gammainc(a,x)/x**self.nh \
            + gamma_upper(2*self.nh,x)/gamma(a) )
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        r = np.sqrt(R**2.+z**2.)
        fac = - self.GMtot * gammainc(3.*self.nh,self.x(r)) / r**3.
        return fac*R, fac*0., fac*z
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return np.sqrt(self.GMtot/r *gammainc(3.*self.nh,self.x(r)))
    def Vesc(self,R,z=0.):
        """
        Escape velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vesc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        return np.sqrt(-2*self.Phi(R,z))
    def sigma(self,R,z=0.):
        """
        Velocity dispersion [kpc/Gyr] assuming isotropic velicity 
        dispersion tensor ... 
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = self.x(r)
        if isinstance(x,list) or isinstance(x,np.ndarray):
            I = []
            for xx in x:
                II = quad(dIdx_Einasto, xx, np.inf, args=(self.nh,),)[0]
                I.append(II)
            I = np.array(I)
        else:
            I = quad(dIdx_Einasto, x, np.inf, args=(self.nh,),)[0]
        sigmasqr = self.GMtot/self.hh*self.nh*np.exp(x) * I 
        return np.sqrt(sigmasqr)
def dIdx_Einasto(x,n):
    """
    Integrand for the integral in the velocity dispersion of Einasto.
    """
    return gammainc(3.*n,x)/(np.exp(x)*x**(n+1.))
        
class MN(object):
    """
    Class that implements Miyamoto & Nagai (1975) disk profile:

        Phi(R,z) = - G M / sqrt{ R^2 + [ a + sqrt(z^2+b^2) ]^2 }
    
    in a cylindrical frame (R,phi,z), where 
    
        M: disk mass
        a: scalelength 
        b: scaleheight.
    
    Syntax:
    
        disk = MN(M,a,b)

    where
    
        M: disk mass [M_sun] (float)
        a: scalelength [kpc] (float)
        b: scaleheight [kpc] (float)

    Attributes:
    
        .Md: disk mass [M_sun]
        .Mh: the same as .Md, but for the purpose of keeping the notation
            for "host" mass consistent with the other profile classes
        .a: disk scalelength [kpc]
        .b: disk scaleheight [kpc]
        .rhalf: half-mass radius [kpc]
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at (R,z) 
        .M(R,z=0.): mass [M_sun] within radius r=sqrt(R^2+z^2),
            defined as M(r) = r Vcirc(r,z=0)^2 / G
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2 + z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at (R,z)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at (R,z=0), defined as
            sqrt(R d Phi(R,z=0.)/ d R)
        .sigma(R,z=0.): velocity dispersion [kpc/Gyr] at (R,z) 
    
    HISTORY: Arthur Fangzhou Jiang (2016-11-03, HUJI)
             Arthur Fangzhou Jiang (2019-08-27, UCSC)
    """
    def __init__(self,M,a,b):
        """
        Initialize Miyamoto-Nagai disk profile
        
        Syntax:
        
            disk = MN(M,a,b)
        
        where 
        
            M: disk mass [M_sun], 
            a: disk scalelength [kpc]
            b: disk scaleheight [kpc]
        """
        # input attributes
        self.Md = M
        self.Mh = self.Md 
        self.a = a
        self.b = b
        #
        # supportive attributes repeatedly used by following methods
        self.GMd = cfg.G * self.Md
        self.rhalf = np.sqrt(a*b) # <<< to be updated
    def s1sqr(self,z):
        """
        Auxilary method that computes (a + sqrt(z^2+b^2))^2 at height z.
        
        Syntax:
        
            .s1sqr(z) 
        
        where
        
            z: z-coordinate [kpc] (float or array)
        """
        return (self.a + self.s2(z))**2.
    def s2(self,z):
        """
        Auxilary method that computes zeta = sqrt(z^2+b^2) at height z.
            
        Syntax:
        
            .s2(z)
        
        where 
        
            z: z-coordinate [kpc] (float or array)
        """
        return np.sqrt(z**2. + self.b**2)             
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at (R,z).
            
        Syntax:
        
            .rho(R,z)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        Rsqr = R**2.
        s1sqr = self.s1sqr(z)
        s2 = self.s2(z)
        return self.Md * self.b**2. * (self.a*Rsqr+(self.a+3.*s2)*s1sqr)\
            / (cfg.FourPi * (Rsqr+s1sqr)**2.5 * s2**3.) 
    def M(self,R,z=0.):
        """
        Mass [M_sun] within spherical radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0):   
        
        where
                
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)  
        """
        r = np.sqrt(R**2.+z**2.) 
        return self.Md * r**3. / (r**2.+(self.a+self.b)**2.)**1.5
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        r = np.sqrt(R**2.+z**2.)
        return 3./(cfg.FourPi*r**3.) * self.M(R,z) 
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .tdyn(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z)) 
    def Phi(self,R,z=0.):
        """
        Potential [(kpc/Gyr)^2] at (R,z).
            
        Syntax:
        
            .Phi(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        Rsqr = R**2.
        s1sqr= self.s1sqr(z)
        return -self.GMd / np.sqrt(Rsqr+s1sqr)
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        Rsqr = R**2.
        s1sqr= self.s1sqr(z)   
        s1 = np.sqrt(s1sqr)
        s2 = self.s2(z)
        fac = -self.GMd / (Rsqr+s1sqr)**1.5
        return fac*R, fac*0., fac*z * s1/s2
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at (R,z=0.), defined as 
            
            V_circ(R,z=0.) = sqrt(R d Phi(R,z=0.)/ d R)
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0.)
                
        Note that only z=0 is meaningful. Because circular velocity is 
        the speed of a satellite on a circular orbit, and for a disk 
        potential, a circular orbit is only possible at z=0. 
        """
        return np.sqrt(R*-self.fgrav(R,z)[0])
    def Vesc(self,R,z=0.):
        """
        Escape velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vesc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        return np.sqrt(-2*self.Phi(R,z))
    def sigma(self,R,z=0.):
        """
        Velocity dispersion [kpc/Gyr] at (R,z), following 
        Ciotti & Pellegrini 1996 (CP96).
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0.) 
        
        Note that this is at the same time the R-direction and the 
        z-direction velocity dispersion, as we implicitly assumed
        that the distribution function of the disk potential depends only
        on the isolating integrals E and L_z. If we further assume 
        isotropy, then it is also the phi-direction velocity dispersion.
        (See CP96 eqs 11-17 for more.)
        """
        Rsqr = R**2.
        s1sqr = self.s1sqr(z)
        s2 = self.s2(z)
        sigmasqr = cfg.G*self.Md**2 *self.b**2 /(8.*np.pi*self.rho(R,z))\
            * s1sqr / ( s2**2. * (Rsqr + s1sqr)**3.)
        return np.sqrt(sigmasqr)
    def Vphi(self,R,z=0):
        """
        The mean azimuthal velocity [kpc/Gyr] at (R,z), following 
        Ciotti & Pellegrini 1996 eq.17.
        
        Syntax: 
        
            .Vphi(R,z=0)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0.) 
        
        Note that this is different from the circular velocity by an 
        amount of asymmetric drift, i.e.,
        
            V_a = V_circ - V_phi.
            
        Note that we have made the assumption of isotropy. 
        """
        Rsqr = R**2.
        s1sqr = self.s1sqr(z)
        s2 = self.s2(z)
        Vphisqr = cfg.G*self.Md**2 *self.a *self.b**2 / \
            (cfg.FourPi*self.rho(R,z)) * Rsqr/(s2**3. *(Rsqr+s1sqr)**3.)
        return np.sqrt(Vphisqr)
        
class Hernquist(object):
    """
    Class that implements the Hernquist (1990) profile:

        rho(r) = M / (2 pi a^3) / [x (1+x)^3], x = r/a  
    
    in a cylindrical frame (R,phi,z), where 
    
        M: total mass
        a: scale radius
    
    Syntax:
    
        baryon = Hernquist(M,a)

    where
    
        M: baryon mass [M_sun] (float)
        a: scalelength [kpc] (float)

    Attributes:
    
        .Mb: baryon mass [M_sun]
        .Mh: the same as .Md, but for the purpose of keeping the notation
            for "host" mass consistent with the other profile classes
        .a: scalelength [kpc]
        .r0: the same as .a
        .rho0: characteristic density, M/(2 pi a^3) [M_sun/kpc^3]
        .rhalf: half-mass radius [kpc]
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at (R,z) 
        .M(R,z=0.): mass [M_sun] within radius r=sqrt(R^2+z^2),
            defined as M(r) = r Vcirc(r,z=0)^2 / G
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2 + z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at (R,z)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at (R,z=0), defined as
            sqrt(R d Phi(R,z=0.)/ d R)
        .sigma(R,z=0.): velocity dispersion [kpc/Gyr] at (R,z) 
    
    HISTORY: Arthur Fangzhou Jiang (2020-09-09, Caltech)
    """
    def __init__(self,M,a):
        """
        Initialize Hernquist profile
        
        Syntax:
        
            baryon = Hernquist(M,a)
        
        where 
        
            M: baryon mass [M_sun], 
            a: scale radius [kpc]
        """
        # input attributes
        self.Mb = M
        self.Mh = self.Mb 
        self.a = a
        self.r0 = a
        # 
        # derived attributes
        self.rho0 = M/(cfg.TwoPi*a**3)
        self.rhalf = 2.414213562373095 * a      
        #
        # supportive attributes repeatedly used by following methods
        self.GMb = cfg.G * M 
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at (R,z).
            
        Syntax:
        
            .rho(R,z)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.a
        return self.rho0 / (x * (1.+x)**3)
    def M(self,R,z=0.):
        """
        Mass [M_sun] within spherical radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0):   
        
        where
                
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)  
        """
        r = np.sqrt(R**2.+z**2.) 
        return self.Mb * r**2 / (r+self.a)**2
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        r = np.sqrt(R**2.+z**2.)
        return 3./(cfg.FourPi*r**3.) * self.M(R,z) 
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .tdyn(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z)) 
    def Phi(self,R,z=0.):
        """
        Potential [(kpc/Gyr)^2] at (R,z).
            
        Syntax:
        
            .Phi(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return -self.GMb / (r+self.a)
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        pass
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at (R,z=0.), defined as 
            
            V_circ(R,z=0.) = sqrt(R d Phi(R,z=0.)/ d R)
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0.)
                
        Note that only z=0 is meaningful. Because circular velocity is 
        the speed of a satellite on a circular orbit, and for a disk 
        potential, a circular orbit is only possible at z=0. 
        """
        r = np.sqrt(R**2.+z**2.)
        return np.sqrt(cfg.G*self.M(r)/r)
    def Vesc(self,R,z=0.):
        """
        Escape velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vesc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        return np.sqrt(-2*self.Phi(R,z))
    def sigma(self,R,z=0.):
        """
        Velocity dispersion [kpc/Gyr] assuming isotropic velicity 
        dispersion tensor ... 
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0.) 
        
        Note that this is at the same time the R-direction and the 
        z-direction velocity dispersion, as we implicitly assumed
        that the distribution function of the disk potential depends only
        on the isolating integrals E and L_z. If we further assume 
        isotropy, then it is also the phi-direction velocity dispersion.
        (See CP96 eqs 11-17 for more.)
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.a
        sigmasqr = cfg.GMb/(12.*self.a)*(12.*x*(1.+x)**3*np.log(1.+1./x)\
            -(x/(1.+x))*(25.+52.*x+42.*x**2+12.*x**3)) 
        return np.sqrt(sigmasqr)

class exp(object):
    """
    Class that implements the exponential disk profile:

        Sigma(r) = Sigma_0 exp(-x), x = r/a  
    
    in a cylindrical frame (R,phi,z), where 
    
        M: total mass
        a: scale radius
    
    Syntax:
    
        disk = exp(M,a)

    where
    
        M: baryon mass [M_sun] (float)
        a: scale radius [kpc] (float)

    Attributes:
    
        .Md: baryon mass [M_sun]
        .Mb: the same as .Md
        .Mh: the same as .Md, but for the purpose of keeping the notation
            for "host" mass consistent with the other profile classes
        .a: scalelength [kpc]
        .Rd: the same as .a
        .Sigma0: central surface density, M/(2 pi a^2) [M_sun/kpc^2]
        .rhalf: half-mass radius [kpc]
        
    Methods:
    
        .Sigma(R,z=0.): surface density [M_sun kpc^-2] at (R,z) 
        .M(R,z=0.): mass [M_sun] within radius r=sqrt(R^2+z^2),
            defined as M(r) = r Vcirc(r,z=0)^2 / G
    
    Note: incomplete, other methods and attributes to be added...
    
    HISTORY: Arthur Fangzhou Jiang (2020-09-09, Caltech)
             Zixiang Jia (2025-5-1, Peking University)
    """
    def __init__(self,M,a):
        """
        Initialize Hernquist profile
        
        Syntax:
        
            baryon = Hernquist(M,a)
        
        where 
        
            M: baryon mass [M_sun], 
            a: scale radius [kpc]
        """
        # input attributes
        self.Md = M
        self.Mb = M
        self.Mh = M
        self.a = a
        self.r0 = a
        # 
        # derived attributes
        self.Sigma0 = M/(cfg.TwoPi*a**2)
        self.rhalf = 1.678 * a      
        #
        # supportive attributes repeatedly used by following methods
        self.GMd = cfg.G * M 
    def Sigma(self,R):
        """
        Surface density [M_sun kpc^-2] at R.
            
        Syntax:
        
            .Sigma(R,z)
        
        where
        
            R: R-coordinate [kpc] (float or array)
        """
        x = R/self.a
        return self.Sigma0 * np.exp(-x)
    def M(self,R,z=0.):
        """
        Mass [M_sun] within spherical radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0):   
        
        where
                
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)  
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r/self.a
        return self.Md * (1.-(1.+x)*np.exp(-x))
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at (R,z=0.), defined as 
            
            V_circ(R,z=0.) = sqrt(R d Phi(R,z=0.)/ d R)
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0.)
                
        Note that only z=0 is meaningful. Because circular velocity is 
        the speed of a satellite on a circular orbit, and for a disk 
        potential, a circular orbit is only possible at z=0. 
        """
        r = np.sqrt(R**2.+z**2.)
        return np.sqrt(cfg.G*self.M(r)/r)
    
class DC14(object):
    """
    Class that implements the Di Cintio et al. (2014) empirical profile 
    for dark matter haloes from cosmological simulations:
    
        rho(R,z) = rho_s / ( (r/r_s)^gamma * (1+(r/r_s)^alpha)^((beta-gamma)/alpha) )
    
    in a cylindrical frame (R,phi,z), where 
    
        r = sqrt(R^2 + z^2)
        r_s: scale radius
        (alpha, beta, gamma): shape parameters that depend on the stellar-to-halo 
            mass ratio M_star / M_halo
    
    The profile transitions from inner slope gamma to outer slope beta, 
    with transition sharpness controlled by alpha.
    
    Syntax:
    
        halo = DC14(Mh, c2, Ms, Delta=200., z=0.)
        
    where 
    
        Mh: halo mass [M_sun], defined as spherical overdensity of Delta 
            times critical density (float)
        c2: halo concentration c200 (float) 
        Ms: stellar mass [M_sun] (float)
        Delta: average overdensity of the halo, in multiples of the 
            critical density of the Universe (float) (default 200.)
        z: redshift (float) (default 0.)
    
    Attributes:
    
        .Mh: halo mass [M_sun]
        .Ms: stellar mass [M_sun]
        .c2: halo concentration, defined as rh/r2
        .Deltah: spherical overdensity wrt instantaneous critical density
        .z: redshift
        .ch: halo concentration (same as c2 for this model)
        .X: log10 of stellar-to-halo mass ratio
        .alpha: transition sharpness parameter
        .beta: outer slope parameter
        .gamma: inner slope parameter
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: average density of halo [M_sun kpc^-3]
        .rh: halo radius within which density is Delta times rhoc [kpc]
        .r2: radius where the density slope is -2 [kpc]
        .c: concentration parameter defined as rh/rs (different from c2 due to profile shape)
        .rs: scale radius [kpc]
        .cali: normalization integral of the profile
        .rhos: characteristic density [M_sun kpc^-3]
        .rho0: central density parameter [M_sun kpc^-3] (same as rhos)
        .s001: logarithmic density slope at 0.01 halo radius
        
    Methods:
    
        .g(x): auxiliary function for integration
        .rho(R,z=0.): density [M_sun kpc^-3] at radius r=sqrt(R^2+z^2)
        .s(R,z=0.): logarithmic density slope at radius r=sqrt(R^2+z^2)
        .M(R,z=0.): mass [M_sun] enclosed in radius r=sqrt(R^2+z^2)
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
    
    HISTORY: Zixiang Jia (2025-5-1)
    """

    def __init__(self,Mh,c2,Ms,Delta=200.,z=0.):
        # input attributes
        self.Mh = Mh 
        self.Ms = Ms
        self.c2 = c2 # concentration c200
        self.Deltah = Delta
        self.z = z
        # derived attributes
        self.ch = self.c2
        self.X = np.log10(self.Ms / self.Mh)
        self.alpha = 2.94 - np.log10(10.**(-1.08*(self.X+2.33))+10.**(2.29*(self.X+2.33)))
        self.beta = 4.23 + 1.34*self.X + 0.26*self.X**2
        self.gamma = -0.06 + np.log10(10.**(-0.68*(self.X+2.56))+10.**(self.X+2.56))
        self.rhoc = co.rhoc(z,cfg.h,cfg.Om,cfg.OL)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3.*self.Mh / (cfg.FourPi*self.rhoh))**(1./3.)
        self.r2 = self.rh / self.c2
        self.c = self.c2 * ((2.-self.gamma)/(self.beta-2.))**(1./self.alpha)
        self.rs = self.rh / self.c
        self.cali = quad(self.g,0,self.c)[0]
        self.rhos = self.Mh / (cfg.FourPi*self.rs**3.*self.cali)
        self.rho0 = self.rhos
        self.s001 = self.s(0.01*self.rh)
    def g(self,x):
        """
        Syntax:
    
            .g(x)
    
        where 
    
            x: dimensionless radius r/r_s (float or array)
        """
        return x**(2.-self.gamma) * (1+x**self.alpha)**(-(self.beta-self.gamma)/self.alpha)
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .rho(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        return self.rho0 / x**self.gamma / (1+x**self.alpha)**((self.beta-self.gamma)/self.alpha)
    def s(self,R,z=0.):
        """
        Logarithmic density slope 
            
            - d ln rho / d ln r 
        
        at radius r = sqrt(R^2 + z^2). 
        
        Syntax:
        
            .s(R,z=0.)

        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        xalpha = x**self.alpha
        return self.gamma + (self.beta-self.gamma)*xalpha/(1.+xalpha)
    def M(self,R,z=0.):
        """
        Mass [M_sun] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        if isinstance(R,float) or isinstance(R,int):
            r = np.sqrt(R**2.+z**2.) 
            x = r / self.rs
            res = self.Mh * quad(self.g,0,x)[0] / self.cali
        elif isinstance(R,np.ndarray):
            res = np.copy(R)*0.
            r = np.sqrt(R**2.+z**2.) 
            x = r / self.rs
            length = len(R)
            for i in range(length):
                res[i] = self.Mh * quad(self.g,0,x[i])[0] / self.cali
        elif isinstance(R,list):
            res = []
            length = len(R)
            for i in range(length):
                r = np.sqrt(R[i]**2.+z**2.) 
                x = r / self.rs
                res.append(self.Mh * quad(self.g,0,x)[0] / self.cali)
        return res
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.) 
        return np.sqrt(cfg.G*self.M(R,z)/r)
    
    
#--- functions dealing with composite potential (i.e., potential list)---

def rho(potential,R,z=0.):
    """
    Density [M_sun/kpc^3], at location (R,z) in an axisymmetric potential
    which consists of either a single component or multiple components.
    
    Syntax:

        rho(potential,R,z=0.)
        
    where
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r)  
    
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the density at 
    (R,z) in this combined halo+disk host, we use: 
    
        rho([halo,disk],R,z)
    """
    if not isinstance(potential, list): # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential] 
    sum = 0.
    for p in potential:
        sum += p.rho(R,z)
    return sum
    
def s(potential,R,z=0.):
    """
    Logarithmic density slope 
            
        - d ln rho / d ln r 
        
    at radius r = sqrt(R^2 + z^2). 
        
    Syntax:
        
        s(potential,R,z=0.)

    where
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r)  
    
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the density at 
    (R,z) in this combined halo+disk host, we use: 
    
        s([halo,disk],R,z)
    """
    r = np.sqrt(R**2.+z**2.)
    r1 = r * (1.+cfg.eps)
    r2 = r * (1.-cfg.eps)
    return -np.log(rho(potential,r1)/rho(potential,r2)) / np.log(r1/r2)
    
def M(potential,R,z=0.):
    """
    Mass [M_sun] within spherical radius r = sqrt(R^2 + z^2) in an 
    axisymmetric potential which consists of either a single component or 
    multiple components.
            
    Syntax:
        
        M(potential,R,z=0):   
        
    where
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r)  
    
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the mass within  
    r = sqrt(R^2 + z^2) in this combined halo+disk host, we use: 
    
        M([halo,disk],R,z)
    """
    if not isinstance(potential, list): # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential] 
    sum = 0.
    for p in potential:
        sum += p.M(R,z)
    return sum
    
def rhobar(potential,R,z=0.):
    """
    Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2) in 
    an axisymmetric potential which consists of either a single component 
    or multiple components.
    
    Syntax:
    
        rhobar(potential,R,z=0.)
        
    where 
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r)  
            
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the mean density 
    within r = sqrt(R^2 + z^2) in this combined halo+disk host, we use: 
    
        rhobar([halo,disk],R,z) 
    """
    r = np.sqrt(R**2.+z**2.)
    return 3./(cfg.FourPi*r**3.) * M(potential,R,z)

def tdyn(potential,R,z=0.):
    """
    Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).
    
    Syntax:
        
        tdyn(potential, R, z=0.)
            
    where
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r) 
    
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the dynamical time 
    within r = sqrt(R^2 + z^2) in this combined halo+disk host, we use: 
    
        tdyn([halo,disk],R,z) 
    """
    return np.sqrt(cfg.ThreePiOverSixteenG / rhobar(potential,R,z))

def Phi(potential,R,z=0.):
    """
    Potential [(kpc/Gyr)^2] at (R,z) in an axisymmetric potential
    which consists of either a single component or multiple components.
            
    Syntax:
        
        Phi(potential,R,z=0):   
        
    where
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r)  
    
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the gravitational
    potential at (R,z) in this combined halo+disk host, we use: 
    
        Phi([halo,disk],R,z)
    """
    if not isinstance(potential, list): # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential] 
    sum = 0.
    for p in potential:
        sum += p.Phi(R,z)
    return sum
    
def Vcirc(potential,R,z=0.):
    """
    Circular velocity [kpc/Gyr] at (R,z=0), defined as 
            
        V_circ(R,z=0) = sqrt(R d Phi(R,z=0)/ d R)
    
    in an axisymmetric potential which consists of either a single 
    component or multiple components.
            
    Syntax:
        
        Vcirc(potential,R,z=0):   
        
    where
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r)  
    
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the circular 
    velocity at (R,z) in this combined halo+disk host, we use: 
    
        Vcirc([halo,disk],R,z)
    """
    R1 = R*(1.+cfg.eps)
    R2 = R*(1.-cfg.eps)
    Phi1 = Phi(potential,R1,z)
    Phi2 = Phi(potential,R2,z)
    dPhidR = (Phi1-Phi2) / (R1-R2)
    return np.sqrt(R * dPhidR)

def Vesc(potential,r,z=0.):
    '''
    Find the escape velocity of a halo at radius r.
    
    We can devide the gravitation potential into inner part and outer part, and
    the former can be replaced by Vcirc, the later can only be calculated through integration.
    
    Vesc=sqrt(2halo.Vcirc(r)**2+8*Pi*G*integrate.quad(halo.rho(r)*r,r,Infinity))
    Here upper limit is 1Mpc, G = 4.3009e-6 [kpc*(km/s)^2*Msun^(-1)]
    '''
    return  potential.Vesc(r,z)#[kpc/Gyr]
        
def sigma(potential,R,z=0.):
    """
    1D velocity dispersion [kpc/Gyr] at (R,z=0), in an axisymmetric 
    potential which consists of either a single component or multiple 
    components. For composite potential, the velocity dispersion is the
    quadratic sum of that of individual components
    
        sigma^2 = Sum sigma_i^2
            
    Syntax:
        
        sigma(potential,R,z=0):   
        
    where
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r)  
    
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the circular 
    velocity at (R,z) in this combined halo+disk host, we use: 
    
        sigma([halo,disk],R,z)
    """
    if not isinstance(potential, list): # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential] 
    sum = 0.
    for p in potential:
        sum += p.sigma(R,z)**2
    return np.sqrt(sum)

def fDF(potential,xv,m):
    """
    Dynamical-friction (DF) acceleration [(kpc/Gyr)^2 kpc^-1] given 
    satellite mass, phase-space coordinate, and axisymmetric host
    potential:
    
        f_DF = -4piG^2 m Sum_i rho_i(R,z)F(<|V_i|)ln(Lambda_i)V_i/|V_i|^3  
    
    where
        
        V_i: relative velocity (vector) of the satellite with respect to 
            the host component i
        F(<|V_i|) = erf(X) - 2X/sqrt{pi} exp(-X^2) with 
            X = |V_i| / (sqrt{2} sigma(R,z))
        ln(Lambda_i): Coulomb log of host component i 
        
    Syntax:
    
        fDF(potential,xv,m)
          
    where 
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates in a cylindrical frame
            [R,phi,z,VR,Vphi,Vz] 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] 
            (numpy array)
        m: satellite mass [M_sun] (float)
            
    Return: 
        
        R-component of DF acceleration (float), 
        phi-component of DF acceleration (float), 
        z-component of DF acceleration (float)

    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the DF acceleration
    experienced by a satellite of mass m at xv in this combined 
    halo+disk host, we do: 
    
        fDF([halo,disk],xv,m)
    
    Note: for a composite potential, we compute the DF acceleration 
    exerted by each component separately, and sum them up as the 
    total DF acceleration. This implicitly assumes that 
        1. each component has a Maxwellian velocity distribution,
        2. the velocity dispersion of each component is not affected by
           other components 
        3. the Coulomb log of each component can be treated individually.
    All these assumptions are not warranted, but there is no trivial, 
    better way to go, see e.g., Penarrubia+2010 (MN,406,1290) Appendix A.
    """
    if not isinstance(potential, list): # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential] 
    #
    R, phi, z, VR, Vphi, Vz = xv
    #
    fac = -cfg.FourPiGsqr * m # common factor in f_DF 
    sR = 0. # sum of R-component of DF accelerations 
    sphi = 0. # ... phi- ...
    sz = 0. # ... z- ...
    for p in potential:
        if isinstance(p,(MN,)): # i.e., if component p is a disk
            lnL = 0.5
            #lnL = 0.0 # <<< test: turn off disk's DF 
            VrelR = VR
            Vrelphi = Vphi - p.Vphi(R,z)
            Vrelz = Vz
        else: # i.e., if component p is a spherical component
            lnL = np.log(p.Mh/m)
            VrelR = VR
            Vrelphi = Vphi
            Vrelz = Vz
        Vrel = np.sqrt(VrelR**2.+Vrelphi**2.+Vrelz**2.)
        Vrel = max(Vrel,cfg.eps) # safety
        X = Vrel / (cfg.Root2 * p.sigma(R,z))
        fac_s = p.rho(R,z) * lnL * ( erf(X) - \
            cfg.TwoOverRootPi*X*np.exp(-X**2.) ) / Vrel**3 
        sR += fac_s * VrelR 
        sphi += fac_s * Vrelphi 
        sz += fac_s * Vrelz 
    return fac*sR, fac*sphi, fac*sz
    
def fRP(potential,xv,sigmamx,Xd=1.):
    """
    Ram-pressure (RP) acceleration [(kpc/Gyr)^2 kpc^-1] of a satellite 
    due to dark-matter self-interaction (Kummer+18 eq.18), in an 
    axisymmetric host potential:
    
        f_RP = - X_d (sigma/m_x) rho_i(R,z) |V_i| V_i
    
    where
        
        V_i: relative velocity (vector) of the satellite with respect to 
            the host component i;
        X_d(V,v_esc): an order-unity factor depending on subhalo's 
            orbital velocity and own escape velocity; 
        sigma/m_x: cross section over dark-matter particle mass.
        
    Syntax:
    
        fRP(potential,xv,sigmamx,Xd=1.)
          
    where 
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates in a cylindrical frame
            [R,phi,z,VR,Vphi,Vz] 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] 
            (numpy array)
        sigmamx: self-interaction cross section over particle mass 
            [cm^2/g] or [2.0884262122368293e-10 kpc^2/M_sun] (default=1.)
        Xd: deceleration fraction (default=1.)
            
    Return: 
        
        R-component of RP acceleration (float), 
        phi-component of RP acceleration (float), 
        z-component of RP acceleration (float)

    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the RP acceleration
    experienced by a satellite of self-interaction cross section of 
    sigmamx, at xv, in this combined halo+disk host, we do: 
    
        fRP([halo,disk],xv,sigmamx,Xd=1.)
    
    Note: for a composite potential, we compute the RP acceleration 
    exerted by each dark-matter component separately, and sum them up as 
    the total RP acceleration. We skip any baryon component, such as MN
    disk, unless the MN disk is a dark-matter disk. 
    """
    if not isinstance(potential, list): # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential] 
    #
    R, phi, z, VR, Vphi, Vz = xv
    # 
    fac = -Xd * (sigmamx*2.0884262122368293e-10) # common factor in f_RP
    sR = 0. # sum of R-component of RP accelerations 
    sphi = 0. # ... phi- ...
    sz = 0. # ... z- ...
    for p in potential:
        if isinstance(p,(MN,)): # i.e., if component p is a baryon disk
            # then there is no RP from it -- this is achieved by setting
            # the relative velocities to zero.
            VrelR = 0.
            Vrelphi = 0.
            Vrelz = 0.
        else: # i.e., if component p is not a disk, i.e., a spherical 
            # dark-matter component, we add its contribution
            VrelR = VR
            Vrelphi = Vphi
            Vrelz = Vz
        Vrel = np.sqrt(VrelR**2.+Vrelphi**2.+Vrelz**2.)
        Vrel = max(Vrel,cfg.eps) # safety
        fac_s = p.rho(R,z) * Vrel 
        sR += fac_s * VrelR 
        sphi += fac_s * Vrelphi 
        sz += fac_s * Vrelz 
    return fac*sR, fac*sphi, fac*sz
        
def ftot(potential,xv,m=None,sigmamx=None,Xd=None):
    """
    Total acceleration [(kpc/Gyr)^2 kpc^-1] at phase-space coordinate xv, 
    in an axisymmetric potential. Here "total" means gravitational 
    acceleration plus dynamical-friction acceleration.
    
    Syntax:

        ftot(potential,xv,m=None,sigmamx=None,Xd=None)
        
    where 
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates in a cylindrical frame
            [R,phi,z,VR,Vphi,Vz] 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] 
            (numpy array)
        m: satellite mass [M_sun] (float) 
            (default is None; if provided, dynamical friction is on)
        sigmamx: SIDM cross section [cm^2/g] 
            (default is None; if provided, ram-pressure drag is on)
        Xd: coefficient of ram-pressure deceleration as in Kummer+18
            (default is None; if sigmamx provided, must provide)
    
    Return: 
    
        fR: R-component of total (grav+DF+RP) acceleration (float), 
        fphi: phi-component of total (grav+DF+RP) acceleration (float), 
        fz: z-component of total (grav+DF+RP) acceleration (float)
        
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the total 
    acceleration experienced by a satellite of mass m, self-interaction
    cross-section sigmamx, at location xv, in this combined halo+disk 
    host, we do: 
    
        ftot([halo,disk],xv,m,sigmamx,Xd)
    """
    if not isinstance(potential, list): # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential] 
    #
    R, phi, z, VR, Vphi, Vz = xv
    #
    fR,fphi,fz = 0.,0.,0.
    for p in potential:
        fR_tmp, fphi_tmp, fz_tmp = p.fgrav(R,z)
        fR += fR_tmp
        fphi += fphi_tmp
        fz += fz_tmp
    # 
    if m is None: # i.e., if dynamical friction is ignored
        fDFR, fDFphi, fDFz = 0.,0.,0.
    else:
        fDFR, fDFphi, fDFz = fDF(potential,xv,m)
    #
    if sigmamx is None: # i.e., if ram-pressure drag is ignored
        fRPR, fRPphi, fRPz = 0.,0.,0.
    else:
        fRPR, fRPphi, fRPz = fRP(potential,xv,sigmamx,Xd)
    return fR+fDFR+fRPR, fphi+fDFphi+fRPphi, fz+fDFz+fRPz 

def EnergyAngMomGivenRpRa(potential,rp,ra):
    """
    Compute the specific orbital energy [(kpc/Gyr)^2] and the specific 
    orbital angular momentum [kpc(kpc/Gyr)] given two radii along the 
    orbit, e.g., the pericenter and the apocenter radii, using 
    Binney & Tremaine (2008 Eq.3.14)
    
    Syntax:
    
        EnergyAngMomGivenRaRp(potential,rp,ra)
        
    where
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        rp: pericenter radius [kpc] (float)
        ra: apocenter radius [kpc] (float)
    
    Return:
    
        E [(kpc/Gyr)^2], L [kpc(kpc/Gyr)]
    """
    Phip = Phi(potential,rp)
    Phia = Phi(potential,ra)
    upsqr = 1./rp**2
    uasqr = 1./ra**2
    L = np.sqrt(2.*(Phip-Phia)/(uasqr-upsqr))
    E = uasqr/(uasqr-upsqr)*Phip - upsqr/(uasqr-upsqr)*Phia 
    return E,L
    
#---for solving SIDM profile
def f(y,t,a,b):
    """
    Returns right-hand-side functions of the ODEs for solving for the
    dimensionless SIDM potential h(t), as in Kaplinghat+14:
    ** Assuming a Hernquist baryon profile
    
        h'(t) = g(t)    
        g'(t) = - [ 2g(t)/t + b/t + a exp(h)/(1-t)^4 ]
        
    where 
    
        t := x/(1+x) with x = r/r_0 is dimensionless radius
        h(t) := - phi(t) / sigma_0^2 is dimensionless potential
        g(t) := h'(t) 
        a := 4 pi G rho_dm0 r_0^2 / sigma_0^2
        b := 4 pi G rho_b0 r_0^2 / sigma_0^2
        
    with rho_dm0, rho_b0 the central DM and baryon density respectively,
    sigma_0 the velocity dispersion in the isothermal core, and r_0 the
    Hernquist scale radius of the baryon distribution.
        
    Syntax:
            
        f(y,t,a,b)
            
    where 
        
        y: [h(t),g(t)] (list or array)
        t: x/(1+x) with x = r/r_0 (float)
        a: 4 pi G rho_dm0 r_0^2 / sigma_0^2 (float)
        b: 4 pi G rho_b0 r_0^2 / sigma_0^2 (float)

    Return: the list of dy/dt:
    
        [g, - (2g+b)/t - a exp(h)/(1-t)^4]
        
    Note that this function shall be used in
    
        scipy.integrate.odeint(f,y0,t,args=(a,b))
    """
    h,g = y
    dydt = [g, -(2.*g+b)/t -a*np.exp(h)/(1.-t)**4]
    return dydt
    
def h(x,a,b):
    """
    Solve for dimensionless SIDM potential profile, h(x), using the 
    Kaplinghat+14 method, given the parameters:
    ** Assuming a Hernquist baryon profile
    
        h(x) := - phi(x) / sigma_0^2, x = r/r_0
        a := 4 pi G rho_dm0 r_0^2 / sigma_0^2
        b := 4 pi G rho_b0 r_0^2 / sigma_0^2
    
    with rho_dm0, rho_b0 the central DM and baryon density respectively,
    sigma_0 the velocity dispersion in the isothermal core, and r_0 the
    Hernquist scale radius of the baryon distribution. Note that the SIDM
    density profile is
    
        rho(x) = rho_dm0 exp[h(x)].
        
    Note that for odeint to work, the t variable as in 
        
        odeint(f,[0.,-b/2.],t,args=(a,b))
    
    must be an array and must have the initial-value point as the first 
    element of this array. 
    
    Syntax:
    
        h(x,a,b)
        
    where
    
        x: radius in units of r_0 in ascending order (float or array)
        a: 4 pi G rho_dm0 r_0^2 / sigma_0^2 (float)
        b: 4 pi G rho_b0 r_0^2 / sigma_0^2 (float)
        
    Return: h(x)
    """
    if np.isscalar(x):
        x = np.array([1e-6,x])
    else:
        x = np.append(1e-6,x)
    t = x/(1.+x) 
    sol = odeint(f,[0.,-b/2.],t,args=(a,b))
    return sol[1:,0]

def f_exp(y,t,a,b):
    """
    Returns right-hand-side functions of the ODEs for solving for the
    dimensionless SIDM potential h(t), as in Kaplinghat+14:
    ** Assuming a exponential baryon profile

        h'(t) = g(t)    
        g'(t) = - [ 2g(t)/t + a exp(h)/(1-t)^4 + b exp(-t/(1-t))/t/(1-t)**3]
        
    where 
    
        t := x/(1+x) with x = r/r_0 is dimensionless radius
        h(t) := - phi(t) / sigma_0^2 is dimensionless potential
        g(t) := h'(t) 
        a := 4 pi G rho_dm0 r_d^2 / sigma_0^2
        b := G M_b / sigma_0^2 /r_d
        .
    with rho_dm0, M_b the central DM and baryon mass respectively,
    sigma_0 the velocity dispersion in the isothermal core, and r_d the
    scale radius of the exponential baryon distribution.
        
    Syntax:
            
        f(y,t,a,b)
            
    where 
        
        y: [h(t),g(t)] (list or array)
        t: x/(1+x) with x = r/r_0 (float)
        a := 4 pi G rho_dm0 r_d^2 / sigma_0^2
        b := G M_b / sigma_0^2 /r_d

    Return: the list of dy/dt:
    
        [g, g'(t)]
        
    Note that this function shall be used in
    
        scipy.integrate.odeint(f,y0,t,args=(a,b))
    """
    h,g = y
    t0 = 1.-t
    dydt = [g,-(2.*g/t + a*np.exp(h)/t0**4 + b*np.exp(-t/t0)/t/t0**3) ]
    return dydt

def h_exp(x,a,b):
    """
    Solve for dimensionless SIDM potential profile, h(x), using the 
    Kaplinghat+14 method, given the parameters:
    ** Assuming a exponential baryon profile

        h(x) := - phi(x) / sigma_0^2, x = r/r_d
        a := 4 pi G rho_dm0 r_d^2 / sigma_0^2
        b := G M_b/ sigma_0^2 /r_d
    
    with rho_dm0, M_b the central DM and baryon mass respectively,
    sigma_0 the velocity dispersion in the isothermal core, and r_d the
    scale radius of the exponential baryon distribution. Note that the SIDM
    density profile is
    
        rho(x) = rho_dm0 exp[h(x)].
        
    Note that for odeint to work, the t variable as in 
        
        odeint(f,[0.,-b/4.],t,args=(a,b))
    
    must be an array and must have the initial-value point as the first 
    element of this array. 
    
    Syntax:
    
        h(x,a,b)
        
    where
    
        x: radius in units of r_0 in ascending order (float or array)
        a := 4 pi G rho_dm0 r_d^2 / sigma_0^2 (float)
        b := G M_b / sigma_0^2 /r_d (float)
        
    Return: h(x)
    """
    if np.isscalar(x):
        x = np.array([1e-6,x])
    else:
        x = np.append(1e-6,x)
    t = x/(1.+x) 
    sol = odeint(f_exp,[0.,-b/2.],t,args=(a,b))
    return sol[1:,0]

def delta(p,rhob0,r0,rhoCDM1,MCDM1,r):
    """
    Given p=[rho_dm0,sigma_0], evaluate the relative error in stitching
    the isothermal-core profile to the outer CDM-like profile
    ** Assuming a Hernquist baryon profile
    
        delta = sqrt( delta_rho^2 + delta_M^2 )
        
    where 
    
        delta_rho = | rho_iso(r_1) - rho_CDM(r_1) | / rho_CDM(r_1)
        detla_M = | M_iso(r_1) - M_CDM(r_1) | / M_CDM(r_1)
        
    Syntax:
    
        delta(p,rho1,M1,r0,rhoCDM1,MCDM1,r)
        
    where
    
        p: [log(rho_dm0),log(sigma_0)] in [M_sun/kpc^3, kpc/Gyr] (array)
        rhob0: Hernquist central density of baryons [M_sun/kpc^3] (float)
        r0: Hernquist scale radius of baryon distribution [kpc] (float)
        rho1: CDM density to match at r_1 [M_sun/kpc^3] (float)
        M1: CDM enclosed mass to match at r_1 [M_sun] (float)
        r: radii between 0 and r_0, where we compute the SIDM profile
           [kpc] (array), e.g., np.logspace(-3.,np.log10(r1),500)
    """
    rhodm0 = 10.**p[0]
    sigma0 = 10.**p[1]
    a = cfg.FourPiG * r0**2 *rhodm0 / sigma0**2
    b = cfg.FourPiG * r0**2 *rhob0 / sigma0**2
    rho = rhodm0*np.exp(h(r/r0,a,b))
    M = Miso(r,rho)
    drho = (rho[-1] - rhoCDM1) / rhoCDM1    
    dM = (M[-1] - MCDM1) / MCDM1
    return drho**2 + dM**2

def delta_exp(p,Mb,r0,rhoCDM1,MCDM1,r):
    """
    Given p=[rho_dm0,sigma_0], evaluate the relative error in stitching
    the isothermal-core profile to the outer CDM-like profile
    ** Assuming a exponential baryon profile

        delta = sqrt( delta_rho^2 + delta_M^2 )
        
    where 
    
        delta_rho = | rho_iso(r_1) - rho_CDM(r_1) | / rho_CDM(r_1)
        detla_M = | M_iso(r_1) - M_CDM(r_1) | / M_CDM(r_1)
        
    Syntax:
    
        delta(p,rho1,M1,r0,rhoCDM1,MCDM1,r)
        
    where
    
        p: [log(rho_dm0),log(sigma_0)] in [M_sun/kpc^3, kpc/Gyr] (array)
        rhob0: Hernquist central density of baryons [M_sun/kpc^3] (float)
        r0: Hernquist scale radius of baryon distribution [kpc] (float)
        rho1: CDM density to match at r_1 [M_sun/kpc^3] (float)
        M1: CDM enclosed mass to match at r_1 [M_sun] (float)
        r: radii between 0 and r_0, where we compute the SIDM profile
           [kpc] (array), e.g., np.logspace(-3.,np.log10(r1),500)
    """
    rhodm0 = 10.**p[0]
    sigma0 = 10.**p[1]
    a = cfg.FourPiG * r0**2 *rhodm0 / sigma0**2
    b = cfg.G * Mb / sigma0**2 / r0
    rho = rhodm0*np.exp(h_exp(r/r0,a,b))
    M = Miso(r,rho)
    drho = (rho[-1] - rhoCDM1) / rhoCDM1    
    dM = (M[-1] - MCDM1) / MCDM1
    return drho**2 + dM**2

def r1(potential,sigmamx=1.,tage=1.,disk=None):
    """
    The characteristic radius of a SIDM halo at which an average particle
    is scattered once during the age of the halo, defined as the solution
    to (e.g., Kaplinghat+16 eq.1)
    
        (4/sqrt{pi}) rho(r) sigma(r) (sigma/m_x) t_age = 1
        
    where  Gamma(r) := rho(r) sigma(r) (sigma/m_x) is the scattering rate
    with rho(r) the density profile, sigma(r) the velocity dispersion, 
    and sigma/m_x the self-interaction cross-section per particle mass. 

    If the baryon proifle (disk) is given, adiabatic contraction of r1 will
    be computed.

    Caution that if you need to compute r1 when p > p_merge (p=sigmamx*tage),
    you should input the mirrored value p' = 2*p_merge - p.

    Syntax:
    
        r1(potential,sigmamx=1.,tage=1.)
        
    where
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        sigmamx: self-interaction cross-section per particle mass 
            [cm^2/g] or [2.08889e-10 kpc^2/M_sun] (default=1.)
        tage: halo age [Gyr], somewhat arbitrary, e.g., lookback time to
            the formation epoch of the halo, where "formation" can be 
            defined as the most recent major merger or the time of 
            reaching half of the current mass.
        disk: baryon profile (default=None)

    return
        If disk is None: 
                         characteristic radius [kpc] (float),
                         DM density at r1 [M_sun/kpc^3] (float),
                         DM mass within r1 [M_sun] (float)
        If disk is provided: 
                         the three quantities after adiabatic contraction
    """
    # define search interval for root-finding [a, b]
    a = cfg.Rres  # minimum radius (resolution limit)
    b = 2000.     # maximum radius (an extreme large value)
    
    # evaluate function at interval boundaries
    fa = Findr1(a, potential, sigmamx, tage)
    fb = Findr1(b, potential, sigmamx, tage)
    
    # check if root exists in interval [a,b]
    if fa * fb > 0.:  # same sign -> no root in interval
        r = cfg.Rres  # fallback to resolution limit
    else:
        # use Brent's method to find root (r where Findr1 = 0)
        r = brentq(Findr1, a, b, args=(potential, sigmamx, tage),
                   xtol=0.001, rtol=1e-5, maxiter=1000)
    
    if disk is None:
        # handle case without disk profile
        return r, potential.rho(r), potential.M(r)
    else:
        # with disk, compute adiabatic contraction
        r1_f, rho1_f, Mdm1_f = gh.r1_direct_contra(r, potential, disk)
        return r1_f, rho1_f, Mdm1_f
    
def Findr1(r,potential,sigmamx,tage):
    """
    Auxiliary function for the function "r1", which returns the 
        
        left-hand side  -  right-hand side
        
    of the equation 
    
        4/sqrt{pi} rho(r) sigma(r) (sigma/m_x) t_age = 1
    """
    return cfg.FourOverRootPi * rho(potential,r) * sigma(potential,r) *\
        (sigmamx*2.08889e-10) * tage - 1.     
        
def Miso(r,rho):
    """
    Enclosed mass profile of the SIDM isothermal core, given the radii 
    and density profile array.
    
    Syntax:
    
        Miso(r,rho)
        
    where
    
        r: radii at which the density profile is evaluated [kpc] (array)
        rho: density profile [M_sun/kpc^3] (array)
        
    Return: the mass profile corresponding to the input density profile
        [M_sun] (array of the same length as r and rho)
    """
    rtmp = np.append(0.,r[:-1]) 
    dr = r - rtmp
    rhoave = np.append(rho[0],0.5*(rho[1:]+rho[:-1]))
    dM = cfg.FourPi * r**2 *dr * rhoave
    return dM.cumsum()

def tmerge(halo_init,disk=None):
    '''
    Compute the (normalized) time at which the high and low density 
    solution branch given by the Jeans model merge according to 
    Jia, et al.,2026.
    
    This function finds the critical timescale for the onset of core collapse
    in SIDM haloes, where the high-density (core-collapse) and low-density 
    (core-growth) solutions become indistinguishable.

    This approach is based on a characteristic
    feature of the Jeans model: when p > p_merge, the model yields
    only physically invalid solutions, whereas for p < p_merge, at least
    one valid solution can be obtained through sufficient searching.

    Note that in this function, the mirroring technique for core-collapse
    solution must NOT be implemented, because the characteristic
    feature of the Jeans model holds only for a not mirrored p.

    ** If you find that this function gives you wrong results, try to
       change the initial guesses or add another search in the function
       checkmerge defined below.
    
    Syntax:

        tmerge(halo_init,disk=None)

    where

        halo_init: initial halo profile (e.g. NFW class instance)
                   Represents the initial DM-only halo before SIDM effects
                   and adiabatic contraction
        disk: optional disk component (Hernquist or Exp class instance)
              If None, uses a dummy disk for DM-only cases
              
    Return:
        tnor_merge: normalized merge time (dimensionless)
        product_merge: physical merge time [Gyr*cm^2/g]
        
    Note: For DM-only cases, no need to input disk.
    '''
    # prepare some quantities
    a0 = 4/np.pi**0.5 # constant for computing normalized time
    rs = halo_init.rs
    rhos = halo_init.rho(rs)*4.
    M0 = 4*np.pi*rs**3.*rhos
    v0 = np.sqrt(M0/rs*cfg.G)
    trans = a0*v0*rhos*2.08889e-10 # conversion factor for normalized time

    # for DM only cases, create a dummy disk
    if disk is None: disk = Hernquist(1.,1000.)

    # compute tmerge by bisection in log space
    lgtnor_lo = -2. # lower bound in log10(normalized time)
    lgtnor_hi = 4.1 # upper bound in log10(normalized time)

    # make sure the lower boundary is not extremely low that no solution can be found
    product_lo = 10.**lgtnor_lo/trans
    r1_lo,rhoCDM1,MCDM1 = r1(halo_init,sigmamx=1.,tage=product_lo,disk=disk)
    merge_lo = checkmerge(r1_lo,rhoCDM1,MCDM1,halo_init,disk)

    # iterate to find a valid lower boundary
    while merge_lo and lgtnor_lo < lgtnor_hi:
        lgtnor_lo += 0.1 # increment lower bound
        product_lo = 10.**lgtnor_lo/trans
        r1_lo,rhoCDM1,MCDM1 = r1(halo_init,sigmamx=1.,tage=product_lo,disk=disk)
        merge_lo = checkmerge(r1_lo,rhoCDM1,MCDM1,halo_init,disk)
    if merge_lo: print('WARNING: cannot find a lower boundary when searching for tmerge!!!')

    # perform bisection to locate merge point
    dlgtnor = lgtnor_hi - lgtnor_lo
    while dlgtnor > 1e-3 : # continue until tolerance reached
        lgtnor_mid = 0.5*(lgtnor_lo + lgtnor_hi)
        product_mid = 10.**lgtnor_mid/trans

        # evaluate if solutions merge at this timescale
        r1_mid,rho_mid,M_mid = r1(halo_init,sigmamx=1.,tage=product_mid,disk=disk)
        merge = checkmerge(r1_mid,rho_mid,M_mid,halo_init,disk)

        # update bisection interval
        if merge:
            lgtnor_hi = lgtnor_mid # merge occurs -> move upper bound down
        else:
            lgtnor_lo = lgtnor_mid # no merge -> move lower bound up
        dlgtnor = lgtnor_hi - lgtnor_lo

    # final results at merge point
    tnor_merge = 10.**(0.5*(lgtnor_hi + lgtnor_lo))
    product_merge = tnor_merge/trans

    return tnor_merge,product_merge

def checkmerge(r1,rho1,M1,halo,disk):
    '''
    Check if the two solution branches given by the Jeans model merge.
    
    When the two solution branches merge, the Jeans model yields only 
    physically invalid solutions, characterised by delta^2 > 1e-5. 
    Otherwise it can always find at least one valid solution.

    Here
        delta = sqrt( delta_rho^2 + delta_M^2 )
    and
        delta_rho = | rho_iso(r_1) - rho_CDM(r_1) | / rho_CDM(r_1)
        detla_M = | M_iso(r_1) - M_CDM(r_1) | / M_CDM(r_1)

    ** If you find that this function gives you wrong results, try to
       change the initial guesses or add another search.
    
    Syntax:
        checkmerge(r1, rho1, M1, halo, disk)
        
    where
        r1: characteristic radius of an SIDM halo (float)
        rho1: DM density at r1 [M_sun/kpc^3] (float)
        M1: DM mass within r1 [M_sun] (float)
        halo: halo profile object for CDM outskirts (density profile object)
        disk: galaxy disk profile (Hernquist object)
    
    Return:
        True if solutions have merged (only core-collapse branch exists),
        False if both low and high density solutions exist.
    '''

    # prepare a few quantities
    sigmaCDM1 = halo.sigma(r1)
    _,rhoCDMres,_ = gh.r1_direct_contra(cfg.Rres,halo,disk)

    # define the radius array
    r = np.logspace(-3.,np.log10(r1),500)

    # specify the searching range
    lgrhodm0_bounds = (np.log10(rho1),np.log10(rhoCDMres))
    if lgrhodm0_bounds[0] >= lgrhodm0_bounds[1]: 
        return False # p is very small, r1->0, SIDM->CDM
    lgsigma0_bounds = (np.log10(0.5*sigmaCDM1),np.log10(4.0*sigmaCDM1))

    # define multiple initial guesses for optimization
    lgsigma0_init = np.log10(sigmaCDM1)
    
    # The initial central density span a wide parameter space to
    # ensure a thorough search.
    lgrhodm0_init1 = 0.91*np.log10(rho1)+0.09*np.log10(rhoCDMres)
    lgrhodm0_init2 =  0.12*np.log10(rho1) + 0.88*np.log10(rhoCDMres)
    lgrhodm0_init3 = 0.81*np.log10(rho1)+0.19*np.log10(rhoCDMres)
    lgrhodm0_init4 = 0.5*np.log10(rho1)+0.5*np.log10(rhoCDMres)

    initial_guess1 = [lgrhodm0_init1,lgsigma0_init]
    initial_guess2 = [lgrhodm0_init2,lgsigma0_init]
    initial_guess3 = [lgrhodm0_init3,lgsigma0_init]
    initial_guess4 = [lgrhodm0_init4,lgsigma0_init]

    # search for a valid solution to check if the two solution branches have merged.
    merge = False # True when 0 valid solution is found for the system

    sol = _search_for_solution(initial_guess1,lgrhodm0_bounds,lgsigma0_bounds,disk,rho1,M1,r)
    delta2_1 = sol[-1]
    if delta2_1 > 1e-5:
        sol = _search_for_solution(initial_guess2,lgrhodm0_bounds,lgsigma0_bounds,disk,rho1,M1,r)
        delta2_2 = sol[-1]
        if delta2_2 > 1e-5: 
            sol = _search_for_solution(initial_guess3,lgrhodm0_bounds,lgsigma0_bounds,disk,rho1,M1,r)
            delta2_3 = sol[-1]
            if delta2_3 > 1e-5:
                sol = _search_for_solution(initial_guess4,lgrhodm0_bounds,lgsigma0_bounds,disk,rho1,M1,r)
                delta2_4 = sol[-1]
                if delta2_4 >1e-5: merge = True
    return merge

def stitchSIDMcore_original(r1,halo,disk,r_list=None,N=500):
    """
    Find the isothermal SIDM core that is stitched smoothly to the 
    CDM-like outskirt. 
    
    The original version in Jiang, et al., 2023. This function cannot 
    identify high-density or low-density solutions, and do not work
    when the two solution branches merge. Support Hernquist profile
    only for disk.
    
    Syntax:
    
        stitchSIDMcore(r1,h,N=500)
        
    where
    
        r1: the characteristic radius of a SIDM halo at which an average 
            particle is scattered once during the age of the halo [kpc]
            (float)
        halo: the halo profile for the CDM-like outskirt 
            (a density profile object)
        disk: the galaxy profile (a Hernquist object)
        r_list: positions that we need to know the corresponding Vcirc.
        N: length of the isothermal-core profile to be returned
            (int, default=500)
    
    Return:
        central DM density [M_sun/kpc^3] (float),
        central DM velocity dispersion [kpc/Gyr] (float),
        density profile out to r1 [M_sun/kpc^3] (array of length N),
        circular velocity profile out to r1 [M_sun] (array of length N),
        radii for the profile [kpc] (array of length N)
        circular velocity at radii of r_list
    """
    # prepare a few quantities
    sigmaCDM1 = halo.sigma(r1)
    rhoCDM1 = halo.rho(r1)
    MCDM1 = halo.M(r1)
    rhoCDMres = halo.rho(cfg.Rres)
    rhob0 = disk.rho0
    r0 = disk.r0
    # specify the initial guess and the searching range
    weight = 0.8
    lgrhodm0_init = weight*np.log10(rhoCDM1)+(1.-weight)*np.log10(rhoCDMres)
    lgsigma0_init = np.log10(sigmaCDM1)
    lgrhodm0_lo = np.log10(rhoCDM1)
    lgrhodm0_hi = np.log10(rhoCDMres)
    lgsigma0_lo = np.log10(0.5*sigmaCDM1)
    lgsigma0_hi = np.log10(2.0*sigmaCDM1)
    # define the radius array (insert r_list into rtemp)
    if r_list is None: r_list = []
    r, key = _create_radius_array(r1, r_list, N)  
    # minimize
    res = minimize(delta,[lgrhodm0_init,lgsigma0_init],
        args=(rhob0,r0,rhoCDM1,MCDM1,r),
        bounds=((lgrhodm0_lo,lgrhodm0_hi),(lgsigma0_lo,lgsigma0_hi)),
        )
    # compute the profile to be returned
    delta2 = res.fun
    rhodm0 = 10.**res.x[0]
    sigma0 = 10.**res.x[1]
    a = cfg.FourPiG * r0**2 *rhodm0 / sigma0**2
    b = cfg.FourPiG * r0**2 *rhob0 / sigma0**2
    rho = rhodm0 * np.exp(h(r/r0,a,b))
    M = Miso(r,rho)
    Vc = np.sqrt(cfg.G*M/r)
    # compute the circular velocity at radii of r_list
    Vc1 = []
    for i in key:
        Vc1.append(Vc[i])
    Vc1 = np.array(Vc1)
    rho1 = []
    for i in key:
        rho1.append(rho[i])
    rho1 = np.array(rho1)
    return rhodm0,sigma0,rho,Vc,r,Vc1,rho1,delta2  ## notice that Vc1 here are only for r < r1 

def stitchSIDMcore2(r1,rho1,M1,halo,disk,r_list=None,N=500):
    """
    Find the isothermal SIDM core that is stitched smoothly to the 
    CDM-like outskirt.

    An improved version, which can check wheter the two solution branches
    have merged and identify both if not. This funcion can work without
    knowing p_merge, but slower and less robust than stitchSIDMcore_given_pmerge.
    
    Syntax:
    
        stitchSIDMcore2(r1,rho1,M1,halo,r_list,N=500)
        
    where
    
        r1: the characteristic radius of a SIDM halo at which an average 
            particle is scattered once during the age of the halo [kpc]
            (float)
        rho1: DM density at r1 (float)
        M1: enclosed DM mass within r1 (float)
        halo: the halo profile for the CDM-like outskirt 
            (a density profile object)
        disk: the galaxy profile (a Hernquist object)
        r_list: positions that we need to compute the corresponding Vcirc (a list or np.adarray)
        N: length of the isothermal-core profile to be returned
            (int, default=500)
    
    Return:
        low density solution (list)
        high density solution (list)
        the state of two solution branches: if merge, True; else False (bool)
        wheter two different solutions are identified (bool)
        
        A low/high-density solution contains:
        
        central DM density [M_sun/kpc^3] (float),
        central DM velocity dispersion [kpc/Gyr] (float),
        density profile out to r1 [M_sun/kpc^3] (array of length N + len(r_list)),
        circular velocity profile out to r1 [M_sun] (array of length N + len(r_list)),
        radii for the profile [kpc] (array of length N + len(r_list))
        circular velocity at radii of r_list (array of length len(r_list))
        density at radii of r_list (array of length len(r_list))
        delta^2 (float)
    
    ** If merge = True, both solutions are physically invalid (delta2 > 1e-5).
       If find2sol = True, will return the low-density solution and high-density solution.
       If merge = False, find2sol = False, only 1 solution is found. Typically this happens
       when the p=sigma*tage is very samll. And the solution is the low-density solution.
    """

    #### preparations for solution search ####

    # prepare a few quantities
    sigmaCDM1 = halo.sigma(r1) # approximate sigma(r1_contra) using sigma(r1_init)
    _,rhoCDMres,_ = gh.r1_direct_contra(cfg.Rres,halo,disk)

    # define the radius array (insert r_list into rtemp)
    if r_list is None: r_list = []
    r, key = _create_radius_array(r1, r_list, N)
    
    # specify the searching range and initial guesses
    lgrhodm0_bounds = (np.log10(rho1),np.log10(rhoCDMres))
    lgsigma0_bounds = (np.log10(0.5*sigmaCDM1),np.log10(4.0*sigmaCDM1))

    lgrhodm0_initial_bounds = [np.log10(rho1),np.log10(rhoCDMres)]
    lgsigma0_initial_bounds = [np.log10(0.8*sigmaCDM1),np.log10(1.2*sigmaCDM1)] # range for generating initial guesses

    initial_guess1 = _generate_initial_guess(lgrhodm0_initial_bounds,lgsigma0_initial_bounds)
    initial_guess2 = _generate_initial_guess(lgrhodm0_initial_bounds,lgsigma0_initial_bounds)
    
    #### search for solutions ####

    # set up flags
    merge = False # True when finding 0 valid solution for the system
    find2sol = False # True when finding 2 valid solution with different central density for the system

    # check if the system has merged and try searching for 1 solution
    sol1 = _search_for_solution(initial_guess1,lgrhodm0_bounds,lgsigma0_bounds,disk,rho1,M1,r)
    sol2 = _search_for_solution(initial_guess2,lgrhodm0_bounds,lgsigma0_bounds,disk,rho1,M1,r)
    if sol1[-1] > 1e-5 and sol2[-1] > 1e-5:
        initial_guess3 = _generate_initial_guess(lgrhodm0_initial_bounds,lgsigma0_initial_bounds)
        sol3 = _search_for_solution(initial_guess3,lgrhodm0_bounds,lgsigma0_bounds,disk,rho1,M1,r)
        merge = sol3[-1] > 1e-5
        if not merge: sol1 = sol3
    elif sol1[-1] < 1e-5 and sol2[-1] < 1e-5 and np.abs(np.log10(sol1[0]/sol2[0])) > 1e-3: 
        # find two valid solutions, and their central density is different
        find2sol = True  
    
    # if not find two solutions or merged, launch a loop to find another solution
    if  (not find2sol) and (not merge): 
        sol_valid = sol1 if sol1[-1] < 1e-5 else sol2 # stand for the valid solution
        # try to find another solution
        Nloop = 0
        while (not find2sol) and Nloop < 20:
            initial_guess = _generate_initial_guess(lgrhodm0_initial_bounds,lgsigma0_initial_bounds)
            sol_temp = _search_for_solution(initial_guess,lgrhodm0_bounds,lgsigma0_bounds,disk,rho1,M1,r)
            if np.abs(np.log10(sol_valid[0]/sol_temp[0])) > 1e-3 and sol_temp[-1] < 1e-5: find2sol = True
            Nloop += 1             
        # manage results
        sol1 = sol_valid # if not find two solutions, let sol1 always be the valid one
        sol2 = sol_temp

    #### manage outputs ####

    if find2sol and sol1[0] > sol2[0]: 
        # let sol1 be the low-density solution, sol2 be the high-density solution
        sol1,sol2 = sol2,sol1 
    lowrhosol = _compute_profile_for_solution(sol1,r,key,disk)
    highrhosol = _compute_profile_for_solution(sol2,r,key,disk)

    return lowrhosol,highrhosol,merge,find2sol

def stitchSIDMcore_given_pmerge(r1,rho1,M1,halo,disk,p,pmerge,
                                rhomerge=None,r_list=None,N=500):
    """
    Find the isothermal SIDM core that is stitched smoothly to the 
    CDM-like outskirt. See appendix A in Jia, et al., 2026 for details.

    An improved version, which can more effeciently and robustly 
    identify the proper solution for the input p. Namely it returns
    low-density solution when p < pmerge and mirrored high-density
    (core-collapse) solution when p > pmerge. To make it work, pmerge
    must be given. 

    This algorithm divides the parameter space of central density rhodm0 
    and product of effective cross section and halo age p into four regimes
    (See appendix A in Jia, et al., 2026 for details):

        A: rhodm0 > rhomerge, p < p1. Early core-growth. The central 
           density of low-density solution is larger than rhomerge.
           We assume high-density solution does not exist at regime A.
        B: rhodm0 > rhomerge, p1 < p < pmerge. The regime that 
           high-density solution exists. The central density of 
           high-density solution is higher than rhomerge.        
        C: rhodm0 < rhomerge, p1 < p < pmerge. Core-growth and 
           maximal-core. The central density of low-density solution 
           is lower than rhomerge.
        D: rhodm0 > rhomerge, p > pmerge. Core-collapse. The solution 
           is appoximated by the high-density solution at regime C.
    
    where,

        p1 : the product at which the central density of low-density 
             solution is equal to rhomerge.
        pmerge: the product at which the two solution branches of Jeans model merge
        rhomerge: central DM density of the halo at pmerge

    !! Caution that when using this function, DON'T INPUT a mirrored
       p when p > pmerge. You should give the initial value. But for the input
       r1, you SHOULD use the mirrored value when computing it.
    
    Syntax:
    
        stitchSIDMcore_given_pmerge(r1,rho1,M1,halo,disk,p,pmerge,
                                rhomerge=None,r_list=None,N=500)
        
    where
    
        r1: the characteristic radius of a SIDM halo at which an average 
            particle is scattered once during the age of the halo [kpc]
            (float)
        rho1: DM density at r1 (float)
        M1: enclosed DM mass within r1 (float)
        halo: the halo profile for the CDM-like outskirt 
            (a density profile object)
        disk: the galaxy profile (a Hernquist object)
        p: product of effective cross section and halo age (float)
        pmerge: the product at which the two solution branches of Jeans model merge (float)
        rhomerge: central DM density of the halo at pmerge (float)
        r_list: positions that we need to compute the corresponding Vcirc (a list or np.adarray)
        N: length of the isothermal-core profile to be returned
            (int, default=500)
    
    Return:
        solution of the Jeans model (list)
        
        which contains:
        
        central DM density [M_sun/kpc^3] (float),
        central DM velocity dispersion [kpc/Gyr] (float),
        density profile out to r1 [M_sun/kpc^3] (array of length N + len(r_list)),
        circular velocity profile out to r1 [M_sun] (array of length N + len(r_list)),
        radii for the profile [kpc] (array of length N + len(r_list))
        circular velocity at radii of r_list (array of length len(r_list))
        density at radii of r_list (array of length len(r_list))
        delta^2 (float)
    """
        
    # prepare a few quantities
    sigmaCDM1 = halo.sigma(r1) # approximate sigma(r1_contra) using sigma(r1_init)
    _,rhoCDMres,_ = gh.r1_direct_contra(cfg.Rres,halo,disk) # density at a very small r

    # define the radius array (insert r_list into rtemp)
    if r_list is None: r_list = []
    r, key = _create_radius_array(r1, r_list, N)
    
    # specify the searching range and initial guess range
    lgrhodm0_bounds = (np.log10(rho1),np.log10(rhoCDMres))
    lgsigma0_bounds = (np.log10(0.5*sigmaCDM1),np.log10(4.0*sigmaCDM1))
    lgsigma0_initial_bounds = (np.log10(0.8*sigmaCDM1),np.log10(1.2*sigmaCDM1))

    # compute rhomerge if it's not given
    if rhomerge is None: rhomerge = _compute_rhodm0_merge(halo,disk,pmerge=pmerge)

    # partition the search intervals of central density for the two solution branches
    lgrhodm0_lowrho_bounds = (np.log10(rho1),np.log10(rhomerge))
    lgrhodm0_highrho_bounds = (np.log10(rhomerge),np.log10(rhoCDMres)) 

    #### search for solutions ####

    # p > pmerge, return the high-density solution
    if p > 2*pmerge:
        # In this case the halo collapses, and Jeans model cannot work
        sol_temp = [np.log10(rhoCDMres),sigmaCDM1,1] # randomly chosen value for rhodm0 and sigma0
        solution = _compute_profile_for_solution(sol_temp,r,key,disk)
        return solution
    elif p > pmerge:
        # correspond to regime D, approximate the solution using the high-density solution in regime B.
        for Nattemp in range(20):
            initial_guess = _generate_initial_guess(lgrhodm0_highrho_bounds,lgsigma0_initial_bounds)
            sol = _search_for_solution(initial_guess,lgrhodm0_highrho_bounds,lgsigma0_bounds,disk,rho1,M1,r)
            if sol[-1] < 1e-5: break

    # p < pmerge, return the low-density solution
    else: 
        if lgrhodm0_lowrho_bounds[0] >= lgrhodm0_lowrho_bounds[1]:
            # Deal with the special case that the rhomerge is very low.
            # When rho1 > rhomerge and p < pmerge, the density at r1 is very high and r1 
            # should be very small, corresponding to a low p (early core-growth). Then 
            # we assume in this case there is no high-density solution. 
            # Correspond to regime A.
            for Nattemp in range(20):
                # In this case, p is very small, the halo is at early core-growth stage,
                # and the central density is high, locating in the high-density interval.
                initial_guess = _generate_initial_guess(lgrhodm0_highrho_bounds,lgsigma0_initial_bounds)
                sol = _search_for_solution(initial_guess,lgrhodm0_highrho_bounds,lgsigma0_bounds,disk,rho1,M1,r)
                if sol[-1] < 1e-5: break
        else:
            # Try searching for a solution with rhodm0 < rhomerge, or at regime C.
            # By this step, we avoid computing the precise value of p1.
            for Nattemp in range(20):
                initial_guess = _generate_initial_guess(lgrhodm0_lowrho_bounds,lgsigma0_initial_bounds)
                sol_temp = _search_for_solution(initial_guess,lgrhodm0_lowrho_bounds,lgsigma0_bounds,disk,rho1,M1,r)
                if sol_temp[-1] < 1e-5: break
            
            if sol_temp[-1] < 1e-5:
                # find a valid low-density solution at regime C. 
                sol = sol_temp
            else:
                # Do not find a valid low-density solution at regime C after thorough search.
                # The solution must locate at regime A.
                for Nattemp in range(20):
                    initial_guess = _generate_initial_guess(lgrhodm0_highrho_bounds,lgsigma0_initial_bounds)
                    sol = _search_for_solution(initial_guess,lgrhodm0_highrho_bounds,lgsigma0_bounds,disk,rho1,M1,r)
                    if sol[-1] < 1e-5: break
    
    #### manage outputs ####

    solution = _compute_profile_for_solution(sol,r,key,disk)

    return solution

def _create_radius_array(r1, r_list, N):
    """
    Create a combined radius array for SIDM profile computation,
    incorporating both user-specified radii and a logarithmic grid.
    
    This function generates a radial grid that spans from a small inner
    radius to the matching radius r1, ensuring that user-specified radii
    (e.g., for observables) are included in the grid.
    
    Syntax:
        r, key = _create_radius_array(r1, r_list, N)
    
    where:
        r1: characteristic radius [kpc] (float)
        r_list: list of user-specified radii to include [kpc] (list or np.nparray)
        N: number of points in the logarithmic grid (int)
    
    Return:
        r: combined radius array [kpc], sorted in ascending order
        key: indices of the user-specified radii in the combined array
             (empty list if r_list is empty)
    """
    if len(r_list) == 0: return np.logspace(-3.0, np.log10(r1), N), []
    r_log = np.logspace(-3.0, np.log10(r1), N)

    # Combine user-specified radii with logarithmic grid
    r_combined = np.sort(np.unique(np.concatenate([r_log, r_list])))

    # Find indices of r_list values in combined array
    key_indices = np.where(np.isin(r_combined, r_list))[0]
    return r_combined, key_indices

def _generate_initial_guess(lgrhodm0_initial_bounds,lgsigma0_initial_bounds):
    """
    Generate a random initial guess for the Jeans equation optimization
    within specified bounds.
    
    This helper function creates random starting points in parameter space
    for finding SIDM core solutions via numerical optimization. Random
    sampling helps avoid getting stuck in saddle points.
    
    Syntax:
        _generate_initial_guess(lgrhodm0_initial_bounds, lgsigma0_initial_bounds)
        
    where

        lgrhodm0_initial_bounds: tuple (min, max) for log10(central DM density)
                                 [log10(M_sun/kpc^3)]
        lgsigma0_initial_bounds: tuple (min, max) for log10(central velocity dispersion)
                                 [log10(kpc/Gyr)]
    
    Return:
        List containing [lgrhodm0, lgsigma0] as random initial guess
    """
    if lgrhodm0_initial_bounds[0] > lgrhodm0_initial_bounds[1]:
        # physically invalid search boundary, return a randomly chosen result
        return [lgrhodm0_initial_bounds[0],lgsigma0_initial_bounds[0]]
    lgrhodm0_random = np.random.uniform(lgrhodm0_initial_bounds[0], lgrhodm0_initial_bounds[1])
    lgsigma0_random = np.random.uniform(lgsigma0_initial_bounds[0], lgsigma0_initial_bounds[1])
    return [lgrhodm0_random, lgsigma0_random]

def _compute_rhodm0_merge(halo,disk,pmerge=None):
    """
    Find the central DM density at the merge point by solving the Jeans
    equation with multiple random initial guesses.
    
    This function attempts to find a valid solution at the critical 
    merge timescale, where the low-density and high-density
    solution branches coalesce. It uses randomized starting points to
    improve robustness in the optimization landscape.
    
    Syntax:
        _compute_rhodm0_merge(halo,disk,pmerge)
    
    where
        halo: the halo profile for the CDM-like outskirt 
            (a density profile object)
        disk: galaxy disk profile (Hernquist or exp object)
        pmerge: the product value when the two solution branch of
               the Jeans model merge.
    
    Return:
        Central DM density at merge point [M_sun/kpc^3] (float)
        Returns None if no solution found after maximum attempts
    """
    # compute pmerge if pmerge is not given
    if pmerge == None: _,pmerge = tmerge(halo,disk)

    # compute r1 at pmerge
    r1merge,rho1,M1 = r1(halo,sigmamx=1,tage=pmerge,disk=disk)

    # prepare a few quantities
    sigmaCDM1 = halo.sigma(r1merge) # approximate sigma(r1_contra) using sigma(r1_init)
    _,rhoCDMres,_ = gh.r1_direct_contra(cfg.Rres,halo,disk) # density at a very small r

    # define the radius array
    r = np.logspace(-3,np.log10(r1merge),500)
    
    # specify the searching range and initial guess range
    lgrhodm0_bounds = (np.log10(rho1),np.log10(rhoCDMres))
    lgsigma0_bounds = (np.log10(0.5*sigmaCDM1),np.log10(4.0*sigmaCDM1))
    lgsigma0_initial_bounds = (np.log10(0.8*sigmaCDM1),np.log10(1.2*sigmaCDM1))

    # initialization
    Findsolution,Nloop = False,0

    while not Findsolution:
        # generate random initial guess within bounds
        initial_guess = _generate_initial_guess(lgrhodm0_bounds,lgsigma0_initial_bounds)

        # search for the solution
        if isinstance(disk,Hernquist):
            rhob0,r0 = disk.rho0,disk.r0
            res = minimize(delta,initial_guess,
                args=(rhob0,r0,rho1,M1,r),
                bounds=(lgrhodm0_bounds,lgsigma0_bounds),
                method = 'Nelder-Mead'
                )
        elif isinstance(disk,exp):
            Mb,r0 = disk.Mb,disk.r0
            res = minimize(delta_exp,initial_guess,
                    args=(Mb,r0,rho1,M1,r),
                    bounds=(lgrhodm0_bounds,lgsigma0_bounds),
                    method = 'Nelder-Mead'
                    )
            
        # evaluate solution quality
        delta2 = res.fun
        rhodm0 = 10.**res.x[0]

        # check if the solution is physically valid
        if delta2 < 1e-3: 
            # Here the threshold is larger than 1e-5, because the system is at the critical stage
            Findsolution = True 
        Nloop += 1
        if Nloop > 20: 
            print("WARNING: cannot find a solution at pmerge!")
            break

    return rhodm0

def _search_for_solution(initial_guess,lgrhodm0_bounds,lgsigma0_bounds,disk,rho1,M1,r):
    """
    Search for a solution by minimizing the Jeans equation mismatch.
    This function performs a single optimization attempt to find central
    density and velocity dispersion that satisfy the Jeans equation with
    boundary conditions at r1.
    
    Syntax:
        _search_for_solution(initial_guess, lgrhodm0_bounds, lgsigma0_bounds,
                             disk, rho1, M1, r)
    
    where
        initial_guess: list [lgrhodm0, lgsigma0] as starting point for optimization
        lgrhodm0_bounds: tuple (min, max) for log10(central DM density)
        lgsigma0_bounds: tuple (min, max) for log10(central velocity dispersion)
        disk: baryon profile (Hernquist or exp object)
        rho1: DM density at r1 [M_sun/kpc^3] (float)
        M1: DM mass within r1 [M_sun] (float)
        r: radial grid array [kpc] (np.ndarray)
    
    Return:
        List containing
        - rhodm0: central DM density [M_sun/kpc^3] (float)
        - sigma0: central velocity dispersion [kpc/Gyr] (float)
        - delta2: objective function residual (mismatch squared) (float)
    
    Note: For degenerate cases (lgrhodm0 bounds inconsistent), returns
          a placeholder solution with delta2 > 1e-5 to indicate failure.
    """
    # Deal with the special case that the lower boundary of rhodm0
    # is higher than the upper boundary.
    if lgrhodm0_bounds[0] >= lgrhodm0_bounds[1]: 
        # In this case, this function will return a invalid solution
        # with delta^2 > 1e-5 and other quantaties randomly chosen.
        return [lgrhodm0_bounds[0],lgsigma0_bounds[0],1]
    
    # search for the solution
    if isinstance(disk,Hernquist):
        rhob0,r0 = disk.rho0,disk.r0
        res = minimize(delta,initial_guess,
            args=(rhob0,r0,rho1,M1,r),
            bounds=(lgrhodm0_bounds,lgsigma0_bounds),
            method = 'Nelder-Mead'
            )
    elif isinstance(disk,exp):
        Mb,r0 = disk.Mb,disk.r0
        res = minimize(delta_exp,initial_guess,
                args=(Mb,r0,rho1,M1,r),
                bounds=(lgrhodm0_bounds,lgsigma0_bounds),
                method = 'Nelder-Mead'
                )
    
    # Extract optimization results
    delta2 = res.fun
    rhodm0 = 10.**res.x[0]
    sigma0 = 10.**res.x[1]

    return [rhodm0,sigma0,delta2]
        
def _compute_profile_for_solution(sol,r,key,disk):
    """
    Compute the full SIDM halo profile from solution parameters.
    Given optimized central density and velocity dispersion, this function
    reconstructs the complete density and circular velocity profiles for
    the SIDM inner halo, assuming an isothermal solution matched to the 
    outer CDM profile at r1.
    
    Syntax:
        _compute_profile_for_solution(sol, r, key, disk)
        
    where
        sol: solution list from _search_for_solution containing
             [rhodm0, sigma0, delta2]
        r: radial grid array out to r1 [kpc] (np.ndarray)
        key: indices specifying which radii correspond to requested r_list (np.ndarray)
        disk: baryon profile (Hernquist or exp object)
    
    Return:
        List containing
        - rhodm0: central DM density [M_sun/kpc^3] (float)
        - sigma0: central velocity dispersion [kpc/Gyr] (float)
        - rho: density profile array [M_sun/kpc^3] (np.ndarray of length len(r))
        - Vc: circular velocity profile array [kpc/Gyr] (np.ndarray of length len(r))
        - r: radial grid array [kpc] (same as input)
        - Vc1: circular velocity at requested radii [kpc/Gyr] (np.ndarray of length len(key))
        - rho1: density at requested radii [M_sun/kpc^3] (np.ndarray of length len(key))
        - delta2: objective function residual (float)
    
    Note: Vc1 and rho1 values are only valid for radii r < r1 where the
          isothermal solution applies. Beyond r1, the profile transitions
          to the outer CDM halo.
    """
    # unpack solution parameters
    rhodm0,sigma0,delta2 = sol

    # compute SIDM density profile based on baryon profile
    if isinstance(disk,exp):
        Mb,r0 = disk.Mb,disk.r0
        a = cfg.FourPiG * r0**2 *rhodm0 / sigma0**2
        b = cfg.G * Mb / sigma0**2 / r0
        rho = rhodm0 * np.exp(h_exp(r/r0,a,b))
    elif isinstance(disk,Hernquist):
        rhob0,r0 = disk.rho0,disk.r0
        a = cfg.FourPiG * r0**2 *rhodm0 / sigma0**2
        b = cfg.FourPiG * r0**2 *rhob0 / sigma0**2
        rho = rhodm0 * np.exp(h(r/r0,a,b))

    # compute mass and circular velocity profile by integrating density
    M = Miso(r,rho)
    Vc = np.sqrt(cfg.G*M/r)

    # extract values at requested radii
    Vc1 = Vc[key] 
    rho1 = rho[key] 

    return [rhodm0,sigma0,rho,Vc,r,Vc1,rho1,delta2]

#-- compute effective SIDM cross section in Yang, et al.,2022
def func_sigmaeff(u, ke, sigma0):
    """
    Integrand for effective cross section calculation.
    Returns integrand value for given u, ke, and sigma0.
    Here the integration over angles in Eq8 in Jia, et al., 2026/
    Eq4.2 in Yang, et al., 2022 has been performed.

    Here,
    Veff = 0.64 * Vmax
    ke = Veff^2/(2*omega^2) 
    k = v^2/(2*omega^2)
    u = k/(4*ke)

    And
    sigmaeff = sigma0/(1024*ke^4)*\Int_0^{/inf} (-2*k+(1+k)*ln(1+2k))*exp(-k/(4*ke))*dk
             = sigma0/(256*ke^3)*\Int_0^{/inf} (-8*ke*u+(1+4*ke*u)*ln(1+8*ke*u))*exp(-u)*dk
    
    The upper limit of the integration should be the escape velocity,
    but we have confirmed that the infinity is a good approximation.
    
    Args:
        u: integration variable (dimensionless velocity)
        ke: dimensionless parameter = Veff^2/(2*omega^2)
        sigma0: static cross section [cm^2/g]
    
    Returns:
        Integrand value at u
    """
    # Replace k with 4*ke*u in the cross section formula
    # Note: reliable for omega <~ 5000 km/s; unstable at extremely high omega
    term1 = -8 * ke * u
    term2 = (1 + 4 * ke * u) * np.log(1 + 8 * ke * u)
    return sigma0 / 256. / ke**3 * (term1 + term2) * np.exp(-u)

def compute_sigmaeff(Vmax, sigma0, omega):
    """
    Compute effective cross section for velocity-dependent SIDM
    using Eq8 in Jia, et al., 2026/ Eq4.2 in Yang, et al., 2022
    
    Args:
        Vmax: maximum circular velocity [km/s] (float, array, or list)
        sigma0: static cross section [cm^2/g] (float)
        omega: scale velocity for velocity-dependence [km/s] (float)
    
    Returns:
        sigmaeff: effective cross section [cm^2/g]
    """
    # Effective velocity: Veff = 0.64 * Vmax
    Veff = 0.64 * Vmax
    # Dimensionless parameter: ke = Veff^2/(2*omega^2)
    ke = Veff**2 / (2. * omega**2)
    
    # Convert list to numpy array for uniform handling
    if isinstance(Vmax, list):
        Vmax = np.array(Vmax)
    
    # Initialize result array
    if isinstance(Vmax, np.ndarray):
        sigmaeff = np.zeros_like(Vmax, dtype=float)
    else:  # scalar case
        sigmaeff = 0.0
    
    # Handle both scalar and array inputs efficiently
    if np.isscalar(Vmax) or isinstance(Vmax, float) or isinstance(Vmax, int):
        # Scalar case: single integration
        sigmaeff, error = integrate.quad(func_sigmaeff, 0, np.inf, args=(ke, sigma0))
    else:
        # Array case: loop through each element
        for i in range(len(Vmax)):
            sigmaeff[i], error = integrate.quad(
                func_sigmaeff, 0, np.inf, args=(ke[i], sigma0)
            )
    
    return sigmaeff

def create_sigmaeff_vmax_interpolation(Vmax_range, sigma0, omega, n_points=100, save_path=None):
    """
    Create a 1D interpolation table for sigmaeff as a function of Vmax for fixed sigma0 and omega.
    
    Args:
        Vmax_range: Range of Vmax (min, max) [km/s]
        sigma0: Fixed static cross section [cm^2/g] (float)
        omega: Fixed scale velocity [km/s] (float)
        n_points: Number of sampling points for Vmax grid
        save_path: Path to save the interpolation table (optional)
    
    Returns:
        interp_func: 1D interpolation function sigmaeff = f(Vmax)
        Vmax_grid: Array of Vmax grid points
        sigmaeff_grid: Array of corresponding sigmaeff values
    """
    
    # Create Vmax grid (logarithmically spaced for better coverage)
    Vmax_min, Vmax_max = Vmax_range
    Vmax_grid = np.logspace(np.log10(Vmax_min), np.log10(Vmax_max), n_points)
    
    # Initialize sigmaeff array
    sigmaeff_grid = np.zeros(n_points)
    
    # Calculate sigmaeff for each Vmax point
    for i, Vmax in enumerate(Vmax_grid):
        sigmaeff_grid[i] = compute_sigmaeff(Vmax, sigma0, omega)
    
    # Create interpolation function
    interp_func = interp1d(Vmax_grid, sigmaeff_grid, 
                          kind='cubic',  # Cubic interpolation for smoothness
                          bounds_error=False,
                          fill_value=(sigmaeff_grid[0], sigmaeff_grid[-1]))
    
    # Save data if requested
    if save_path:
        import pickle
        save_data = {
            'Vmax_grid': Vmax_grid,
            'sigmaeff_grid': sigmaeff_grid,
            'sigma0': sigma0,
            'omega': omega,
            'Vmax_range': Vmax_range,
            'interp_func': interp_func
        }
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

    return interp_func, Vmax_grid, sigmaeff_grid
