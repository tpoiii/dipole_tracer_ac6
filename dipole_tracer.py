"""
test particle tracer in a dipole field
Uses Schulz and Lanzerotti 1974 and Northrop and Teller 1960


module variables:
debug (bool) - turn on debugging
    make fly_in_scene run 10x faster
N_POOL (int) - number of processes to use (default: 4)
VWdata - global VWtable that holds V and W integrals on grid

"""

import os
import warnings
from multiprocessing import Pool,Manager
import numpy as np
from scipy import constants
from scipy.integrate import quad,trapz,IntegrationWarning
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import cumtrapz
from scipy.special import erf
import matplotlib.pyplot as plt
import pickle
import datetime as dt
from tictoc import tic,toc

N_POOL = 4 # number of processes to use

debug = False

# physical constants
B0nT = 31e3 # dipole field strength, nT (S&L value)
akm = 6371 # Earth radius, km
cSI = constants.c # speed of light m/s
meSI = constants.m_e # electron mass, kg
eSI = constants.e # elementary charge, C
eV = constants.e # electron Volt, J
keV = 1e3*eV # keV, J
MeV = 1e6*eV # MeV, J
mekeV = meSI*cSI**2/keV
meMeV = mekeV/1e3


VWdata = None # global VWtable object

def beNice():
    """ check process nice level. 
    Renice to achieve a nice level of 5 if needed """
    nice = os.nice(0) # add zero to nice and return it
    if nice < 5:
        print('Renicing by %d' % (5-nice))
        os.nice(5-nice) # increment nice to 5

# dipole magnetic field and derivatives
# conventions: 
#  location is passed as L,theta
#  theta in radians
#  B in nT
#  distance (ds) in km
#  vectors are returned as (Br,Btheta) tuple
def Bmag(L,theta):
    """ return magnitude of B in nT at specificed L, theta
        Bmag(L,theta)
        theta in radians
    """
    B = B0nT/L**3*np.sqrt(1+3*np.cos(theta)**2)/np.sin(theta)**6
    return B
    
def Bvec(L,theta):
    """ return (Br,Btheta) in nT at specificed L, theta
        Bvec(L,theta)
        theta in radians
    """
    A = -B0nT/L**3/np.sin(theta)**6 # prefactor
    return (2*np.cos(theta)*A,np.sin(theta)*A)

def gradB(L,theta):
    """ return (dB/dr,dB/dtheta) in (nT/km,nT) at specificed L, theta
        gradB(L,theta)
        theta in radians
    """
    B = Bmag(L,theta)
    A = 3*B/L/akm # prefactor
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    return (-A/sintheta**2,-A*costheta/sintheta/(1+3*costheta**2))

def dBds(L,theta):
    """ return dB/ds in nT/km at specified L,theta
        dBds(L,theta)
        theta in radians
    """
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    return 3*B0nT/L**4/akm*costheta*(1+7*costheta**2)/sintheta**8/(1+3*costheta**2)

def dBvecds(L,theta):
    """ return (dBr/ds,dBtheta/ds) in nT/km at specified L,theta
        dBvecds(L,theta)
        theta in radians
    """
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    A = -3*B0nT/L**4/akm/sintheta**8/np.sqrt(1+3*costheta**2)
    return(A*(1+3*costheta**2),A*sintheta*costheta)

def dbds(L,theta):
    """ return (dbr/ds,dbtheta/ds) in 1/km at specified L,theta
        dbds(L,theta)
        theta in radians
    """
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    A = -3./L/akm/sintheta/(1+3*costheta**2)**2
    return (A*sintheta*(1+5*costheta**2),-4*A*costheta**3)

def Vint(lam,lambdam):
    """
    V = Vint(lam,lambdam)
    returns V integral, which is part of the normalized travel time
    from the equator to magnetic latitude lambda for a particle with
    mirror latitude lambdam
    lam,lambdam are scalars in radians    
    Note: does not handle lambdam=0 case
    """
    if (lambdam == 0) or (lam == 0):
        return 0.0
    if lam>lambdam:
        lam = lambdam
    C = np.sqrt(1+3*np.sin(lambdam)**2)/np.cos(lambdam)**6
    func = lambda lami :  np.cos(lami)*np.sqrt(1+3*np.sin(lami)**2)/np.sqrt(np.maximum(0,1-np.sqrt(1+3*np.sin(lami)**2)/C/np.cos(lami)**6))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=IntegrationWarning)
        res = quad(func,0,lam) # execution time: ~1 kHz

    return res[0]

def Wint(lam,lambdam):
    """
    W = Wint(lam,lambdam)
    returns W integral, which is part of the normalized drift distance
    as a particle moves from equator to magnetic latitude lambda
    particle has mirror latitude lambdam
    lam,lambdam are scalars in radians    
    Note: does not handle lambdam=0 case
    """
    if (lambdam == 0) or (lam == 0):
        return 0.0
    if lam>lambdam:
        lam = lambdam
    C = np.sqrt(1+3*np.sin(lambdam)**2)/np.cos(lambdam)**6
    BBm = lambda lami : np.sqrt(1+3*np.sin(lami)**2)/C/np.cos(lami)**6 # B/Bm
    func = lambda lami :  3*(1+np.sin(lami)**2)/np.cos(lami)**3/(1+3*np.sin(lami)**2)*(1-2/BBm(lami))/np.sqrt(np.maximum(0,1-BBm(lami)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=IntegrationWarning)
        res = quad(func,0,lam) # execution time: ~1 kHz

    return res[0]


class VWtable(object):
    """ table = VWtable(Nv=300,maxLambdam=None,store=True)
    class for holding pre-computed table of V and W integrals
    and providing interpolation functions
    Parameters:
        Nv - number of latitudes in latitude grid
        maxLambdam - maximum latitude in grid, radians (default is L=15 at Earth's surface)
        store - precompute and store V,W table vs do on demand every time
    data properties:
        Ilambda - latitude grid, radians (Nv,)
        Ialphaeq - equivalent equatorial pitch angle for mirror lats, radians (Nv,)
        if store==True:
        IVint - V integral from Vint (Nv,Nv) IVint[i,j] = Vint(Ilambda[j],Ilambda[i])
        IWint - W integral from Wint (Nv,Nv) IWint[i,j] = Wint(Ilambda[j],Ilambda[i])
        (Note IVint and IWint at lam>lambdam are replaced by values lam=lambdam)
    Methods:
        Vint(lam,lambdam) - perform V integral by quadrature
        Wint(lam,lambdam) - perform W integral by quadrature 
        V(lam,lambdam) - interpolate V by 2-d linear interpolation
        W(lam,lambdam) - interpolate W by 2-d linear interpolation        
    """
    def __init__(self,Nv=300,maxLambdam=None,store=True):
        if Nv is None:
            Nv = 300
        self.Nv = Nv
        
        if maxLambdam is None:
            maxLambdam = np.arccos(15**-0.5) # Earth's surface at L=15
        self.maxLambdam = maxLambdam
        
        self.store = store

        Ilambda = np.linspace(0,maxLambdam,Nv)
        y = (1+3*np.sin(Ilambda)**2)**-0.5*np.cos(Ilambda)**3
        Ialphaeq = np.arcsin(y)
        self.Ilambda = Ilambda
        self.Ialphaeq = Ialphaeq        
        if self.store:
            IVint = np.full((Nv,Nv),np.nan)
            IWint = np.full((Nv,Nv),np.nan)
            for (ilam,lami) in enumerate(Ilambda):
                for (jlam,lamj) in enumerate(Ilambda):
                    if lami <= lamj:
                        IVint[jlam,ilam] = Vint(lami,lamj)
                        IWint[jlam,ilam] = Wint(lami,lamj)
                    else:
                        IVint[jlam,ilam] = IVint[jlam,jlam]
                        IWint[jlam,ilam] = IWint[jlam,jlam]
            self.IVint = IVint
            self.IWint = IWint

    def V(self,lam,lambdam):
        """
        V = V(lam,lambdam)
        see Vint
        returns stored or interpolated value based on self.store
        """
        if self.store:
            return interp2d(self.Ilambda,self.Ilambda,self.IVint,kind='linear',copy=False)(lam,lambdam)
        else:
            if np.isscalar(lam):
                return Vint(lam,lambdam)
            else:
                return np.vectorize(Vint)(lam,lambdam)

    def W(self,lam,lambdam):
        """
        W = W(lam,lambdam)
        see Wint
        returns stored or interpolated value based on self.store
        """
        if self.store:
            return interp2d(self.Ilambda,self.Ilambda,self.IWint,kind='linear',copy=False)(lam,lambdam)
        else:
            if np.isscalar(lam):
                return Wint(lam,lambdam)
            else:
                return np.vectorize(Wint)(lam,lambdam)
class VWCache(object):
    """
    cache of V,W integrals
    VWCache(filename=None,V=None,W=None,local=False)    
    if filename provided and exists, loads from file (pickle format)
    remembers filename for future load/save calls
    Can pass in V,W shared dicts when used in parallel context
    Can pass in local to turn off sharing across processes
    methods: load, save, Vint,Wint
    """
    def __init__(self,filename=None,V=None,W=None,local=False):
        self.filename = filename
        self.local = local
        self.manager = None
        if (V is None) and (W is None):
            if local:
                self.V = {}
                self.W = {}
            else:
                self.manager = Manager() # manage shared dicts
                self.V = self.manager.dict()
                self.W = self.manager.dict()
        else:
            self.V = V
            self.W = W
        if (filename is not None) and os.path.isfile(filename):
            self.load(filename)
    def load(self,filename=None):
        """ .load(filename=None)
        loads cache from filename (pickle format)
        remembers filename if provided
        """
        if not os.path.exists(filename):
            raise Exception('Cannot load from %s, does not exist' % filename)
        self.filename = filename
        with open(filename,'rb') as file:
            (V,W) = pickle.load(file)
        # clear then update to not mess up parallelized dict setup
        self.V.clear()
        self.V.update(V)
        self.W.clear()
        self.W.update(W)
    def save(self,filename=None):
        """ .save(filename=None)
        save cache to file (pickle format)
        remembers filename if provided
        """
        if filename is None:
            filename = self.filename
        if filename is None:
            raise Exception('No filename provided to save function')
        self.filename = filename
        with open(filename,'wb') as file:
            pickle.dump((dict(self.V),dict(self.W)),file)
        
    def _make_key(self,lam,lambdam):
        """.make_key(lam,lambdam)
        makes an immuatble dict key from the local (lam) and mirror latitude (lambdam)
        """
        #return (lam,lambdam)
        return (lam,lambdam)
    def Vint(self,lam,lambdam):
        """
        V = Vint(lam,lambdam)
        see Vint module function
        returns stored or computed value
        """
        key = self._make_key(lam,lambdam)
        if key in self.V:
            return self.V[key]
        else:
            val = Vint(lam,lambdam)
            self.V[key] = val
            return val

    def Wint(self,lam,lambdam):
        """
        W = Wint(lam,lambdam)
        see Wint module function
        returns stored or interpolated value based on self.store
        """
        key = self._make_key(lam,lambdam)
        if key in self.W:
            return self.W[key]
        else:
            val = Wint(lam,lambdam)
            self.W[key] = val
            return val
        
def alphaeq2lambdam(alphaeq):
    """
    lambdam = alphaeq2lambdam(alphaeq)
    convert equatorial pitch angle to mirror latitude
    alphaeq in radians (N,) or scalar
    lambdam in radians same shape as alphaeq
    """
    if not np.isscalar(alphaeq):
        return np.vectorize(alphaeq2lambdam)(alphaeq)
    # sin(alphaeq)**2 = cos(lam)**6/sqrt(1+3*sin(lam)**2)
    y = np.sin(alphaeq)**2
    lam = np.arccos(y**(1/6)) # initial guess
    func = lambda lam : y-np.cos(lam)**6/np.sqrt(1+3*np.sin(lam)**2)
    res = fsolve(func,lam)
    return res[0]

class Particle:
    """
    Particle class holds particle's invariants
    (state variables held by ODE solver)
    part = Particle(L,Bm,energykeV=None,M=None,species='e-')
    L is drift shell (modified 3rd invariant)
    Bm is mirror magnetic field in nT
    provide energykeV or M, M in MeV/G
    has properties:
        species - species string (e.g., 'e-')
        q - signed charge, C
        m0 - rest mass, kg
        m0keV - rest mass, keV/c^2
        m0MeV - rest mass, MeV/c^2
        keV - kinetic energy, keV
        MeV - kinetic energy, MeV
        gamma - relativistic factor
        M - 1st invariant, MeV/G
        MSI - 1st invariant in J/T
        L - drift shell
        Bm - mirror magnetic field, nT
        BmG - mirror magnetic field, G
        Beq - equatoria magnetic field, nT
        alphaeq - equatorial pitch angle, rad
        alphaeq_deg - equatorial pitch angle, degrees
        v - velocity, km/s
        m - relativistic mass, kg
        p - momentum, kg m /s
    methods:
        vpar(theta) - returns vparallel at given theta        
        uphi(theta) - returns drift speed at given theta
        
    """
    def __init__(self,L,Bm,energykeV=None,M=None,species='e-'):
        self.L = L
        self.Bm = Bm # nT
        self.BmG = Bm/1e5 # Gauss

        if species == 'e-':
            self.species = species
            self.q = -eSI
            self.m0 = meSI
        else:
            raise Exception('Species %s not implemented' % species)

        self.m0keV = self.m0*cSI**2/keV
        self.m0MeV = self.m0keV/1e3

        if M is None:
            self.keV = energykeV
            self.gamma = 1+energykeV/self.m0keV
            self.M = (self.gamma**2-1)*self.m0MeV/2/self.BmG
        else:
            self.M = M
            self.gamma = np.sqrt(1+2*M*self.BmG/self.m0MeV)
            self.keV = (self.gamma-1)*self.m0keV
            
        self.v = cSI*np.sqrt(1-self.gamma**-2) # m/s
        self.m = self.m0*self.gamma # kg
        self.p = self.m*self.v # kg*m/s
        self.MeV = self.keV/1e3
        self.MSI = self.M*MeV*1e4 # M in J/T
        self.Beq = B0nT/L**3 # nT
        self.alphaeq = np.arcsin(np.sqrt(self.Beq/Bm)) # radians
        self.alphaeq_deg = np.degrees(self.alphaeq)
        
    def __str__(self):
        return ('%s: %s %g keV L=%g Bm=%g (M=%g,alphaeq=%g)' % (self.__class__.__name__,self.species,self.keV,self.L,self.Bm,self.M,self.alphaeq_deg))
    def __repr__(self):
        return self.__str__()
    def vpar(self,theta):
        """
        vpar = .vpar(theta)
        theta in radians
        vpar in m/s
        returns unsigned parallel velocity at theta
        """
        B = Bmag(self.L,theta)
        vpar = np.sqrt(np.maximum(0,1-B/self.Bm))*self.v
        return vpar
    def uphi(self,theta):
        """
        uphi = .uphi(theta)
        theta in radians
        uphi in m/s
        returns drift speed at theta
        """
        BSI = Bmag(self.L,theta)*1e-9 # nT -> T
        BvecSI = np.array(Bvec(self.L,theta))*1e-9 # nT -> T
        bhat = BvecSI/BSI
        gradBSI = np.array(gradB(self.L,theta))*1e-9*1e-3 # nT/km -> T/m
        vpar = self.vpar(theta) # m/s
        dbdsSI = np.array(dbds(self.L,theta))*1e-3 # 1/km -> 1/m
        
        gradBterm = self.MSI/self.gamma/self.q/BSI*gradBSI
        curvBterm = self.m0*self.gamma*vpar**2/self.q/BSI*dbdsSI
        total = gradBterm+curvBterm
        uphi = bhat[0]*total[1]-bhat[1]*total[0] # cross product
        
        return uphi
    def bounce(self,Nv=None):
        """
            bounce = .bounce(Nv=None)
                Nv - number of latitudes to use (defaults to VWdata.Nv)
            uses global VWdata
            return a dict containing a bounce motion and supporting data
            fields:
                t - travel times from northward equator crossing (Nt,)
                theta - colatitude, radians (Nt,)
                lambda - latitude, radians (Nt,)
                phi - longitude traveled, radians (Nt,)
                alpha - local pitch angle, radians (Nt,)
                tint_func - function tint_func(lam) = travel time (seconds) to positive latitude lam (radians)
                pint_func - function pint_func(lam) = longitude drift (radians) while bouncing to positive latitude lam (radians)
                tbounce - bounce period, seconds (scalar)
                dphi - longitude drift in quarter bounce, radians (scalar)
                lambdam - mirror latitude, radians (scalar)
                
        """
        global VWdata
        if VWdata is None:
            VWdata = VWtable(Nv=Nv)
        if Nv is None:
            Nv = VWdata.Nv
        lambdam = alphaeq2lambdam(self.alphaeq)
        Ilambda = np.linspace(0,lambdam,Nv)
        
        vint = VWdata.V(Ilambda,lambdam)
        wint = VWdata.W(Ilambda,lambdam)
        tint = vint*self.L*akm*1e3/self.v # travel time vs lambda, seconds    
        pint = wint*self.MSI/self.L/(akm*1e3)/self.gamma/self.q/self.v # phi distance, radians
        tbounce = tint[-1]*4 # bounce period, seconds
            
        tint_func = interp1d(Ilambda,tint,'linear',fill_value=(tint[0],tint[-1]),bounds_error=False)
        pint_func = interp1d(Ilambda,pint,'linear',fill_value=(pint[0],pint[-1]),bounds_error=False)
        
        # construct a bounce movement starting northbound at the equator
        # use -2:0:-1 indexing to provide reverse indexing excluding end points for equatorward legs
        bounce = {'t':np.concatenate((tint,tbounce/2-tint[-2:0:-1],tbounce/2+tint,tbounce-tint[-2:0:-1]),axis=0),
                  'lambda':np.concatenate((Ilambda,Ilambda[-2:0:-1],-Ilambda,-Ilambda[-2:0:-1]),axis=0),
                  'phi':np.concatenate((pint,2*pint[-1]-pint[-2:0:-1],pint[-1]*2+pint,pint[-1]*4-pint[-2:0:-1]),axis=0),              
                  }
        bounce['theta'] = np.pi/2-bounce['lambda']
        bounce['alpha'] = np.arccos(self.vpar(bounce['theta'])/self.v) # acute
        ioblique = np.abs(bounce['t']/tbounce-1/2)<1/4 # southbound
        bounce['alpha'][ioblique] = np.pi-bounce['alpha'][ioblique] # oblique (southbound)
        bounce['tbounce'] = tbounce
        bounce['dphi'] = pint[-1]
        bounce['tint_func'] = tint_func
        bounce['pint_func'] = pint_func
        bounce['lambdam'] = lambdam
        return bounce

def state2xyz(L,theta,phi):
    """
    xyz = stat2xyz(L,theta,phi)
    return (3,N) ndarray of positions, in RE given L, theta, phi
    L,theta,phi are broadcast compatible scalar or 1-D arrays
    theta,phi in radians
    Note: if theta,phi are all scalar, then xyz will be (3,)
    """
    st = np.sin(theta)
    r = L*st**2
    x = r*np.cos(phi)*st
    y = r*np.sin(phi)*st
    z = r*np.cos(theta)
    return np.array((x,y,z))
    
def xyz2state(x,y=None,z=None):
    """
    (L,theta,phi) = xyz2state(xyz)
    (L,theta,phi) = xyz2state(x,y,z)
    return L,theta,phi for cartesian location in RE
    theta,phi in radians
    xyz is (3,) or (3,N)
    x,y,z are broadcast compatible scalar or 1-D arrays
    Note: if xyz is (3,) or if all x,y,z are scalar, then L,theta,phi will be scalar
    Note: L,theta,phi will be the broadcast shape of x,y,z or xyz[0]
    """
    if y is None: # xyz supplied
        z = x[2]
        y = x[1]
        x = x[0]
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
    L = r/np.sin(theta)**2
    return (L,theta,phi)


def gdz2xyz(alt,lat,lon):
    """
    xyz = gdz2xyz(alt,lat,lon)
    alt - altitude km
    lat - latitude, deg
    lon - longitude, deg
    alt,lat, lon are scalar or (N,)
    xyz is (3,) or (3,N) in RE
    """
    r = 1.0+alt/akm
    latrad = np.radians(lat)
    lonrad = np.radians(lon)
    z = r*np.sin(latrad)
    x = r*np.cos(lonrad)*np.cos(latrad)
    y = r*np.sin(lonrad)*np.cos(latrad)
    return np.array((x,y,z))


def backtrace(t0,L,theta0,phi0,alpha0,energykeV,species,phi_stop,sources=None,atm_alt=100,fluxOnly=False):
    """
    result = trace_particle(theta0,phi0,alpha0,energykeV,L,species,phi_stop,sources=None,atm_alt=100,fluxOnly=False)
    t0 - initial time, seconds
    L, theta0,phi0 - initial location, angles in radians (scalars)
    alpha0 - initial local pitch angle, radians (scalar)
    energykeV - kinetic energy, keV (scalar)
    species - species string
    phi_stop - change in longitude at which to stop, radians (scalar)
    sources - optional list SourceRegion objects
    atm_alt - atmospheric loss cone altitude, km
    fluxOnly - only return flux instead of entire result dictionary
    uses global VWdata
    result - dictionary
        flux - scalar - flux from sources
        t, theta, phi - (N,) location of particle
        alpha - (N,) local pitch angle of particle
        lambda - (N,) pi/2-theta (magnetic latitude)
    """
    
    if sources is None:
        if fluxOnly:
            return 0.0 # no sources -> no flux
        sources = []
    
    global VWdata
    if VWdata is None:
        VWdata = VWtable()
    
    Bm = Bmag(L,theta0)/np.sin(alpha0)**2
    part = Particle(L,Bm,energykeV=energykeV,species=species)
    
    # loss cone
    # r = L*cos(lam)**2, r = a+atm
    lambda_lc = np.arccos(((1+atm_alt/akm)/L)**0.5)
    
    bounce = part.bounce()
    tbounce = bounce['tbounce'] # bounce period
    lambdam = bounce['lambdam'] # mirror latitude
    lam = np.pi/2-theta0 # current latitude    
    
    blc = lambdam >lambda_lc # particle in bounce loss cone
    
    if fluxOnly and blc and not ((theta0<=np.pi/2) and (alpha0<=np.pi/2)):
        return 0.0 # only continue blc+fluxOnly if N hemi and northbound f/ eq
    
    # figure out phase
    # t - time from equator to where particle is now
    # dphi - drift since equator
    tlc = -np.inf # last time in loss cone
    lost = blc # lost to atmosphere
    lost_later = False
    if theta0 <= np.pi/2: # northern hemisphere
        if alpha0 <= np.pi/2: # northbound to atmosphere
            t = bounce['tint_func'](lam)
            dphi = bounce['pint_func'](lam)
            if blc:
                lost = False # makes it to equator, but...
                lost_later = True # lost b4 most recent N'ward eq crossing
        else: # southbound to equator
            t = tbounce/2-bounce['tint_func'](lam)
            dphi = bounce['dphi']*2-bounce['pint_func'](lam)
            if blc:
                if lam > lambda_lc: # in N loss cone
                    tlc = tbounce/4 # assume motion starts at N mirror point
                else:
                    tlc = tbounce/2-bounce['tint_func'](lambda_lc) # time to N lc
    else: # sourthern hemisphere
        if alpha0 <= np.pi/2:  # northbound to equator
            t = tbounce-bounce['tint_func'](-lam)
            dphi = bounce['dphi']*4-bounce['pint_func'](-lam)
            if blc:
                if lam < -lambda_lc: # in S loss cone
                    tlc = 3*tbounce/4 # assume motion starts at S mirror point
                else:
                    tlc = tbounce-bounce['tint_func'](lambda_lc) # time to S lc
        else: # southbound from equator
            t = tbounce/2+bounce['tint_func'](-lam)
            dphi = bounce['dphi']*2+bounce['pint_func'](-lam)
            if blc:
                tlc = bounce['tint_func'](lambda_lc) # time to N lc
    
    
    
    if fluxOnly: # don't need to worry about tlc
        # offset equatorial values by starting values
        teq = t0-t
        phieq = phi0-dphi
        # compute source fluxes
        flux = 0
        xyz = state2xyz(L,np.pi/2,phieq)
        for source in sources:
            flux += source.getFlux(L,phieq,energykeV,teq,xyz=xyz)
        if lost_later: # Done: particle will be lost before any more bounces
            return flux
        # Now just offset by full bounces and add sources
        while np.abs(phieq-phi0) < phi_stop:
            teq -= tbounce
            phieq -= bounce['phi'][-1]
            xyz = state2xyz(L,np.pi/2,phieq)
            for source in sources:
                flux += source.getFlux(L,phieq,energykeV,teq,xyz=xyz)
        return flux
    
    
    i = (bounce['t'] > tlc) & (bounce['t'] < t) # between last loss cone and now
    
    result = {'t':[t0],'theta':[theta0],'phi':[phi0],'alpha':[alpha0],'flux':0}

    result['t'].extend(result['t'][-1]+bounce['t'][i][::-1]-t)
    result['theta'].extend(bounce['theta'][i][::-1])
    result['phi'].extend(result['phi'][-1]+bounce['phi'][i][::-1]-dphi)
    result['alpha'].extend(bounce['alpha'][i][::-1])
    
    # now add whole bounces, excluding redundant last point
    while (not lost) and (np.abs(result['phi'][-1]-result['phi'][0])< phi_stop):
        result['t'].extend(result['t'][-1]+bounce['t'][-2::-1]-tbounce)
        result['theta'].extend(bounce['theta'][-2::-1])
        result['phi'].extend(result['phi'][-1]+bounce['phi'][-2::-1]-bounce['phi'][-1])
        result['alpha'].extend(bounce['alpha'][-2::-1])
        if lost_later: # lost on prior bounce
            lost = True
            lost_later = False
            # truncate to last entry into atmosphere (first index because time reversed)
            i = np.nonzero(np.abs(np.pi/2-np.array(result['theta']))>lambda_lc)[0][0]
            for k in ['t','theta','phi','alpha']:
                result[k] = result[k][0:i]
    
    # now reverse arrays to be in time order
    # and turn into numpy arrays
    for k in ['t','theta','phi','alpha']:
        result[k].reverse()
        result[k] = np.array(result[k])

    result['lambda'] = np.pi/2-result['theta']
    
    for i in np.nonzero((result['lambda']==0) & (result['alpha']<=90))[0]:
        xyz = state2xyz(L,np.pi/2,result['phi'][i])
        for source in sources:
            result['flux'] += source.getFlux(L,result['phi'][i],energykeV,result['t'][i],xyz=xyz)
        
    return result

class SourceRegion(object):
    """
    Equatorial Source Region (uniform)
    region = SourceRegion(L0,phi0,rkm,E0,J0,period,width,t0=None,color='b',cluster=None)
    L0,phi0 - center of region phi0 in radians
    rkm - radius of region in km
    E0 - e-folding energy in keV
    J0 - intensity: flux is J = J0*exp(-E/E0) for E in keV, J in #/cm^2/s/sr/keV
    period - time between pulse starts
    width - pulse width, seconds
    t0 - start time of reference pulse, seconds (None to randomize)
    color - color for plotting
    cluster - parent cluster
    all inputs are scalars
    """
    def __init__(self,L0,phi0,rkm,E0,J0,period,width,t0=None,color='b',cluster=None):
        self.L0 = L0
        self.phi0 = phi0
        self.rkm = rkm
        self.type = 'uniform'
        self.E0 = E0
        self.J0 = J0
        self.period = period
        self.width = width
        if t0 is None:
            t0 = -randomState.random()*width
        self.t0 = t0
        self.color = color
        self.cluster = cluster
        xyz =  state2xyz(L0,np.pi/2,phi0)
        self.x = xyz[0]
        self.y = xyz[1]
        self.set_radius(rkm)
    def toDict(self):
        """ d = toDict()
        express critical properties as a dictionary (for saving to JSON)
        """
        d = { 'L0':self.L0,'phi0':self.phi0,'rkm':self.rkm,
                'E0':self.E0,'J0':self.J0,'period':self.period,
                'width':self.width,'t0':self.t0,'color':self.color,
                'type':self.type,'cluster':None
                }
        if self.cluster:
            d['cluster'] = self.cluster.toDict()
        return d
    @property
    def max_extent_km(self):
        """
        return max extent
        """
        return self.rkm # step function edge
    @property
    def luminosity(self):
        """
        return integral of flux over area (just using J0 for flux)
        """
        return self.J0*self.rkm**2*np.pi # flat circle
    def _set_limits(self):
        """
        set min_L,max_L,min_phi,max_phi
        """
        self.min_L = self.L0-self.max_extent_km/akm
        self.max_L = self.L0+self.max_extent_km/akm
        self.min_phi = self.phi0-self.max_extent_km/(self.L0*akm)
        self.max_phi = self.phi0+self.max_extent_km/(self.L0*akm)
        
    def random_radius(self):
        """ r = random_radius()
        generate a random radius value, likelihood scaled by local flux
        acounts for rdr weighting
        """
        # integral of 2*pi*rdr = 2*pi*r^2/2 = pi*r^2
        # total probability is pi*rkm^2
        # u = pi*r^2 / (pi*rkm^2) = (r/rkm)^2
        # r = rkm*sqrt(u)
        u = randomState.random()
        r = self.rkm*np.sqrt(u)
        return np.maximum(np.minimum(r,0),self.max_extent_km)
    
    def set_radius(self,newr):
        """ set the radius to a new value in km
        set_radius(rkm)
        """ 
        self.rkm = newr
        self._set_limits()
        
    def inLrange(self,L):
        """
        bool = inLrange(L)
        returns true only if L is in the L range of this source
        """
        return (L>= self.min_L) & (L <= self.max_L)

    def inPhiRange(self,phi):
        """
        bool = inPhiRange(phi)
        returns true only if phi is in the phi range of this source
        phi in radians
        """
        return (phi>= self.min_phi) & (phi <= self.max_phi)
        
    def _rflux(self,r):
        """
        radial flux dependence
        """
        return 1.0
    
    def getFlux(self,L,phi,EkeV,t,xyz=None):
        """
        flux = getFlux(L,phi,EkeV,t,xyz=None)
        returns flux contributed by this source region at specified L, phi, Energy, time
        L,phi location of particle (at equator), phi in radians
        xyz - optional input precomputed x,y,z position
        EkeV particle energy in keV
        t - time of equatorial crossing
        flux - flux contributed by this source region #/cm^2/s/sr/keV
        all scalars
        """
        if xyz is None:
            xyz = state2xyz(L,np.pi/2,phi)
        r = np.sqrt((xyz[0]-self.x)**2+(xyz[1]-self.y)**2)*akm # RE -> km
        if (r > self.max_extent_km):
            return 0.0 # particle outside source region
        dt = np.mod(t-self.t0,self.period) # time since t0, modulo pulse period (mod deals with negative correctly)
        if dt < self.width:
            return self.J0*np.exp(-EkeV/self.E0)*self._rflux(r) # source is active
        else:
            return 0.0 # source is inactive
    def plot(self):
        """
        add region to current axes
        a dot at the center
        a dashed circle at r0
        a solid circle at the max extent
        colored by self.color
        triggers draw of parent cluster if first source in cluster
        """
        t = np.linspace(0,np.pi*2,200) # angels for region boundary
        plt.plot(self.x,self.y,'.',color=self.color)
        plt.plot(self.x+self.rkm/akm*np.cos(t),self.y+self.rkm/akm*np.sin(t),'--',color=self.color)
        plt.plot(self.x+self.max_extent_km/akm*np.cos(t),self.y+self.max_extent_km/akm*np.sin(t),'-',color=self.color)
        if self.cluster and (self.cluster[0] == self):
            self.cluster.plot()
    def path_length(self,L):
        """
        plen = path_length(L)
        return drift path length (in km) through source region for L shell at the equator
        scaled by radial flux dependence
        """
        scalar = np.isscalar(L) # remember initial type
        L = np.array(L)
        if scalar:
            plen = np.array([0])
        else:
            plen = np.zeros(L.shape)
        iL = self.inLrange(L)
        if np.any(iL):
            plen[iL] = 2*np.sqrt(self.rkm**2-(akm*(L[iL]-self.L0))**2)
        if scalar:
            return plen[0]
        else:
            return plen
    def steady_state(self,L,theta,Emid=None,dosResp=None,chan='DOS1',atm_alt=100):
        """
            A = steady_state(L,theta,Emid=E)            
            c = steady_state(L,theta,dosResp=dosResp,chan='DOS1')
            return the steady-state ratio (A) or rate (c) at L,theta
            (L and theta can both be arrays or both scalars)
            c = A*integral{R(E)*J0*exp(-E/E0) dE}
            Emid is the 50% level of the weighed response (see get_energy_limits)
            dosResp - dict with arrays keV and <chan> giving cm^2 sr vs energy
            chan (optional) - which channel to analyze, default is DOS1
            atm_alt (optional) - altitude of atmosphere (BLC) in km, default is 100
        """
        if np.isscalar(theta) != np.isscalar(L):
            raise Exception('L and theta must be both arrays or both scalars')

        plen = self.path_length(L)
        if np.all(plen==0):
            return plen # not conjugate
        
        # get Emid and equatorial rate (ce)
        ce = 1 # ignore
        if dosResp is None:
            if Emid is None:
                raise Exception('At least one of Emid or R required')
        else:
            elimits = get_energy_limits(dosResp,self.E0,chan)
            ce = elimits['RJ'][-1]*self.J0
            if Emid is None:
                Emid = elimits['Emid']
        
        # work on locally-mirroring particles at point nearest L0
        if np.isscalar(theta):
            thetam = theta
        else:
            i0 = np.argmin(np.abs(np.array(L)-self.L0))
            thetam = theta[i0]
        lambdam = np.pi/2-thetam
            
        Bm = Bmag(self.L0,thetam)
        
        part = Particle(L,Bm,energykeV=Emid)
        Vm = Vint(lambdam,lambdam)
        Wm = Wint(lambdam,lambdam)
        tbounce = 4*Vm*L*akm*1e3/part.v # bounce period seconds
        phibounce = 4*Wm*part.MSI/L/(akm*1e3)/part.gamma/part.q/part.v # radians per bounce
        tdrift = 2*np.pi/phibounce*tbounce # drift period, seconds
        
        # loss cone fraction
        lambda_lc = np.arccos(((1+atm_alt/akm)/L)**0.5) # loss cone latitude, radians
        Blc = Bmag(self.L0,np.pi/2-lambda_lc)
        alc = np.arcsin(np.sqrt(Bm/Blc)) # local pitch angle of loss cone, radians
        # Fdlc = [integral_{alc}^{pi/2} sin(a)da]/[integral_{0}^{pi/2} sin(a)da]
        # Fdlc = np.cos(alc) # fraction not in BLC is in DLC
        
        A = plen/(2*np.pi*akm*L)*self.width/self.period*(4*np.pi*np.cos(alc))*tdrift/tbounce
        if np.isscalar(theta) and np.isscalar(L):
            A = A[0]
        return A*ce # ce is 1 when dosResp not supplied

class GaussianSourceRegion(SourceRegion):
    """
    Equatorial Source Region (gaussian)
    source region with flux that falls off exp(-(r/r0)^2) to 3 r0
    same inputs as SourceRegion
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.type = 'gaussian'
    @property
    def max_extent_km(self):
        """
        return max extent
        """
        return 3*self.rkm # 3 standard deviations
    def _rflux(self,r):
        """
        radial flux dependence
        exp(-(r/r0)^2)
        """
        return np.exp(-(r/self.rkm)**2)
    @property
    def luminosity(self):
        """
        return integral of flux over area (just using J0 for flux)
        """
        # integral of r*exp(-0.5*(r/r0)^2) = -r0^2*exp(-0.5*(r/r0)^2)
        # definite integral from 0 to r: r0^2 -r0^2*exp(-0.5*(r/r0)^2) = r0^2(1-exp(-0.5*(r/r0)^2))
        # definite integral from 0 to 3*r0:
        # r0^2*(1-exp(-0.5*3^2))
        return self.J0*np.pi*self.rkm**2*(1-np.exp(-0.5*(self.max_extent_km/self.rkm)**2))
    def path_length(self,L):
        """
        plen = path_length(L)
        return drift path length (in km) through source region for L shell at the equator
        scaled by radial flux dependence
        """
        scalar = np.isscalar(L) # remember initial type
        L = np.array(L)
        if scalar:
            plen = np.array([0])
        else:
            plen = np.zeros(L.shape)
        iL = self.inLrange(L)
        if np.any(iL):
            d2 = ((L[iL]-self.L0)*akm/self.rkm)**2
            plen[iL] = np.sqrt(2*np.pi)*self.rkm*np.exp(-d2/2)*erf(np.sqrt(9/2-d2/2))
        if scalar:
            return plen[0]
        else:
            return plen
    def random_radius(self):
        """ r = random_radius()
        generate a random radius value, likelihood scaled by local flux
        acounts for rdr weighting
        """
        # integral of 2*pi*exp(-0.5*(r/r0)^2) = -r0^2*exp(-0.5*(r/r0)^2)*2*pi = -2*pi*r0^2*exp(-0.5*(r/r0)^2)
        # definite integral from 0 to r: 2*pi*r0^2(1-exp(-0.5*(r/r0)^2))
        # total probability is 2*pi*r0^2*(1-exp(-0.5*3^2))
        # u = 2*pi*r0^2*(1-exp(-0.5*(r/r0)^2)) / 2*pi*r0^2*(1-exp(-0.5*3^2)) = (1-exp(-0.5*(r/r0)^2))/(1-exp(-0.5*3^2))
        # u*(1-exp(-0.5*3^2)) = 1-exp(-0.5*(r/r0)^2)
        # 1-u*(1-exp(-0.5*3^2)) = exp(-0.5*(r/r0)^2)
        # log(1-u*(1-exp(-0.5*3^2))) = -0.5*(r/r0)^2
        # r = r0*sqrt(-2*log(1-u*(1-exp(-0.5*3^2))))
        u = randomState.random()
        r = self.rkm*np.sqrt(-2*np.log(1-u*(1-np.exp(-0.5*(self.max_extent_km/self.rkm)**2))))
        return np.maximum(np.minimum(r,self.max_extent_km),0)

def makeSourceRegion(L0,phi0,rkm,E0,J0,period,width,t0=None,color='b',type='uniform',cluster=None):
    """
    Equatorial Source Region
    region = SourceRegion(L0,phi0,rkm,E0,J0,period,width,t0=None,color='b',cluster=None)
    L0,phi0 - center of region phi0 in radians
    rkm - radius of region in km
    E0 - e-folding energy in keV
    J0 - intensity: flux is J = J0*exp(-E/E0) for E in keV, J in #/cm^2/s/sr/keV
    period - time between pulse starts
    width - pulse width, seconds
    t0 - start time of reference pulse, seconds (None to randomize)
    type - 'uniform' (default) or 'gaussian'
    color - color for plotting
    cluster - parent cluster
    all inputs are scalars
    """
    if type == 'uniform':
        return SourceRegion(L0,phi0,rkm,E0,J0,period,width,t0=t0,color=color,cluster=cluster)
    elif type == 'gaussian':
        return GaussianSourceRegion(L0,phi0,rkm,E0,J0,period,width,t0=t0,color=color,cluster=cluster)
    else:
        raise Exception('Unrecognized SourceRegion type %s' % type)


class SourceCluster(list):
    """
    Cluster of Equatorial Source Regions
    cluster = SourceRegion(L0,phi0,cluster_rkm,region_rkm,E0,J0,period,width,t0=None,color='b')
    cluster is a subclass of list
    L0,phi0 - center of cluster phi0 in radians
    cluster_rkm - radius of cluster in km
    region_rkm - radius of regions within cluster in km
    E0 - e-folding energy in keV
    J0 - intensity: flux is J = J0*exp(-E/E0) for E in keV, J in #/cm^2/s/sr/keV
    period - time between pulse starts
    width - pulse width, seconds
    t0 - start time of reference pulse, seconds (None to randomize)
    type - 'uniform' or 'gaussian' (default)
    color - color for plotting
    all inputs are scalars
    """
    def __init__(self,L0,phi0,cluster_rkm,region_rkm,E0,J0,period,width,t0=None,color='b',type='gaussian'):
        super().__init__() # empty list
        self.L0 = L0
        self.phi0 = phi0
        self.cluster_rkm = cluster_rkm
        self.region_rkm = region_rkm
        self.E0 = E0
        self.J0 = J0
        self.period = period
        self.width = width
        self.t0 = t0
        self.color = color
        self.type = type
        
         # make proxy source region for entire cluster
        cluster = makeSourceRegion(L0,phi0,cluster_rkm,E0,J0,period,width,t0=t0,color=color,type=type)
        self.proxy = cluster
        
        # copy virtual parameters
        self.max_extent_km = cluster.max_extent_km
        self.min_L = cluster.min_L
        self.max_L = cluster.max_L
        self.min_phi = cluster.min_phi
        self.max_phi = cluster.max_phi
        self.x = cluster.x
        self.y = cluster.y
        
        luminosity_goal = cluster.luminosity
        luminosity = 0
        while luminosity < luminosity_goal:
            r = cluster.random_radius() # generate random point within cluster, probability based on flux
            angle = np.pi*2*randomState.random() # random rotation relative to center
            # offset from cluster center by r,angle
            x = cluster.x+r*np.cos(angle)/akm
            y = cluster.y+r*np.sin(angle)/akm
            for s in self: # loop over existing sources
                dist = np.sqrt((x-s.x)**2+(y-s.y)**2)*akm # reject close sources in proportion to dist/region_rkm
                if (dist < region_rkm) and (randomState.random()>dist/region_rkm): # as d/r -> 1, never reject
                    break
            else: # for-else. only execute this if no break in loop over existing sources
                # make source and add to self
                (L,theta,phi) = xyz2state(x,y,0)
                region = makeSourceRegion(L,phi,region_rkm,E0,J0,period,width,t0=t0,color=color,type=type,cluster=self)
                self.append(region)
                luminosity += region.luminosity

        # scale down regions to preserve total luminosity
        lscale = luminosity_goal/luminosity 
        rscale = np.sqrt(lscale) # luminosity scales with area ~ r^2
        for p in self:
            p.set_radius(region_rkm*rscale)
            
            
    def toDict(self):
        """ d = toDict()
        express critical properties as a dictionary (for saving to JSON)
        """
        d = self.proxy.toDict()
        d['region_rkm'] = self.region_rkm
        d['cluster_rkm'] = self.cluster_rkm
        del d['rkm']
        return d
    def plot(self):
        """
        add cluster radii to current axes
        a dashed circle at r0
        a solid circle at the max extent
        colored by self.color
        """
        self.proxy.plot()

def fake_ac6(L1,L2,t0=0.0,dt=0.1,lon=0.0,alt=600):
    """
    make fake ac6 trajectory
    ac6 = fake_ac6(L1,L2,t0=0.0,dt=0.1,lon=0.0,alt=600)
    L1,L2 starting and ending L values
    t0 = start time, seconds
    dt = time step, seconds
    lon = longitude, degrees
    alt = altitude, km
    all inputes scalar
    ac6 - dict
        alt - altitude, km (Nt,)
        lat - latitude, degrees (Nt,)
        lon - longitude, degrees (Nt,)
        t - time, seconds (Nt,)
        L - L value, (Nt,)
        theta,phi - dipole coordinates, radians (Nt,)
        x,y,z - 3-D position, RE (Nt,)
        Xeq,Yeq - equatorial position, RE (Nt,)
    """
    v = 7.8 # orbital speed, km/s
    R = akm+alt
    dlatdt = np.degrees(v/R) # degrees/s
    # L = r*cos(lat)**2
    lat1 = np.degrees(np.arccos((L1*akm/R)**-0.5))
    lat2 = np.degrees(np.arccos((L2*akm/R)**-0.5))
    ac6 = {}
    ac6['lat'] = np.linspace(lat1,lat2,np.ceil(np.abs(lat2-lat1)/dlatdt/dt))
    ac6['t']= t0+(ac6['lat']-lat1)/dlatdt
    ac6['lon'] = np.zeros(ac6['t'].shape)+lon
    ac6['alt'] = np.zeros(ac6['t'].shape)+alt
    (ac6['x'],ac6['y'],ac6['z']) = gdz2xyz(ac6['alt'],ac6['lat'],ac6['lon'])
    (ac6['L'],ac6['theta'],ac6['phi']) = xyz2state(ac6['x'],ac6['y'],ac6['z'])
    ac6['Xeq'] = ac6['L']*np.cos(ac6['phi'])
    ac6['Yeq'] = ac6['L']*np.sin(ac6['phi'])
    return ac6

def read_ac6_iso_response_csv(filename):
    """
    iso = read_ac6_iso_response_csv(filename)
    read AC6 isotropic response CSV file
    filename = filename to ready, a CSV ASCII file
    expecting 4 columns, like this
        Energy MeV,DOS1,DOS2,DOS3 cm^2 sr
        0.0101158,0,0,0
        0.0103514,0,0,0
        ...
    returns np dict with fields:
        MeV - energy in MeV (NE,)
        keV - energy in keV (NE,)
        DOS1 - DOS1 response cm^s sr (NE,)
        DOS2 - DOS2 response cm^s sr (NE,)
        DOS3 - DOS3 response cm^s sr (NE,)
    """
    if not os.path.exists(filename):
        raise Exception('%s does not exist' % filename)
    iso = np.genfromtxt(filename,delimiter=',',skip_header=1,names=('MeV','DOS1','DOS2','DOS3'))
    iso = {v:iso[v] for v in iso.dtype.names} # convert to dict
    iso['keV'] = iso['MeV']*1e3 # add keV field
    return iso

def ListGet(x,dt,nan=np.nan):
    """
    y = ListGet(x,dt,nan=np.nan)
    x - (Nt,)
    dt - scalar integer
    y - (Nt,)
    y[i] = x[i+dt]
    nan's pad out the rest
    (use nan=0 for ints)
    """
    Nx = len(x)
    y = np.full((Nx,),nan)
    if dt>0:
        y[:-dt] = x[dt:].copy()
    elif dt==0:
        y = x.copy()
    else: # dt<0
        y[-dt:] = x[0:dt].copy()
    return y

def ConvMean(x,t0,t1,F=2/3):
    """
    y = ConvMean(x,t0,t1,F=2/3)
    convolution mean
    x - (Nt,)
    t0,t1 - scalar index offsets
    F - scalar fraction on [0,1]
    y - (Nt,)
    y[i] = average of x[i+t1] to x[i+t2] inclusive
    set to NaN when available data fraction for average is less than F
    """
    x = np.array(x,dtype=float)
    Nx = len(x)
    Nsum = t1-t0+1
    v = np.ones((Nsum,))
    xGood = np.isfinite(x)
    x[np.logical_not(xGood)] = 0
    xGood = np.array(xGood,dtype=float)
    xsum = np.convolve(x,v,mode='full')
    xcount = np.convolve(xGood,v,mode='full')
    xmean = xsum/xcount
    xmean[xcount < F*Nsum] = np.nan
    
    # length of xmean is Nsum + Nx-1
    y = ListGet(xmean,t0+Nsum-1)
    
    return y[0:Nx]
    

def _fly_in_scene_one(args):
    """ helper for fly_in_scene for parallelization, just a wrapper for backtrace"""
    (t0,L,theta,phi,alpha,energykeV,phi_stop,sources) = args
    return backtrace(t0,L,theta,phi,alpha,energykeV,'e-',phi_stop,sources,fluxOnly=True)

def scene_boundaries(ac6,sources):
    """result = scene_boundaries(ac6,sources)
    compute scene boundaries based on ac6 trajectory and sources
    ac6 - ac6 trajectory data dict
    sources  list of SourceRegion objects
    result - dict including
        IL - (Nt,) bool array of ac6['L'] values with sources on drift shell
        min_L,max_L - L range of sources (scalars)
        min_phi,max_phi - phi range of sources+ac6 (scalars)
    """
    Nt = len(ac6['L'])
    # figure out L range of active sources
    IL = np.full((Nt,),False)
    if sources is not None:
        for region in sources:
            IL = IL | region.inLrange(ac6['L'])
    min_phi = ac6['phi'][IL].min() # min phi for ac6
    max_phi = ac6['phi'][IL].max() # max phi for ac6
    if sources is not None:
        for region in sources:
            dphi = region.max_extent_km/(region.L0*akm) # phi extent of region (half width)
            min_phi = np.minimum(region.phi0-dphi,min_phi) # min phi of region
            max_phi = np.maximum(region.phi0+dphi,max_phi) # max phi of region

    min_L = ac6['L'][IL].min()
    max_L = ac6['L'][IL].max()
    return {'IL':IL,'min_L':min_L,'max_L':max_L,'min_phi':min_phi,'max_phi':max_phi}

def get_energy_limits(dosResp,E0,chan='DOS1'):
    """
    elimits = get_energy_limits(dos1resp,E0,chan='DOS1')
    dosResp - dict with arrays keV and <chan> giving cm^2 sr vs energy
    E0 = e-folding energy, keV. Spectrum is exp(-E/E0)
    chan (optional) - which channel to analyze
    elimits - dict with fields:
        Rj - DOS1*exp(-E/E0), the spectrum-weighted response
        ipeak - index of peak Rj
        Epeak - energy of peak Rj
        Rjpeak - Rj at peak
        JkeV - bool array of energies at which response is >1% of peak
        JEmin - minimum energy in JkeV
        JEmax - maximum energy in JkeV
        RJ - cumulative integral of Rj
        RJ1 - RJ normalized to end at 1
        imid - index where RJ1~0.5
        Emid - energy where RJ1~0.5
        Rjmid - Rj where RJ1~0.5
        fracRJkeV - fraction of response contained in JkeV energies
        E0 - echo of E0
        chan - echo of chan (e.g.,'DOS1')
    """
    EkeV = dosResp['keV']
    Rj = dosResp[chan]*np.exp(-EkeV/E0) # weighted response
    ipeak = np.argmax(Rj)
    Epeak = EkeV[ipeak]
    Rjpeak = Rj[ipeak]
    JkeV = Rj>=Rj.max()/100
    RJ = cumtrapz(Rj,dosResp['keV'],initial=0) # cumulative weighted response
    RJ1 = RJ/RJ[-1] # noramlized cumulative weighted response
    imid = np.argmin(np.abs(RJ1-0.5))
    Emid = EkeV[imid] # 50% energy for cumulative response
    Rjmid = Rj[imid] # response * flux at Emid
    JEmin = EkeV[JkeV].min()
    JEmax = EkeV[JkeV].max()
    fracRJkeV = RJ1[JkeV][-1]-RJ1[JkeV][0]
    
    return {'Rj':Rj,'ipeak':ipeak,'Epeak':Epeak,'Rjpeak':Rjpeak,'JkeV':JkeV,
            'RJ':RJ,'RJ1':RJ1,'imid':imid,'Emid':Emid,'Rjmid':Rjmid,
            'JEmin':JEmin,'JEmax':JEmax,'fracRJkeV':fracRJkeV,'E0':E0,'chan':chan}


_fly_in_scene_spectrum_globals = None # global dict used by wrapper
def fly_in_scene_spectrum(args):
    """flux = fly_in_scene_spectrum((it,alpha0))
    flux = fly_in_scene_spectrum((it,alpha0,cacheV,cacheW))
    pulls ac6, keV phi_stop, and sources from global dict _fly_in_scene_spectrum
    returns flux from sources by backtracing
    computes Vint and Wint on the fly or uses cacheV and cacheW to biuld a VWCache object
    atm_alt (loss cone alt, km) can also be provided in _fly_in_scene_spectrum
      otherwise defaults to 100 km
    flux is (NE,)
    """
    if len(args) == 2:
        (it,alpha0) = args
        VWcache = None
    else:
        (it,alpha0,cacheV,cacheW) = args
        VWcache = VWCache(V=cacheV,W=cacheW)
        
    ac6 = _fly_in_scene_spectrum_globals['ac6']
    t0 = ac6['t'][it]
    L = ac6['L'][it]
    theta0 = ac6['theta'][it]
    phi0 = ac6['phi'][it]
    energykeV = _fly_in_scene_spectrum_globals['keV']
    phi_stop = _fly_in_scene_spectrum_globals['phi_stop']
    sources = _fly_in_scene_spectrum_globals['sources']
    if 'VWcache' in _fly_in_scene_spectrum_globals: # non-parallel VWcache
        VWcache = _fly_in_scene_spectrum_globals['VWcache']

    NE = len(energykeV)
    flux = np.zeros((NE,))
    if sources is None:
        return flux # no sources, flux is zero
    
    atm_alt = 100 # loss cone altitude, km
    if 'atm_alt' in _fly_in_scene_spectrum_globals:
        atm_alt = _fly_in_scene_spectrum_globals['atm_alt']

    lam0 = np.pi/2-theta0 # initial latitude    
    
    # mirror field
    Bm = Bmag(L,theta0)/np.sin(alpha0)**2
    Beq = B0nT/L**3
    alpha_eq = np.arcsin(np.sqrt(Beq/Bm))
    lambdam = alphaeq2lambdam(alpha_eq) # mirror latitude

    # loss cone
    # r = L*cos(lam)**2, r = a+atm
    lambda_lc = np.arccos(((1+atm_alt/akm)/L)**0.5)

    lost_later = False # lost in bounce prior to N'ward crossing equator
    if lambdam >lambda_lc: # particle in bounce loss cone
        if ((lam0>=0) and (alpha0<=np.pi/2)):
            # only continue blc if N hemi and northbound f/ eq
            lost_later = True
        else:
            # lost before crossing eq N'bound
            return flux 

    # V,W integrals
    if VWcache:
        V = VWcache.Vint(np.abs(lam0),lambdam)
        W = VWcache.Wint(np.abs(lam0),lambdam)
        Vm = VWcache.Vint(lambdam,lambdam)
        Wm = VWcache.Wint(lambdam,lambdam)
    else:
        V = Vint(np.abs(lam0),lambdam)
        W = Wint(np.abs(lam0),lambdam)
        Vm = Vint(lambdam,lambdam)
        Wm = Wint(lambdam,lambdam)

    tlamv = V # dimensionless time from eq to |lam| in N hemi
    philamv = W # dimensionless phi drift moving from eq to |lam| in N hemi
    tbouncev = 4*Vm # dimensionless bounce period 
    phibouncev = 4*Wm # dimensionless phi drift over bounce motion
    
   # figure out phase
    # tv - dimensionless time from equator to where particle is now times v
    # dphiv - dimensionless drift since equator
    if lam0>=0: # northern hemisphere
        if alpha0 <= np.pi/2: # northbound to atmosphere
            tv = tlamv
            dphiv = philamv
        else: # southbound to equator
            tv = tbouncev/2-tlamv
            dphiv = phibouncev/2-philamv
    else: # sourthern hemisphere
        if alpha0 <= np.pi/2:  # northbound to equator
            tv = tbouncev-tlamv
            dphiv = phibouncev-philamv
        else: # southbound from equator
            tv = tbouncev/2+tlamv
            dphiv = phibouncev/2+philamv
    
    
    for (iE,E) in enumerate(energykeV):
        part = Particle(L,Bm=Bm,energykeV=E,species='e-')
        # time,phi to lambda
        t = tv*L*akm*1e3/part.v # seconds
        dphi = dphiv*part.MSI/L/(akm*1e3)/part.gamma/part.q/part.v # radians
        # time, phi for full bounce motion
        tbounce = tbouncev*L*akm*1e3/part.v # seconds
        phibounce = phibouncev*part.MSI/L/(akm*1e3)/part.gamma/part.q/part.v # radians
        
        teq = t0-t
        phieq = phi0-dphi
        
        xyz = None # on-demand precomputing of xyz for speedup
        for source in sources:
            if source.inPhiRange(phieq):
                if xyz is None:
                    xyz = state2xyz(L,np.pi/2,phieq)
                flux[iE] += source.getFlux(L,phieq,E,teq,xyz=xyz)
        if lost_later: # Done: particle will be lost before any more bounces
            continue
        # Now just offset by full bounces and add sources
        while np.abs(phieq-phi0) < phi_stop:
            teq -= tbounce
            phieq -= phibounce
            xyz = None  # on-demand precomputing of xyz for speedup
            for source in sources:
                if source.inPhiRange(phieq):
                    if xyz is None:
                        xyz = state2xyz(L,np.pi/2,phieq)
                    flux[iE] += source.getFlux(L,phieq,E,teq,xyz=xyz)
    
    return flux

def fly_in_scene2(ac6,sources,energykeV,response,min_alpha_deg=5,VWcache=None,Na=100):
    """ result = fly_in_scene2(ac6,sources,energykeV,response,min_alpha_deg=5,VWcache=None,Na=None)
    fly an ac6 through a scene, uses parallelization through Pool
    (optimized, does not use VWtable, all V,W calculations done on the fly)
    ac6 - ac6 trajectory data dict
    sources  list of SourceRegion objects
    energykeV - (NE,) energy grid, keV
    response - (NE,) response weights, cm^2 sr
    min_alpha_deg - (scalar) minimum local pitch angle to include in integral, degrees
    VWcache - a VWCache object used to speed up V,W integrals by re-using them
    Na - number of local pitch angles to sample for integral (defaults to 100)
    result - dict including
        t - (Nt,) time (from ac6), seconds
        keV - (NE,) energy (from energy keV), keV
        alpha - (Na,) local pitch angle grid, radians
        fedu - (Nt,NE,Na) unidirectional differential flux
        fedo - (Nt,NE) omnidirectional differential flux
        rate - (Nt,) count rate in dosimeter based on response
        min_L,max_L - L range of sources (scalars)
        min_phi,max_phi - phi range of sources+ac6 (scalars)
    """

    Nt = len(ac6['t'])
    NE = len(energykeV)
    if Na is None:
        Na = 100 # number of local pitch angles to sample
    Ialpha_deg = np.linspace(min_alpha_deg,180-min_alpha_deg,Na)
    Ialpha = np.radians(Ialpha_deg)
    
    boundaries = scene_boundaries(ac6,sources)
    IL = boundaries['IL']
    min_L = boundaries['min_L']
    max_L = boundaries['max_L']
    min_phi = boundaries['min_phi']
    max_phi = boundaries['max_phi']
    phi_stop = max_phi-min_phi
    
    # prepare arguments for _fly_in_scene_one, a wrapper for backtrace
    iL = np.nonzero(IL)[0]
    cases = np.full((len(iL),Na),None)
    global _fly_in_scene_spectrum_globals
    _fly_in_scene_spectrum_globals = {'ac6':ac6,'keV':energykeV,'phi_stop':phi_stop,'sources':sources}
    for (i,ialpha) in np.ndindex(cases.shape):
        if VWcache and not VWcache.local:
            cases[i,ialpha] = (iL[i],Ialpha[ialpha],VWcache.V,VWcache.W)
        else:
            cases[i,ialpha] = (iL[i],Ialpha[ialpha])

    if VWcache and VWcache.local: # share local cache (hopefully faster once precomputed vals are stored)
        _fly_in_scene_spectrum_globals['VWcache'] = VWcache
            
    print('%d cases w/ %d energies each' % (cases.size,NE))
    beNice()
    if N_POOL>1:
        with Pool(N_POOL) as p:
            result = p.map(fly_in_scene_spectrum,cases.ravel())
    else:
        result = list(map(fly_in_scene_spectrum,cases.ravel()))
    # result is a list len(iL)*Na long with elements that are NE lists
    tstart = tic()
    fedu = np.zeros((Nt,Na,NE))    
    fedu[IL,:,:] = np.reshape(result,(len(iL),Na,NE))
    fedu = np.transpose(fedu,(0,2,1)) # order Nt,NE,Na
    
    fedo = trapz(fedu*np.sin(np.reshape(Ialpha,(1,1,Na))),Ialpha,axis=2)*2*np.pi # (Nt,NE)
    rate = trapz(np.reshape(response,(1,NE))*fedo,energykeV,axis=1) # (Nt,)
    print('post-processing %g seconds' % toc(tstart))
    return {'t':ac6['t'],'keV':energykeV,'alpha':Ialpha,'fedu':fedu,'fedo':fedo,'rate':rate,
            'min_L':min_L,'max_L':max_L,'min_phi':min_phi,'max_phi':max_phi}


def fly_in_scene(ac6,sources,energykeV,response,min_alpha_deg=5):
    """ result = fly_in_scene(ac6,sources,energykeV,response,min_alpha_deg=5)
    fly an ac6 through a scene, uses parallelization through Pool
    ac6 - ac6 trajectory data dict
    sources  list of SourceRegion objects
    energykeV - (NE,) energy grid, keV
    response - (NE,) response weights, cm^2 sr
    min_alpha_deg - (scalar) minimum local pitch angle to include in integral, degrees
    uses global VWdata - (VWtable) precomputed V,W integrals, new one will be created if not supplied
    result - dict including
        t - (Nt,) time (from ac6), seconds
        keV - (NE,) energy (from energy keV), keV
        alpha - (Na,) local pitch angle grid, radians
        fedu - (Nt,NE,Na) unidirectional differential flux
        fedo - (Nt,NE) omnidirectional differential flux
        rate - (Nt,) count rate in dosimeter based on response
        min_L,max_L - L range of sources (scalars)
        min_phi,max_phi - phi range of sources+ac6 (scalars)
    """

    Nt = len(ac6['t'])
    NE = len(energykeV)
    if debug:
        Na = 10
    else:
        Na = 100
    Ialpha_deg = np.linspace(min_alpha_deg,180-min_alpha_deg,Na)
    Ialpha = np.radians(Ialpha_deg)
    
    boundaries = scene_boundaries(ac6,sources)
    IL = boundaries['IL']
    min_L = boundaries['min_L']
    max_L = boundaries['max_L']
    min_phi = boundaries['min_phi']
    max_phi = boundaries['max_phi']
    phi_stop = max_phi-min_phi
    
    # prepare arguments for _fly_in_scene_one, a wrapper for backtrace
    iL = np.nonzero(IL)[0]
    cases = np.full((len(iL),NE,Na),None)
    for (i,iE,ialpha) in np.ndindex(cases.shape):
        cases[i,iE,ialpha] = (ac6['t'][iL[i]],ac6['L'][iL[i]],ac6['theta'][iL[i]],ac6['phi'][iL[i]],Ialpha[ialpha],energykeV[iE],phi_stop,sources)
    print('%d cases' % cases.size)
    beNice()
    with Pool(N_POOL) as p:
        result = p.map(_fly_in_scene_one,cases.ravel())
    fedu = np.zeros((Nt,NE,Na))
    fedu[IL,:,:] = np.reshape(result,cases.shape)
    fedo = trapz(fedu*np.sin(np.reshape(Ialpha,(1,1,Na))),Ialpha,axis=2)*2*np.pi # (Nt,NE)
    rate = trapz(np.reshape(response,(1,NE))*fedo,energykeV,axis=1) # (Nt,)
    return {'t':ac6['t'],'keV':energykeV,'alpha':Ialpha,'fedu':fedu,'fedo':fedo,'rate':rate,
            'min_L':min_L,'max_L':max_L,'min_phi':min_phi,'max_phi':max_phi}



        
# read attitude file
# data = np.genfromtxt('data/AC6-B_20150207_L2_att_V03.csv',delimiter=',',missing_values=('-1e+31'),names=True)
    
def plot_scene(ac6,sources,min_L=None,max_L=None,min_phi=None,max_phi=None,trace_shells=True):
    """plot_scene(ac6,sources,min_L=None,max_L=None,min_phi=None,max_phi=None,trace_shells=True)
    plot a scene in the equatorial plane:
        green line for ac6
        black lines trace drift path from sources to ac6
        circles for microburst source regions
        
    ac6 - ac6 trajectory data dict
    sources  list of SourceRegion objects
    min_L,max_L - min,max L bounds of ac6 trace
    min_phi,max_phi - min,max phi bounds of traces from sources to ac6
    trace_shells - trace drift shell from source to ac6
    """

    boundaries = scene_boundaries(ac6,sources)
    if min_L is None:
        min_L = boundaries['min_L']
    if max_L is None:
        max_L = boundaries['max_L']
    if min_phi is None:
        min_phi = boundaries['min_phi']
    if max_phi is None:
        max_phi = boundaries['max_phi']
        
    plt.figure()
    plt.grid('on')
    plt.axis('equal')
    plt.xlabel('X, R$_E$')
    plt.ylabel('Y, R$_E$')
    # draw sources
    if sources is not None:
        for region in sources:
            if trace_shells:
                phi_stop = interp1d(ac6['L'],ac6['phi'],fill_value='extrapolate',bounds_error=False)(region.L0)
                phi = np.linspace(region.phi0,phi_stop,100)
                plt.plot(region.L0*np.cos(phi),region.L0*np.sin(phi),'k-') # arc through region
            region.plot()
    iL = (ac6['L'] >= min_L) & (ac6['L'] <= max_L)        
    plt.plot(ac6['Xeq'][iL],ac6['Yeq'][iL],'g-')
    ax = plt.axis()
    # draw L shells
    phi = np.linspace(0,np.pi*2,1000)
    if np.ceil(min_L)-np.floor(max_L)+1 < 5:
        step = 0.5
    else:
        step = 1
    for L in np.arange(np.ceil(min_L/step),np.floor(max_L/step)+1)*step:
        plt.plot(L*np.cos(phi),L*np.sin(phi),'--',color='gray',zorder=0) # draw underneath
    plt.axis(ax)
    

randomState = None    
def seedRNG(seed=None):
    """
    randomState = seedRNG(seed=None)
    seed - 32-bit uint to use as seed (defaults to 12345)
    seeds the module's private random number generator
    returns a RandomState that can be used to get random or randint
    """
    if seed is None:
        seed = 12345
    global randomState
    randomState = np.random.RandomState(seed)
    return randomState
    
if randomState is None:
    seedRNG()     
    
    
def load_rbsp_plasmapause(filename):
    """
    filename = full file name, filename in data/ subfolder, or just probe A or B
    if just probe is given, the filename is like data/rbspa_inner_plasmapause_list2.dat.txt
    returns a dict with arrays 'Inbound','Outbound','OrbitNumber'. 
    Inbound/Outbound are plasmapause crossing times
    """
    if len(filename) == 1:
        probe = filename.lower()
        filename = 'data/rbsp%s_inner_plasmapause_list2.dat.txt' % probe
    if not os.path.exists(filename) and os.path.exists('data/'+filename):
        filename = 'data/'+filename
    if not os.path.exists(filename):
        raise Exception('Cannot load from %s, does not exist' % filename)
    
    # # Source file = /home/iwc/rbsp/ppause-sebastian-rbspa.txt
    # # Spacecraft = RBSP-A
    # # Generated: 2016/12/22T23:27Z
    # # Columns:
    # #   Outbound inner edge of plasmapause
    # #   Inbound inner edge of plasmapause
    # #   Orbit number
    # 2012-09-14T20:38:28.5  2012-09-14T22:53:09.9     42  
    # 2012-09-15T04:57:17.4  2012-09-15T09:09:56.8     43 
    str2date = lambda x: dt.datetime.strptime(x.decode("utf-8"), '%Y-%m-%dT%H:%M:%S.%f')
    data = np.genfromtxt(filename,converters={0: str2date, 1: str2date})
    data = {'Outbound':data['f0'],'Inbound':data['f1'],'OrbitNum':data['f2'],'filename':filename}
    return data

if __name__ == '__main__':
    from odc_util import BouncePeriod, DriftPeriod 
    
    N_POOL = 40
    
    # make figures folder if needed
    if not os.path.exists('figures'):
        os.mkdir('figures')
    
    
    plt.close('all')
    
    plt.figure()
    for (veh,style) in [('A','-'),('B','--')]:
        iso = read_ac6_iso_response_csv('data/ac6%s_response_iso.csv' % veh.lower())
        dos1resp = {'keV':iso['MeV']*1e3,'DOS1':iso['DOS1']} # save for later
        plt.loglog(dos1resp['keV'],dos1resp['DOS1'],style,label='AC6-%s' % veh)
    plt.xlabel('keV')
    plt.ylabel('cm$^2$ sr')
    plt.legend()
    
    ac6 = fake_ac6(4,8,0,dt=1/50)
    
    
    energykeV = 40
    L = 6.6
    
    driftP = DriftPeriod('e-',energykeV,65,L)
    bounceP = BouncePeriod('e-',energykeV,65,L)
    
    #phi0 = -0.5*bounceP/driftP*np.pi*2
    phi0 = np.radians(1)
    
    region = SourceRegion(L,phi0,50,10,1,5,0.3,0)
    
    plot_scene(ac6,[region],min_L=region.L0-1,max_L=region.L0+1)
    dx = np.linspace(-200/akm,200/akm,40)
    xI,yI = np.meshgrid(region.x+dx,region.y+dx)
    zI = np.zeros(xI.shape)
    (LI,thetaI,phiI) = xyz2state(xI,yI,zI)
    for ix,iy in np.ndindex(xI.shape):
        if region.getFlux(LI[ix,iy],phiI[ix,iy],energykeV,0)>0:
            plt.plot(xI[ix,iy],yI[ix,iy],'r.')

    VWdata = VWtable()
    
    Ilambda = np.linspace(0,VWdata.maxLambdam,VWdata.Nv*2)
    ilam = VWdata.Nv//2
    lambdam = VWdata.Ilambda[ilam]
    v = np.array([Vint(np.minimum(lam,lambdam),lambdam) for lam in Ilambda])
    v2 = VWdata.V(Ilambda,lambdam)
    
    plt.figure()
    plt.plot(Ilambda,v,'k.-',Ilambda,v2,'r-')
    print('Max error: %g (%g relative)' % (abs(v-v2).max(),abs(v2/v-1)[v>0].max()))


    species = 'e-'

    # test particle at 66 degrees north
    alt = 600 # km
    lat = 67 # deg
    lon = 0 # deg

    xyz = gdz2xyz(alt,lat,lon)
    (L,theta0,phi0) = xyz2state(xyz)
    
    JkeV = (dos1resp['keV'] >= 30) & (dos1resp['keV']<=300)
    if debug:
        JkeV = (dos1resp['keV'] >= 30) & (dos1resp['keV']<=40)
    IkeV = dos1resp['keV'][JkeV]
    NE = len(IkeV) # number of energies
    phi_stop = np.radians(0.5) # dummy test value
    
    energykeV = IkeV[0]
    alpha0 = np.radians(80)
    alphaeq = np.arcsin(np.sin(alpha0)*np.sqrt(Bmag(L,np.pi/2)/Bmag(L,theta0)))
    bounceP = BouncePeriod('e-',energykeV/1e3,alphaeq,L)
    driftP = DriftPeriod('e-',energykeV/1e3,alphaeq,L)
    result = backtrace(0,L,theta0,phi0,alpha0,energykeV,'e-',phi_stop)
    plt.figure()
    plt.subplot(2,1,1)
    for key in ['lambda','theta','alpha']:
        plt.plot(result['t']/bounceP,np.degrees(result[key]),label=key)
    plt.legend()
    plt.subplot(2,1,2)
    for key in ['phi']:
        plt.plot(result['t']/bounceP,np.degrees(result[key]),label=key)
    
    
    #region = SourceRegion(L0,phi0,rkm,E0,J0,period,width,t0)
    region = SourceRegion(5,phi0-np.radians(0.1),50,10,1,10,0.1,0)
    bounds = scene_boundaries(ac6,[region])
    phi_stop = bounds['max_phi']-bounds['min_phi']
    flux = np.zeros(ac6['t'].shape)
    for i in range(len(ac6['t'])):
        if not bounds['IL'][i]:
            continue
        result = backtrace(ac6['t'][i],ac6['L'][i],ac6['theta'][i],ac6['phi'][i],np.radians(80),energykeV,'e-',phi_stop,[region])
        flux[i] = result['flux']
        fluxi = backtrace(ac6['t'][i],ac6['L'][i],ac6['theta'][i],ac6['phi'][i],np.radians(80),energykeV,'e-',phi_stop,[region],fluxOnly=True)
        if ((flux[i]>0) or (fluxi>0)) and (np.abs(flux[i]-fluxi)/(flux[i]+fluxi)>1e-5):
            print('WARNING: fluxOnly answer differs from full backtraces on case %d: %g,%g' % (i,flux[i],fluxi))
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(ac6['t'],ac6['Xeq'],label='Xeq')
    plt.ylabel('X, Re')
    plt.subplot(2,1,2)
    plt.plot(ac6['t'],flux)
    plt.xlabel('t, seconds')
    plt.ylabel('flux')

    tic()
    result2 = fly_in_scene2(ac6,[region],dos1resp['keV'][JkeV],dos1resp['DOS1'][JkeV])
    toc()
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(ac6['t'],ac6['Xeq'],label='Xeq')
    plt.ylabel('X, Re')
    plt.title('Method 2')
    plt.subplot(2,1,2)
    plt.plot(ac6['t'],result2['rate'])
    plt.xlabel('t, seconds')
    plt.ylabel('dos1rate')

    tic()
    result = fly_in_scene(ac6,[region],dos1resp['keV'][JkeV],dos1resp['DOS1'][JkeV])
    toc()
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(ac6['t'],ac6['Xeq'],label='Xeq')
    plt.ylabel('X, Re')
    plt.title('Method 1')
    plt.subplot(2,1,2)
    plt.plot(ac6['t'],result['rate'])
    plt.xlabel('t, seconds')
    plt.ylabel('dos1rate')

    if debug:  
        VWdata = VWtable(store=False)
        tic()
        result3 = fly_in_scene(ac6,[region],dos1resp['keV'][JkeV],dos1resp['DOS1'][JkeV])
        toc()
        plt.figure()    
        plt.plot(ac6['t'],result['rate'],'k-',ac6['t'],result2['rate'],'r--',ac6['t'],result3['rate'],'b:')
        plt.xlim([56.4,57.2])
        plt.legend(['Backtrace w/ Stored Table','V,W Direct','Backtrace w/o Stored'])
    else:
        plt.figure()    
        plt.plot(ac6['t'],result['rate'],'k-',ac6['t'],result2['rate'],'r:')
        plt.xlim([56.4,57.2])
        plt.legend(['Backtrace w/ Stored Table','V,W Direct'])
    

    