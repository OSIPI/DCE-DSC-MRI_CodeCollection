"""
July 2024 by Natalia Korobova, Oliver Gurney-Champion and Matthew Orton
n.korobova@amsterdamumc.nl
o.j.gurney-champion@amsterdamumc.nl

Contains the modification of Extended Tofts model for MRI acquisitions oversampling the center of k-space, such as radial, spiral, PROPELLER trajectories.
Modification assumes that the image contrast is averaged over an acquisition time of one dynamic as a result of oversampling the center of k-space.
Modification is based on previously published code by OG_MO_AUMC_ICR_RMH_NL_UK from the OSIPI GitHub.

requirements:
scipy
joblib
matplotlib
numpy
"""

from scipy.optimize import curve_fit
import numpy as np
from joblib import Parallel, delayed
from matplotlib.pylab import *

def aifPopHN(Hct = 0.4):
    # defines plasma curve; note this is a population-based AIF for H&N patients (https://doi.org/10.2967/jnumed.116.174433). Please fit the AIF to determine your ideal ab, mb, ae and me parameters. You can use func aif and fit to your AIF for that.
    aif = {'ab': 3.646/(1-Hct), 'mb': 25.5671, 'ae': 1.53, 'me': 0.2130, 't0': 0.1}
    return aif

def aifPopPMB(Hct = 0.4):
    # defines plasma curve; note this is a population-based AIF based on https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21066 and https://iopscience.iop.org/article/10.1088/0031-9155/53/5/005
    # NB the PMB paper neglected to include the Hct in the parameter ab, so we include this adjustment here with an assumed Hct
    aif = {'ab': 2.84/(1-Hct), 'mb': 22.8, 'ae': 1.36, 'me': 0.171, 't0': 0.2}
    return aif

def con_to_R1eff(C, R1map, relaxivity):
    """
    Calculates effective R1 relaxation time after contrast injection

    Parameters
    ----------
      C : numpy.array
        Concentration of contrast agent over time
      R1map : numpy.array
        Baseline (precontrast) R1 values
      relaxivity : int
    """
    assert (relaxivity > 0.0)
    return R1map + relaxivity * C

def r1eff_to_conc(R1eff, R1map, relaxivity):
    """
    Calculates contrast agent concentration over time

    Parameters
    ----------
      R1eff : numpy.array
        Effective R1 relaxation time after contrast injection
      R1map : numpy.array
        Baseline (precontrast) R1 values
      relaxivity : int
    """
    conc = (R1eff - R1map) / relaxivity
    return conc

def r1eff_to_dce(M0, R, TR, flip):
    """
    Calculates signal over time

    Parameters
    ----------
      M0 : numpy.array
      R : numpy.array
        Effective R1 relaxation time after contrast injection
      TR : float
      flip : float
        Flip angle in radians
    """
    A = ((1 - exp(-TR * R)) * sin(flip)) / (1 - cos(flip) * exp(-TR * R))
    return A*M0

def dce_to_r1eff(S, S0num, R1, TR, flip):
    """
    Calculates effective R1 relaxation time after contrast injection

    Parameters
    ----------
      S : numpy.array
        Signal over time
      S0num: int
        Number of frames before contrast injection
      R1: numpy.array
        Baseline (precontrast) R1 values
      TR : float
      flip : float
        Flip angle in radians
    """
    assert(flip > 0.0)
    assert(TR > 0.0 and TR < 1.0)
    S0 = S[:S0num].mean(axis=0)
    S0 = np.repeat(np.expand_dims(S0,axis=0),np.shape(S)[0],axis=0)
    A = np.divide(S, S0, out=np.zeros_like(S), where=S0!=0)
    E0 = exp(-R1 * TR)
    nom = (1.0 - A + A*E0 - E0*cos(flip))
    denom = (1.0 - A*cos(flip) + A*E0*cos(flip) - E0*cos(flip))
    E = np.divide(nom, denom, out=np.zeros_like(nom), where=denom!=0)
    E[E<=0] = 1
    R = (-1.0 / TR) * log(E)
    return R

def fit_tofts_model(descrete_time, measurement, delay, aif, jobs, X0 = (1.5, 1.0, 0.5, 0.5), bounds = ((0.0, 0.5, 0.0, 0.0), (3.0, 1.5, 1.0, 1.0))):
    """
    Solves the Extended Tofts model for each voxel and returns model parameters using the computationally efficient AIF as described by Orton et al. 2008 in https://doi.org/10.1088/0031-9155/53/5/005

    Parameters
    ----------
      descrete_time : numpy.array
        Time samples at which Ct was measured
      measurement : numpy.array
        Concentration of contrast agent as N x T matrix, with N the number of curves to fit and T the number of time points
      delay: int
        Delay, acquisiotion time of one dynamic image, typically d = descrete_time[1] - descrete_time[0]
        When d = 0, conventional extended Tofts model is used, without the correction for time-averaged signal
      aif: dict
        The aif model parameters in library, including ab, ae, mb, me and t0
      X0 : tuple
        Optional. Initial guess of parameters ke, dt, ve, vp
      bounds : tuple
        Optional. Fit boundaries for  ke, dt, ve, vp
    """
    popt_default = [0, 0, 0, 0]

    if len(X0) == 4:
        def parfun(idx):
            fit_func = lambda tt, ke, dt, ve, vp: Cosine4AIF_ExtKety_Conv(tt, aif[idx], ke, dt, ve, vp, delay)
            try:
                popt, pcov = curve_fit(fit_func, descrete_time, measurement[idx, :], p0 = X0, bounds = bounds, maxfev=5000)
            except RuntimeError:
                print('ya')
                popt = popt_default
            return popt
    else:
        def parfun(idx):
            fit_func = lambda tt, ke, dt, ve, vp: Cosine4AIF_ExtKety_Conv(tt, aif[idx], ke, dt, ve, vp, delay)
            try:
                popt, pcov = curve_fit(fit_func, descrete_time, measurement[idx, :], p0 = X0[idx], bounds = bounds, maxfev=5000)
            except RuntimeError:
                print('ya')
                popt = popt_default
            return popt
    idxs = range(len(measurement))
    output = Parallel(n_jobs=jobs, verbose=50)(delayed(parfun)(i) for i in idxs)

    return np.transpose(output)

def fit_aif(t, Caif, delay, jobs, X0, bounds = ((0, 0, 0, 0, 0), (inf, inf, inf, inf, inf))):
    """
    Calculates the AIF model parameters given a AIF curve

    Parameters
    ----------
      Caif : numpy.array
        Concentration curve of the AIF over time
      t : numpy.array
        Time samples at which Caif is measured
    """
    popt_default = [0, 0, 0, 0, 0]
    if len(X0) == 5:
        def parfun(idx):
            fit_func = lambda t, ab, ae, mb, me, dt: Cosine4AIF_Conv(t, ab, ae, mb, me, dt, delay)
            try:
                popt, pcov = curve_fit(fit_func, t, Caif[idx, :], p0 = X0, bounds = bounds, maxfev=5000)
            except RuntimeError:
                print('ya')
                popt = popt_default
            return {'ab': popt[0], 'ae': popt[1], 'mb': popt[2], 'me': popt[3], 't0': popt[4]}
    else:
        def parfun(idx):
            fit_func = lambda t, ab, ae, mb, me, dt: Cosine4AIF_Conv(t, ab, ae, mb, me, dt, delay)
            try:
                popt, pcov = curve_fit(fit_func, t, Caif[idx, :], p0 = X0[idx], bounds = bounds, maxfev=5000)
            except RuntimeError:
                print('ya')
                popt = popt_default
            return {'ab': popt[0], 'ae': popt[1], 'mb': popt[2], 'me': popt[3], 't0': popt[4]}
    idxs = range(len(Caif))
    output = Parallel(n_jobs=jobs, verbose=50)(delayed(parfun)(i) for i in idxs)
    return output

def Cosine4AIF_ExtKety_Conv(t,aif,ke,dt,ve,vp,d):
    """
    Calculates concentration of contrast agent over time
    Modified version of Cosine4AIF_ExtKety published by OG_MO_AUMC_ICR_RMH_NL_UK in OSIPI GitHub

    Parameters
    ----------
      t : numpy.array
        Time samples at which concentration is sampled
      aif : dict
        The aif model parameters in library, including ab, ae, mb, me and t0
      ke, dt, ve, vp: float
        Extended Tofts parameters
      d: float
        Delay, acquisiotion time of one dynamic image, typically d = t[1] - t[0]
        When d = 0, conventional extended Tofts model is used, without the correction for time-averaged signal
    """
    t = t - aif['t0'] - dt

    if d == 0:
        cpBolus = aif['ab']*CosineBolus(t,aif['mb'])
        cpWashout = aif['ab']*aif['ae']*ConvBolusExp(t,aif['mb'],aif['me'])
        ceBolus = ke*aif['ab']*ConvBolusExp(t,aif['mb'],ke)
        ceWashout = ke*aif['ab']*aif['ae']*ConvBolusExpExp(t,aif['mb'],aif['me'],ke)
    else:
        cpBolus = aif['ab']*ConvBolusRect(t,aif['mb'],d) / d
        cpWashout = aif['ab']*aif['ae']*ConvBolusExpRect(t,aif['mb'],aif['me'],d)/ d
        ceBolus = ke*aif['ab']*ConvBolusExpRect(t,aif['mb'],ke,d)/d
        ceWashout = ke*aif['ab']*aif['ae']*ConvBolusExpExpRect(t,aif['mb'],aif['me'],ke,d) / d
    
    
    cp = cpBolus + cpWashout
    ce = ceBolus + ceWashout

    ct = np.zeros(np.shape(t))
    ct[t > 0] = vp * cp[t > 0] + ve * ce[t > 0]
    
    return ct

def Cosine4AIF_Conv(t,ab,ae,mb,me,t0,d):

    t = t - t0
    
    if d == 0:
        cpBolus = ab*CosineBolus(t,mb)
        cpWashout = ab*ae*ConvBolusExp(t,mb,me)
    else:
        cpBolus = ab*ConvBolusRect(t,mb,d) / d
        cpWashout = ab*ae*ConvBolusExpRect(t,mb,me,d)/ d

    cp = cpBolus + cpWashout
    
    return cp

def ConvBolusRect(t,m,d):
    y = ConvBolusExp(t, m, 0) - ConvBolusExp(t-d, m, 0)
    return y

def ConvBolusExpRect(t, m, k, d):
    y = ConvBolusExpExp(t,m,k,0) - ConvBolusExpExp(t-d,m,k,0)
    return y

def ConvBolusExpExpRect(t,m,k1,k2,d):
    a = ConvBolusExpExpExp(t,m,k1,k2,0)
    b = ConvBolusExpExpExp(t-d,m,k1,k2,0)
    y = a - b
    return y

def CosineBolus(t,m):
    z = array(m * t)
    I = (z >= 0) & (z < (2 * pi))
    y = np.zeros(np.shape(t))
    y[I] = 1 - cos(z[I])
    return y

def ConvBolusExp(t,m,k):
    tB = 2 * pi / m
    t=array(t)
    I1 = (t > 0) & (t < tB)
    I2 = t >= tB

    y = np.zeros(np.shape(t))

    y[I1] = multiply(t[I1],SpecialCosineExp(k*t[I1], m*t[I1]))
    y[I2] = tB*SpecialCosineExp(k*tB, m*tB)*exp(-k*(t[I2]-tB))

    return y

def ConvBolusExpExp(t,m,k1,k2):
    tol = 1e-4

    tT = tol / abs(k2 - k1)
    tT = array(tT)
    Ig = (t > 0) & (t < tT)
    Ie = t >= tT
    y = np.zeros(np.shape(t))
    y[Ig] = ConvBolusGamma(t[Ig], m, 0.5 * (k1 + k2))
    y1 = ConvBolusExp(t[Ie], m, k1)
    y2 = ConvBolusExp(t[Ie], m, k2)
    y[Ie] = (y1 - y2) / (k2 - k1)

    return y

def ConvBolusExpExpExp(t,m,k1,k2,k3):
    
    k = sort([k1,k2,k3])
    k1 = k[0]
    k2 = k[1]
    k3 = k[2]
    
    y = log10(k2-k1) - log10(k3-k2)
    tT1 = 10**(-3 + 0.7*abs(y))/(k3-k1)
    tT2 = 10**(-4.5)/(k3-k1)
    
    ID = (t>tT1)
    IG1 = (t<=tT1) & (t>tT2)
    IG2 = (t<=tT2)

    y = np.zeros(np.shape(t))
    
    fD1 = (ConvBolusExp(t[ID],m,k1) - ConvBolusExp(t[ID],m,k2))/(k2-k1)
    fD2 = (ConvBolusExp(t[ID],m,k2) - ConvBolusExp(t[ID],m,k3))/(k3-k2)
    y[ID] = (fD1 - fD2)/(k3-k1)
    
    k12 = 0.5*(k1 + k2)
    k13 = 0.5*(k1 + k3)
    k23 = 0.5*(k2 + k3)
    
    if k2 < k13:
        fDA = (ConvBolusExp(t[IG1],m,k12) - ConvBolusExp(t[IG1],m,k3))/(k3-k12)
        fA1 = ConvBolusGamma(t[IG1],m,k12)
        y[IG1] = (fA1 - fDA)/(k3-k12)
    else:
        fDA = (ConvBolusExp(t[IG1],m,k1) - ConvBolusExp(t[IG1],m,k23))/(k23-k1)
        fA3 = ConvBolusGamma(t[IG1],m,k23)
        y[IG1] = (fDA - fA3)/(k23-k1)
        
    if k2<k13:
        k123 = 0.5*(k12 + k3)
    else:
        k123 = 0.5*(k1 + k23)

    y[IG2] = ConvBolusGamma2(t[IG2],m,k123)
    
    return y

def ConvBolusGamma(t,m,k):
    tB = 2 * pi / m
    tB=array(tB)
    y = np.zeros(np.shape(t))
    I1 = (t > 0) & (t < tB)
    I2 = t >= tB

    ce = SpecialCosineExp(k * tB, m * tB)
    cg = SpecialCosineGamma(k * tB, m * tB)

    y[I1] = square(t[I1]) * SpecialCosineGamma(k * t[I1], m * t[I1])
    y[I2] = tB * multiply(((t[I2] - tB) * ce + tB * cg), exp(-k * (t[I2] - tB)))

    return y

def ConvBolusGamma2(t,m,k):
    tB = 2 * pi / m
    y = np.zeros(np.shape(t))
    I1 = (t > 0) & (t < tB)
    I2 = t >= tB
    
    
    ce = SpecialCosineExp(k*tB, m*tB)
    cg = SpecialCosineGamma(k*tB, m*tB)
    cg2 = SpecialCosineGamma2(k*tB, m*tB)
    
    y[I1] = 0.5*t[I1]**3 * SpecialCosineGamma2(k*t[I1], m*t[I1])
    y[I2] = 0.5*tB*((t[I2]-tB)**2 * ce + 2*(t[I2]-tB) * tB * cg + tB**2 * cg2) * exp(-k*(t[I2]-tB))
    
    return y

def SpecialCosineGamma(x, y):
    x=array(x)
    y=array(y)
    x2 = square(x)
    y2 = square(y)
    expTerm = multiply(3+divide(square(y), square(x)), (1 - exp(-x))) \
                   - multiply(divide(square(y) + square(x),x), exp(-x))
    trigTerm = multiply((square(x) - square(y)), (1 - cos(y))) - multiply(multiply(2 * x,y), sin(y))
    f = divide((trigTerm + multiply(square(y), expTerm)), square(square(y) + square(x)))

    return f

def SpecialCosineGamma2(x,y):
    
    x=array(x)
    y=array(y)
    
    ex = 1.5
    ey = 1
    
    I0  = (x>ex)  & (y>ey)
    Ix  = (x<=ex) & (y>ey)
    Iy  = (x>ex)  & (y<=ey)
    Ixy = (x<=ex) & (y<=ey)
    
    expTerm = np.zeros(np.shape(x))
    trigTerm = np.zeros(np.shape(x))
    f = np.zeros(np.shape(x))

    I0y = I0 + Iy
    expTerm[I0y] = 2*(6*x[I0y]**4 + 3*y[I0y]**2 * x[I0y]**2 + y[I0y]**4) * (1 - exp(-x[I0y])) / x[I0y]**3 \
                   - 2*exp(-x[I0y]) * (x[I0y]**2 + y[I0y]**2) * (3*x[I0y]**2 + y[I0y]**2) / x[I0y]**2 \
                   - exp(-x[I0y]) * (x[I0y]**2 + y[I0y]**2)**2 / x[I0y]

    
    yE2 = y[Ix]**2
    yE4 = yE2**2
    xE = x[Ix]
    
    if Ix.any():
        expTerm[Ix] = 1*yE4/3  + 2*(0*(0-3)-1)*yE2 + 0 \
                                    - xE/2*(2*yE4/4  + 2*(1*(1-3)-1)*yE2 + 0 \
                                    - xE/3*(3*yE4/5  + 2*(2*(2-3)-1)*yE2 + 3*2*(-2)*(-3) \
                                    - xE/4*(4*yE4/6  + 2*(3*(3-3)-1)*yE2 + 4*3*(-1)*(-2) \
                                    - xE/5*(5*yE4/7  + 2*(4*(4-3)-1)*yE2 + 5*4*(0)*(-1) \
                                    - xE/6*(6*yE4/8  + 2*(5*(5-3)-1)*yE2 + 6*5*1*0 \
                                    - xE/7*(7*yE4/9  + 2*(6*(6-3)-1)*yE2 + 7*6*2*1 \
                                    - xE/8*(8*yE4/10 + 2*(7*(7-3)-1)*yE2 + 8*7*3*2 \
                                    - xE/9*(9*yE4/11 + 2*(8*(8-3)-1)*yE2 + 9*8*4*3 \
                                    - xE/10*(10*yE4/12 + 2*(9*(9-3)-1)*yE2 + 10*9*5*4 \
                                    - xE/11*(11*yE4/13 + 2*(10*(10-3)-1)*yE2 + 11*10*6*5 \
                                    - xE/12*(12*yE4/14 + 2*(11*(11-3)-1)*yE2 + 12*11*7*6 \
                                    - xE/13*(13*yE4/15 + 2*(12*(12-3)-1)*yE2 + 13*12*8*7 \
                                    - xE/14*(14*yE4/16 + 2*(13*(13-3)-1)*yE2 + 14*13*9*8 \
                                    - xE/15*(15*yE4/17 + 2*(14*(14-3)-1)*yE2 + 15*14*10*9 \
                                    - xE/16*(16*yE4/18 + 2*(15*(15-3)-1)*yE2 + 16*15*11*10 \
                                    - xE/17*(17*yE4/19 + 2*(16*(16-3)-1)*yE2 + 17*16*12*11 \
                                    - xE/18*(18*yE4/20 + 2*(17*(17-3)-1)*yE2 + 18*17*13*12 \
                                    - xE/19*(19*yE4/21 + 2*(18*(18-3)-1)*yE2 + 19*18*14*13 \
                                    - xE/20*(20*yE4/22 + 2*(19*(19-3)-1)*yE2 + 20*19*15*14 \
                                    - xE/21*(21*yE4/23 + 2*(20*(20-3)-1)*yE2 + 21*20*16*15 \
                                    - xE/22*(22*yE4/24 + 2*(21*(21-3)-1)*yE2 + 22*21*17*16 \
                                    - xE/23*(23*yE4/25 + 2*(22*(22-3)-1)*yE2 + 23*22*18*17 \
                                    - xE/24*(24*yE4/26 + 2*(23*(23-3)-1)*yE2 + 24*23*19*18 \
                                    - xE/25*(25*yE4/27 + 2*(24*(24-3)-1)*yE2 + 25*24*20*19 \
                                    - xE/26*(26*yE4/28 + 2*(25*(25-3)-1)*yE2 + 26*25*21*20 \
                                    )))))))))))))))))))))))))
    I0x = I0 + Ix
    trigTerm[I0x] = 2*x[I0x]*(x[I0x]**2 - 3*y[I0x]**2)*(1 - cos(y[I0x])) - 2*y[I0x]*(3*x[I0x]**2 - y[I0x]**2)*sin(y[I0x])
    yT2 = y[Iy]**2
    xT3 = -x[Iy]**3
    xT2 = 6*x[Iy]**2
    xT = 6*x[Iy]
    
    if Iy.any():
        trigTerm[Iy] =   -2*yT2/(1*2)*  (xT3 + 1*xT2 - 0 - \
                            yT2/(3*4)*  (xT3 + 2*xT2 - 2*3*xT + 4*1*2*3  - \
                            yT2/(5*6)*  (xT3 + 3*xT2 - 3*5*xT + 4*2*3*5  - \
                            yT2/(7*8)*  (xT3 + 4*xT2 - 4*7*xT + 4*3*4*7  - \
                            yT2/(9*10)* (xT3 + 5*xT2 - 5*9*xT + 4*4*5*9  - \
                            yT2/(11*12)*(xT3 + 6*xT2 - 6*11*xT + 4*5*6*11 - \
                            yT2/(13*14)*(xT3 + 7*xT2 - 7*13*xT + 4*6*7*13 - \
                            yT2/(15*16)*(xT3 + 8*xT2 - 8*15*xT + 4*7*8*15 - \
                            yT2/(17*18)*(xT3 + 9*xT2 - 9*17*xT + 4*8*9*17 - \
                            yT2/(19*20)*(xT3 + 10*xT2 - 10*19*xT + 4*9*10*19))))))))))
    
    I0xy = I0 + Ix + Iy
    f[I0xy] = (trigTerm[I0xy] + y[I0xy]**2 * expTerm[I0xy]) / (y[I0xy]**2 + x[I0xy]**2)**3
    yH2 = y[Ixy]**2
    xH = x[Ixy]

    yH2 = y[Ixy]**2
    xH = x[Ixy]
    
    if Ixy.any():
        f[Ixy] =    yH2/(2*3*4*5)*(1*2-xH/6* (2*3-xH/7 *(3*4-xH/8* (4*5-xH/9 *(5*6-xH/10*(6*7-xH/11*(7*8-xH/12*(8*9-xH/13*(9*10-xH/14*(10*11-xH/15*(11*12-xH/16*(12*13-xH/17*(13*14-xH/18*(14*15-xH/19*(15*16-xH/20.*(16*17-xH/21.*(17*18-xH/22.*(18*19-xH/23.*(19*20-xH/24)))))))))))))))))) - \
                    yH2/(6*7)*    (1*2-xH/8* (2*3-xH/9 *(3*4-xH/10*(4*5-xH/11*(5*6-xH/12*(6*7-xH/13*(7*8-xH/14*(8*9-xH/15*(9*10-xH/16*(10*11-xH/17*(11*12-xH/18*(12*13-xH/19*(13*14-xH/20*(14*15-xH/21*(15*16-xH/22.*(16*17-xH/23.*(17*18-xH/24.*(18*19-xH/25.*(19*20-xH/26)))))))))))))))))) - \
                    yH2/(8*9)*    (1*2-xH/10*(2*3-xH/11*(3*4-xH/12*(4*5-xH/13*(5*6-xH/14*(6*7-xH/15*(7*8-xH/16*(8*9-xH/17*(9*10-xH/18*(10*11-xH/19*(11*12-xH/20*(12*13-xH/21*(13*14-xH/22*(14*15-xH/23*(15*16-xH/24.*(16*17-xH/25.*(17*18-xH/26.*(18*19-xH/27.*(19*20-xH/28)))))))))))))))))) - \
                    yH2/(10*11)*  (1*2-xH/12*(2*3-xH/13*(3*4-xH/14*(4*5-xH/15*(5*6-xH/16*(6*7-xH/17*(7*8-xH/18*(8*9-xH/19*(9*10-xH/20*(10*11-xH/21*(11*12-xH/22*(12*13-xH/23*(13*14-xH/24*(14*15-xH/25*(15*16-xH/26.*(16*17-xH/27.*(17*18-xH/28.*(18*19-xH/29.*(19*20-xH/30)))))))))))))))))) - \
                    yH2/(12*13)*  (1*2-xH/14*(2*3-xH/15*(3*4-xH/16*(4*5-xH/17*(5*6-xH/18*(6*7-xH/19*(7*8-xH/20*(8*9-xH/21*(9*10-xH/22*(10*11-xH/23*(11*12-xH/24*(12*13-xH/25*(13*14-xH/26*(14*15-xH/27*(15*16-xH/28.*(16*17-xH/29.*(17*18-xH/30.*(18*19-xH/31.*(19*20-xH/32)))))))))))))))))) - \
                    yH2/(14*15)*  (1*2-xH/16*(2*3-xH/17*(3*4-xH/18*(4*5-xH/19*(5*6-xH/20*(6*7-xH/21*(7*8-xH/22*(8*9-xH/23*(9*10-xH/24*(10*11-xH/25*(11*12-xH/26*(12*13-xH/27*(13*14-xH/28*(14*15-xH/29*(15*16-xH/30.*(16*17-xH/31.*(17*18-xH/32.*(18*19-xH/33.*(19*20-xH/34)))))))))))))))))) - \
                    yH2/(16*17)*  (1*2-xH/18*(2*3-xH/19*(3*4-xH/20*(4*5-xH/21*(5*6-xH/22*(6*7-xH/23*(7*8-xH/24*(8*9-xH/25*(9*10-xH/26*(10*11-xH/27*(11*12-xH/28*(12*13-xH/29*(13*14-xH/30*(14*15-xH/31*(15*16-xH/32.*(16*17-xH/33.*(17*18-xH/34.*(18*19-xH/35.*(19*20-xH/36))))))))))))))))))  \
                    )))))))
    return f

def SpecialCosineExp(x,y):
    x=array(x)
    y=array(y)
    
    ex = 0.6
    ey = 0.45
    
    I0 = ((x > ex)  & (y > ey))
    Ix = ((x <= ex) & (y > ey))
    Iy = ((x > ex)  & (y <= ey))
    Ixy = ((x <= ex) & (y <= ey))
    
    expTerm = np.zeros(x.size)
    trigTerm = np.zeros(x.size)
    f = np.zeros(x.size)
    
    I0y = I0 + Iy
    
    
    if I0y.any():
        expTerm[I0y] = (1 - exp(-x[I0y])) / x[I0y]
    xH = x[Ix]
    
    if Ix.any():
        expTerm[Ix] = 1 - xH/2*(\
                      1 - xH/3*(\
                      1 - xH/4*(\
                      1 - xH/5*(\
                      1 - xH/6*(\
                      1 - xH/7*(\
                      1 - xH/8*(\
                      1 - xH/9*(\
                      1 - xH/10*(\
                      1 - xH/11*(\
                      1 - xH/12*(\
                      1 - xH/13*(\
                      1 - xH/14*(\
                      1 - xH/15)))))))))))))
    
    I0x = I0 + Ix
    trigTerm[I0x] = x[I0x]*(1 - cos(y[I0x])) - y[I0x]*sin(y[I0x])

    yH2 = y[Iy]**2
    xH = x[Iy]
    
    if Iy.any():
        trigTerm[Iy] =   yH2/(1*2)*(xH - 2 - \
                         yH2/(3*4)*(xH - 4 - \
                         yH2/(5*6)*(xH - 6 - \
                         yH2/(7*8)*(xH - 8 - \
                         yH2/(9*10)*(xH - 10 - \
                         yH2/(11*12)*(xH - 12 - \
                         yH2/(13*14)*(xH - 14)))))))
            
    I0xy = I0 + Ix + Iy
    f[I0xy] = (trigTerm[I0xy] + y[I0xy]**2*expTerm[I0xy]) / (x[I0xy]**2+y[I0xy]**2)

    xH = x[Ixy];
    yH2 = y[Ixy]**2
    
    
    if Ixy.any():
        f[Ixy] =   yH2/(2*3)*(1 - xH/4*(1-xH/5*(1-xH/6*(1-xH/7*(1-xH/8*(1-xH/9*(1-xH/10*(1-xH/11*(1-xH/12*(1-xH/13*(1-xH/14*(1-xH/15))))))))))) - \
                   yH2/(4*5)*(1 - xH/6*(1-xH/7*(1-xH/8*(1-xH/9*(1-xH/10*(1-xH/11*(1-xH/12*(1-xH/13*(1-xH/14*(1-xH/15*(1-xH/16*(1-xH/17))))))))))) - \
                   yH2/(6*7)*(1 - xH/8*(1-xH/9*(1-xH/10*(1-xH/11*(1-xH/12*(1-xH/13*(1-xH/14*(1-xH/15*(1-xH/16*(1-xH/17*(1-xH/18*(1-xH/19))))))))))) - \
                   yH2/(8*9)*(1 - xH/10*(1-xH/11*(1-xH/12*(1-xH/13*(1-xH/14*(1-xH/15*(1-xH/16*(1-xH/17*(1-xH/18*(1-xH/19*(1-xH/20*(1-xH/21))))))))))) - \
                   yH2/(10*11)*(1 - xH/12*(1-xH/13*(1-xH/14*(1-xH/15*(1-xH/16*(1-xH/17*(1-xH/18*(1-xH/19*(1-xH/20*(1-xH/21*(1-xH/22*(1-xH/23))))))))))) - \
                   yH2/(12*13)*(1 - xH/14*(1-xH/15*(1-xH/16*(1-xH/17*(1-xH/18*(1-xH/19*(1-xH/20*(1-xH/21*(1-xH/22*(1-xH/23*(1-xH/24*(1-xH/25)))))))))))  \
                   ))))))

    return f
