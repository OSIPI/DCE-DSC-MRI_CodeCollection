"""
March 2021 by Oliver Gurney-Champion and Matthew Orton
o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion
Solves the Extended Tofts model for each voxel and returns model parameters using the computationally efficient AIF as described by Orton et al. 2008 in https://doi.org/10.1088/0031-9155/53/5/005
The Cosine8 AIF is further described by Rata et al. 2016 in https://doi.org/10.1007/s00330-015-4012-9

Copyright (C) 2021 by Oliver Gurney-Champion and Matthew Orton

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

requirements:
scipy
joblib
matplotlib
numpy
"""

from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from matplotlib.pylab import *
import numpy as np

def aifPopHN(Hct = 0.4):
    # defines plasma curve; note this is a population-based AIF for H&N patients (https://doi.org/10.2967/jnumed.116.174433). Please fit the AIF to determine your ideal ab, mb, ae and me parameters. You can use func aif and fit to your AIF for that.
    aif = {'ab': 3.646/(1-Hct), 'mb': 25.5671, 'ae': 1.53, 'me': 0.2130, 't0': 0.1}
    return aif

def aifPopPMB(Hct = 0.4):
    # defines plasma curve; note this is a population-based AIF based on https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21066 and https://iopscience.iop.org/article/10.1088/0031-9155/53/5/005
    # NB the PMB paper neglected to include the Hct in the parameter ab, so we include this adjustment here with an assumed Hct
    aif = {'ab': 2.84/(1-Hct), 'mb': 22.8, 'ae': 1.36, 'me': 0.171, 't0': 0.2}
    return aif

def Cosine4AIF(t,ab,ae,mb,me,dt):
    # input AIF model parameters; output AIF curve
    t = t - dt
    cpBolus = ab*CosineBolus(t,mb)
    cpWashout = ab*ae*ConvBolusExp(t,mb,me)
    cp = cpBolus + cpWashout
    return cp

def Cosine8AIF(t, ab, ar, ae, mb, mm, mr, me, tr, dt):
    t = t - dt
    cpBolus = ab*mm*ConvBolusExp(t, mb, mm)
    cpRecirc = ab*ar*CosineBolus(t - tr, mr)
    cpWashout = ab*ae*ConvBolusExp(t - tr, mr, me)
    cp = cpBolus + cpRecirc + cpWashout
    return cp

def fit_aif(Caif,t, model='Cosine8'):
    '''
    Calculates the AIF model parameters given a AIF curve
    :param Caif: Concentration curve of the AIF over time
    :param t: Time samples at which Caif is measured
    :return cp: library containing the AIF hyper parameters
    '''
    if model=='Cosine4':
        fit_func = lambda t, ab, ae, mb, me, dt: Cosine4AIF(t,ab,ae,mb,me,dt)
        X0 = (9,1.5,23,0.1,0.5)
        popt, _ = curve_fit(fit_func, t, Caif, p0=X0, bounds=(0, inf))
        aif = {'ab': popt[0], 'ae': popt[1], 'mb': popt[2], 'me': popt[3], 't0': popt[4]}
        fit_curve = Cosine4AIF(t, popt[0], popt[1], popt[2], popt[3], popt[4])

    if model=='Cosine8':
        fit_func = lambda t, ab, ar, ae, mb, mm, mr, me, tr, dt: Cosine8AIF(t, ab, ar, ae, mb, mm, mr, me, tr, dt)
        X0 = (10,0.05,0.3,50,12,10,0.2,0.1,0.45)
        popt, _ = curve_fit(fit_func, t, Caif, p0=X0, bounds=(0, inf))
        aif = {'ab': popt[0], 'ar': popt[1], 'ae': popt[2], 'mb': popt[3], 'mm': popt[4], 'mr': popt[5], 'me': popt[6], 'tr': popt[7], 't0': popt[8]}
        fit_curve = Cosine8AIF(t, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])

    #import matplotlib.pyplot as plt
    #plt.plot(t, fit_curve)
    #plt.plot(t, Caif, marker='.', linestyle='')
    #plt.show()

    return aif

def fit_tofts_model(Ct, t, aif, idxs=None, X0 = (0.6, 1, 0.03, 0.0025), bounds = ((0.0, 0, 0.0, 0.0), (5.0, 2, 1.0, 1.0)),jobs=4, model='Cosine8'):
    '''
    Solves the Extended Tofts model for each voxel and returns model parameters using the computationally efficient AIF as described by Orton et al. 2008 in https://doi.org/10.1088/0031-9155/53/5/005.
    :param Ct: Concentration curve as a N x T matrix, with N the number of curves to fit and T the number of time points
    :param t: Time samples at which Ct is measured
    :param aif: The aif model parameters in library, including ab, ae, mb, me and t0. can be obtained by fitting Cosine4AIF
    :param idxs: optional parameter determining which idxs to fit
    :param X0: initial guess of parameters ke, dt, ve, vp
    :param bounds: fit boundaries for  ke, dt, ve, vp
    :return output: matrix with EXtended Tofts parameters per voxel: ke,dt,ve,vp
    '''
    N, ndyn = Ct.shape
    if idxs is None:
        idxs = range(N)
    if model=='Cosine4':
        fit_func = lambda tt, ke, dt, ve, vp: Cosine4AIF_ExtKety(tt, aif, ke, dt, ve, vp)
    if model=='Cosine8':
        fit_func = lambda tt, ke, dt, ve, vp: Cosine8AIF_ExtKety(tt, aif, ke, dt, ve, vp)
    popt_default = [0, 0, 0, 0]
    print('fitting %d voxels' % len(idxs))
    if len(idxs)<2:
        output, pcov = curve_fit(fit_func, t, Ct[0], p0=X0, bounds=bounds)
        return output
    else:
        def parfun(idx, timing=t):
            if any(np.isnan(Ct[idx, :])):
                popt = popt_default
            else:
                try:
                    popt, pcov = curve_fit(fit_func, timing, Ct[idx, :], p0=X0, bounds=bounds)
                except RuntimeError:
                    popt = popt_default
            return popt
        output = Parallel(n_jobs=jobs, verbose=5)(delayed(parfun)(i) for i in idxs)
        return np.transpose(output)


def enhance(signal,delay=20,sds=1,percentage=10):
    """
    quick tool to check whether voxels are in deed enhancing.

    :param signal: Array with the input signal for different voxels
    :param delay: optional parameter which indicates the expected contrast arrival point.
    :param sds: an optional parameter determining number of SDs the signal needs to at least increase to de detected as enhancing voxel.
    :param percentage: an optional parameter determining the percentage of signal that needs to go over the threshold.

    :return selected: The selected enhancing voxel indices
    """
    stds = np.std(signal[:,:delay-5],axis=1)
    means = np.mean(signal[:,:delay-5],axis=1)
    cutoff = np.array(means+sds*stds)
    selects = signal[:,delay+5:] < np.repeat(np.expand_dims(cutoff,1),np.shape(signal)[1]-delay-5,axis=1)
    selected = np.sum(selects,1) < (percentage/100*(np.shape(signal)[1]-delay-5))
    return selected

def dce_to_r1eff(S, R1, TR, flip, baselinescans=9):
    """
    Go from DCE seignal over time to R1 over time.

    :param S: Array with the input signal for different voxels
    :param R1: Baseline R1.
    :param TR: Repetition time of the sequence.
    :param flip: The flip angle at which measured.

    :return : The selected enhancing voxel indices
    """
    S0 = np.mean(S[:,1:baselinescans],axis=1)*(1-np.exp(-R1*TR)*cos(flip)) / \
         (sin(flip)*(1-np.exp(-TR*R1)))
    S0 = np.repeat(np.expand_dims(S0,axis=1),np.shape(S)[1],axis=1)
    return - log((S0*sin(flip) - S)/(S0*sin(flip)-S*cos(flip)))/TR

def Cosine4AIF_ExtKety(t,aif,ke,dt,ve,vp):
    # The Cosine4 AIF is further described by Orton et al. 2008 in https://doi.org/10.1088/0031-9155/53/5/005
    # offset time array
    t = t - aif['t0'] - dt

    cpBolus = aif['ab']*CosineBolus(t,aif['mb'])
    cpWashout = aif['ab']*aif['ae']*ConvBolusExp(t,aif['mb'],aif['me'])
    ceBolus = ke*aif['ab']*ConvBolusExp(t,aif['mb'],ke)
    ceWashout = ke*aif['ab']*aif['ae']*ConvBolusExpExp(t,aif['mb'],aif['me'],ke)

    cp = cpBolus + cpWashout
    ce = ceBolus + ceWashout

    ct = np.zeros(np.shape(t))
    ct[t > 0] = vp * cp[t > 0] + ve * ce[t > 0]

    return ct

def Cosine8AIF_ExtKety(t, aif, ke, dt, ve, vp):
    # The Cosine8 AIF is further described by Rata et al. 2016 in https://doi.org/10.1007/s00330-015-4012-9
    # offset time array
    t = t - aif['t0'] - dt

    cpBolus = aif['ab'] * aif['mm'] * ConvBolusExp(t, aif['mb'], aif['mm'])
    cpRecirc = aif['ab'] * aif['ar'] * CosineBolus(t - aif['tr'], aif['mr'])
    cpWashout = aif['ab'] * aif['ae'] * ConvBolusExp(t - aif['tr'], aif['mr'], aif['me'])

    ceBolus = ke*aif['ab']*aif['mm']*ConvBolusExpExp(t,aif['mb'],aif['mm'],ke)
    ceRecirc = ke*aif['ab'] * aif['ar']*ConvBolusExp(t-aif['tr'],aif['mr'],ke)
    ceWashout = ke*aif['ab'] * aif['ae']*ConvBolusExpExp(t-aif['tr'],aif['mr'],aif['me'],ke)

    cp = cpBolus + cpRecirc + cpWashout
    ce = ceBolus + ceRecirc + ceWashout

    ct = np.zeros(np.shape(t))
    ct[t > 0] = vp * cp[t > 0] + ve * ce[t > 0]

    return ct


def CosineBolus(t,m):
    z = array(m * t)
    I = (z >= 0) & (z < (2 * pi))
    y = np.zeros(np.shape(t))
    y[I] = 1 - cos(z[I])

    return y

def ConvBolusExp(t,m,k):
    tB = 2 * pi / m
    tB=tB
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

def SpecialCosineGamma(x, y):
    x=array(x)
    y=array(y)
    expTerm = multiply(3+divide(square(y), square(x)), (1 - exp(-x))) \
                   - multiply(divide(square(y) + square(x),x), exp(-x))
    trigTerm = multiply((square(x) - square(y)), (1 - cos(y))) - multiply(multiply(2 * x,y), sin(y))
    f = divide((trigTerm + multiply(square(y), expTerm)), square(square(y) + square(x)))

    return f

def SpecialCosineExp(x,y):
    x=array(x)
    y=array(y)
    expTerm = divide((1 - exp(-x)), x)
    trigTerm = multiply(x, (1 - cos(y))) - multiply(y, sin(y))
    f = divide((trigTerm + multiply(square(y), expTerm)), (square(x) + square(y)))
    return f


# some helper functions to simulate data
def con_to_R1eff(C, R1map, relaxivity):
    assert(relaxivity > 0.0)
    return R1map + relaxivity * C

def r1eff_to_dce(R, TR, flip):
    S=((1-exp(-TR*R)) * sin(flip))/(1-cos(flip)*exp(-TR*R))
    return S

def R1_two_fas(images, flip_angles, TR):
    ''' Create T1 map from multiflip images '''
    inshape = images.shape
    nangles = inshape[-1]
    n = np.prod(inshape[:-1])
    images = np.reshape(images, (n, nangles))
    assert(nangles == 2)
    assert(len(flip_angles) == 2)
    signal_scale = abs(images).max()
    images = images / signal_scale
    R1map = zeros(n)
    c1=cos(flip_angles[0])
    c2=cos(flip_angles[1])
    s1=sin(flip_angles[0])
    s2=sin(flip_angles[1])
    rho=images[:,1]/images[:,0]
    for j in range(n):
        if images[j,:].mean() > 0.05:
            try:
                R1map[j]= np.log((rho[j]*s1*c2-c1*s2)/(rho[j]*s1-s2))/TR
            except RuntimeError:
                R1map[j] = 0
                print(j)
    return (R1map)

def R1_VFA(images, flip_angles, TR,jobs=4,bounds=((0,0),(np.Inf,10))):
    ''' Create T1 map from multiflip images '''
    inshape = images.shape
    nangles = inshape[-1]
    assert(nangles == len(flip_angles))
    try:
        signal_scale = np.max(abs(images),axis=1) + 0.01
        images = images / np.repeat(np.expand_dims(signal_scale,1),nangles,axis=1)
    except:
        signal_scale = np.max(abs(images), axis=0) + 0.01
        images = images / signal_scale
    fit_func = lambda alpha, M0, T1: R1_afo_FA(alpha,M0,T1,TR)
    popt_default = [0, 0]
    X0=[5,1]
    idxs = range(inshape[0])
    print('fitting ' + str(inshape[0]) + ' voxels')
    if np.ndim(images) is 1:
        output, pcov = curve_fit(fit_func, flip_angles, images, p0=X0,bounds=bounds)
        return output[1]
    else:
        def parfun(idx, alpha=flip_angles):
            if any(np.isnan(images[idx, :])):
                popt = popt_default
            else:
                try:
                    popt, pcov = curve_fit(fit_func, alpha, images[idx, :], p0=X0,bounds=bounds)
                except RuntimeError:
                    popt = popt_default
            return popt
        if jobs == 1:
            output = np.zeros((len(idxs),2))
            for idx in idxs:
                try:
                    popt, pcov = curve_fit(fit_func, flip_angles, images[idx, :], p0=X0,bounds=bounds)
                    output[idx] = popt
                except:
                    output[idx] = popt_default
        else:
            output = Parallel(n_jobs=jobs, verbose=5)(delayed(parfun)(i) for i in idxs)
        return np.transpose(output)

def R1_afo_FA(alpha,M0,T1,TR):
    E1=exp(-TR/T1)
    return M0 * (1-E1)*sin(alpha)/(1-cos(alpha)*E1)

def r1eff_to_conc(R1eff, R1map, relaxivity):
    return (R1eff - R1map) / relaxivity

