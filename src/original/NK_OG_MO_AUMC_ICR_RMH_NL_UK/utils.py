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

import numpy as np
from joblib import Parallel, delayed
import DCE_correction as dce

def generate_parameters(Nsamples, vp_uniform, state = 3):
    rg = np.random.RandomState(state)
    ke = rg.uniform(0, 3, Nsamples)
    dt = rg.uniform(0.5, 1.00, Nsamples)
    ve = rg.uniform(0, 1, Nsamples)
    if vp_uniform:
        vp = rg.uniform(0, 1.0, Nsamples)
    else:
        vp = rg.exponential(0.025, Nsamples)

    return ke,dt,ve,vp

def generate_aif(Nsamples, state = 3):
    aif = []
    ab,mb,ae,me,t0 = [2.84/(1-0.4), 22.8, 1.36, 0.171,0.5]
    rg = np.random.RandomState(state)
    ab = rg.uniform(ab * 0.8, ab * 1.2, Nsamples)
    mb = rg.uniform(mb * 0.8, mb * 1.2, Nsamples)
    ae = rg.uniform(ae * 0.8, ae * 1.2, Nsamples)
    me = rg.uniform(me * 0.8, me * 1.2, Nsamples)
    t0 = rg.uniform(t0 * 0.8, t0 * 1.2, Nsamples)
    for aa in range(Nsamples):
        aif.append({'ab': ab[aa], 'mb': mb[aa], 'ae': ae[aa], 'me': me[aa], 't0': t0[aa]})
    return aif

def calc_aif(timing,aif,delay,jobs):
    def func(idx):
        ab, ae, mb, me, t0 = [aif[idx]['ab'], aif[idx]['ae'], aif[idx]['mb'], aif[idx]['me'], aif[idx]['t0']]
        Cp = dce.Cosine4AIF_Conv(timing, ab, ae, mb, me, t0, delay)
        Cp = np.nan_to_num(Cp)
        return Cp
    Nsamples = len(aif)
    Cp_voxels = Parallel(n_jobs=jobs)(delayed(func)(i) for i in range(Nsamples))
    Cp_voxels = np.array(Cp_voxels)
    return Cp_voxels

def calc_concentration(timing,aif,ke,dt,ve,vp,delay,jobs,snr):
    def calc_noise(snr, signal):
        P_noise = np.mean(signal) / snr
        return np.random.normal(0, P_noise, len(signal))
    def func(idx):
        C = dce.Cosine4AIF_ExtKety_Conv(timing, aif[idx], ke[idx], dt[idx], ve[idx], vp[idx], delay)
        C = np.nan_to_num(C)
        if snr!=-1:
            C += calc_noise(snr, C)
        return C
    Nsamples = len(ke)
    C_voxels = Parallel(n_jobs=jobs)(delayed(func)(i) for i in range(Nsamples))
    C_voxels = np.array(C_voxels)
    return C_voxels
