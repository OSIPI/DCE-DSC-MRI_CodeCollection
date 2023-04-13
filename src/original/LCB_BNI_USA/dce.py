"""Functions for DCE processing.

Created 15 October 2021
@authors: Laura Bell
@institution: Barrow Neurological Institute

Translated from jupyter notebook DSCpipeline.ipynb by Michael Thrippleton

Functions:
    estimate_delta_R1
    signal_to_conc
    tofts_model
    fit_tofts
"""

import numpy as np
from scipy.optimize import curve_fit


def estimate_delta_R1(s, basepts_range, t10, tr, fa):
    """Estimate change in R1 for DCE time series (SPGR or GE-EPI sequence).

    Parameters
    ----------
    s : ndarray
        1D signal array.
    basepts_range : list
        2-element list indicating [first, last] index to use for baseline
        signal estimation.
    t10 : float
        Pre-contrast T1 (s).
    tr : float
        Repetition time (s).
    fa : float
        Flip angle (deg).

    Returns
    -------
    s0 : float
        Fully relaxed signal.
    dR1 : ndarray
        1D array of Delta R1 time series (s^-1).

    """
    fa_rad = np.deg2rad(fa)
    R10 = 1 / t10
    s_base = np.mean(s[basepts_range[0] : basepts_range[1] + 1])
    s0 = (((np.exp(-R10 * tr) * np.cos(fa_rad) * s_base) - s_base) / np.sin(fa_rad)) / (
        np.exp(-R10 * tr) - 1.0
    )
    sin_fa_s0 = np.sin(fa_rad) * s0
    a = sin_fa_s0 - s
    b = sin_fa_s0 - np.cos(fa_rad) * s
    c = sin_fa_s0 - np.cos(fa_rad) * s_base
    d = sin_fa_s0 - s_base

    dR1 = -1 / tr * np.log((a / b) * (c / d))
    return s0, dR1


def signal_to_conc(s, basepts_range, t10, tr, fa, r1):
    """Convert DCE signal to concentration time series (SPGR or GE-EPI).

    Parameters
    ----------
    s : ndarray
        1D signal array.
    basepts_range : list
        2-element list indicating [first, last] index to use for baseline
        signal estimation.
    t10 : float
        Pre-contrast T1 (s).
    tr : float
        Repetition time (s).
    fa : float
        Flip angle (deg).
    r1 : float
        R1 relaxivity (s^-1 mM^-1).

    Returns
    -------
    C_t : ndarray
        1D array concentration time series (mM).

    """
    s0, dR1 = estimate_delta_R1(s, basepts_range, t10, tr, fa)
    C_t = dR1 / r1
    return C_t


def tofts_model(time, cp, ktrans, ve):
    """Forward Tofts model.

    Parameters
    ----------
    time : ndarray
        1D array of time points (s).
    cp : ndarray
        1D array of AIF plasma concentrations (mM).
    ktrans : float
        KTrans (min^-1).
    ve : float
        vE.

    Returns
    -------
    Ct : ndarray
        1D array of tissue concentrations (mM).

    """
    Ct = np.zeros_like(time)
    ktrans_per_s = ktrans / 60
    for tau in range(time.size):
        t = time[0 : tau + 1]
        cp_t = cp[0 : tau + 1]
        cp_t_exp = cp_t * np.exp((-ktrans_per_s / ve) * (t[-1] - t))
        if tau == 0:
            Ct[tau] = ktrans_per_s * 0
        else:
            Ct[tau] = ktrans_per_s * np.trapz(cp_t_exp, t)
    return Ct


def fit_tofts(
    time,
    Ct,
    cp,
    ktrans_0=120e-4,
    ve_0=0.2,
    ktrans_bounds=(60e-7, 120),
    ve_bounds=(0.01, 1),
):
    """Fit Tofts model to concentration data.

    Parameters
    ----------
    time : ndarray
        1D array of time points (s).
    Ct : ndarray
        1D array of tissue concentrations (mM).
    cp : ndarray
        1D array of AIF plasma concentrations (mM).
    ktrans_0 : float, optional
        KTrans starting value (min^-1). The default is 120e-4.
    ve_0 : float, optional
        vE starting value. The default is 0.2.
    ktrans_bounds : tuple, optional
        KTrans (low, upper) bounds (min^-1). The default is (60e-7, 120).
    ve_bounds : tuple, optional
        vE (lower, upper) bounds. The default is (0.01, 1).

    Returns
    -------
    ktrans_opt : float
        KTrans estimte (min^-1).
    ve_opt : float
        vE estimate.
    Ct_fit : ndarray
        1D array of tissue concentrations corresponding to model fit (mM).

    """
    bounds = (
        (ktrans_bounds[0] / 60, ve_bounds[0]),
        (ktrans_bounds[1] / 60, ve_bounds[1]),
    )
    pval, pcov = curve_fit(
        lambda t, ktrans_per_s, ve: tofts_model(t, cp, ktrans_per_s * 60, ve),
        time,
        Ct,
        p0=[ktrans_0 / 60, ve_0],
        bounds=bounds,
    )
    ktrans_opt = pval[0] * 60
    ve_opt = pval[1]
    Ct_fit = tofts_model(time, cp, ktrans_opt, ve_opt)
    return ktrans_opt, ve_opt, Ct_fit
