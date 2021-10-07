"""Functions to fit MRI SPGR signal to obtain T1.

Created 28 September 2020
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Functions:
    fit_vfa_2_point: obtain T1 using analytical formula based on two images
    fit_vfa_linear: obtain T1 using linear regression
    fit_vfa_nonlinear: obtain T1 using non-linear least squares fit

"""

import numpy as np
from scipy.optimize import curve_fit

from .utils.utilities import minimize_global


def fit_vfa_2_point(s, fa, tr):
    """Return T1 based on 2 SPGR signals with different FA but same TR and TE.

    Parameters
    ----------
    s : ndarray
        1D array containing 2 signal values.
    fa : ndarray
        1D array containing 2 FA values (deg).
    tr : float
        TR for SPGR sequence (s).

    Returns
    -------
    s0 : float
         Equilibrium signal.
    t1 : float
        T1 estimate (s).

    """
    fa_rad = np.pi*fa/180

    with np.errstate(divide='ignore', invalid='ignore'):
        sr = s[0] / s[1]
        t1 = tr / np.log(
            (sr*np.sin(fa_rad[1])*np.cos(fa_rad[0]) -
             np.sin(fa_rad[0])*np.cos(fa_rad[1])) /
            (sr*np.sin(fa_rad[1]) - np.sin(fa_rad[0])))
        s0 = s[0] * ((1-np.exp(-tr/t1)*np.cos(fa_rad[0])) /
                     ((1-np.exp(-tr/t1))*np.sin(fa_rad[0])))

    t1 = np.nan if ~np.isreal(t1) | (t1 <= 0) | np.isinf(t1) else t1
    s0 = np.nan if (s0 <= 0) | np.isinf(s0) else s0

    return s0, t1


def fit_vfa_linear(s, fa, tr):
    """Return T1 based on VFA signals using linear regression.

    Parameters
    ----------
        s: ndarray
           1D array of signals.
        fa: ndarray
                1D array of flip angles (deg).
        tr: float
            Repetition time (s).

    Returns
    -------
        t1: float
            T1 estimate (s).
        s0: float
            Equilibrium signal.
    """
    fa_rad = np.pi*fa/180

    y = s / np.sin(fa_rad)
    x = s / np.tan(fa_rad)
    A = np.stack([x, np.ones(x.shape)], axis=1)
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    is_valid = (intercept > 0) and (0. < slope < 1.)
    t1, s0 = (-tr/np.log(slope),
              intercept/(1-slope)) if is_valid else (np.nan, np.nan)

    return s0, t1


def fit_vfa_nonlinear(s, fa, tr):
    """Return T1 based on VFA signals using NLLS fitting.

    Parameters
    ----------
        s: ndarray
           1D array of signals.
        fa: ndarray
                1D array of flip angles (deg).
        tr: float
            Repetition time (s).

    Returns
    -------
        t1: float
            T1 estimate (s).
        s0: float
            Equilibrium signal.
    """

    # use linear fit to obtain initial guess
    p_linear = np.array(fit_vfa_linear(s, fa, tr))

    # calculate typical values for scaling
    p_typ = np.array([s[0] / spgr_signal(1., 1., tr, fa[0]), 1.])

    # if linear result is valid, use this otherwise use typical values
    p0 = p_linear if (~np.isnan(p_linear[0]) &
                      ~np.isnan(p_linear[1])) else p_typ
    p0_scaled = p0 / p_typ

    popt_scaled, pcov = curve_fit(
        lambda x_fa, p_s0, p_t1: spgr_signal(
            p_s0 * p_typ[0], p_t1 * p_typ[1], tr, x_fa), fa, s,
        p0=p0_scaled, sigma=None, absolute_sigma=False, check_finite=True,
        bounds=(1e-5, np.inf), method='trf', jac=None)

    popt = p_typ * popt_scaled
    s0, t1 = popt[0], popt[1]
    return s0, t1


def fit_hifi(s, esp, ti, n, b, a, td, centre, weights=None, k_fixed = None, **minimizer_kwargs):
    n_scans = len(s)   
    is_ir = ~np.isnan(ti)
    is_spgr = ~is_ir
    idx_ir = np.where(is_ir)[0]
    idx_spgr = np.where(is_spgr)[0]
    n_ir = idx_ir.size
    n_spgr = idx_spgr.size
    if weights is None:
        weights = np.ones(n_scans)

    # quick linear s0 and T1 estimate
    if n_spgr > 1 and np.all(np.isclose(esp[idx_spgr], esp[idx_spgr[0]])): # use VFA linear method
        s0_vfa, t1_vfa = fit_vfa_linear(s[is_spgr], b[is_spgr],
                                          esp[idx_spgr[0]])
        #print(f"{s[is_spgr]}, {b[is_spgr]}, {esp[is_spgr]}")
        #print(f"{s0_vfa, t1_vfa}")
        if ~np.isnan(s0_vfa) & ~np.isnan(t1_vfa):
            s0_init, t1_init = s0_vfa, t1_vfa
            #print(f"initial s0, t1 (using vfa): {s0_init, t1_init}")
        else: # if invalid, assume T1=1
            t1_init = 1
            s0_init = s[idx_spgr[0]] / spgr_signal(1, t1_init,
                                                   esp[idx_spgr[0]],
                                                   b[idx_spgr[0]])
            #print(f"initial s0, t1 (using vfa, invalid): {s0_init, t1_init}")
    elif n_spgr == 1:
            t1_init = 1
            s0_init = s[idx_spgr[0]] / spgr_signal(1, t1_init,
                                                   esp[idx_spgr[0]],
                                                   b[idx_spgr[0]])
            #print(f"initial s0, t1 (using first spgr): {s0_init, t1_init}")
    else:
        t1_init = 1
        s0_init = s[0] / irspgr_signal(1, t1_init, esp[0], ti[0], n[0], b[0],
                                       180, td[0], centre[0])
        #print(f"initial s0, t1 (using first ir-spgr): {s0_init, t1_init}")
    
    if k_fixed is None:
        k_init = 1
        bounds = [(0,np.inf), (0, np.inf), (0, np.inf)]
    else:
        k_init = k_fixed
        bounds = [(0,np.inf), (0, np.inf), (1, 1)]
    
    x_scalefactor = np.array([t1_init, s0_init, k_init]) # t1, s0, k_fa
    x_0_norm_all = [np.array([1, 1, 1])]
    
    # now perform non-linear fit
    def cost(x_norm, *args):
        t1_try, s0_try, k_fa_try = x_norm * x_scalefactor
        s_try = np.zeros(n_scans)
        s_try[is_ir] = irspgr_signal(s0_try, t1_try, esp[is_ir], ti[is_ir],
                                      n[is_ir], k_fa_try*b[is_ir], a[is_ir],
                                      td[is_ir], centre[is_ir])
        s_try[is_spgr] = spgr_signal(s0_try, t1_try, esp[is_spgr],
                                     k_fa_try*b[is_spgr])
        ssq = np.sum(weights * ((s_try - s)**2))
        return ssq

    # SLOW vs. matlab (10ms) - try different method
    result = minimize_global(cost, x_0_norm_all, args=None,
                             bounds=bounds,
                             method='trust-ncg',
                             **minimizer_kwargs)
    
    if result.success is False:
        raise ArithmeticError(f'Unable to calculate T1'
                              f': {result.message}')
    #print(result)

    t1_opt, s0_opt, k_fa_opt = result.x * x_scalefactor
    s_opt = np.zeros(n_scans)
    s_opt[is_ir] = irspgr_signal(s0_opt, t1_opt, esp[is_ir], ti[is_ir],
                                      n[is_ir], k_fa_opt*b[is_ir], a[is_ir],
                                      td[is_ir], centre[is_ir])
    s_opt[is_spgr] = spgr_signal(s0_opt, t1_opt, esp[is_spgr],
                                 k_fa_opt*b[is_spgr])
    s_opt[weights == 0] = np.nan

    return t1_opt, s0_opt, k_fa_opt, s_opt


def spgr_signal(s0, t1, tr, fa):
    """Return signal for SPGR sequence.

    Parameters
    ----------
        s0 : float
             Equilibrium signal.
        t1 : float
             T1 value (s).
        tr : float
             TR value (s).
        fa : float
                 Flip angle (deg).

    Returns
    -------
        s : float
            Steady-state SPGR signal.
    """
    fa_rad = np.pi*fa/180

    e = np.exp(-tr/t1)
    s = s0 * (((1-e)*np.sin(fa_rad)) /
              (1-e*np.cos(fa_rad)))

    return s


def irspgr_signal(s0, t1, esp, ti, n, b, a=180, td=0, centre=0.5):
    """Return signal for IR-SPGR sequence.

    Uses formula by Deichmann et al. (2000) to account for modified
    apparent relaxation rate during the pulse train. Note inversion is assumed
    to be ideal.

    Parameters
    ----------
        s0 : float
             Equilibrium signal.
        t1 : float
             T1 value (s).
        esp : float
             Echo spacing (s). For SPGR, this is the TR.
        ti : float
             Inversion time (s). Note this is the actual time delay between the
             inversion pulse and the start of the echo train. The effective TI
             may be different, e.g for linear phase encoding of the echo train.
        n : int
            Number of excitation pulses per inversion pulse
        b : float
            Readout pulse flip angle (deg)
        a : float
            Inversion pulse flip angle (deg). Not used in this implementation.
        td : float
             Delay between end of readout train and the next inversion (s).
        centre : float
                 Time in readout train when centre of k-space is acquired,
                 expressed as a fraction of the readout duration. e.g. = 0 for
                 centric phase encoding, = 0.5 for linear phase encoding.

    Returns
    -------
        s : float
            Steady-state IR-SPGR signal.
    """
    b_rad = np.pi*b/180
    tau = esp * n
    t1_star = (1/t1 - 1/esp*np.log(np.cos(b_rad)))**-1
    m0_star = s0 * ((1-np.exp(-esp/t1)) / (1-np.exp(-esp/t1_star)))

    r1 = -tau/t1_star
    e1 = np.exp(r1)
    e2 = np.exp(-td/t1)
    e3 = np.exp(-ti/t1)

    a1 = m0_star * (1-e1)
    a2 = s0 * (1 - e2)
    a3 = s0 * (1 - e3)

    a = a3 - a2*e3 - a1*e2*e3
    b = -e1*e2*e3

    m1 = a/(1-b)

    s = np.abs((
        m0_star + (m1-m0_star)*np.exp(centre*r1))*np.sin(b_rad))

    return s