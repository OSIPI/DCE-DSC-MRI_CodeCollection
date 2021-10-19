"""Functions to fit MRI SPGR signal to obtain T1.

Created 28 September 2020
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Functions:
    fit_vfa_2_point: obtain T1 using analytical formula based on two images
    fit_vfa_linear: obtain T1 using linear regression
    fit_vfa_nonlinear: obtain T1 using non-linear least squares fit
    fit_hifi: obtain T1 by fitting a combination of SPGR and IR-SPGR scans
    spgr_signal: get SPGR signal
    irspgr_signal: get IR-SPGR signal
"""

import numpy as np
from scipy.optimize import curve_fit, least_squares


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


def fit_hifi(s, esp, ti, n, b, td, centre, k_fixed=None):
    """Fit any combination of (IR-)SPGR scans to estimate T1.

    Note perfect inversion is assumed for IR-SPGR.

    Parameters
    ----------
    s : ndarray
        Array of signal intensities (1 float per acquisition).
    esp : ndarray
        Echo spacings (s, 1 float per acquisition). Equivalent to TR for SPGR
        scans.
    ti : ndarray
        Inversion times (s, 1 per acquisition). Note this is the actual time
        delay between the inversion pulse and the start of the echo train. The
        effective TI may be different, e.g for linear phase encoding of the
        echo train. For SPGR, set to np.nan
    n : ndarray
        Number of excitation pulses per inversion pulse (1 int per acquisition)
        . For SPGR, set to np.nan
    b : ndarray
        Excitation flip angles (deg, 1 float per acquisition).
    td : ndarray
        Delay between readout train and next inversion pulse (s, 1 float per
        acquisition).
    centre : ndarray
        Times in readout train when centre of k-space is acquired, expressed
        as a fraction of the readout duration. e.g. = 0 for centric phase
        encoding, = 0.5 for linear phase encoding (float, 1 per acquisition).
    k_fixed : float, optional
        Value to which k_fa (actual/nominal flip angle) is fixed. If set to
        None (default) then the value of k_fa is optimised.

    Returns
    -------
    tuple (t1_opt, s0_opt, k_fa_opt, s_opt)
        t1_opt: float
                T1 estimate (s).
        s0_opt: float
                Equilibrium signal estimate.
        k_fa_opt: float
                  k_fa (actual/nominal flip angle) estimate.

    """
    # get information about the scans
    n_scans = len(s)
    is_ir = ~np.isnan(ti)
    is_spgr = ~is_ir
    idx_spgr = np.where(is_spgr)[0]
    n_spgr = idx_spgr.size

    # First get a quick linear T1 estimate
    # If >1 SPGR, use linear VFA fit
    if n_spgr > 1 and np.all(np.isclose(esp[idx_spgr], esp[idx_spgr[0]])):
        s0_vfa, t1_vfa = fit_vfa_linear(s[is_spgr], b[is_spgr],
                                        esp[idx_spgr[0]])
        if ~np.isnan(s0_vfa) and ~np.isnan(t1_vfa):
            s0_init, t1_init = s0_vfa, t1_vfa
        else:  # if result invalid, assume T1=1
            t1_init = 1
            s0_init = s[idx_spgr[0]] / spgr_signal(1, t1_init,
                                                   esp[idx_spgr[0]],
                                                   b[idx_spgr[0]])
    # If 1 SPGR, assume T1=1 and estimate s0 based on this scan
    elif n_spgr == 1:
        t1_init = 1
        s0_init = s[idx_spgr[0]] / spgr_signal(1, t1_init,
                                               esp[idx_spgr[0]],
                                               b[idx_spgr[0]])
    # If all scans are IR-SPGR, assume T1=1 and estimate s0 based on 1st scan
    else:
        t1_init = 1
        s0_init = s[0] / irspgr_signal(1, t1_init, esp[0], ti[0], n[0], b[0],
                                       180, td[0], centre[0])

    # Prepare for non-linear fit
    if k_fixed is None:
        k_init = 1
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
    else:
        k_init = k_fixed
        bounds = ([0, 0, 1], [np.inf, np.inf, 1])
    x_0 = np.array([t1_init, s0_init, k_init])

    def residuals(x):
        t1_try, s0_try, k_fa_try = x
        s_try = np.zeros(n_scans)
        s_try[is_ir] = irspgr_signal(s0_try, t1_try, esp[is_ir], ti[is_ir],
                                     n[is_ir], k_fa_try*b[is_ir], td[is_ir],
                                     centre[is_ir])
        s_try[is_spgr] = spgr_signal(s0_try, t1_try, esp[is_spgr],
                                     k_fa_try*b[is_spgr])
        return s_try - s

    result = least_squares(residuals, x_0, bounds=bounds, method='trf',
                           x_scale=(t1_init, s0_init, k_init)
                           )
    if result.success is False:
        raise ArithmeticError(f'Unable to fit HIFI data'
                              f': {result.message}')

    t1_opt, s0_opt, k_fa_opt = result.x
    s_opt = np.zeros(n_scans)
    s_opt[is_ir] = irspgr_signal(s0_opt, t1_opt, esp[is_ir], ti[is_ir],
                                 n[is_ir], k_fa_opt*b[is_ir], td[is_ir],
                                 centre[is_ir])
    s_opt[is_spgr] = spgr_signal(s0_opt, t1_opt, esp[is_spgr],
                                 k_fa_opt*b[is_spgr])

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


def irspgr_signal(s0, t1, esp, ti, n, b, td=0, centre=0.5):
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