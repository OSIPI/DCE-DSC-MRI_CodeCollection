"""Functions to convert between quantities and fit DCE-MRI data.

Created 28 September 2020
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Functions:
    sig_to_enh
    enh_to_conc
    conc_to_enh
    conc_to_pkp
    enh_to_pkp
    pkp_to_enh
    volume_fractions
    minimize_global
"""


import numpy as np
from scipy.optimize import root
from utils.utilities import least_squares_global


def sig_to_enh(s, base_idx):
    """Convert signal data to enhancement.

    Parameters
    ----------
    s : ndarray
        1D float array containing signal time series
    base_idx : list
        list of integers indicating the baseline time points.

    Returns
    -------
    enh : ndarray
        1D float array containing enhancement time series (%)
    """
    s_pre = np.mean(s[base_idx])
    enh = 100.*((s - s_pre)/s_pre)
    return enh


def enh_to_conc(enh, k, R10, c_to_r_model, signal_model):
    """Estimate concentration time series from enhancements.

    Assumptions:
        -fast-water-exchange limit.
        -see conc_to_enh

    Parameters
    ----------
    enh : ndarray
        1D float array containing enhancement time series (%)
    k : float
        B1 correction factor (actual/nominal flip angle)
    R10 : float
        Pre-contrast R1 relaxation rate (s^-1)
    c_to_r_model : c_to_r_model
        Model describing the concentration-relaxation relationship.
    signal_model : signal_model
        Model descriibing the relaxation-signal relationship.

    Returns
    -------
    C_t : ndarray
        1D float array containing tissue concentration time series (mM),
        specifically the mMol of tracer per unit tissue volume.

    """
    # Define function to fit for one time point
    def enh_to_conc_single(e):
        # Find the C where measured-predicted enhancement = 0
        res = root(lambda c:
                   e - conc_to_enh(c, k, R10, c_to_r_model, signal_model),
                   x0=0, method='hybr', options={'maxfev': 1000, 'xtol': 1e-7})
        if res.success is False:
            raise ArithmeticError(
                f'Unable to find concentration: {res.message}')
        return min(res.x)
    # Loop through all time points
    C_t = np.asarray([enh_to_conc_single(e) for e in enh])
    return C_t


def conc_to_enh(C_t, k, R10, c_to_r_model, signal_model):
    """Forward model to convert concentration to enhancement.

    Assumptions:
        -Fast-water-exchange limit.
        -Assumes R20=0 for convenience, which may not be valid for all
        sequences
        -R2* calculation not presently implemented. Assumes R2=R2*

    Parameters
    ----------
    C_t : ndarray
        1D float array containing tissue concentration time series (mM),
        specifically the mMol of tracer per unit tissue volume.
    k : float
        B1 correction factor (actual/nominal flip angle)
    R10 : float
        Pre-contrast R1 relaxation rate (s^-1)
    c_to_r_model : c_to_r_model
        Model describing the concentration-relaxation relationship.
    signal_model : signal_model
        Model descriibing the relaxation-signal relationship.

    Returns
    -------
    enh : ndarray
        1D float array containing enhancement time series (%)
    """
    R1 = c_to_r_model.R1(R10, C_t)
    R2 = c_to_r_model.R2(0, C_t)  # can assume R20=0 for existing signal models
    s_pre = signal_model.R_to_s(s0=1., R1=R10, R2=0, R2s=0, k=k)
    s_post = signal_model.R_to_s(s0=1., R1=R1, R2=R2, R2s=R2, k=k)
    enh = 100. * ((s_post - s_pre) / s_pre)
    return enh


def conc_to_pkp(C_t, pk_model, pk_pars_0=None, weights=None):
    """Fit concentration-time series to obtain pharmacokinetic parameters.

    Uses non-linear least squares optimisation.

    Assumptions:
        -Fast-water-exchange limit
        -See conc_to_enh

    Parameters
    ----------
    C_t : ndarray
        1D float array containing tissue concentration time series (mM),
        specifically the mMol of tracer per unit tissue volume.
    pk_model : pk_model
        Pharmacokinetic model used to predict tracer distribution.
    pk_pars_0 : list, optional
        list of dicts containing starting values of pharmacokinetic parameters.
        If there are >1 dicts then the optimisation will be run multiple times
        and the global minimum used.
        Example: [{'vp': 0.1, 'ps': 1e-3, 've': 0.5}]
        Defaults to values in pk_model.typical_vals.
    weights : ndarray, optional
        1D float array of weightings to use for sum-of-squares calculation.
        Can be used to "exclude" data points from optimisation.
        Defaults to equal weighting for all points.

    Returns
    -------
    tuple (pk_pars_opt, Ct_fit)
        pk_pars_opt : dict of optimal pharmacokinetic parameters,
            Example: {'vp': 0.1, 'ps': 1e-3, 've': 0.5}
        Ct_fit : 1D ndarray of floats containing best-fit tissue
            concentration-time series (mM).
    """
    if pk_pars_0 is None:
        pk_pars_0 = [pk_model.pkp_dict(pk_model.typical_vals)]
    if weights is None:
        weights = np.ones(C_t.shape)

    # Convert initial pars from list of dicts to list of arrays
    x_0_all = [pk_model.pkp_array(pars) for pars in pk_pars_0]

    def residuals(x):
        C_t_try, _C_cp, _C_e = pk_model.conc(*x)
        return weights * (C_t_try - C_t)

    result = least_squares_global(residuals, x_0_all, method='trf',
                                  bounds=pk_model.bounds,
                                  x_scale=(pk_model.typical_vals))

    if result.success is False:
        raise ArithmeticError(f'Unable to calculate pharmacokinetic parameters'
                              f': {result.message}')
    pk_pars_opt = pk_model.pkp_dict(result.x)  # convert parameters to dict
    check_ve_vp_sum(pk_pars_opt)
    Ct_fit, _C_cp, _C_e = pk_model.conc(*result.x)
    Ct_fit[weights == 0] = np.nan

    return pk_pars_opt, Ct_fit


def enh_to_pkp(enh, hct, k, R10_tissue, R10_blood, pk_model, c_to_r_model,
               water_ex_model, signal_model, pk_pars_0=None, weights=None):
    """Fit signal enhancement curve to obtain pharamacokinetic parameters.

    Any combination of signal, pharmacokinetic, relaxivity and water exchange
    models may be used.

    Assumptions:
        -R2 and R2* effects neglected.

    Parameters
    ----------
    enh : ndarray
        1D float array containing enhancement time series (%)
    hct : float
        Capillary haematocrit.
    k : float
        B1 correction factor (actual/nominal flip angle)
    R10_tissue : float
        Pre-contrast R1 relaxation rate for tissue (s^-1)
    R10_blood : float
        Pre-contrast R1 relaxation rate for capillary blood (s^-1). Used to
        estimate R10 for each tissue compartment. AIF R10 value is typically
        used.
    pk_model : pk_model
        Pharmacokinetic model used to predict tracer distribution.
    c_to_r_model : c_to_r_model
        Model describing the concentration-relaxation relationship.
    water_ex_model : water_ex_model
        Model to predict one or more exponential relaxation components given
        the relaxation rates for each compartment and water exchange behaviour.
    signal_model : signal_model
        Model descriibing the relaxation-signal relationship.
    pk_pars_0 : list, optional
        List of dicts containing starting values of pharmacokinetic parameters.
        If there are >1 dicts then the optimisation will be run multiple times
        and the global minimum used.
        Example: [{'vp': 0.1, 'ps': 1e-3, 've': 0.5}]
        Defaults to values in pk_model.typical_vals.
    weights : ndarray, optional
        1D float array of weightings to use for sum-of-squares calculation.
        Can be used to "exclude" data points from optimisation.
        Defaults to equal weighting for all points.

    Returns
    -------
    tuple (pk_pars_opt, Ct_fit)
        pk_pars_opt : dict of optimal pharmacokinetic parameters,
            Example: {'vp': 0.1, 'ps': 1e-3, 've': 0.5}
        enh_fit : 1D ndarray of floats containing best-fit tissue
            enhancement-time series (%).

    """
    if pk_pars_0 is None:  # get default initial estimates if none provided
        pk_pars_0 = [pk_model.pkp_dict(pk_model.typical_vals)]
    if weights is None:
        weights = np.ones(enh.shape)

    # get initial estimates as array, then scale
    x_0_all = [pk_model.pkp_array(pars) for pars in pk_pars_0]

    def residuals(x):
        pk_pars_try = pk_model.pkp_dict(x)
        enh_try = pkp_to_enh(pk_pars_try, hct, k, R10_tissue, R10_blood,
                             pk_model, c_to_r_model, water_ex_model,
                             signal_model)
        return weights * (enh_try - enh)

    # minimise the cost function
    result = least_squares_global(residuals, x_0_all, method='trf',
                                  bounds=pk_model.bounds,
                                  x_scale=(pk_model.typical_vals))
    if result.success is False:
        raise ArithmeticError(f'Unable to calculate pharmacokinetic parameters'
                              f': {result.message}')

    # generate optimal parameters (as dict) and predicted enh
    pk_pars_opt = pk_model.pkp_dict(result.x)
    check_ve_vp_sum(pk_pars_opt)
    enh_fit = pkp_to_enh(pk_pars_opt, hct, k, R10_tissue, R10_blood, pk_model,
                         c_to_r_model, water_ex_model, signal_model)
    enh_fit[weights == 0] = np.nan

    return pk_pars_opt, enh_fit


def pkp_to_enh(pk_pars, hct, k, R10_tissue, R10_blood, pk_model, c_to_r_model,
               water_ex_model, signal_model):
    """Forward model to generate enhancement from pharmacokinetic parameters.

    Any combination of signal, pharmacokinetic, relaxivity and water exchange
    models may be used.

    Assumptions:
        -R2 and R2* effects neglected.
        -spin population fraction = volume fraction
        -Pre-contrast fast water exchange
        -Pre-contrast R1 in capillary blood = R1 in VIF blood
        -Pre-contrast R1 equal in EES and intracellular space
        -Same concentration-relaxation relationship for all compartments

    Parameters
    ----------
    pk_pars : dict
        Pharmacokinetic parameters required by the model.
        Example: [{'vp': 0.1, 'ps': 1e-3, 've': 0.5}].
    hct : float
        Capillary haematocrit.
    k : k
        B1 correction factor (actual/nominal flip angle).
    R10_tissue : float
        Pre-contrast R1 relaxation rate for tissue (s^-1)
    R10_blood : float
        Pre-contrast R1 relaxation rate for capillary blood (s^-1). Used to
        estimate R10 for each tissue compartment. AIF R10 value is typically
        used.
    pk_model : pk_model
        Pharmacokinetic model used to predict tracer distribution.
    c_to_r_model : c_to_r_model
        Model describing the concentration-relaxation relationship.
    water_ex_model : water_ex_model
        Model to predict one or more exponential relaxation components given
        the relaxation rates for each compartment and water exchange behaviour.
    signal_model : signal_model
        Model descriibing the relaxation-signal relationship.

    Returns
    -------
    enh : ndarray
        1D float array containing enhancement time series (%).

    """
    # get volume fractions and spin population fractions
    v = volume_fractions(pk_pars, hct)
    p = v

    # calculate pre-contrast R10 in each compartment
    R10_extravasc = (R10_tissue-p['b']*R10_blood)/(1-p['b'])
    R10 = {'b': R10_blood,
           'e': R10_extravasc,
           'i': R10_extravasc}
    # calculate R10 exponential components
    R10_components, p0_components = water_ex_model.R1_components(p, R10)

    # calculate Gd concentration in each tissue compartment
    C_t, C_cp, C_e = pk_model.conc(**pk_pars)
    c = {'b': C_cp / v['b'],
         'e': C_e / v['e'],
         'i': np.zeros(C_e.shape),
         }

    # calculate R1 in each tissue compartment
    R1 = {'b': c_to_r_model.R1(R10['b'], c['b']),
          'e': c_to_r_model.R1(R10['e'], c['e']),
          'i': c_to_r_model.R1(R10['i'], c['i'])}

    # calculate R1 exponential components
    R1_components, p_components = water_ex_model.R1_components(p, R1)

    # calculate pre- and post-Gd signal, summed over relaxation components
    s_pre = np.sum([
        p0_c * signal_model.R_to_s(1, R10_components[i], k=k)
        for i, p0_c in enumerate(p0_components)], 0)
    s_post = np.sum([
        p_c * signal_model.R_to_s(1, R1_components[i], k=k)
        for i, p_c in enumerate(p_components)], 0)
    enh = 100. * (s_post - s_pre) / s_pre

    return enh


def volume_fractions(pk_pars, hct):
    """Calculate complete set of tissue volume fractions.

    Calculates a complete set of tissue volume fractions, including any not
    specified by the pharmacokinetic model.
    Example 1: The Tofts model does not specify vp, therefore assuming the
    original "weakly vascularised" interpretation, vb = 0 and
    vi = 1 - ve
    Example 2: The Patlak model does not specify ve, just a single
    extravascular compartment with ve = 1 - vb and vi = 0.

    Assumptions:
        This function encodes a set of assumptions required to predict
        signal enhancement in the presence of non-fast water exchange. This
        could be modified to incorporate additional information or alternative
        assumptions/interpretations.

    Parameters
    ----------
    pk_pars : dict
        Pharmacokinetic parameters required by the model.
        Example: [{'vp': 0.1, 'ps': 1e-3}].
    hct : float
        Capillary haematocrit.

    Returns
    -------
    v : dict
        Complete set of tissue volume fractions (blood, EES, intracellular).
        Example: {'b': 0.1, 'e': 0.4, 'i': 0.5}

    """
    # if vp exists, calculate vb, otherwise assume vb = 0
    if 'vp' in pk_pars:
        vb = pk_pars['vp'] / (1 - hct)
    else:
        vb = 0

    # if ve exists define vi as remaining volume, otherwise assume vi = 0
    if 've' in pk_pars:
        ve = pk_pars['ve']
        vi = 1 - vb - ve
    else:
        ve = 1 - vb
        vi = 0

    v = {'b': vb, 'e': ve, 'i': vi}
    return v


def check_ve_vp_sum(pk_pars):
    # check vp + ve <= 1
    if (('vp' in pk_pars) and ('ve' in pk_pars)) and (
            pk_pars['vp'] + pk_pars['ve'] > 1):
        v_tot = pk_pars['vp'] + pk_pars['ve']
        raise ValueError(f'vp + ve = {v_tot}!')
