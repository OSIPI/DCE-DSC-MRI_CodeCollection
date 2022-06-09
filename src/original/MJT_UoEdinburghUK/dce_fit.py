"""Classes and functions to convert between quantities and fit DCE-MRI data.

Created 28 September 2020
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Classes:
    SigToEnh
    EnhToConc
    ConcToPKP
    EnhToConcSPGR
    EnhToPKP
    PatlakLinear

Functions:
    conc_to_enh
    pkp_to_enh
    volume_fractions
    check_ve_vp_sum
"""

import numpy as np
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from .fitting import Fitter
from .pk_models import Patlak
from .utils.utilities import least_squares_global


class SigToEnh(Fitter):
    """Convert signal to enhancement.

    Subclass of Fitter. Calculates the enhancement of each volume relative to
    the mean over baseline volumes.
    """

    def __init__(self, base_idx):
        """

        Args:
            base_idx (array-like): indices corresponding to baseline volumes.
        """
        self.base_idx = base_idx

    def output_info(self):
        """Get output info. Overrides superclass method.
        """
        return ('enh', True),

    def proc(self, s):
        """Calculate enhancement time series. Overrides superclass method.

        Args:
            s (array): 1D signal array

        Returns:
            ndarray: 1D array of enhancements (%)
        """
        if any(np.isnan(s)):
            raise ValueError(
                f'Unable to calculate enhancements: nan arguments received.')
        s_pre = np.mean(s[self.base_idx])
        if s_pre <= 0:
            raise ArithmeticError('Baseline signal is zero or negative.')
        enh = np.empty(s.shape, dtype=np.float32)
        enh[:] = 100. * ((s - s_pre) / s_pre) if s_pre > 0 else np.nan
        return enh


class EnhToConc(Fitter):
    """Convert enhancement to concentration.

    Subclass of Fitter. Calculates points on the enh vs. conc curve,
    interpolates and uses this to "look up" concentration values given the
    enhancement values. It assumes the fast water exchange limit.
    """

    def __init__(self, c_to_r_model, signal_model, C_min=-0.5, C_max=30,
                 n_samples=1000):
        """

        Args:
            c_to_r_model (CRModel): concentration to relaxation
                relationship
            signal_model (SignalModel): relaxation to signal relationship
            C_min (float, optional): minimum value of concentration to look for
            C_max (float, optional): maximum value of concentration to look for
            n_samples (int, optional): number of points to sample the enh-conc
                function, prior to interpolation
        """
        self.c_to_r_model = c_to_r_model
        self.signal_model = signal_model
        self.C_min = C_min
        self.C_max = C_max
        self.C_samples = np.linspace(C_min, C_max, n_samples)

    def output_info(self):
        """Get output info. Overrides superclass method.
        """
        return ('C_t', True),

    def proc(self, enh, t10, k_fa=1):
        """Calculate concentration time series. Overrides superclass method.

        Args:
            enh (ndarray): 1D array of enhancements (%)
            t10 (float): tissue T10 (s)
            k_fa (float, optional): B1 correction factor (actual/nominal flip
            angle). Defaults to 1.

        Returns:
            ndarray: 1D array of tissue concentrations (mM)
        """
        if any(np.isnan(enh)) or np.isnan(t10) or np.isnan(k_fa):
            raise ValueError(
                f'Unable to calculate concentration: nan arguments received.')
        e_samples = conc_to_enh(self.C_samples, t10, k_fa, self.c_to_r_model,
                                self.signal_model)
        C_st = self.C_samples[np.concatenate((argrelextrema(e_samples,
                                                            np.greater)[0],
                                              argrelextrema(e_samples, np.less)[
                                                  0]))]
        C_lb = self.C_min if C_st[C_st <= 0].size == 0 else max(C_st[C_st <= 0])
        C_ub = self.C_max if C_st[C_st > 0].size == 0 else min(C_st[C_st > 0])
        points_allowed = (C_lb <= self.C_samples) & (self.C_samples <= C_ub)
        C_allowed = self.C_samples[points_allowed]
        e_allowed = e_samples[points_allowed]
        C_func = interp1d(e_allowed, C_allowed, kind='quadratic',
                          bounds_error=True)
        return C_func(enh)


class EnhToConcSPGR(Fitter):
    """Convert enhancement to concentration.

    Subclass of Fitter. Uses analytical formula for SPGR signal,
    excluding T2* effects and assuming the fast water exchange limit. This
    approach is faster than EnhToConc.
    """

    def __init__(self, tr, fa, r1):
        """

        Args:
            tr (float): repetition time (s)
            fa (float): flip angle (deg)
            r1 (float): R1 relaxivity (s^-1 mM^-1)
        """
        self.tr = tr
        self.fa = fa * np.pi/180
        self.r1 = r1

    def output_info(self):
        """Get output info. Overrides superclass method.
        """
        return ('C_t', True),

    def proc(self, enh, t10, k_fa=1):
        """Calculate concentration time series. Overrides superclass method.

        Args:
            enh (ndarray): 1D array of enhancements (%)
            t10 (float): tissue T10 (s)
            k_fa (float, optional): B1 correction factor (actual/nominal flip
            angle). Defaults to 1.

        Returns:
            ndarray: 1D array of tissue concentrations (mM)
        """
        if any(np.isnan(enh)) or np.isnan(t10) or np.isnan(k_fa):
            raise ValueError(
                f'Unable to calculate concentration: nan arguments received.')
        cos_fa_true = np.cos(k_fa * self.fa)
        exp_r10_tr = np.exp(self.tr/t10)
        C_t = -np.log((exp_r10_tr * (enh-100*cos_fa_true-enh*exp_r10_tr+100)) /
                      (100 * exp_r10_tr + enh * cos_fa_true - 100 * exp_r10_tr *
                       cos_fa_true - enh * exp_r10_tr * cos_fa_true)
                      ) / (self.tr * self.r1)
        return C_t


class ConcToPKP(Fitter):
    """Fit tissue concentrations using pharmacokinetic model.

    Subclass of Fitter.
    """
    def __init__(self, pk_model, pk_pars_0=None, weights=None):
        """
        Args:
            pk_model (PkModel): Pharmacokinetic model used to predict tracer
                distribution.
            pk_pars_0 (list, optional): list of dicts containing starting values
                of pharmacokinetic parameters. If there are >1 dicts then the
                optimisation will be run multiple times and the global minimum
                used.
                Example: [{'vp': 0.1, 'ps': 1e-3, 've': 0.5}]
                Defaults to values in PkModel.typical_vals.
            weights (ndarray, optional): 1D float array of weightings to use
                for sum-of-squares calculation. Can be used to "exclude" data
                points from optimisation. Defaults to equal weighting for all
                points.
        """
        self.pk_model = pk_model
        if pk_pars_0 is None:
            self.pk_pars_0 = [pk_model.pkp_dict(pk_model.typical_vals)]
        else:
            self.pk_pars_0 = pk_pars_0
        if weights is None:
            self.weights = np.ones(pk_model.n)
        else:
            self.weights = weights
        # Convert initial pars from list of dicts to list of arrays
        self.x_0_all = [pk_model.pkp_array(pars) for pars in self.pk_pars_0]

    def output_info(self):
        """Get output info. Overrides superclass method.
        """
        # outputs are pharmacokinetic parameters + fitted concentration
        return tuple([(name, False) for name in
                      self.pk_model.parameter_names]) + (('Ct_fit', True),)

    def proc(self, C_t):
        """Fit tissue concentration time series. Overrides superclass method.
        Args:
            C_t (ndarray): 1D float array containing tissue concentration
            time series (mM), specifically the mMol of tracer per unit tissue
            volume.

        Returns:
            tuple: (pk_par_1, pk_par_2, ..., Ct_fit)
            pk_par_i (float): fitted parameters (in the order given in
                self.PkModel.parameter_names)
            Ct_fit (ndarray): best-fit tissue concentration (mM).
        """
        if any(np.isnan(C_t)):
            raise ValueError(f'Unable to fit model: nan arguments received.')
        result = least_squares_global(self.__residuals, self.x_0_all,
                                      args=(C_t,), method='trf',
                                      bounds=self.pk_model.bounds,
                                      x_scale=self.pk_model.typical_vals)
        if result.success is False:
            raise ArithmeticError(
                f'Unable to calculate pharmacokinetic parameters'
                f': {result.message}')
        pk_pars_opt = self.pk_model.pkp_dict(result.x)
        check_ve_vp_sum(pk_pars_opt)
        Ct_fit, _C_cp, _C_e = self.pk_model.conc(*result.x)
        Ct_fit[self.weights == 0] = np.nan
        return tuple(result.x) + (Ct_fit,)

    def __residuals(self, x, C_t):
        C_t_try, _C_cp, _C_e = self.pk_model.conc(*x)
        res = self.weights * (C_t_try - C_t)
        return res


class EnhToPKP(Fitter):
    """Fit tissue enhancement curves using pharmacokinetic model.

    Subclass of Fitter. Fits tissue enhancements for specified combination of
    relaxivity model, water exchange model, sequence and pharmacokinetic
    model.
    Uses the following forward model:
        PkModel predicts CA concentrations in tissue compartments
        CRModel estimates relaxation rates in tissue compartments
        WaterExModel estimates exponential relaxation components
        SignalModel estimates MRI signal
    R2 and R2* effects neglected.
    """
    def __init__(self, hct, pk_model, t10_blood, c_to_r_model, water_ex_model,
                 signal_model, pk_pars_0=None, weights=None):
        """
        Args:
            hct (float): Capillary haematocrit
            pk_model (PkModel): Pharmacokinetic model used to predict tracer
                distribution.
            t10_blood (float): Pre-contrast T1 relaxation rate for capillary
                blood (s). Used to estimate T10 for each tissue compartment. AIF
                T10 value is typically used.
            c_to_r_model (CRModel): Model describing concentration-
                relaxation relationship.
            water_ex_model (WaterExModel): Model to predict one or more
                exponential relaxation components given the relaxation rates for
                each compartment and water exchange behaviour.
            signal_model (SignalModel): Model descriibing the
                relaxation-signal relationship.
            pk_pars_0 (list, optional): List of dicts containing starting
                values of pharmacokinetic parameters. If there are >1 dicts
                then the optimisation will be run multiple times and the
                global minimum used.
                Example: [{'vp': 0.1, 'ps': 1e-3, 've': 0.5}]
                Defaults to values in PkModel.typical_vals.
            weights (ndarray, optional): 1D float array of weightings to use for
                sum-of-squares calculation. Can be used to "exclude" data
                points from optimisation. Defaults to equal weighting for all
                points.
        """
        self.hct = hct
        self.pk_model = pk_model
        self.t10_blood = t10_blood
        self.c_to_r_model = c_to_r_model
        self.water_ex_model = water_ex_model
        self.signal_model = signal_model
        if pk_pars_0 is None:
            self.pk_pars_0 = [pk_model.pkp_dict(pk_model.typical_vals)]
        else:
            self.pk_pars_0 = pk_pars_0
        if weights is None:
            self.weights = np.ones(pk_model.n)
        else:
            self.weights = weights
        # Convert initial pars from list of dicts to list of arrays
        self.x_0_all = [pk_model.pkp_array(pars) for pars in self.pk_pars_0]

    def output_info(self):
        """Get output info. Overrides superclass method.
        """
        # outputs are pharmacokinetic parameters + fitted enhancement
        return tuple([(name, False) for name in
                      self.pk_model.parameter_names]) + (('enh_fit', True),)

    def proc(self, enh, k_fa, t10_tissue):
        """Fit enhancement time series. Overrides superclass method.

    Args:
        enh (ndarray): 1D float array of enhancement time series (%)
        k_fa (float): B1 correction factor (actual/nominal flip angle)
        t10_tissue(float): Pre-contrast T1 relaxation rate for tissue (s)

    Returns
    -------
    tuple (pk_pars_opt, Ct_fit)
        pk_pars_opt : dict of optimal pharmacokinetic parameters,
            Example: {'vp': 0.1, 'ps': 1e-3, 've': 0.5}
        enh_fit : 1D ndarray of floats containing best-fit tissue
            enhancement-time series (%).

    """
        if any(np.isnan(enh)) or np.isnan(t10_tissue) or np.isnan(k_fa):
            raise ValueError(f'Unable to fit model: nan arguments received.')
        result = least_squares_global(self.__residuals, self.x_0_all,
                                      args=(k_fa, t10_tissue, enh),
                                      method='trf',
                                      bounds=self.pk_model.bounds,
                                      x_scale=self.pk_model.typical_vals)
        if result.success is False:
            raise ArithmeticError(
                f'Unable to calculate pharmacokinetic parameters'
                f': {result.message}')
        pk_pars_opt = self.pk_model.pkp_dict(result.x)
        check_ve_vp_sum(pk_pars_opt)
        enh_fit = pkp_to_enh(pk_pars_opt, self.hct, k_fa, t10_tissue,
                             self.t10_blood, self.pk_model, self.c_to_r_model,
                             self.water_ex_model, self.signal_model)
        enh_fit[self.weights == 0] = np.nan
        return tuple(result.x) + (enh_fit,)

    def __residuals(self, x, k_fa, t10_tissue, enh):
        pk_pars_try = self.pk_model.pkp_dict(x)
        enh_try = pkp_to_enh(pk_pars_try, self.hct, k_fa, t10_tissue,
                             self.t10_blood, self.pk_model, self.c_to_r_model,
                             self.water_ex_model, self.signal_model)
        return self.weights * (enh_try - enh)


class PatlakLinear(Fitter):
    """Fit tissue concentrations using Patlak model.

    Subclass of Fitter.
    Uses multiple linear regression fitting. This is faster than non-linear
    fitting but more reliable than the traditional "graphical Patlak" method.
    """
    def __init__(self, t, aif, upsample_factor=1, include=None):
        """
        Args:
            t (ndarray): 1D float array of times (s) at which concentration
                should be calculated. Normally these are the times at which
                data points were measured. The sequence of times does not
                have to start at zero.
        aif (aifs.AIF): AIF object to use.
        upsample_factor (int, optional): The IRF and AIF are upsampled by
            this factor when calculating concentration. For non-uniform
            temporal resolution, the smallest time difference between time
            points is divided by this number. The default is 1.
        include (ndarray, optional): 1D float array of True/False or 1/0
                indicating which points to include in the linear regression.
                Defaults to None, in which case all points are included.
        """
        self.t = t
        self.aif = aif
        if include is None:
            self.include = np.ones(t.size).astype(bool)
        else:
            self.include = include.astype(bool)
        # Create a Patlak object and use to create the regressors
        # These are simply the vascular and EES contributions to
        # concentration when vp=1 and ps=1, i.e. the AIF and its integral.
        _, reg_vp, reg_ps = Patlak(t, aif, upsample_factor).conc(vp=1, ps=1)
        # combine regressors into a matrix
        self.regs = np.stack([reg_vp, reg_ps],
                             axis=1)
        self.regs_incl = np.stack([reg_vp[self.include], reg_ps[self.include]],
                             axis=1)

    def output_info(self):
        """Get output info. Overrides superclass method.
        """
        # outputs are pharmacokinetic parameters + fitted concentration
        return ('vp', False), ('ps', False), ('Ct_fit', True)

    def proc(self, C_t):
        """Fit tissue concentration time series. Overrides superclass method.
        Args:
            C_t (ndarray): 1D float array containing tissue concentration
            time series (mM), specifically the mMol of tracer per unit tissue
            volume.

        Returns:
            tuple: vp, ps, Ct_fit
                vp (float): blood plasma volume fraction (fraction)
                ps (float): permeability-surface area product (min^-1)
                Ct_fit (ndarray): 1D array of floats containing fitted tissue
                    concentrations (mM)
        """
        if any(np.isnan(C_t[self.include])):
            raise ValueError(f'Unable to fit model: nan arguments received.')

        # do ML regression
        try:
            coeffs = np.linalg.lstsq(
                self.regs_incl, C_t[self.include], rcond=None)[0]
        except LinAlgError:
            raise ArithmeticError(
                f'Unable to calculate pharmacokinetic parameters')
        vp, ps = coeffs
        Ct_fit = self.regs @ coeffs
        Ct_fit[~self.include] = np.nan
        return vp, ps, Ct_fit


def conc_to_enh(C_t, t10, k, c_to_r_model, signal_model):
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
    t10 : float
        Pre-contrast R1 relaxation rate (s^-1)
    c_to_r_model : CRModel
        Model describing the concentration-relaxation relationship.
    signal_model : SignalModel
        Model descriibing the relaxation-signal relationship.

    Returns
    -------
    enh : ndarray
        1D float array containing enhancement time series (%)
    """
    R10 = 1 / t10
    R1 = c_to_r_model.R1(R10, C_t)
    R2 = c_to_r_model.R2(0, C_t)  # can assume R20=0 for existing signal models
    s_pre = signal_model.R_to_s(s0=1., R1=R10, R2=0, R2s=0, k_fa=k)
    s_post = signal_model.R_to_s(s0=1., R1=R1, R2=R2, R2s=R2, k_fa=k)
    enh = 100. * ((s_post - s_pre) / s_pre)
    return enh


def pkp_to_enh(pk_pars, hct, k_fa, t10_tissue, t10_blood, pk_model,
               c_to_r_model, water_ex_model, signal_model):
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
    k_fa : k
        B1 correction factor (actual/nominal flip angle).
    t10_tissue : float
        Pre-contrast t1 relaxation rate for tissue (s)
    t10_blood : float
        Pre-contrast t1 relaxation rate for capillary blood (s). Used to
        estimate t10 for each tissue compartment. AIF t10 value is typically
        used.
    pk_model : PkModel
        Pharmacokinetic model used to predict tracer distribution.
    c_to_r_model : CRModel
        Model describing the concentration-relaxation relationship.
    water_ex_model : WaterExModel
        Model to predict one or more exponential relaxation components given
        the relaxation rates for each compartment and water exchange behaviour.
    signal_model : SignalModel
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
    R10_blood, R10_tissue = 1/t10_blood, 1/t10_tissue
    R10_extravasc = (R10_tissue - p['b'] * R10_blood) / (1 - p['b'])
    R10 = {'b': R10_blood, 'e': R10_extravasc, 'i': R10_extravasc}
    # calculate R10 exponential components
    R10_components, p0_components = water_ex_model.R1_components(p, R10)

    # calculate Gd concentration in each tissue compartment
    C_t, C_cp, C_e = pk_model.conc(**pk_pars)
    c = {'b': C_cp / v['b'], 'e': C_e / v['e'], 'i': np.zeros(C_e.shape), }

    # calculate R1 in each tissue compartment
    R1 = {'b': c_to_r_model.R1(R10['b'], c['b']),
          'e': c_to_r_model.R1(R10['e'], c['e']),
          'i': c_to_r_model.R1(R10['i'], c['i'])}

    # calculate R1 exponential components
    R1_components, p_components = water_ex_model.R1_components(p, R1)

    # calculate pre- and post-Gd signal, summed over relaxation components
    s_pre = np.sum(
        [p0_c * signal_model.R_to_s(1, R10_components[i], k_fa=k_fa) for i, p0_c in
         enumerate(p0_components)], 0)
    s_post = np.sum(
        [p_c * signal_model.R_to_s(1, R1_components[i], k_fa=k_fa) for i, p_c in
         enumerate(p_components)], 0)
    enh = 100. * (s_post - s_pre) / s_pre
    return enh


def volume_fractions(pk_pars, hct):
    """Calculate complete set of tissue volume fractions.

    Calculates a complete set of tissue volume fractions, including any not
    specified by the pharmacokinetic model.
    Example 1: The Tofts model does not specify vp, therefore (assuming the
    original "weakly vascularised" interpretation of the model), vb = 0 and
    vi = 1 - ve
    Example 2: The Patlak model does not specify ve, just a single
    extravascular compartment with ve = 1 - vb. Implicitly, vi = 0.

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
    """Check that ve+vp <= 1.

    Although this constraint could be implemented by the fitting algorithm,
    this would mean using a slower algorithm.

    Args:
        pk_pars (dict): Pharmacokinetic parameters.
            Example: [{'vp': 0.1, 'ps': 1e-3, 've': 0.5}].
    Raises:
        ValueError: if pk_pars inculdes vp and ve, and their sum is > 1
    """
    # check vp + ve <= 1
    if (('vp' in pk_pars) and ('ve' in pk_pars)) and (
            pk_pars['vp'] + pk_pars['ve'] > 1):
        v_tot = pk_pars['vp'] + pk_pars['ve']
        raise ArithmeticError(f'vp + ve = {v_tot}!')
