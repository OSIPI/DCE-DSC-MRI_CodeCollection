"""Pharmacokinetic models.

Created 28 September 2020
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Classes: pk_model and associated subclasses
Functions: interpolate_time_series
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import LinearConstraint


class pk_model(ABC):
    """Abstract base class for pharmacokinetic models.

    Subclasses correspond to specific models (e.g. Tofts). The main purpose of
    a pk_model object is to return tracer concentration as a function of the
    model parameters.

    The main purpose of these classes is to provide a means to calculate tracer
    concentrations in the tissue compartments, as a function of the model
    parameters and the AIF. These are calculated by interpolating the AIF
    concentration and convolving with the impulse response functions (IRF).

    The approximate interpolated time resolution can be
    passed to the constructor as dt_interp_request, otherwise the spacing
    between the first two time points is used for interpolation.

    Attributes
    ----------
    t : np.ndarray
        1D array of time points (s) at which concentrations should be
        calculated
    aif : aifs.aif
        AIF object that will be used to calculate tissue concentrations
    dt_interp : float
        spacing of time points following interpolation (s)
    t_interp : np.ndarray
        interpolated time points (s)
    c_ap_interp : np.ndarray
        interpolated arterial plasma concentration time series (mM)
    n_interp : int
        number of interpolated time points
    n : int
        number of time points requested
    typical_vals : np.ndarray
        typical parameter values as 1D array (e.g. for scaling)
    constraints : scipy.optimize.LinearConstraint
        fitting constraints
    PARAMETERS : list
        list of parameter names
    TYPICAL_VALS : np.ndarray
        default typical parameter values as 1D array (e.g. for scaling)
    CONSTRAINTS : list
        list of scipy.optimize.LinearConstraint objects
        default constraints to use for the pharmacokinetic model

    Methods
    -------
    conc(*pk_pars, **pk_pars_kw)
        get concentrations in each tissue compartment at requested times
    irf
        get impulse response function for plasma and EES compartments
    pkp_array(pkp_dict)
        convert parameters from dict to array format
    pkp_dict(pkp_array)
        convert parameters from array to dict format
    """

    #  The following class variables should be overridden by derived classes
    PARAMETER_NAMES = None
    TYPICAL_VALS = None
    CONSTRAINTS = None

    def __init__(self, t, aif, dt_interp_request=None):
        """docstring."""
        self.t = t
        self.aif = aif

        if dt_interp_request is None:
            dt_interp_request = self.t[1] - self.t[0]
        self.dt_interp, self.t_interp = \
            interpolate_time_series(dt_interp_request, t)
        # get AIF concentration at interpolated time points
        self.c_ap_interp = aif.c_ap(self.t_interp)
        self.n_interp = self.t_interp.size
        self.n = self.t.size
        self.typical_vals = type(self).TYPICAL_VALS
        self.constraints = type(self).CONSTRAINTS

    def conc(self, *pk_pars, **pk_pars_kw):
        """Get concentration time series as function of model parameters.

        Parameters can be supplied either as individual arguments or as a dict.
        This superclass implementation is used for all subclasses.

        Parameters
        ----------
        *pk_pars, **pk_pars_kw : float
            Pharmacokinetic parameters, supplied either as positional arguments
            (in the order specified in PARAMETERS) or as keyword arguments.
            Possible parameters:
                vp : blood plasma volume fraction (fraction)
                ve : extravascular extracellular volume fraction (fraction)
                ps : permeability-surface area product (min^-1)
                fp : blood plasma flow rate (ml/100ml/min)
                ktrans : volume transfer constant (min^-1)

        Returns
        -------
        C_t : ndarray
            1D array of floats containing time series of tissue concentrations
            (mM).
        C_cp : ndarray
            1D array of floats containing time series of capillary plasma
            concentrations (mM). Note: concentration is per unit tissue
            volume.
        C_e : ndarray
            1D array of floats containing time series of EES concentrations
            (mM). Note: concentration is per unit tissue volume.
        """
        # Calculate IRF (using subclass implementation)
        irf_cp, irf_e = self.irf(*pk_pars, **pk_pars_kw)
        irf_cp[0] /= 2
        irf_e[0] /= 2

        # Do the convolutions, taking only results in the required time range
        C_cp_interp = self.dt_interp * np.convolve(
            self.c_ap_interp, irf_cp, mode='full')[:self.n_interp]
        C_e_interp = self.dt_interp * np.convolve(
            self.c_ap_interp, irf_e, mode='full')[:self.n_interp]

        # Resample concentrations at the measured time points
        C_cp = np.interp(self.t, self.t_interp, C_cp_interp)
        C_e = np.interp(self.t, self.t_interp, C_e_interp)

        C_t = C_cp + C_e

        return C_t, C_cp, C_e

    @abstractmethod
    def irf(self):
        """Get IRF. Method is overriden in subclasses for specific models."""
        pass

    def pkp_array(self, pkp_dict):
        """Convert pharmacokineetic parameters from dict to array format.

        Parameters
        ----------
        pkp_dict : dict
            Dict of pharmacokinetic parameters.

        Returns
        -------
        TYPE : ndarray
            1D array of pharmacokinetic parameters in the order specified by
            PARAMETERS. Irrelevant input parameters are ignored.

        """
        return np.array([pkp_dict[p] for p in type(self).PARAMETER_NAMES])

    def pkp_dict(self, pkp_array):
        """Convert phamacokinetic parameters from array to dict format.

        Parameters
        ----------
        pkp_array : ndarray
            1D array of pharmacokinetic parameters in the order specified by
            PARAMETERS.

        Returns
        -------
        TYPE : dict
            Dict of pharmacokinetic parameters.

        """
        return dict(zip(type(self).PARAMETER_NAMES, pkp_array))
        pass


class steady_state_vp(pk_model):
    """Steady-state vp model subclass.

    Tracer is confined to a single blood plasma compartment with same
    concentration as for AIF.
    Parameters: vp
    """

    PARAMETER_NAMES = ('vp',)
    TYPICAL_VALS = np.array([0.1])
    CONSTRAINTS = [LinearConstraint(TYPICAL_VALS * np.array(
        [[1]]),  # 0 < vp <= 1
        np.array([1e-8]),
        np.array([1]),
        keep_feasible=True)]

    def irf(self, vp, **kwargs):
        """Get IRF for this model. Overrides superclass method."""
        # calculate irf for capillary plasma (delta function centred at t=0)
        irf_cp = np.zeros(self.n_interp, dtype=float)
        irf_cp[0] = 2. * vp / self.dt_interp

        # calculate irf for the EES (zero)
        irf_e = np.zeros(self.n_interp, dtype=float)

        return irf_cp, irf_e


class patlak(pk_model):
    """Patlak model subclass.

    Tracer is present in the blood plasma compartment with same concentration
    as AIF; one-way leakage into EES is permitted.
    Parameters: vp, ps
    """

    PARAMETER_NAMES = ('vp', 'ps')
    TYPICAL_VALS = np.array([0.1, 1.e-3])
    CONSTRAINTS = [LinearConstraint(TYPICAL_VALS * np.array(
        [[1, 0],  # 0 < vp <= 1
         [0, 1]]),  # -1e-3 < ps <= 1
        np.array([1e-8, -1e-3]),
        np.array([1, 1]),
        keep_feasible=True)]

    def irf(self, vp, ps, **kwargs):
        """Get IRF for this model. Overrides superclass method."""
        # calculate irf for capillary plasma (delta function centred at t=0)
        irf_cp = np.zeros(self.n_interp, dtype=float)
        irf_cp[0] = 2. * vp / self.dt_interp

        # calculate irf for the EES (constant term)
        irf_e = np.ones(self.n_interp, dtype=float) * (1./60.) * ps

        return irf_cp, irf_e


class extended_tofts(pk_model):
    """Extended tofts model subclass.

    Tracer is present in the blood plasma compartment with same concentration
    as AIF; two-way leakage between blood plasma and EES is permitted.
    Parameters: vp, ps, ve
    """

    PARAMETER_NAMES = ('vp', 'ps', 've')
    TYPICAL_VALS = np.array([0.1, 1e-3, 0.2])
    CONSTRAINTS = [LinearConstraint(TYPICAL_VALS * np.array(
        [[1, 0, 0],  # 0 < vp <= 1
         [0, 1, 0],  # -1e-3 < ps <= 1
         [1, 0, 1],  # 0 < vp + ve <= 1
         [0, 0, 1]]),  # 0 < ve < 1
        np.array([1e-8, -1e-3, 1e-8, 1e-8]),
        np.array([1, 1, 1, 1]),
        keep_feasible=True)]

    def irf(self, vp, ps, ve, **kwargs):
        """Get IRF for this model. Overrides superclass method."""
        # calculate irf for capillary plasma (delta function centred at t=0)
        irf_cp = np.zeros(self.n_interp, dtype=float)
        irf_cp[0] = 2. * vp / self.dt_interp

        # calculate irf for the EES
        irf_e = (1./60.) * ps * np.exp(-(self.t_interp * ps)/(60. * ve))

        return irf_cp, irf_e


class tcum(pk_model):
    """Two-compartment uptake model subclass.

    Tracer flows from AIF to the blood plasma compartment; one-way leakage
    from blood plasma to the EES is permitted.
    Parameters: vp, ps, fp
    """

    PARAMETER_NAMES = ('vp', 'ps', 'fp')
    TYPICAL_VALS = np.array([0.1, 0.05, 50.])
    CONSTRAINTS = [LinearConstraint(TYPICAL_VALS * np.array(
        [[1, 0, 0],  # 0 < vp <= 1
         [0, 1, 0],  # -1e-3 < ps <= 1
         [0, 0, 1]]),  # 0 < Fp < 200
         np.array([1e-8, -1e-3, 1e-8]),
         np.array([1, 1, 200]),
         keep_feasible=True)]

    def irf(self, vp, ps, fp, **kwargs):
        """Get IRF for this model. Overrides superclass method."""
        fp_per_s = fp / (60. * 100.)
        ps_per_s = ps / 60.
        tp = vp / (fp_per_s + ps_per_s)
        ktrans = ps_per_s / (1 + ps_per_s/fp_per_s)

        # calculate irf for capillary plasma
        irf_cp = fp_per_s * np.exp(-self.t_interp/tp)

        # calculate irf for the EES
        irf_e = ktrans * (1 - np.exp(-self.t_interp/tp))

        return irf_cp, irf_e


class tcxm(pk_model):
    """Two-compartment exchange model subclass.

    Tracer flows from AIF to the blood plasma compartment; two-way leakage
    between blood plasma and EES is permitted.
    Parameters: vp, ps, ve, fp
    """

    PARAMETER_NAMES = ('vp', 'ps', 've', 'fp')
    TYPICAL_VALS = np.array([0.1, 0.05, 0.5, 50.])
    CONSTRAINTS = [LinearConstraint(TYPICAL_VALS * np.array(
        [[1, 0, 1, 0],  # 0 < vp + ve <= 1
         [1, 0, 0, 0],  # 0 < vp <= 1
         [0, 1, 0, 0],  # -1e-3 < ps <= 1
         [0, 0, 1, 0],  # 0 < ve <= 1
         [0, 0, 0, 1]]),  # 0 < Fp < 200
        np.array([1e-8, 1e-8, -1e-3, 1e-8, 1e-8]),
        np.array([1, 1, 1, 1, 200]),
        keep_feasible=True)]

    def irf(self, vp, ps, ve, fp, **kwargs):
        """Get IRF for this model. Overrides superclass method."""
        fp_per_s = fp / (60. * 100.)
        ps_per_s = ps / 60.
        v = ve + vp
        T = v / fp_per_s
        tc = vp / fp_per_s
        te = ve / ps_per_s
        sig_p = ((T + te) + np.sqrt((T + te)**2 - (4 * tc * te)))/(2 * tc * te)
        sig_n = ((T + te) - np.sqrt((T + te)**2 - (4 * tc * te)))/(2 * tc * te)

        # calculate irf for capillary plasma
        irf_cp = vp * sig_p * sig_n * (
             (1 - te*sig_n) * np.exp(-self.t_interp*sig_n) + (te*sig_p - 1.)
             * np.exp(-self.t_interp*sig_p)
             ) / (sig_p - sig_n)

        # calculate irf for the EES
        irf_e = ve * sig_p * sig_n * (np.exp(-self.t_interp*sig_n)
                                      - np.exp(-self.t_interp*sig_p)
                                      ) / (sig_p - sig_n)

        return irf_cp, irf_e


class tofts(pk_model):
    """Tofts model subclass.

    Tracer flows from AIF to the EES via a negligible blood plasma compartment;
    two-way leakage between blood plasma and the EES is permitted.
    Parameters: ktrans, ve
    """

    PARAMETER_NAMES = ('ktrans', 've')
    TYPICAL_VALS = np.array([1e-2, 0.2])
    CONSTRAINTS = [LinearConstraint(TYPICAL_VALS * np.array(
        [[1, 0],  # -1e-3 < ktrans <= 1.0
         [0, 0]]),  # 0 < ve < 1
        np.array([1e-3, 0]),
        np.array([1, 1]),
        keep_feasible=True)]

    def irf(self, ktrans, ve, **kwargs):
        """Get IRF for this model. Overrides superclass method."""
        ktrans_per_s = ktrans / 60.

        # calculate irf for capillary plasma (zeros)
        irf_cp = np.zeros(self.n_interp, dtype=float)

        # calculate irf for the EES
        irf_e = ktrans_per_s * np.exp(-self.t_interp * ktrans_per_s/ve)

        return irf_cp, irf_e


def interpolate_time_series(dt_required, t):
    """
    Interpolate a series of time points.

    Parameters
    ----------
    dt_required : float
        Requested spacing between interpolated time points.
    t : ndarray
        1D array of floats containing original time points.

    Returns
    -------
    dt_actual : float
        Spacing between interpolated time points
        (approximately equal to dt_required).
    t_interp : ndarray
        1D array of floats containing timepoints with spacing dt_actual,
        starting at dt_actual/2 and ending at the max(t).

    """
    # interpolate time series t to evenly spaced values from dt/2 to max(t)
    max_t = np.max(t)
    n_interp = np.round(max_t/dt_required + 0.5).astype(int)
    dt_actual = max_t/(n_interp-0.5)
    t_interp = np.linspace(0.5*dt_actual, max_t, num=n_interp)

    return dt_actual, t_interp
