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
        This is only defined if the arterial delay is fixed.
    n_interp : int
        number of interpolated time points
    n : int
        number of time points at which to predict concentration
    fixed_delay : float
        Fixed delay applied to AIF concentration. For variable AIF delay, this
        is set to None and delay is supplied as an argument to the conc method.
    parameter_names : tuple
        names of variable parameters
    typical_vals : np.ndarray
        typical parameter values as 1D array (e.g. for scaling)
    bounds : tuple
        2-tuple giving lower and upper bounds for variable parameters. Refer to
        documentation for scipy.optimize.least_squares

    Class variables
    ---------------
    LOWER_BOUNDS: tuple
        default lower bounds for model parameters
    UPPER_BOUNDS: tuple
        default upper bounds for model parameters
    PARAMETER_NAME : tuple
        tuple containing parameter names
    TYPICAL_VALS : np.ndarray
        default typical parameter values as 1D array (e.g. for scaling)

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
    LOWER_BOUNDS = None
    UPPER_BOUNDS = None

    def __init__(self, t, aif, upsample_factor=1, fixed_delay=0):
        """Construct pk_model object.

        Parameters
        ----------
        t : ndarray
            1D float array of times at which concentration should be
            calculated. Normally these are the times at which data points were
            measured. The sequence of times does not have to start at zero.
        aif : aifs.aif
            aif object to use.
        upsample_factor : int, optional
            The IRF and AIF are upsampled by this factor when calculating
            concentration. For non-uniform temporal resolution, the smallest
            time difference between time points is divided by this number.
            The default is 1.
        fixed_delay : float, optional
            Fixed delay to apply to AIF, reflecting the arterial arrival time.
            The default is 0. If set to None, the AIF delay is assumed to be
            a variable parameter.
        """
        self.t = t
        self.n = self.t.size
        t_spacings = np.diff(t)
        self.isUniform = np.all(np.isclose(t_spacings, t_spacings[0]))
        self.aif = aif
        self.upsample_factor = upsample_factor
        self.dt_upsample = np.min(t_spacings) / upsample_factor
        self.t_upsample = np.linspace(t[0], t[-1], self.dt_upsample)
        self.tau_upsample = self.t_upsample - t[0]
        self.n_upsample = self.t_upsample.size
        self.fixed_delay = fixed_delay

        # set variable parameters and bounds, depending whether AIF delay fixed
        if fixed_delay is None:  # add AIF delay as a variable parameter
            self.parameter_names = type(self).PARAMETER_NAMES + ('delay',)
            self.typical_vals = np.append(type(self).TYPICAL_VALS, 1)
            self.bounds = (type(self).LOWER_BOUNDS + (-10,),
                           type(self).UPPER_BOUNDS + (10,))
        else:  # AIF delay is fixed; store AIF as vector for speed
            self.parameter_names = type(self).PARAMETER_NAMES
            self.typical_vals = type(self).TYPICAL_VALS
            self.bounds = (type(self).LOWER_BOUNDS, type(self).UPPER_BOUNDS)
            self.c_ap_upsample = aif.c_ap(self.t_upsample - fixed_delay)

    def conc(self, *pars, **pars_kw):
        """Get concentration time series as function of model parameters.

        Parameters can be supplied either as individual arguments or as a dict.
        This superclass implementation is used for all subclasses. The required
        parameters are defined by self.parameter_names.

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
                delay : AIF delay (s)

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
        # If delay is supplied as argument, use it to shift the AIF
        if self.fixed_delay is None:
            delay = pars_kw['delay'] if pars_kw else pars[-1]
            c_ap_upsample = self.aif.c_ap(self.t_upsample - delay)
        else:  # otherwise, use the stored AIF
            c_ap_upsample = self.c_ap_upsample

        # Calculate IRF (using subclass implementation)
        irf_cp, irf_e = self.irf(*pars, **pars_kw)
        irf_cp[[0, -1]] /= 2
        irf_e[[0, -1]] /= 2

        # Do the convolutions, taking only results in the required time range
        C_cp_upsample = self.dt_upsample * np.convolve(
            c_ap_upsample, irf_cp, mode='full')[:self.n_upsample]
        C_e_upsample = self.dt_upsample * np.convolve(
            c_ap_upsample, irf_e, mode='full')[:self.n_upsample]

        # Downsample concentrations back to the measured time points
        C_cp = np.interp(self.t, self.t_upsample, C_cp_upsample)
        C_e = np.interp(self.t, self.t_upsample, C_e_upsample)

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
        return np.array([pkp_dict[p] for p in self.parameter_names])

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
        return dict(zip(self.parameter_names, pkp_array))
        pass


class steady_state_vp(pk_model):
    """Steady-state vp model subclass.

    Tracer is confined to a single blood plasma compartment with same
    concentration as for AIF.
    Parameters: vp
    """

    PARAMETER_NAMES = ('vp',)
    TYPICAL_VALS = np.array([0.1])
    LOWER_BOUNDS = (0,)
    UPPER_BOUNDS = (1,)

    def irf(self, vp, **kwargs):
        """Get IRF for this model. Overrides superclass method."""
        # calculate irf for capillary plasma (delta function centred at t=0)
        irf_cp = np.zeros(self.n_upsample, dtype=float)
        irf_cp[0] = 2. * vp / self.dt_upsample

        # calculate irf for the EES (zero)
        irf_e = np.zeros(self.n_upsample, dtype=float)

        return irf_cp, irf_e


class patlak(pk_model):
    """Patlak model subclass.

    Tracer is present in the blood plasma compartment with same concentration
    as AIF; one-way leakage into EES is permitted.
    Parameters: vp, ps
    """

    PARAMETER_NAMES = ('vp', 'ps')
    TYPICAL_VALS = np.array([0.1, 1.e-3])
    LOWER_BOUNDS = (0, -1e-3)
    UPPER_BOUNDS = (1, 1)

    def irf(self, vp, ps, **kwargs):
        """Get IRF for this model. Overrides superclass method."""
        # calculate irf for capillary plasma (delta function centred at t=0)
        irf_cp = np.zeros(self.n_upsample, dtype=float)
        irf_cp[0] = 2. * vp / self.dt_upsample

        # calculate irf for the EES (constant term)
        irf_e = np.ones(self.n_upsample, dtype=float) * (1./60.) * ps

        return irf_cp, irf_e


class extended_tofts(pk_model):
    """Extended tofts model subclass.

    Tracer is present in the blood plasma compartment with same concentration
    as AIF; two-way leakage between blood plasma and EES is permitted.
    Parameters: vp, ps, ve
    """

    PARAMETER_NAMES = ('vp', 'ps', 've')
    TYPICAL_VALS = np.array([0.1, 1e-3, 0.2])
    LOWER_BOUNDS = (0, -1e-3, 1e-8)
    UPPER_BOUNDS = (1, 1, 1)

    def irf(self, vp, ps, ve, **kwargs):
        """Get IRF for this model. Overrides superclass method."""
        # calculate irf for capillary plasma (delta function centred at t=0)
        irf_cp = np.zeros(self.n_upsample, dtype=float)
        irf_cp[0] = 2. * vp / self.dt_upsample

        # calculate irf for the EES
        irf_e = (1./60.) * ps * np.exp(-(self.tau_upsample * ps)/(60. * ve))

        return irf_cp, irf_e


class tcum(pk_model):
    """Two-compartment uptake model subclass.

    Tracer flows from AIF to the blood plasma compartment; one-way leakage
    from blood plasma to the EES is permitted.
    Parameters: vp, ps, fp
    """

    PARAMETER_NAMES = ('vp', 'ps', 'fp')
    TYPICAL_VALS = np.array([0.1, 0.05, 50.])
    LOWER_BOUNDS = (1e-8, -1e-3, 1e-8)
    UPPER_BOUNDS = (1, 1, 200)

    def irf(self, vp, ps, fp, **kwargs):
        """Get IRF for this model. Overrides superclass method."""
        fp_per_s = fp / (60. * 100.)
        ps_per_s = ps / 60.
        tp = vp / (fp_per_s + ps_per_s)
        ktrans = ps_per_s / (1 + ps_per_s/fp_per_s)

        # calculate irf for capillary plasma
        irf_cp = fp_per_s * np.exp(-self.tau_upsample/tp)

        # calculate irf for the EES
        irf_e = ktrans * (1 - np.exp(-self.tau_upsample/tp))

        return irf_cp, irf_e


class tcxm(pk_model):
    """Two-compartment exchange model subclass.

    Tracer flows from AIF to the blood plasma compartment; two-way leakage
    between blood plasma and EES is permitted.
    Parameters: vp, ps, ve, fp
    """

    PARAMETER_NAMES = ('vp', 'ps', 've', 'fp')
    TYPICAL_VALS = np.array([0.1, 0.05, 0.5, 50.])
    LOWER_BOUNDS = (1e-8, -1e-3, 1e-8, 1e-8)
    UPPER_BOUNDS = (1, 1, 1, 200)

    def irf(self, vp, ps, ve, fp, *args, **kwargs):
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
             (1 - te*sig_n) * np.exp(-self.tau_upsample*sig_n) + (te*sig_p - 1.)
             * np.exp(-self.tau_upsample*sig_p)
             ) / (sig_p - sig_n)

        # calculate irf for the EES
        irf_e = ve * sig_p * sig_n * (np.exp(-self.tau_upsample*sig_n)
                                      - np.exp(-self.tau_upsample*sig_p)
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
    LOWER_BOUNDS = (-1e-3, 1e-8)
    UPPER_BOUNDS = (1, 1)

    def irf(self, ktrans, ve, **kwargs):
        """Get IRF for this model. Overrides superclass method."""
        ktrans_per_s = ktrans / 60.

        # calculate irf for capillary plasma (zeros)
        irf_cp = np.zeros(self.n_upsample, dtype=float)

        # calculate irf for the EES
        irf_e = ktrans_per_s * np.exp(-self.tau_upsample * ktrans_per_s/ve)

        return irf_cp, irf_e
