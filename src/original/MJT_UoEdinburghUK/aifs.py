"""AIFs.

Created 28 September 2020
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Classes: aif and derived subclasses:
    patient_specific
    parker_like
    parker
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import interp1d


class aif(ABC):
    """Abstract base class for arterial input functions.

    Subclasses correspond to types of AIF, e.g. population-average functions
    and patient-specific AIFs based on input data.
    The main purpose of the aif class is to return the tracer concentration in
    arterial plasma at any time points.

    Methods
    -------
    c_ap(t) : get the tracer concentration in arterial plasma at time(s) t (s)
    """

    @abstractmethod
    def c_ap(self, t):
        """Get the tracer concentration in arterial plasma at arbitrary times.

        Parameters
        ----------
        t : ndarray
            1D array of floats containing times (s) at which to calculate AIF
            concentration.

        Returns
        -------
        c_ap : ndarray
            1D array of floats containing tracer concentrations (mM) in AIF
            blood plasma at times t.
        """
        pass


class patient_specific(aif):
    """Patient-specific AIF subclass.

    Constructed using time-concentration data, typically obtained from
    experimental measurements. The c_ap method returns AIF
    concentration at any requested time points using interpolation.

    Attributes
    ----------
    t_data : ndarray
        1D float array of time points (s) at which AIF concentration data are
        provided
    c_ap_data : ndarray
        1D float array of concentration data (mM)
    c_ap_func : interp1d
        interpolation function to generate AIF concentration
    """

    def __init__(self, t_data, c_ap_data):
        self.t_data = t_data
        self.c_ap_data = c_ap_data
        self.c_ap_func = interp1d(t_data, c_ap_data,
                                  kind='quadratic', bounds_error=False,
                                  fill_value=(0, c_ap_data[-1]))

    def c_ap(self, t):
        """Get AIF plasma concentration(t). Overrides superclass method."""
        #  calculate concentration(t) using interpolation function
        c_ap = self.c_ap_func(t)
        return c_ap


class parker_like(aif):
    """Parker-like AIF subclass.

    Generate AIF concentrations using a mathematical function that is based
    on the Parker population-average function but with two exponential terms.
    Parameters default to the original Parker function.

    Attributes
    ----------
    hct : float
        Arterial haematocrit
    a1, a2, t1, t2, sigma1, sigma2, s, tau, alpha, beta, alpha2, beta2 : float
        AIF function parameters
    t_start : float
        Start time (s). The AIF function is time-shifted by this delay.
    """

    def __init__(self, hct, a1=0.809, a2=0.330, t1=0.17046, t2=0.365,
                 sigma1=0.0563, sigma2=0.132, s=38.078, tau=0.483,
                 alpha=0, beta=0, alpha2=1.050, beta2=0.1685, t_start=0):
        self.a1, self.a2, self.t1, self.t2 = a1, a2, t1, t2
        self.sigma1, self.sigma2, self.s, self.tau = sigma1, sigma2, s, tau
        self.alpha, self.alpha2 = alpha, alpha2
        self.beta, self.beta2 = beta, beta2
        self.hct = hct
        self.t_start = t_start

    def c_ap(self, t):
        """Get AIF plasma concentration(t). Overrides superclass method."""
        t_mins = (t - self.t_start) / 60.

        # calculate c(t) for arterial blood
        c_ab = (self.a1/(self.sigma1*np.sqrt(2.*np.pi))) * \
            np.exp(-((t_mins-self.t1)**2)/(2.*self.sigma1**2)) + \
            (self.a2/(self.sigma2*np.sqrt(2.*np.pi))) * \
            np.exp(-((t_mins-self.t2)**2)/(2.*self.sigma2**2)) + \
            (self.alpha*np.exp(-self.beta*t_mins) +
             self.alpha2*np.exp(-self.beta2*t_mins)) / \
            (1+np.exp(-self.s*(t_mins-self.tau)))

        c_ap = c_ab / (1 - self.hct)
        c_ap[t < self.t_start] = 0.
        return c_ap


class parker(parker_like):
    """Parker AIF (subclass of parker_like).

    Generate AIF concentrations using Parker population-average function.

    Attributes
    ----------
    hct : float
        Arterial haematocrit
    a1, a2, t1, t2, sigma1, sigma2, s, tau, alpha, beta, alpha2, beta2 : float
        AIF function parameters
    t_start : float
        Start time (s). The AIF function is time-shifted by this delay.
    """

    def __init__(self, hct, t_start=0):
        super().__init__(hct, t_start=t_start)
