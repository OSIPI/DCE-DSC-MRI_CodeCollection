"""AIFs.

Created 28 September 2020
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Classes: AIF and derived subclasses:
    PatientSpecific
    ParkerLike
    Parker
    ManningFast
    ManningSlow
    Heye
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import interp1d


class AIF(ABC):
    """Abstract base class for arterial input functions.

    Subclasses correspond to types of AIF, e.g. population-average functions
    and patient-specific AIFs based on input data.
    The main purpose of the AIF class is to return the tracer concentration in
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


class PatientSpecific(AIF):
    """Patient-specific AIF subclass.

    Constructed using time-concentration data, typically obtained from
    experimental measurements. The c_ap method returns AIF
    concentration at any requested time points using interpolation.
    """

    def __init__(self, t_data, c_ap_data):
        """

        Args:
            t_data (ndarray): 1D float array of time points (s) at which
                input AIF concentration data are provided
            c_ap_data (ndarray): 1D float array of input concentration data (
            mM).
        """
        self.t_data = t_data
        self.c_ap_data = c_ap_data
        self.c_ap_func = interp1d(
            t_data,
            c_ap_data,
            kind="quadratic",
            bounds_error=False,
            fill_value=(0, c_ap_data[-1]),
        )

    def c_ap(self, t):
        """Get AIF plasma concentration(t). Overrides superclass method."""
        #  calculate concentration(t) using interpolation function
        c_ap = self.c_ap_func(t)
        return c_ap


class ParkerLike(AIF):
    """Parker-like AIF subclass.

    Generate AIF concentrations using a mathematical function that is based
    on the Parker population-average function but with two exponential terms.
    Parameters default to the original Parker function.
    """

    def __init__(
        self,
        hct,
        a1=0.809,
        a2=0.330,
        t1=0.17046,
        t2=0.365,
        sigma1=0.0563,
        sigma2=0.132,
        s=38.078,
        tau=0.483,
        alpha=0,
        beta=0,
        alpha2=1.050,
        beta2=0.1685,
        scale_factor=1,
        t_start=0,
    ):
        """

        Args:
            hct (float): Arterial haematocrit
            a1, a2, t1, t2, sigma1, sigma2, s, tau, alpha, beta, alpha2,
                beta2 (float): AIF function parameters. Default to original
                Parker function values.
            scale_factor (float): Scale factor applied to AIF curve. Defaults to
                1.
            t_start (float): Start time (s). The AIF function is time-shifted by
                this delay. Defaults to 0.
        """
        self.a1, self.a2, self.t1, self.t2 = a1, a2, t1, t2
        self.sigma1, self.sigma2, self.s, self.tau = sigma1, sigma2, s, tau
        self.alpha, self.alpha2 = alpha, alpha2
        self.beta, self.beta2 = beta, beta2
        self.hct = hct
        self.scale_factor = scale_factor
        self.t_start = t_start

    def c_ap(self, t):
        """Get AIF plasma concentration(t). Overrides superclass method."""
        t_mins = (t - self.t_start) / 60.0

        # calculate c(t) for arterial blood
        c_ab = (
            (self.a1 / (self.sigma1 * np.sqrt(2.0 * np.pi)))
            * np.exp(-((t_mins - self.t1) ** 2) / (2.0 * self.sigma1**2))
            + (self.a2 / (self.sigma2 * np.sqrt(2.0 * np.pi)))
            * np.exp(-((t_mins - self.t2) ** 2) / (2.0 * self.sigma2**2))
            + (
                self.alpha * np.exp(-self.beta * t_mins)
                + self.alpha2 * np.exp(-self.beta2 * t_mins)
            )
            / (1 + np.exp(-self.s * (t_mins - self.tau)))
        )
        c_ab *= self.scale_factor
        c_ap = c_ab / (1 - self.hct)
        c_ap[t < self.t_start] = 0.0
        return c_ap


class Parker(ParkerLike):
    """Parker AIF (subclass of ParkerLike).

    Generate AIF concentrations using Parker population-average function.
    Reference: Parker et al., Magnetic Resonance in Medicine, 2006
    https://doi.org/10.1002/mrm.21066

    """

    def __init__(self, hct, t_start=0):
        """

        Args:
            hct (float): Arterial haematocrit
            t_start (float): Start time (s). The AIF function is time-shifted by
                this delay. Defaults to 0.
        """
        super().__init__(hct, t_start=t_start)


class ManningFast(ParkerLike):
    """AIF function for DCE-MRI with fast injection and long acquisition time.

    TODO: CHECK PARAMETERS
    Based on Parker AIF and modified to reflect measured AIF in a mild-stroke
    population over a longer acquisition time.
    Reference: Manning et al., Magnetic Resonance in Medicine, 2021
    https://doi.org/10.1002/mrm.28833

    """

    def __init__(self, hct=0.42, t_start=0):
        """

        Args:
            hct (float): Arterial haematocrit. Defaults to 0.42
            t_start (float): Start time (s). The AIF function is time-shifted by
                this delay. Defaults to 0.
        """
        super().__init__(
            hct,
            alpha=0.246,
            alpha2=0.765,
            beta=0.180,
            beta2=0.0240,
            scale_factor=0.89,
            t_start=t_start,
        )


class ManningSlow(PatientSpecific):
    """AIF function for DCE-MRI with fast injection and long acquisition time.

    Based on data from a mild-stroke population acquired with a slow injection.
    AIF concentration increases after t=39.62*3 seconds.
    Reference: Manning et al., Magnetic Resonance in Medicine, 2021
    https://doi.org/10.1002/mrm.28833

    """

    def __init__(self):
        # Define a patient-specific AIF based on reference data.
        # We define first time point as t=3*39.62 to ensure c_ap=0 until the
        # end of the 3rd pre-contrast acquisition.
        c_ap_ref = np.array(
            [
                0.000000,
                0.137956,
                0.719692,
                1.634260,
                2.134626,
                1.875262,
                1.757133,
                1.596487,
                1.470386,
                1.352991,
                1.280691,
                1.206125,
                1.146877,
                1.098958,
                1.056410,
                1.024845,
                0.992435,
                0.969435,
                0.944838,
                0.919047,
                0.899973,
                0.880771,
                0.862782,
                0.844603,
                0.829817,
                0.816528,
                0.800179,
                0.781698,
                0.774622,
                0.754376,
            ]
        )

        t_ref = 39.62 * np.array(
            [
                3.0,
                3.5,
                4.5,
                5.5,
                6.5,
                7.5,
                8.5,
                9.5,
                10.5,
                11.5,
                12.5,
                13.5,
                14.5,
                15.5,
                16.5,
                17.5,
                18.5,
                19.5,
                20.5,
                21.5,
                22.5,
                23.5,
                24.5,
                25.5,
                26.5,
                27.5,
                28.5,
                29.5,
                30.5,
                31.5,
            ]
        )

        super().__init__(t_ref, c_ap_ref)


class Heye(ParkerLike):
    """AIF function for DCE-MRI with fast injection and long acquisition time.

    Based on Parker AIF and modified to reflect measured AIF in a mild-stroke
    population over a longer acquisition time.
    Reference: Heye et al., Neuroimage (2016)
    https://doi.org/10.1016/j.neuroimage.2015.10.018

    """

    def __init__(self, hct=0.45, t_start=0):
        """

        Args:
            hct (float): Arterial haematocrit. Defaults to 0.45
            t_start (float): Start time (s). The AIF function is time-shifted by
                this delay. Defaults to 0.
        """
        super().__init__(
            hct,
            alpha=3.1671,
            alpha2=0.5628,
            beta=1.0165,
            beta2=0.0266,
            scale_factor=1,
            t_start=t_start,
        )
