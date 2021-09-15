"""Signal models.

Created 28 September 2020
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Classes: signal_model and derived subclasses:
    spgr
"""

from abc import ABC, abstractmethod
import numpy as np


class signal_model(ABC):
    """Abstract base class for signal models.

    Subclasses correspond to specific signal models (e.g. SPGR). The purpose of
    these classes is to estimate the MRI signal as a function of relaxation
    parameters.

    The class attributes are the acquisition parameters and are determined by
    the subclass.

    Methods
    -------
    R_to_s(s0, R1, R2, R2s, k): get the signal
    """

    @abstractmethod
    def R_to_s(self, s0, R1, R2, R2s, k):
        """Convert relaxation parameters to signal.

        Parameters
        ----------
        s0 : float
            Equilibrium signal.
        R1 : float
            R1 relaxation rate (s^-1).
        R2 : float
            R2 relaxation rate (s^-1).
        R2s : float
            R2* signal dephasing rate (s^-1).
        k : float
            B1 correction factor, equal to the actual/nominal flip angle.

        Returns
        -------
        s : float
            Signal
        """
        pass


class spgr(signal_model):
    """Signal model subclass for spoiled gradient echo pulse sequence.

    Attributes
    ----------
    tr : float
        repetition time (s)
    fa_rad : float
        flip angle (rad)
    te : float
        echo time (s)
    """

    def __init__(self, tr, fa_rad, te):
        self.tr = tr
        self.fa_rad = fa_rad
        self.te = te

    def R_to_s(self, s0, R1, R2=None, R2s=0, k=1.):
        """Get signal for this model. Overrides superclass method."""
        fa = k * self.fa_rad
        s = s0 * (((1.0-np.exp(-self.tr*R1))*np.sin(fa)) /
                  (1.0-np.exp(-self.tr*R1)*np.cos(fa))
                  ) * np.exp(-self.te*R2s)
        return s
