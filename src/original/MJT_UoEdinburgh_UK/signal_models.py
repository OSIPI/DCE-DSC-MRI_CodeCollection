"""Signal models.

Created 28 September 2020
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Classes: SignalModel and derived subclasses:
    SPGR
"""

from abc import ABC, abstractmethod
import numpy as np


class SignalModel(ABC):
    """Abstract base class for signal models.

    Subclasses correspond to specific signal models (e.g. SPGR). The purpose of
    these classes is to estimate the MRI signal as a function of relaxation
    parameters.

    The class attributes are the acquisition parameters and are determined by
    the subclass.

    """

    @abstractmethod
    def R_to_s(self, s0, R1, R2, R2s, k_fa):
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
        k_fa : float
            B1 correction factor, equal to the actual/nominal flip angle.

        Returns
        -------
        s : float
            Signal
        """
        pass


class SPGR(SignalModel):
    """Signal model subclass for spoiled gradient echo pulse sequence."""

    def __init__(self, tr, fa, te):
        """

        Args:
            tr (float): repetition time (s)
            fa (float): flip angle (deg)
            te (float): echo time (s)
        """
        self.tr = tr
        self.fa = fa * np.pi / 180
        self.te = te

    def R_to_s(self, s0, R1, R2=None, R2s=0, k_fa=1):
        """Get signal for this model. Overrides superclass method."""
        fa = k_fa * self.fa
        s = (
            s0
            * (
                ((1.0 - np.exp(-self.tr * R1)) * np.sin(fa))
                / (1.0 - np.exp(-self.tr * R1) * np.cos(fa))
            )
            * np.exp(-self.te * R2s)
        )
        return s
