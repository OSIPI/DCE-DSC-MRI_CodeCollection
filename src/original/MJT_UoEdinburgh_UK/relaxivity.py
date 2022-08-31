"""Relaxivity models.

Created 28 September 2020
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Classes: CRModel abstract class and derived subclasses:
    CRLinear
"""

from abc import ABC, abstractmethod


class CRModel(ABC):
    """Abstract base class for relaxivity models.

    Subclasses correspond to specific relaxivity models (e.g. linear).
    The main purpose of these classes is to convert tracer concentration to
    relaxation rates.

    """

    @abstractmethod
    def R1(self, R10, c):
        """Get post-contrast R1. Method overriden by subclasses."""
        pass

    @abstractmethod
    def R2(self, R20, c):
        """Get post-contrast R2. Method overriden by subclasses."""
        pass


class CRLinear(CRModel):
    """Linear relaxivity subclass.

    Linear relationship between R1/R2 and concentration.
    """

    def __init__(self, r1, r2):
        """

        Args:
        r1 (float): R1 relaxivity (s^-1 mM^-1)
        r2 (float): R2 relaxivity (s^-1 mM^-1)
        """
        self.r1 = r1
        self.r2 = r2

    def R1(self, R10, c):
        """Get post-contrast R1. Overrides superclass method.

        Parameters
        ----------
        R10 : float
            Pre-contrast R1 (s^-1).
        c : float or ndarray
            Concentration value(s) (mM).

        Returns
        -------
        R1 : float or ndarray
            Post-contrast R1 value(s) (s^-1)
        """
        R1 = R10 + self.r1 * c
        return R1

    def R2(self, R20, c):
        """Get post-contrast R2. Overrides superclass method.

        Parameters
        ----------
        R20 : float
            Pre-contrast R2 (s^-1).
        c : float or ndarray
            Concentration value(s) (mM).

        Returns
        -------
        R2 : float or ndarray
            Post-contrast R2 value(s) (s^-1)
        """
        R2 = R20 + self.r2 * c
        return R2
