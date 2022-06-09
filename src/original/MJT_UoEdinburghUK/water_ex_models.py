"""Water exchange models.

Created 28 September 2020
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Classes: WaterExModel and derived subclasses:
    FXL
    NXL
    NTEXL
"""

from abc import ABC, abstractmethod


class WaterExModel(ABC):
    """Abstract base class for water exchange models.

    Subclasses correspond to specific models (e.g. fast-exchange limit). The
    main purpose of these classes is to estimate the exponential T1 relaxation
    components for the tissue, dependent on the T1 relaxation rates in each
    tissue compartment (blood, EES and intracellular). For example,
    in the fast-exchange limit model, the result is a single T1 component,
    while in the slow exchange limit, the result is 3 T1 components.
    """

    @abstractmethod
    def R1_components(self, p, R1):
        """get the R1 relaxation rates and populations for T1 components.

        Parameters
        ----------
        p : dict
            Spin population fraction for each tissue compartment.
            Example: p = {'b': 0.1, 'e': 0.4, 'i': 0.5}
        R1 : dict
            R1 relaxation rate (s^-1) for each tissue compartment.
            Example: R1 = {'b': 0.6, 'e': 1.0, 'i': 1.0}

        Returns
        -------
        R1 components : list
            List of floats, corresponding to the R1 of each exponential
            relaxation component. The number of items depends on the water
            exchange model.
        p_components: list
            List of floats, corresponding to the spin population fractions of
            each exponential relaxation component.
        """
        pass


class FXL(WaterExModel):
    """Fast water exchange model.

    Water exchange between all compartments is in the fast limit.
    """

    def R1_components(self, p, R1):
        """Get R1 components for this model. Overrides superclass method."""
        R1 = p['b']*R1['b'] + p['e']*R1['e'] + p['i']*R1['i']
        R1_components = [R1]
        p_components = [1.]
        return R1_components, p_components


class NXL(WaterExModel):
    """No-exchange limit water exchange model.

    Water exchange between all compartments is in the slow limit.
    """

    def R1_components(self, p, R1):
        """Get R1 components for this model. Overrides superclass method."""
        R1_components = [R1['b'], R1['e'], R1['i']]
        p_components = [p['b'], p['e'], p['i']]
        return R1_components, p_components


class NTEXL(WaterExModel):
    """No-transendothelial water exchange limit model.

    Water exchange between blood and EES compartments is in the slow limit.
    The EES and intracellular compartments are in the fast-exchange limit and
    behave as a single compartment.
    """

    def R1_components(self, p, R1):
        """Get R1 components for this model. Overrides superclass method."""
        p_ev = p['e'] + p['i']
        R1_ev = (p['e']*R1['e'] + p['i']*R1['i']) / p_ev

        R1_components = [R1['b'], R1_ev]
        p_components = [p['b'], p_ev]
        return R1_components, p_components
