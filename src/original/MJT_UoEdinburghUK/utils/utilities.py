"""Utilities.

Created 6 October 2021
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Functions:
    minimize_global
"""


import numpy as np
from scipy.optimize import minimize


def minimize_global(cost, x_0_all, **minimizer_kwargs):
    """Find global minimum by calling scipy.optimize.minimize multiple times.

    Parameters
    ----------
    cost : function
        Function to be minimised.
    x_0_all : list
        list of 1D ndarrays. Each contains a set of initial parameter values.
    **minimizer_kwargs : optional keyword arguments accepted by minimize

    Returns
    -------
    result : OptimizeResult
        OptimizeResult corresponding to the fitting attempt with the lowest
        minimum.

    """
    results = [minimize(cost, x_0, **minimizer_kwargs) for x_0 in x_0_all]
    costs = [result.fun if result.success is False else np.nan
             for result in results]
    cost = min(costs)
    idx = costs.index(cost)
    result = results[idx]
    return result
