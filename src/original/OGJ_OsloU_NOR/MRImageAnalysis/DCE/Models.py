from .. import math
import numpy as np


def TM(t, C_a, K_trans, k_ep=None, v_e=None, v_p=None):
    if k_ep is None:
        k_ep = Conversion.k_ep(K_trans=K_trans, v_e=v_e)
    tofts = K_trans * np.exp(-t * k_ep)
    return math.NP.convolve(tofts, C_a, t)


def ETM(t, C_a, K_trans, k_ep=None, v_p=None, v_e=None):
    if k_ep is None:
        k_ep = Conversion.k_ep(K_trans=K_trans, v_e=v_e)
    return TM(t, C_a, K_trans, k_ep) + C_a * v_p


def twoCXM(t, C_a, PS=None, F_p=None, v_e=None, v_p=None, K_trans=None, k_ep=None):
    if 0 in [K_trans, PS] and not 0 in [F_p, v_e, v_p]:
        two_compartment_model = F_p * np.exp(-t * F_p / v_p)
    else:
        if PS is None:
            PS = Conversion.PS(F_p=F_p, K_trans=K_trans)
        E = PS / float(PS + F_p)

        e = v_e / float(v_e + v_p)
        Ee = E * e

        tau_pluss = (
            (E - Ee + e)
            / (2.0 * E)
            * (1 + np.sqrt(1 - 4 * (Ee * (1 - E) * (1 - e)) / (E - Ee + e) ** 2))
        )
        tau_minus = (
            (E - Ee + e)
            / (2.0 * E)
            * (1 - np.sqrt(1 - 4 * (Ee * (1 - E) * (1 - e)) / (E - Ee + e) ** 2))
        )

        F_pluss = F_p * (tau_pluss - 1.0) / (tau_pluss - tau_minus)
        F_minus = -F_p * (tau_minus - 1.0) / (tau_pluss - tau_minus)

        K_pluss = F_p / ((v_p + v_e) * tau_minus)
        K_minus = F_p / ((v_p + v_e) * tau_pluss)

        two_compartment_model = F_pluss * np.exp(-t * K_pluss) + F_minus * np.exp(
            -t * K_minus
        )
    return math.NP.convolve(two_compartment_model, C_a, t)


class Conversion:
    def __init__(self):
        pass

    def raiseError(self):
        raise TypeError("Invalid argument")

    @staticmethod
    def k_ep(K_trans=None, v_e=None, PS=None, F_p=None):
        """
        Needs one of the following combinations of paramaters:
        [K_trans, v_e]
        [PS, F_p, v_e]
        """
        try:
            return K_trans / v_e
        except TypeError:
            pass
        try:
            return PS * F_p / (PS + F_p) / v_e
        except TypeError:
            pass

        # if not None in [K_trans, v_e]:
        # 	return K_trans/v_e
        # if not None in [v_e, PS, F_p]:
        # 	return self.K_trans(PS=PS, F_p=F_p)/v_e

    @staticmethod
    def K_trans(PS=None, F_p=None, k_ep=None, v_e=None):
        """
        Needs one of the following combinations of paramaters:
        [PS, F_p]
        [k_ep, v_e]
        """
        try:
            return PS * F_p / (PS + F_p)
        except TypeError:
            raise TypeError("Invalid argument")
        try:
            return k_ep * v_e
        except TypeError:
            raise TypeError("Invalid argument")
        # if not None in [PS, F_p]:
        # 	return PS*F_p/(PS + F_p)
        # if not None in [k_ep, v_e]:
        # 	return k_ep*v_e

    @staticmethod
    def v_e(K_trans=None, k_ep=None):
        return K_trans / k_ep

    @staticmethod
    def PS(F_p, K_trans):
        return F_p * K_trans / (F_p - K_trans)

    @staticmethod
    def F_p(K_trans, PS):
        return K_trans * PS / (PS - K_trans)
