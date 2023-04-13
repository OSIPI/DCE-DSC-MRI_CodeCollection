## AUTHOR: Lucy Kershaw, University of Edinburgh
## Purpose: Module to calculate uptake curves using various different models

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz


# Inputs:
# params	numpy array of model parameters, explained for each individual model
# t 		numpy array of time points in seconds
# AIF		numpy array containing the PLASMA AIF as delta R1 in s^-1 (i.e. no relaxivity usually involved but it doesn't matter because it cancels - just be consistent)
# toff		offset time in seconds between the AIF and the uptake curve, default is zero

# Output:
# G			numpy array of model curve as delta R1 in s^-1 unless the AIF included relaxivity

#####################
# Palak model
#
# params: 	[Ktrans, vp]
# Units:	Ktrans	ml (ml tissue)^-1 s^-1
# 			vp		no units (value is between 0 and 1)
#####################


def Patlak(params, t, AIF, toff=0):
    # Assign parameter names
    Ktrans, vp = params

    # Shift the AIF by the toff (if not zero)
    if toff != 0:
        f = interp1d(t, AIF, kind="linear", bounds_error=False, fill_value=0)
        AIF = (t > toff) * f(t - toff)

    # Use trapezoidal integration for the AIF
    CpIntegral = Ktrans * cumtrapz(AIF, x=t, initial=0)

    # Add Cpvp term
    G = CpIntegral + (AIF * vp)

    return G


#####################
# Kety (Tofts) model
#
# params: 	[Ktrans, ve]
# Units:	Ktrans	ml (ml tissue)^-1 s^-1
# 			ve		no units (value is between 0 and 1)
#####################


def Kety(params, t, AIF, toff=0):
    # Assign parameter names
    Ktrans, ve = params

    # Shift the AIF by the toff (if not zero)
    if toff != 0:
        f = interp1d(t, AIF, kind="linear", bounds_error=False, fill_value=0)
        AIF = (t > toff) * f(t - toff)

    # Calculate the impulse response function
    imp = Ktrans * np.exp(-1 * Ktrans * t / ve)

    # Convolve impulse response with AIF
    convolution = np.convolve(AIF, imp)

    # Discard unwanted points and make sure timespacing is correct
    G = convolution[0 : len(t)] * t[1]

    return G


#####################
# Extended Kety (Tofts) model
#
# params: 	[Ktrans, ve, vp]
# Units:	Ktrans	ml (ml tissue)^-1 s^-1
# 			ve		no units (value is between 0 and 1)
# 			vp		no units (value is between 0 and 1)
#####################


def ExtKety(params, t, AIF, toff=0):
    # Assign parameter names
    Ktrans, ve, vp = params

    # Shift the AIF by the toff (if not zero)
    if toff != 0:
        f = interp1d(t, AIF, kind="linear", bounds_error=False, fill_value=0)
        AIF = (t > toff) * f(t - toff)

    # Calculate the impulse response function
    imp = Ktrans * np.exp(-1 * Ktrans * t / ve)

    # Convolve impulse response with AIF
    convolution = np.convolve(AIF, imp)

    # Discard unwanted points, make sure timespacing is correct and add Cpvp term
    G = convolution[0 : len(t)] * t[1] + (vp * AIF)

    return G


#####################
# Two compartment uptake model
#
# params: 	[E, Fp, vp]
# Units:	E 		no units (value is between 0 and 1)
# 			Fp		ml (ml tissue)^-1 s^-1
# 			vp		no units (value is between 0 and 1)
#####################


def TwoCUM(params, t, AIF, toff=0):
    # Assign parameter names
    E, Fp, vp = params

    # Shift the AIF by the toff (if not zero)
    if toff != 0:
        f = interp1d(t, AIF, kind="linear", bounds_error=False, fill_value=0)
        AIF = (t > toff) * f(t - toff)

    # First calculate the parameter Tp
    Tp = (vp / Fp) * (1 - E)

    # Calculate the impulse response function
    exptTp = np.exp(-1 * t / Tp)
    imp = exptTp * (1 - E) + E

    # Convolve impulse response with AIF and make sure time spacing is correct
    convolution = np.convolve(AIF, imp) * t[1]

    # Discard unwanted points and multiply by Fp
    G = Fp * convolution[0 : len(t)]

    return G


#####################
# Two compartment exchange model
#
# params: 	[E, Fp, vp]
# Units:	E 		no units (value is between 0 and 1)
# 			Fp		ml (ml tissue)^-1 s^-1
# 			ve		no units (value is between 0 and 1)
# 			vp		no units (value is between 0 and 1)
#####################


def TwoCXM(params, t, AIF, toff=0):
    # Assign parameter names
    E, Fp, ve, vp = params

    # Shift the AIF by the toff (if not zero)
    if toff != 0:
        f = interp1d(t, AIF, kind="linear", bounds_error=False, fill_value=0)
        AIF = (t > toff) * f(t - toff)

    # First calculate the parameters TB, TE and Tp
    Tp = (vp / Fp) * (1 - E)
    TE = ve * (1 - E) / (E * Fp)
    TB = vp / Fp

    # And then the impulse response function parameters A, Kplus, Kminus
    Kplus = 0.5 * (
        (1 / Tp) + (1 / TE) + np.sqrt(((1 / Tp) + (1 / TE)) ** 2 - (4 / (TE * TB)))
    )
    Kminus = 0.5 * (
        (1 / Tp) + (1 / TE) - np.sqrt(((1 / Tp) + (1 / TE)) ** 2 - (4 / (TE * TB)))
    )
    A = (Kplus - (1 / TB)) / (Kplus - Kminus)

    # Calculate the impulse response function
    expKplus = np.exp(-1 * t * Kplus)
    expKminus = np.exp(-1 * t * Kminus)

    imp = expKplus + A * (expKminus - expKplus)
    # Calculate the convolution and make sure time spacing is correct
    convolution = np.convolve(AIF, imp) * t[1]

    # Discard unwanted points and multiply by Fp
    G = Fp * convolution[0 : len(t)]

    return G


#####################
# Adiabatic approximation to the Tissue Homogeneity (AATH) model
#
# params: 	[E, Fp, vp]
# Units:	E 		no units (value is between 0 and 1)
# 			Fp		ml (ml tissue)^-1 s^-1
# 			ve		no units (value is between 0 and 1)
# 			vp		no units (value is between 0 and 1)
#####################


def AATH(params, t, AIF, toff=0):
    # Assign parameter names
    E, Fp, ve, vp = params
    Tc = vp / Fp

    # Shift the AIF by the toff (if not zero)
    if toff != 0:
        f = interp1d(t, AIF, kind="linear", bounds_error=False, fill_value=0)
        AIF = (t > toff) * f(t - toff)

    # Calculate the impulse response function
    imp = E * Fp * np.exp(-1 * E * Fp * (t - Tc) / ve)
    if np.round(Tc / t[1]) != 0:
        imp[0 : (round(Tc / t[1]))] = Fp

    # Calculate the convolution and make sure time spacing is correct
    convolution = np.convolve(AIF, imp) * t[1]

    # Discard unwanted points
    G = convolution[0 : len(t)]

    return G
