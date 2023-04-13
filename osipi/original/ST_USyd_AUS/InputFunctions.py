#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 13:19:48 2018

@author: Sirisha Tadimalla
"""
# Libraries
import math
import numpy as np


# Shifts array to the right by n elements
# and inserts n zeros at the beginning of the array
def arr_shift(A, n):
    shift = np.zeros(n)
    A_shifted = np.insert(A, 0, shift)
    A_new = A_shifted[0 : len(A)]
    return A_new


# Population AIF from Parker MRM 2006
def AIF(t0, t):
    # parameter values defined in table 1 (Parker MRM 2006)
    A1 = 0.809
    A2 = 0.330
    T1 = 0.17046
    T2 = 0.365
    sigma1 = 0.0563
    sigma2 = 0.132
    alpha = 1.050
    beta = 0.1685
    s = 38.078
    tau = 0.483

    # Eq. 1 (Parker 2006)

    Ca = [
        (A1 / sigma1)
        * (1 / math.sqrt(2 * math.pi))
        * math.exp(-((i / 60 - T1) ** 2) / (2 * sigma1**2))
        + (A2 / sigma2)
        * (1 / math.sqrt(2 * math.pi))
        * math.exp(-((i / 60 - T2) ** 2) / (2 * sigma2**2))
        + alpha * math.exp(-beta * i / 60) / (1 + math.exp(-s * (i / 60 - tau)))
        for i in t
    ]

    # baseline shift
    Ca = arr_shift(Ca, int(t0 / t[1]) - 1)

    return Ca


# Population AIF from Parker MRM 2006 - modified for a longer injection time
def variableAIF(inj_time, t, t0):
    # Standard AIF (Parker MRM 2006)
    # Injection rate of 3ml/s of a dose of 0.1mmol/kg of CA of concentration 0.5mmol/ml
    # Assuming a standard body weight of 70kg, the injection time comes to
    I = 70 * (1 / 5) * (1 / 3)  # seconds
    Ca = AIF(t0, t)  # standard AIF

    # Number of times the standard AIF must be shifted by I to match the required injection time
    n = int(round(inj_time / I))

    # Calculate AIF for each n
    shift = int(I / t[1])
    Ca_sup = np.zeros(shape=(n + 1, len(Ca)))
    Ca_sup[0] = Ca
    for i in range(1, n + 1):
        Ca_sup[i] = arr_shift(Ca, shift * i)

    Ca_new = (1 / n) * np.sum(Ca_sup, axis=0)

    inj_time = I * n  # Calculate actual injection time

    return Ca_new


# Population AIF for a preclinical case - from McGrath MRM 2009
def preclinicalAIF(t0, t):
    # Model B - parameter values defined in table 1 (McGrath MRM 2009)
    A1 = 3.4
    A2 = 1.81
    k1 = 0.045
    k2 = 0.0015
    t1 = 7

    # Eq. 5 (McGrath MRM 2009)
    Ca = [
        A1 * (i / t1) + A2 * (i / t1)
        if i <= t1
        else A1 * np.exp(-k1 * (i - t1)) + A2 * np.exp(-k2 * (i - t1))
        for i in t
    ]

    # baseline shift
    Ca = arr_shift(Ca, int(t0 / t[1]) - 1)

    return Ca
