import sys, os
import MRImageAnalysis as mri
import matplotlib.pyplot as plt
import numpy as np

savePath = "../results/bat/"
K_trans = 0.07
v_p = 0.02
v_e = 0.2
F_ps = [0.2, 0.4, 0.8]


def f(t, i):
    """
    i is how many dts to move right
    """
    ret = np.zeros(len(t))
    ret[i] = 1
    return ret


t0, C_a0 = mri.DCE.AIF.loadStandard()
dt = t0[1] - t0[0]

# remove the baseline in the AIF
c = 0
for i in (C_a0 > 0) * 1:
    if i == 1:
        first = c
        break
    c += 1
original_BAT = t0[c]
C_a = np.zeros(len(C_a0))
C_a[:-c] = C_a0[c:]
C_a[-c:] = C_a0[-1]


BATs = np.arange(original_BAT - 4, original_BAT + 4, 0.1)

C_a_new = np.zeros((len(BATs), len(C_a)))
for i in range(len(BATs)):
    C_a_new[i] = np.convolve(C_a, f(t0, int(BATs[i] / dt)))[: len(t0)]

TM_values = np.zeros((len(BATs), 1 + 2 * len(F_ps)))
ETM_values = np.zeros((len(BATs), 1 + 3 * len(F_ps)))
twoCXM_values = np.zeros((len(BATs), 1 + 4 * len(F_ps)))

TM_values[:, 0] = BATs - original_BAT
ETM_values[:, 0] = BATs - original_BAT
twoCXM_values[:, 0] = BATs - original_BAT

t0 /= 60  # time in minutes
for i in range(len(F_ps)):
    # create a signal with appropriate values
    S0 = mri.DCE.Models.twoCXM(t0, C_a0, K_trans=K_trans, v_p=v_p, v_e=v_e, F_p=F_ps[i])

    # now downsample the signal so that we get a better dt
    dt = 2 / 60.0
    t, S = mri.math.misc.downSampleAverage(t0, S0, dt)

    # compute the model fit using different AIFs with different BATs
    for j in range(len(BATs)):
        _, C_a = mri.math.misc.downSampleAverage(t0, C_a_new[j], dt)
        for model in ["TM", "ETM", "2CXM"]:
            fit = mri.DCE.Analyze.fitToModel(model, S, t, C_a, showPbar=False)
            if model == "TM":
                TM_values[j, i * 2 + 1] = fit.K_trans
                TM_values[j, i * 2 + 2] = fit.v_e
            if model == "ETM":
                ETM_values[j, i * 3 + 1] = fit.K_trans
                ETM_values[j, i * 3 + 2] = fit.v_e
                ETM_values[j, i * 3 + 3] = fit.v_p
            if model == "2CXM":
                twoCXM_values[j, i * 4 + 1] = fit.K_trans
                twoCXM_values[j, i * 4 + 2] = fit.v_e
                twoCXM_values[j, i * 4 + 3] = fit.v_p
                twoCXM_values[j, i * 4 + 4] = fit.F_p


header = "True values are: K_trans = {}, v_e = {}, v_p = {}.".format(K_trans, v_e, v_p)
header += "\nThe columns are (bat=bolus arrival time):"
header += "\nChange in bat (s)"
header_TM = header
header_ETM = header
header_2CXM = header
for F_p in F_ps:
    header_TM += ", K_trans(F_p = {}), v_e(F_p = {})".format(F_p, F_p)
    header_ETM += ", K_trans(F_p = {}), v_e(F_p = {}), v_p(F_p = {})".format(
        F_p, F_p, F_p
    )
    header_2CXM += (
        ", K_trans(F_p = {}), v_e(F_p = {}), v_p(F_p = {}), F_p(F_p = {})".format(
            F_p, F_p, F_p, F_p
        )
    )

np.savetxt(savePath + "TM_values.txt", TM_values, header=header_TM)
np.savetxt(savePath + "ETM_values.txt", ETM_values, header=header_ETM)
np.savetxt(savePath + "2CXM_values.txt", twoCXM_values, header=header_2CXM)
