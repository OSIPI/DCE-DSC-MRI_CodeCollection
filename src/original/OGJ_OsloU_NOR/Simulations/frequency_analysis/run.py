import MRImageAnalysis as mri
from numpy import fft
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
from imports import *

savePath = "../results/frequency_analysis/"


K_trans = 0.07
F_ps = [0.2, 0.4, 0.8]
v_e = 0.2
v_p = 0.02

"""

		Transfer function

"""


w = np.linspace(0, 10, 100000)  # in s^-1
w *= 60 * 2 * np.pi  # in rad/min
TFunc = TransferFunctions(w=w, K_trans=K_trans, F_p=F_ps[0], v_e=v_e, v_p=v_p)

save = np.zeros((len(w), 6))
save[:, 0] = w
save[:, 1] = TFunc.H_TM()
save[:, 2] = TFunc.H_ETM()
for i in range(len(F_ps)):
    TFunc.F_p = F_ps[i]
    save[:, i + 3] = TFunc.H_2CXM()

header = "True values: K_trans = {} ml/100g/min, v_e = {} ml/100g (%), v_p = {} ml/100g (%).".format(
    K_trans * 100, v_e * 100, v_p * 100
)
header += "\nThen the columns are (with F_p (ml/100g/min) indicated):"
header += "\nangular frequency (rad/min), H_TM, H_ETM"
for F_p in F_ps:
    header += ", H_2CXM(F_p = {})".format(F_p * 100)
header += "\nH_* stands for the transfer function of the given model in units of dB (20log|H(omega)|)"
np.savetxt(savePath + "transfer_functions.txt", save, header=header)

# Find the cutoff values and save to file
cutoffs = np.zeros((3, 7))
TFunc.reset()
for i in range(len(F_ps)):
    TFunc.F_p = F_ps[i]
    cutoffs[i, 0] = F_ps[i]
    cutoffs[i, 1] = TFunc.H_TM(cutoff=True)
    cutoffs[i, 2] = TFunc.H_ETM(cutoff=True)[0]
    cutoffs[i, 3] = TFunc.H_ETM(cutoff=True)[1]
    cutoffs[i, 4] = TFunc.H_2CXM(cutoff=True)[0]
    cutoffs[i, 5] = TFunc.H_2CXM(cutoff=True)[1]
    cutoffs[i, 6] = TFunc.H_2CXM(cutoff=True)[2]
header = "True values: K_trans = {} ml/100g/min, v_e = {} ml/100g (%), v_p = {} ml/100g (%).".format(
    K_trans, v_e, v_p
)
header += "\nThen the columns are:"
header += "\nF_p(ml/100g/min), cutoff TM, cutoff ETM, zero ETM, cutoff 1 2CXM, cutoff 2 2CXM, zero 2CXM"
np.savetxt(savePath + "cutoffs.txt", cutoffs, header=header)

"""

	Simulating with different sampling frequencies

"""


# simulate the signal with different dt
dts = np.arange(0.1, 25.1, 0.1) / 60

t0, C_a0 = mri.DCE.AIF.loadStandard()
t0 /= 60

TM_values = np.zeros((len(dts), 7))
ETM_values = np.zeros((len(dts), 10))
twoCXM_values = np.zeros((len(dts), 13))
TM_values[:, 0] = dts * 60
ETM_values[:, 0] = dts * 60
twoCXM_values[:, 0] = dts * 60

for i in range(len(F_ps)):
    # create high res signal
    S0 = mri.DCE.Models.twoCXM(t0, C_a0, K_trans=K_trans, v_p=v_p, v_e=v_e, F_p=F_ps[i])
    for j in range(len(dts)):
        # downsample signal
        t, C_a = mri.math.misc.downSampleAverage(t0, C_a0, dts[j])
        t, S = mri.math.misc.downSampleAverage(t0, S0, dts[j])

        # to ensure zero baseline signal, prepend zero to the arrays, extending the time array
        C_a = np.append(np.array([0]), C_a)
        S = np.append(np.array([0]), S)
        t = np.append(t, np.array([t[-1] + t[1] - t[0]]))

        # Calculate the model fits
        for model in ["TM", "ETM", "twoCXM"]:
            fit = mri.DCE.Analyze.fitToModel(model, S, t, C_a, showPbar=False)

            if model == "TM":
                TM_values[j, i * 2 + 1] = fit.K_trans * 100
                TM_values[j, i * 2 + 2] = fit.v_e * 100
            if model == "ETM":
                ETM_values[j, i * 3 + 1] = fit.K_trans * 100
                ETM_values[j, i * 3 + 2] = fit.v_e * 100
                ETM_values[j, i * 3 + 3] = fit.v_p * 100
            if model == "twoCXM":
                twoCXM_values[j, i * 4 + 1] = fit.K_trans * 100
                twoCXM_values[j, i * 4 + 2] = fit.v_e * 100
                twoCXM_values[j, i * 4 + 3] = fit.v_p * 100
                twoCXM_values[j, i * 4 + 4] = fit.F_p * 100
header = "True values: K_trans = {} ml/100g/min, v_e = {} ml/100g (%), v_p = {} ml/100g (%).".format(
    K_trans, v_e, v_p
)
header += "\nThe columns contain (with true F_p (ml/100g/min) in parentheses):\ndt (s)"
header_TM = header
header_ETM = header
header_twoCXM = header
for i in range(len(F_ps)):
    header_TM += ", K_trans(F_p = {}), v_e(F_p = {})".format(
        F_ps[i] * 100, F_ps[i] * 100
    )
    header_ETM += ", K_trans(F_p = {}), v_e(F_p = {}), v_p(F_p = {})".format(
        F_ps[i] * 100, F_ps[i] * 100, F_ps[i] * 100
    )
    header_twoCXM += (
        ", K_trans(F_p = {}), v_e(F_p = {}), v_p(F_p = {}), F_p(F_p = {})".format(
            F_ps[i] * 100, F_ps[i] * 100, F_ps[i] * 100, F_ps[i] * 100
        )
    )


np.savetxt(savePath + "TM_values.txt", TM_values, header=header_TM)
np.savetxt(savePath + "ETM_values.txt", ETM_values, header=header_ETM)
np.savetxt(savePath + "twoCXM_values.txt", twoCXM_values, header=header_twoCXM)


with open(savePath + "true.json", "w") as outfile:
    json.dump({"K_trans": K_trans * 100, "v_e": v_e * 100, "v_p": v_p * 100}, outfile)
