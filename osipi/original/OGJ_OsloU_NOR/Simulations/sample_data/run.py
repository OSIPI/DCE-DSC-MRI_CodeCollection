import MRImageAnalysis as mri
import matplotlib.pyplot as plt
import numpy as np
import sys

K_trans = 7 / 100
v_e = 20 / 100
v_p = 2 / 100
F_p = 20 / 100

t0, C_a = mri.DCE.AIF.loadStandard()
t0 /= 60
C_t = mri.DCE.Models.twoCXM(t0, C_a, K_trans=K_trans, v_p=v_p, v_e=v_e, F_p=F_p)

t, C_a = mri.math.misc.downSampleAverage(t0, C_a, 4 / 60)
t, C_t = mri.math.misc.downSampleAverage(t0, C_t, 4 / 60)

C_t_noisy = np.zeros(len(C_t))

SNR = 30
for i in range(len(C_t)):
    sigma = C_t[i] / SNR
    C_t_noisy[i] = C_t[i] + np.random.normal(0, sigma)

data_save_path = "../Report/Figures/data/sample_data/"

save = np.zeros((len(C_t), 4))
save[:, 0] = t
save[:, 1] = C_a
save[:, 2] = C_t
save[:, 3] = C_t_noisy

header = "The columns are:\n"
header += "Time (min), C_a, C_t, C_t_noisy\n"
header += "The input parameters were K_trans={} ml/100g/min, v_e={} ml/100g, v_p={} ml/100g, F_p={} ml/100g/min".format(
    K_trans * 100, v_e * 100, v_p * 100, F_p * 100
)

np.savetxt(data_save_path + "data.txt", save, header=header)
