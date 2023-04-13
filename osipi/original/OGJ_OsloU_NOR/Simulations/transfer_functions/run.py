from imports import *

K_trans_values = [0.03, 0.07, 0.1, 0.13]
v_e_values = [0.05, 0.1, 0.2, 0.3]
v_p_values = [0.01, 0.015, 0.02, 0.03]
F_p_values = [0.2, 0.4, 0.8, 1]

w = np.linspace(0, 10, 100000)
w *= 60 * 2 * np.pi
TFunc = TransferFunctions(w, K_trans=0.07, v_e=0.2, v_p=0.02, F_p=0.2)

save = np.zeros((1 + len(w), 17))
save[1:, 0] = w / 60 / 2 / np.pi

TFunc.reset()
i = -1
for K_trans in K_trans_values:
    i += 1
    TFunc.K_trans = K_trans

    save[0, i + 1] = TFunc.H_2CXM(cutoff=True)[0] / 60
    save[1:, i + 1] = TFunc.H_2CXM()

TFunc.reset()
i = -1
for v_e in v_e_values:
    i += 1
    TFunc.v_e = v_e

    save[0, i + 5] = TFunc.H_2CXM(cutoff=True)[0] / 60
    save[1:, i + 5] = TFunc.H_2CXM()

TFunc.reset()
i = -1
for v_p in v_p_values:
    i += 1
    TFunc.v_p = v_p

    save[0, i + 9] = TFunc.H_2CXM(cutoff=True)[0] / 60
    save[1:, i + 9] = TFunc.H_2CXM()

TFunc.reset()
i = -1
for F_p in F_p_values:
    i += 1
    TFunc.F_p = F_p
    save[0, i + 13] = TFunc.H_2CXM(cutoff=True)[0] / 60
    save[1:, i + 13] = TFunc.H_2CXM()


header = (
    "The first row is the cutoffs. Then the columns contain the transfer functions.\n"
)
header += "freq. (Hz), "
for K_trans in K_trans_values:
    header += "TFunc (K_trans={}), ".format(K_trans * 100)
for v_e in v_e_values:
    header += "TFunc (v_e={}), ".format(v_e * 100)
for v_p in v_p_values:
    header += "TFunc (v_p={}), ".format(v_p * 100)
for F_p in F_p_values:
    header += "TFunc (F_p={}), ".format(F_p * 100)

np.savetxt("../results/transfer_functions/data.txt", save, header=header)
