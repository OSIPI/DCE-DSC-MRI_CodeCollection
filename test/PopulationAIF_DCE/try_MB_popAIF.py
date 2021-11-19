import src.original.OG_MO_AUMC_ICR_RMH.ExtendedTofts.DCE as DCE
import matplotlib.pyplot as plt
import numpy as np

aif_par = DCE.aifPopPMB(0)
print(aif_par)
time = np.arange(0, 5, 4.97/60)
aif = DCE.fit_aif(aif_par, time)

plt.plot(time, fit_curve)
plt.plot(time, Caif, marker='.', linestyle='')
plt.show()