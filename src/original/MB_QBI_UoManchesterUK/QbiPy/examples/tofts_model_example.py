'''
Simple interactive script to demonstrate using QbiPy's tofts_model module

The toft_model (like all QbiPy TK models) uses an Aif object from the dce_aif module.

This object can be used to either:
- generate a Parker AIF that will be auto recomputed at a given delay time
- use a base AIF (eg from an existing array like object) that will be linearly interpolated
- load an AIF from a text file (see module for required format) that will be linearly resampled
'''
#%%
%load_ext autoreload
%autoreload 2
#%%
import matplotlib.pyplot as plt
import numpy as np

#We need the tofts_model and dce_aif modules
from QbiPy.dce_models import tofts_model, dce_aif

#%%
#Create times, from 0 to 5 mins, 5s temp resolution
n_t = 61
t = np.linspace(0, 5, n_t)
aif = dce_aif.Aif(times=t, hct=0.42)
# %%
# Set up some sample parameters
Ktrans = 0.25
v_e = 0.2
v_p = 0.1
tau_a = 0.05

#Compute C(t) using the full model function
Ct1 = tofts_model.concentration_from_model(
  aif, Ktrans, v_e, v_p, tau_a
)[0,]

#Compute C(t) using the wrapper for single voxels. This simplifies the input arguments, skips dimension checks on the inputs and is thus suitable to be used in NLLS optimisers
Ct2 = tofts_model.concentration_from_model_single(
  [Ktrans, v_e, v_p, tau_a], aif
)

#Check the two functions produce the same C(t)
plt.figure()
plt.plot(t, Ct1)
plt.plot(t, Ct2, '--')
plt.legend(['From full Ct function', 'From single wrapper function'])
plt.xlabel('time (mins)')
plt.ylabel('C(t)')
# %%
# An alternative way to set up the Aif is use an existing array, we show this below

#First get the AIF as a 1D array from the previously created object
Ca_t = aif.resample_AIF(0)[0,]

#Now create a new aif object, using the Aif type ARRAY setting
aif2 = dce_aif.Aif(times = t, base_aif=Ca_t, aif_type=dce_aif.AifType.ARRAY)

#Check the new AIF object produces the same C(t)
Ct2 = tofts_model.concentration_from_model_single(
  [Ktrans, v_e, v_p, tau_a], aif
)

plt.figure()
plt.plot(t, Ct1)
plt.plot(t, Ct2, '--')
plt.legend(['From Parker AIF', 'From base array AIF'])
plt.xlabel('time (mins)')
plt.ylabel('C(t)')
# %%
# The module does not implement an NLLS fit, however LLS fitting is included, although this requires knowing the delay time tau. Below we cheat and use the known delay.

# Add some low variance Gaussian noise to our C(t)
Ct_n = Ct1 + np.random.randn(n_t) / 1000

Ktrans_lls, v_e_lls, v_p_lls = tofts_model.solve_LLS(
  Ct_n, aif, tau_a
)

#Check we recover the expected params
print(f'Ktrans: ({Ktrans},{Ktrans_lls:5.4f})')
print(f'v_e: ({v_e},{v_e_lls:5.4f})')
print(f'v_p: ({v_p},{v_p_lls:5.4f})')

# %%
