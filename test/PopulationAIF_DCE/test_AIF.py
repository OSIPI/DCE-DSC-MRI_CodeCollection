import os
import pytest
import numpy as np
import math
import matplotlib.pyplot as pyplot

from src.original.PvH_NKI_NL.AIF.PopulationAIF import ParkerAIF, GeorgiouAIF
#from PopulationAIF import ParkerAIF, GeorgiouAIF


dt = 1 #temp resolution in s
endt = 280 # end time in s
notimepoints = endt/dt
time = np.arange(0,endt+dt, dt)

AIF_G=GeorgiouAIF(time)
pyplot.figure(1)    
pyplot.plot(time.tolist(), AIF_G.tolist(), 'r')
filename = os.path.join(os.path.dirname(__file__), 'data', 'GeorgiouAIF_MRM2018.txt')  
timeaif, aifdata = np.loadtxt(filename, delimiter='\t', unpack=True)
timeaif=timeaif*60  # convert time to seconds

# interpolate to a given time series
last_timeaif = timeaif[len(timeaif)-1]
last_time = time[len(time)-1]
if last_time <= last_timeaif:
        # interpolate to the right time series
    GeorgiouAIF_ref = np.interp(time, timeaif, aifdata)

pyplot.plot(time.tolist(), GeorgiouAIF_ref.tolist(), 'b')

# set the tolerance to use for this dataset
a_tol = 0.05
r_tol = 0.05

def test_Georgiou_AIF(t, ref, atol, rtol):
       
    # prepare input data
    AIF_G=GeorgiouAIF(t)
    np.testing.assert_allclose( [AIF_G], [ref], rtol=rtol, atol=atol )
    
test_Georgiou_AIF(time, GeorgiouAIF_ref, a_tol, r_tol)    

#np.testing.assert_allclose( [AIF_G], [GeorgiouAIF_ref], rtol=a_tol, atol=r_tol,err_msg='difference outside tolerance', verbose=True)


