## AUTHOR: Lucy Kershaw, University of Edinburgh
## Purpose: Module to estimate bolus arrival time, two methods. 
##			BEWARE neither have been successful on all datasets! Test carefully.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d


#####################
# BAT by fitting Kety (Tofts) model to first third of the curve
#
# params: 	t           array of time points
#           AIF         Arterial input function
#           Conc_curve  Concentration curve
#           startguess  Starting guess for model parameters Ktrans, ve, toff
#
# Units:	t           seconds
#           AIF         delta R1 in s^-1
#           SI_curve    delta R1 in s^-1
#           startguess  [Ktrans in ml(ml tissue)^-1 s^-1, ve no units, toff in s]
#
# Output:   toff in s NOTE: This is the time between the rise of the AIF and the rise of the concentration curve 
#####################

def BAT_Kety(t,AIF,Conc_curve):

    #Set starting guess for the optimisation
    startguess=[0.01,0.1,t[7]]

    #Calculate the number of timepoints corresponding to the first third of the curve
    firstthird=int(np.round(len(t)/3))

    #Set sensible bounds for the Kety model parameters
    Ketybnds=((0.00001,10),(0.00001,2),(0.00001,t[30]))

    #Fit the model
    Ketyresult=minimize(Ketyobjfun,startguess,args=(t[0:firstthird],AIF[0:firstthird],Conc_curve[0:firstthird]),bounds=Ketybnds,method='SLSQP',options={'disp':False})
    
    # Set a default value for the toff as zero, check that optimisation was successful and then return the fitted toff.
    toff=0
    if not np.isnan(Ketyresult.x[2]):
        toff=Ketyresult.x[2]
    return Ketyresult


# Helper function - chi for data vs Kety model with toff as a free parameter
def Ketyobjfun(params,t,AIF, Conc_curve):
    # Assign parameter names
    Ktrans, ve, toff = params

    # Shift the AIF by the amount toff
    tnew = t - toff
    f=interp1d(t,AIF,kind='linear',bounds_error=False,fill_value=0)
    AIFnew = (t>toff)*f(t-toff)

    imp=Ktrans*np.exp(-1*Ktrans*t/ve); # Calculate the impulse response function
    convolution=np.convolve(AIFnew,imp) # Convolve impulse response with AIF
    G=convolution[0:len(t)]*t[1]
    
    chi2=np.sum((Conc_curve-G)**2)

    return chi2

####################
# BAT by fitting a piecewise linear quadratic function to the data as in Cheong, L.H., Koh, T.S., Hou, Z., 2003. An automatic approach for estimating bolus arrival 
#           time in dynamic contrast MRI using piecewise continuous regression models. Phys Med Biol 48, N83-8-N83-8.
#
# params:   t           array of time points
#           SI_curve    Signal intensity curve
#           toffmax     Maximum possible toff index to try.  For the AIF use the default of None to calculate
#                       for points up to the peak in the curve.   For uptake curves, tweak according to your data.
#
# Units:    t           seconds
#           SI_curve    delta R1 in s^-1
#
# Output:   toff in s NOTE: This is the time since the *beginning of the curve* that it starts to rise. 
#           To calculate the time offset between the AIF and the Concentration curve you'll need to use 
#           this on both the AIF and the Concentration curve and subtract one rise time from the other
#####################


def BAT_linquad(t, Conc_curve, toffmax=None):

    if toffmax==None:
        toffmax=np.argmax(Conc_curve)
    chi2s=np.zeros(toffmax)

    #For each time point k up to the maximum toffmax, fit a linear-quadratic model
    for k in range(toffmax):
        fit=minimize(chi2_linquad,[0.0001,0.001], args=(t[0:toffmax],k,Conc_curve[0:toffmax]), method='L-BFGS-B',bounds=[(0,10),(0,10)],options={'disp':False})
        lqcurve=fit.x[0]*(t-t[k]) + fit.x[1]*(t-t[k])*(t-t[k])
        lqcurve[0:k]=0
        chi2s[k]=fit.fun

    #Find minimum chi2 and use this as offset into the t array to get offset time
    minidx=np.argmin(chi2s)
    toff=t[minidx]

    return toff



#Helper function
def chi2_linquad(params,t,k,Conc_curve):

    #Construct linear quadratic function
    lqcurve=params[0]*(t-t[k]) + params[1]*(t-t[k])*(t-t[k])
    lqcurve[0:k]=0

    #Calculate chi2 for this function with the curve data
    chi2=np.sum((lqcurve-Conc_curve)*(lqcurve-Conc_curve))
    return chi2
    
