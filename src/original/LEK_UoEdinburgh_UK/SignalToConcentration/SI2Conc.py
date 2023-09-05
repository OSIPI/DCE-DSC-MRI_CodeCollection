
## AUTHOR: Lucy Kershaw, University of Edinburgh
## Purpose: Module to convert signal intensity to concentration (and vice-versa)

import numpy as np


#####################
# FLASH Signal intensity curve to concentration curve 
######################

# Inputs:
	# SIcurve		numpy array of SI values
	# TR			TR for FLASH sequence, in seconds
	# flip			Flip angle for FLASH sequence in degrees
	# T1base		Native T1 corresponding to the baseline signal intensity, in seconds
	# baselinepts	Range of data points to be used for baseline, indexes to start and end at [start,end]
	# S0 			Equilibrium signal, if known. Default is to calculate it here

# Output:
	#H				numpy array of curve as delta R1 in s^-1


def SI2Conc(SIcurve,TR,flip,T1base,baselinepts_range,S0=None):

	# Convert flip angle to radians
	rflip=flip*np.pi/180

	# Convert baseline T1 to baseline R1
	R1base=1/T1base

	# If S0 isn't specified, calculate from baseline
	if S0 is None:
		SIbase=np.mean(SIcurve[baselinepts_range[0]:baselinepts_range[1]])
		S0=CalcM0(SIbase,TR,flip,T1base)
	
	# Now calculate the R1 curve
	R1=np.log(((S0*np.sin(rflip))-SIcurve)/(S0*np.sin(rflip)-(SIcurve*np.cos(rflip))))*(-1/TR)

	# And finally the delta R1 curve
	H=R1-R1base
	#return H
	return H


#####################
# Concentration curve (as deltaR1) to FLASH Signal intensity curve 
######################

# Inputs:
	# deltaR1		numpy array of SI values
	# TR			TR for FLASH sequence, in seconds
	# flip			Flip angle for FLASH sequence in degrees
	# T1base		Native T1 corresponding to the baseline signal intensity, in seconds
	# S0 			Equilibrium signal

# Output:
	#H				numpy array of SI curve



def Conc2SI(deltaR1,TR,flip,T1base,S0):

	# Convert flip angle to radians
	rflip=flip*np.pi/180
	# Convert T1 base to R1 base
	R1base=1/T1base

	# Convert deltaR1 curve to R1 curve
	R1curve=deltaR1+R1base
	# Convert to SI
	SI=FLASH(S0,TR,flip,1/R1curve)

	return SI

######################
# Helper functions to calculate S0 and the FLASH signal intensity equation
######################


def CalcM0(SI, TR, flip, T1):
	#convert flip angle to radians
	rflip=flip*np.pi/180
	#Calculate M0
	S0=SI*(1-np.cos(rflip)*np.exp(-TR/T1))/(np.sin(rflip)*(1-np.exp(-TR/T1)))
	return S0

def FLASH(S0, TR, flip, T1):
	rflip=flip*np.pi/180
	SI=S0*np.sin(rflip)*(1-np.exp(-TR/T1))/(1-np.cos(rflip)*np.exp(-TR/T1))
	return SI