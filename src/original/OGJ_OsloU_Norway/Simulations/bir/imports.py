import numpy as np
import MRImageAnalysis as mri

def dofft(t, y):
	Ts = (t[1]-t[0])
	Fs = 1/Ts;  # sampling rate

	n = len(y) # length of the signal
	k = np.arange(n)
	T = n/Fs
	frq = k/T # two sides frequency range
	frq = frq[range(int(n/2))] # one side frequency range


	Y = np.fft.fft(y)/n # fft computing and normalization
	Y = Y[range(int(n/2))]

	return frq, Y



def twoCXM(t, K_trans, F_p, v_e, v_p):
	PS = mri.DCE.Models.Conversion.PS(F_p=F_p, K_trans=K_trans)
	
	E  = PS/float(PS + F_p)
	e  = v_e/float(v_e + v_p)
	Ee = E*e
	
	tau_pluss = (E - Ee + e)/(2.*E)*(1 + np.sqrt(1 - 4*(Ee*(1-E)*(1-e))/(E - Ee + e)**2 ) )		
	tau_minus = (E - Ee + e)/(2.*E)*(1 - np.sqrt(1 - 4*(Ee*(1-E)*(1-e))/(E - Ee + e)**2 ) )


	F_pluss = F_p*(tau_pluss - 1.)/(tau_pluss - tau_minus)
	F_minus = -F_p*(tau_minus - 1.)/(tau_pluss - tau_minus)

	K_pluss = F_p/((v_p + v_e) * tau_minus)
	K_minus = F_p/((v_p + v_e) * tau_pluss)

	return F_pluss*np.exp(-t*K_pluss) + F_minus*np.exp(-t*K_minus)

	