import MRImageAnalysis as mri
import matplotlib.pyplot as plt
import numpy as np
import sys

class TransferFunctions:
	def __init__(self, w, K_trans, v_e, v_p=None, F_p=None, PS=None):
		self.w = w
		self.defaults = {
			'K_trans': K_trans,
			'F_p'    : F_p,
			'v_p'    : v_p,
			'v_e'    : v_e,
			'PS'     : PS
		}
		self.reset()

	@property
	def k_ep(self):
		return self.K_trans/self.v_e

	def reset(self):
		for p in self.defaults:
			setattr(self, p, self.defaults[p])

	def dB(self, y):
		return 20*np.log10(np.abs(y))

	def H_2CXM(self, cutoff=False):
		if self.PS is None:
			self.PS = mri.DCE.Models.Conversion.PS(F_p=self.F_p, K_trans=self.K_trans)
		
		E  = self.PS/float(self.PS + self.F_p)
		e  = self.v_e/float(self.v_e + self.v_p)
		Ee = E*e
		
		tau_pluss = (E - Ee + e)/(2.*E)*(1 + np.sqrt(1 - 4*(Ee*(1-E)*(1-e))/(E - Ee + e)**2 ) )		
		tau_minus = (E - Ee + e)/(2.*E)*(1 - np.sqrt(1 - 4*(Ee*(1-E)*(1-e))/(E - Ee + e)**2 ) )


		F_pluss = self.F_p*(tau_pluss - 1.)/(tau_pluss - tau_minus)
		F_minus = -self.F_p*(tau_minus - 1.)/(tau_pluss - tau_minus)

		K_pluss = self.F_p/((self.v_p + self.v_e) * tau_minus)
		K_minus = self.F_p/((self.v_p + self.v_e) * tau_pluss)

		i = np.complex(0,1)
		H = F_pluss/(i*self.w + K_pluss) + F_minus/(i*self.w + K_minus)

		if cutoff:
			c1 = 2*np.pi*np.abs(-K_pluss/i)
			c2 = 2*np.pi*np.abs(-K_minus/i)
			zero = 2*np.pi*np.abs(i*(K_pluss*F_minus + K_minus*F_pluss)/(F_pluss + F_minus))
			return c1, c2, zero

		return self.dB(H)

	def H_ETM(self, cutoff=False):
		i = np.complex(0,1)
		H = self.v_p + self.K_trans/(i*self.w + self.k_ep)

		if cutoff:
			c = 2*np.pi*np.abs(-self.k_ep/i)
			zero = 2*np.pi*np.abs(i*(self.K_trans + self.v_p*self.k_ep)/self.v_p)
			return c, zero
		return self.dB(H)

	def H_TM(self, cutoff=False):
		i = np.complex(0,1)
		H = self.K_trans/(i*self.w + self.k_ep)

		if cutoff:
			c = 2*np.pi*np.abs(-self.k_ep/i)
			return c
		return self.dB(H)




