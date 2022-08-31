"""Functions to fit MRI SPGR signal to obtain T1.

Created 28 September 2020
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Classes:
    VFA2Points
    VFALinear
    VFANonLinear
    HIFI

Functions:
    spgr_signal: get SPGR signal
    irspgr_signal: get IR-SPGR signal
"""

import numpy as np
from scipy.optimize import least_squares
from .fitting import Fitter


class VFA2Points(Fitter):
    """Estimate T1 with 2 flip angles.

    Subclass of Fitter.
    """

    def __init__(self, fa, tr):
        """

        Args:
            fa (ndarray): 1D array containing the two flip angles (deg)
            tr: (float): TR (s)
        """
        self.fa = np.asarray(fa)
        self.tr = tr
        self.fa_rad = np.pi * self.fa / 180

    def output_info(self):
        """Get output info. Overrides superclass method.
        """
        return ('s0', False), ('t1', False)

    def proc(self, s, k_fa=1):
        """Estimate T1. Overrides superclass method.

        Args:
            s (ndarray): 1D array containing the two signals
            k_fa (float): B1 correction factor, i.e. actual/nominal flip angle.

        Returns:
            tuple: (s0, t1)
                s0 (float): fully T1-relaxed signal
                t1 (float): T1 (s)

        """
        if any(np.isnan(s)):
            raise ValueError(
                f'Unable to calculate T1: nan signal values received.')
        with np.errstate(divide='ignore', invalid='ignore'):
            fa_true = k_fa * self.fa_rad
            sr = s[0] / s[1]
            t1 = self.tr / np.log(
                (sr * np.sin(fa_true[1]) * np.cos(fa_true[0]) -
                 np.sin(fa_true[0]) * np.cos(fa_true[1])) /
                (sr * np.sin(fa_true[1]) - np.sin(fa_true[0])))
            s0 = s[0] * ((1 - np.exp(-self.tr / t1) * np.cos(fa_true[0])) /
                         ((1 - np.exp(-self.tr / t1)) * np.sin(fa_true[0])))

        if ~np.isreal(t1) | (t1 <= 0) | np.isinf(t1) | (s0 <= 0) | np.isinf(s0):
            raise ArithmeticError('T1 estimation failed.')

        return s0, t1


class VFALinear(Fitter):
    """Linear variable flip angle T1 estimation.

    Subclass of Fitter.
    """

    def __init__(self, fa, tr):
        """

        Args:
            fa (ndarray): 1D array containing the flip angles (deg)
            tr: (float): TR (s)
        """
        self.fa = np.asarray(fa)
        self.tr = tr
        self.fa_rad = np.pi * self.fa / 180

    def output_info(self):
        """Get output info. Overrides superclass method.
        """
        return ('s0', False), ('t1', False)

    def proc(self, s, k_fa=1):
        """Estimate T1. Overrides superclass method.

        Args:
            s (ndarray): 1D array containing the signals
            k_fa (float): B1 correction factor, i.e. actual/nominal flip angle.

        Returns:
            tuple: (s0, t1)
                s0 (float): fully T1-relaxed signal
                t1 (float): T1 (s)

        """
        if any(np.isnan(s)) or np.isnan(k_fa):
            raise ArithmeticError(
                f'Unable to calculate T1: nan signal or k_fa values received.')
        fa_true = k_fa * self.fa_rad
        y = s / np.sin(fa_true)
        x = s / np.tan(fa_true)
        A = np.stack([x, np.ones(x.shape)], axis=1)
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

        if (intercept < 0) or ~(0. < slope < 1.):
            raise ArithmeticError('T1 estimation failed.')

        t1, s0 = -self.tr / np.log(slope), intercept / (1 - slope)

        return s0, t1


class VFANonLinear(Fitter):
    """Non-linear variable flip angle T1 estimation.

    Subclass of Fitter.
    """

    def __init__(self, fa, tr):
        """

        Args:
            fa (ndarray): 1D array containing the flip angles (deg)
            tr: (float): TR (s)
        """
        self.fa = np.asarray(fa)
        self.tr = tr
        self.fa_rad = np.pi * self.fa / 180
        self.linear_fitter = VFALinear(fa, tr)

    def output_info(self):
        """Get output info. Overrides superclass method.
        """
        return ('s0', False), ('t1', False)

    def proc(self, s, k_fa=1):
        """Estimate T1. Overrides superclass method.

        Args:
            s (ndarray): 1D array containing the signals
            k_fa (float): B1 correction factor, i.e. actual/nominal flip angle.

        Returns:
            tuple: (s0, t1)
                s0 (float): fully T1-relaxed signal
                t1 (float): T1 (s)

        """
        if any(np.isnan(s)) or np.isnan(k_fa):
            raise ValueError(
                f'Unable to calculate T1: nan signal or k_fa values received.')
        # use linear fit to obtain initial guess, otherwise start with T1=1
        try:
            x0 = np.array(self.linear_fitter.proc(s, k_fa=k_fa))
        except ArithmeticError:
            x0 = np.array([s[0] / spgr_signal(1., 1., self.tr, k_fa * self.fa[
                0]), 1.])

        result = least_squares(self.__residuals, x0, args=(s, k_fa), bounds=(
            (1e-8, 1e-8), (np.inf, np.inf)), method='trf', x_scale=x0)
        if result.success is False:
            raise ArithmeticError(f'Unable to fit VFA data:'
                                  f' {result.message}')

        s0, t1 = result.x
        return s0, t1

    def __residuals(self, x, s, k_fa):
        s0, t1 = x
        s_est = spgr_signal(s0, t1, self.tr, k_fa * self.fa)
        return s - s_est


class HIFI(Fitter):
    """DESPOT1-HIFI T1 estimation.

    Subclass of Fitter.
    Note: perfect inversion is assumed for IR-SPGR signals.
    """

    def __init__(self, esp, ti, n, b, td, centre):
        """

        Args:
            esp (ndarray):  Echo spacings (s, 1 float per acquisition).
                Equivalent to TR for SPGR scans.
            ti (ndarray): Inversion times (s, 1 per acquisition). Note this
                is the actual time delay between the inversion pulse and the
                start of the echo train. The effective TI may be different,
                e.g for linear phase encoding of the echo train. For SPGR,
                set values to np.nan.
            n (ndarray): Number of excitation pulses per inversion pulse (1
                int value per acquisition). For SPGR, set values to np.nan.
            b (ndarray): Excitation flip angles (deg, 1 float per acquisition).
            td (ndarray): Delay between readout train and next inversion
                pulse (s, 1 float per acquisition). For SPGR, set values to
                np.nan.
            centre (ndarray): Times in readout train when centre of k-space
                is acquired, expressed as a fraction of the readout duration.
                e.g. = 0 for centric phase encoding, = 0.5 for linear phase
                encoding (float, 1 per acquisition). For SPGR, set values to
                np.nan.
        """
        self.esp = esp
        self.ti = ti
        self.n = n
        self.b = b
        self.td = td
        self.centre = centre
        # get information about the scans
        self.n_scans = len(esp)
        self.is_ir = ~np.isnan(ti)
        self.is_spgr = ~self.is_ir
        self.idx_spgr = np.where(self.is_spgr)[0]
        self.n_spgr = self.idx_spgr.size
        self.get_linear_estimate = self.n_spgr > 1 and np.all(
            np.isclose(esp[self.idx_spgr], esp[self.idx_spgr[0]]))
        if self.get_linear_estimate:
            self.linear_fitter = VFALinear(b[self.is_spgr],
                                           esp[self.idx_spgr[0]])
        self.max_k_fa = 90 / max(self.b[self.is_ir]) if any(self.is_ir) else \
            np.inf

    def output_info(self):
        """Get output info. Overrides superclass method.
        """
        return ('s0', False), ('t1', False), ('k_fa', False), ('s_opt', True)

    def proc(self, s, k_fa_fixed=None):
        """Estimate T1 and k_fa. Overrides superclass method.

        Args:
            s (ndarray): 1D array containing the signals
            k_fa_fixed (float): Value to which k_fa (actual/nominal flip
                angle) is fixed. If set to None (default) then the value of k_fa
                is optimised.

        Returns:
            tuple: s0, t1, k_fa, s_opt
                s0 (float): fully T1-relaxed signal
                t1 (float): T1 (s)
                k_fa (float): flip angle correction factor
                s_opt (ndarray): fitted signal intensities
        """
        # First get a quick linear T1 estimate
        if self.get_linear_estimate:  # If >1 SPGR, use linear VFA fit
            i = self.idx_spgr[0]
            try:
                s0_init, t1_init = self.linear_fitter.proc(s[self.is_spgr])
            except ArithmeticError:  # if result invalid, assume T1=1
                t1_init = 1
                s0_init = s[i] / spgr_signal(1, t1_init, self.esp[i], self.b[i])
        # If 1 SPGR scan, assume T1=1 and estimate s0 based on 1st SPGR scan
        elif self.n_spgr == 1:
            i = self.idx_spgr[0]
            t1_init = 1
            s0_init = s[i] / spgr_signal(1, t1_init, self.esp[i], self.b[i])
        # If 0 SPGR scans, assume T1=1 and estimate s0 based on 1st scan
        else:
            t1_init = 1
            s0_init = s[0] / irspgr_signal(1, t1_init, self.esp[0], self.ti[0],
                                           self.n[0], self.b[0], self.td[0],
                                           self.centre[0])
        # Now do a non-linear fit using all scans
        if k_fa_fixed is None:
            k_init = 1
            bounds = ([0, 0, 0], [np.inf, np.inf, self.max_k_fa])
        else:
            k_init = k_fa_fixed
            bounds = ([0, 0, k_fa_fixed-1e-8], [np.inf, np.inf, k_fa_fixed])
        x_0 = np.array([t1_init, s0_init, k_init])
        result = least_squares(self.__residuals, x_0, args=(s,), bounds=bounds,
                               method='trf',
                               x_scale=(t1_init, s0_init, k_init)
                               )
        if not result.success:
            raise ArithmeticError(f'Unable to fit HIFI data: {result.message}')
        t1, s0, k_fa = result.x
        s_opt = self.__signal(result.x)
        return s0, t1, k_fa, s_opt

    def __residuals(self, x, s):
        return s - self.__signal(x)

    def __signal(self, x):
        # calculate signal for all of the (IR-)SPGR scans
        t1, s0, k_fa = x
        s = np.zeros(self.n_scans)
        s[self.is_ir] = irspgr_signal(s0, t1, self.esp[self.is_ir],
                                      self.ti[self.is_ir], self.n[self.is_ir],
                                      k_fa * self.b[self.is_ir],
                                      self.td[self.is_ir],
                                      self.centre[self.is_ir])
        s[self.is_spgr] = spgr_signal(s0, t1, self.esp[self.is_spgr],
                                      k_fa * self.b[self.is_spgr])
        return s


def spgr_signal(s0, t1, tr, fa):
    """Return signal for SPGR sequence.

    Parameters
    ----------
        s0 : float
             Equilibrium signal.
        t1 : float
             T1 value (s).
        tr : float
             TR value (s).
        fa : float
             Flip angle (deg).

    Returns
    -------
        s : float
            Steady-state SPGR signal.
    """
    fa_rad = np.pi * fa / 180

    e = np.exp(-tr / t1)
    s = abs(s0 * (((1 - e) * np.sin(fa_rad)) / (1 - e * np.cos(fa_rad))))

    return s


def irspgr_signal(s0, t1, esp, ti, n, b, td, centre):
    """Return signal for IR-SPGR sequence.

    Uses formula by Deichmann et al. (2000) to account for modified
    apparent relaxation rate during the pulse train. Note inversion is assumed
    to be ideal.

    Parameters
    ----------
        s0 : float
             Equilibrium signal.
        t1 : float
             T1 value (s).
        esp : float
             Echo spacing (s). For SPGR, this is the TR.
        ti : float
             Inversion time (s). Note this is the actual time delay between the
             inversion pulse and the start of the echo train. The effective TI
             may be different, e.g for linear phase encoding of the echo train.
        n : int
            Number of excitation pulses per inversion pulse
        b : float
            Readout pulse flip angle (deg)
        td : float
             Delay between end of readout train and the next inversion (s).
        centre : float
                 Time in readout train when centre of k-space is acquired,
                 expressed as a fraction of the readout duration. e.g. = 0 for
                 centric phase encoding, = 0.5 for linear phase encoding.

    Returns
    -------
        s : float
            Steady-state IR-SPGR signal.
    """
    b_rad = np.pi * b / 180
    tau = esp * n
    t1_star = 1 / (1 / t1 - 1 / esp * np.log(np.cos(b_rad)))
    m0_star = s0 * ((1 - np.exp(-esp / t1)) / (1 - np.exp(-esp / t1_star)))

    r1 = -tau / t1_star
    e1 = np.exp(r1)
    e2 = np.exp(-td / t1)
    e3 = np.exp(-ti / t1)

    a1 = m0_star * (1 - e1)
    a2 = s0 * (1 - e2)
    a3 = s0 * (1 - e3)

    a = a3 - a2 * e3 - a1 * e2 * e3
    b = -e1 * e2 * e3

    m1 = a / (1 - b)

    s = np.abs((m0_star + (m1 - m0_star) * np.exp(centre * r1)) * np.sin(b_rad))
    return s
