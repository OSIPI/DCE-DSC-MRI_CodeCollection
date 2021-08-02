"""
PopulationAIF.py
====================================
This module returns a population AIF given a certain time array
options:
- Parker AIF
- Georgiou AIF (MRM 2018, doi: 10.1002/mrm.27524)
@author p.v.houdt@nki.nl
@lab Van der Heide group (https://www.nki.nl/research/research-groups/uulke-van-der-heide/)
@institute department of Radiation Oncology, the Netherlands Cancer Institute
"""

import numpy
import math

def ParkerAIF(time):
    """
    Function to determine the Parker AIF given a certain time array
    Parameters
    ---------
    time
        time array in seconds.
    AIF
        array of concentration values for whole blood
    citation: Parker et al. Magn Reson Med 2006 https://doi.org/10.1002/mrm.21066
    """

    # parameters taken from Table 1
    A1 = 0.809 #mmol.min
    A2 = 0.330 #mmol.min
    T1 = 0.17046 #min
    T2 = 0.365 #min
    sigma1 = 0.0563 #min
    sigma2 = 0.132 #min
    alpha = 1.050 #mmol
    beta = 0.1685 #min-1
    s = 38.078 #min-1
    tau = 0.483 #min

    # convert min to s
    A1 = A1*60
    A2 = A2*60
    T1 = T1*60
    T2 = T2*60
    sigma1 = sigma1*60
    sigma2 = sigma2*60
    beta = beta/60
    s = s/60
    tau = tau*60

    gaussian1 = A1 / (sigma1 * numpy.sqrt(2 * numpy.pi))* numpy.exp(-numpy.square(time - T1)/(2 * numpy.square(sigma1)))
    gaussian2 = A2 / (sigma2 * numpy.sqrt(2 * numpy.pi))* numpy.exp(-numpy.square(time - T2)/(2 * numpy.square(sigma2)))
    modSigm = (alpha * numpy.exp(-beta * time)) / (1 + numpy.exp(-s * (time - tau)))
    Cb = numpy.add(gaussian1,gaussian2)
    Cb = numpy.add(Cb, modSigm)#whole blood values

    return Cb

class AIFGeorgiouFunctionalParameters():
    A1: float
    m1: float
    A2: float
    m2: float
    A3: float
    m3: float
    alpha: float
    beta: float
    tau: float

    def __init__(self, a1: float = 0, m1: float = 0, a2: float = 0, m2: float = 0, a3: float = 0, m3: float = 0, alpha: float = 0, beta: float = 0, tau: float = 0):
        self.A1 = a1
        self.m1 = m1
        self.A2 = a2
        self.m2 = m2
        self.A3 = a3
        self.m3 = m3
        self.alpha = alpha
        self.beta = beta
        self.tau = tau


def functionalform_GeorgiouAIF(time,par):
    """
    Function to create the Georgiou AIF based on the Horsefield model using input parameters par.
    Implementation based on the equation 1 in the paper and original implementation in Matlab by Georgiou et al.
    references
    Geourgiou et al. MRM 2018, doi: 10.1002/mrm.27524
    Horsfield et al. Phys Med Biol 2009, https://doi.org/10.1088/0031-9155/54/9/023
    Parameters
    ---------
    time
        time array in seconds.
    par
        set of input parameters (original values from Table 3 of paper)
    output
        AIF values for whole blood
    """
    # put time array in minutes
    time = time/60

    no_circ = round(time[len(time)-1] / par.tau)
    Cb = numpy.zeros(len(time))
    for current_circ in range(0, no_circ+1):
        if current_circ < no_circ:
            timeindex = numpy.where((time >= current_circ*par.tau) & (time < (current_circ+1)*par.tau))
        else:
            timeindex = numpy.where(time >= current_circ*par.tau)

        current_time = time[timeindex]

        ftot = numpy.zeros(len(timeindex))
        for k in range(0, current_circ+1):
            alphamod = (k + 1) * par.alpha + k
            timemod = current_time - k * par.tau
            f1 = numpy.power(timemod, alphamod) * numpy.exp(-timemod / par.beta)
            try:
                ans = math.gamma(alphamod + 1)
                f2 = numpy.power(par.beta, alphamod + 1) * math.gamma(alphamod + 1)  # or try use math.gamma?
                f3 = f1 / f2
                ftot = ftot + f3
            except OverflowError:
                ans = float('inf')

        exp1 = par.A1 * numpy.exp(-par.m1 * current_time)
        exp2 = par.A2 * numpy.exp(-par.m2 * current_time)
        exp3 = par.A3 * numpy.exp(-par.m3 * current_time)
        sumexp = numpy.add(exp1, exp2)
        sumexp = numpy.add(sumexp, exp3)
        Cb[timeindex] = sumexp*ftot

    return Cb

def GeorgiouAIF(time):
    """
    Function to determine the Georgiou AIF given a certain time array
    concentration values will be interpolated from original high temporal resolution AIF
    reference Geourgiou et al. MRM 2018, doi: 10.1002/mrm.27524
    xls file from supplemental materials has been taken
    Parameters
    ---------
    time
        time array in seconds.
    AIF
        array of concentration values for whole blood
    """

    # import txt file
    timeaif, aifdata = numpy.loadtxt("PopulationAIF_GeorgiouMRM2018.txt", delimiter='\t', unpack=True)
    timeaif=timeaif*60  # convert time to seconds

    # interpolate to a given time series
    last_timeaif = timeaif[len(timeaif)-1]
    last_time = time[len(time)-1]
    if last_time <= last_timeaif:
        # interpolate to the right time series
        Cb = numpy.interp(time, timeaif, aifdata)
    else:
        # in case the time series of the output time series is longer than the original measurements, we use the parameters of the Horsfield functional form
        # values taken from Table 3 of the paper
        # initiate structured array
        par = AIFGeorgiouFunctionalParameters(a1=0.37, m1=0.11, a2=0.33, m2=1.17, a3=10.06, m3=16.02, alpha=5.26, beta=0.032, tau=0.129)
        #units: mM, min-1, mM, min-1, mM, min-1, -, min, min

        Cb = functionalform_GeorgiouAIF(time, par)

    return Cb
