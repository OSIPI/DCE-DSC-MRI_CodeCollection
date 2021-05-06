"""
ExamplePopulationAIF.py
====================================
Demonstrate the use of PopulationAIF.py and visualize the results
options:
- Parker AIF
- Georgiou AIF (MRM 2018, doi: 10.1002/mrm.27524)

@author p.v.houdt@nki.nl
@lab Van der Heide group (https://www.nki.nl/research/research-groups/uulke-van-der-heide/)
@institute department of Radiation Oncology, the Netherlands Cancer Institute
"""


import matplotlib.pyplot as pyplot
from PopulationAIF import *

try:
    dt = 1 #temp resolution in s
    endt = 280 # end time in s
    notimepoints = endt/dt

    time = numpy.arange(0,endt+dt, dt)

    # Parker AIF
    Cb_Parker = ParkerAIF(time)
    pyplot.figure(1)
    pyplot.plot(time.tolist(), Cb_Parker.tolist())
    #should look similar to Fig. 3 of Parker et al MRM 2006

    # Georgiou AIF
    Cb_Georgiou = GeorgiouAIF(time)
    pyplot.figure(2)
    pyplot.plot(time.tolist(), Cb_Georgiou.tolist())
    pyplot.show()
    # should look similar to Fig.2a of Georgiou et al. MRM 2018



except Exception as e:
    print(e)