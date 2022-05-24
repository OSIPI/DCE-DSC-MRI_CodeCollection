OSIPI DCE-DSC code
====================

This website publishes the results of the test results of the code in [OSIPI Code Collection](https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection).

### About OSIPI TF2.3 and the OSIPI code collection  

The ISMRM Open-Source Initiative for Perfusion Imaging (OSIPI) aims to promote the sharing of perfusion imaging software in order to reduce duplication, improve reproducibility and speed up translation. 
The DCE-DSC-MRI_CodeCollection code library is an ongoing project maintained by Taskforce 2.3 of OSIPI and aims to collect, test and share open-source code for the analysis of dynamic contrast-enhanced (DCE-) and dynamic susceptibility-enhanced (DSC-) MRI for use in research and software development. 
Code contributions can include modules covering one or more steps within the image processing pipeline, for example T1 mapping, converting signal to concentration and pharmacokinetic modelling. 
A further aim of OSIPI is to develop a fully tested and harmonised code library, drawing on the contributions within this repository.


### Scientific validation
The goal of our testing framework is to verify and compare the output of 
code contributions implementing specific functionality. 
A testing framework using the Pytest package and Github Actions was set up for 
automated testing of contributed source code. 
Test functions and test data are located in the /test directory of the 
repository and structured according to category. 
For each contribution a separate test file was created, but contributions 
implementing the same functionality are tested using the same test data and 
tolerances.

for more information please check the [wiki](https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection/wiki/Viewing-the-test-results)

### Purpose of this website
Pytest generates binary pass-fail output dependent on the defined tolerances.
A badge on the Github repository page indicates whether all tests have passed.
The purpose of this website is to provide end users of the contributed 
code with detailed quantitative information regarding the output of specific 
code contributions in different regions of parameter space, and to indicate 
the limits of agreement with the reference values.
In this way, users of the repository can make an informed decision regarding 
which code snippet to use for their own analyses.

### Remarks
Code output may depend on the assumptions and methods used in any given 
implementation and on the nature of the test data. Furthermore, while we 
believe that the reference values are reliable, they do not represent a 
gold standard. Therefore, it is not the purpose of this website to rank or 
recommend individual code contributions.

If you would like to contribute to this part of the OSIPI initiative please email the contacts listed on our [website](https://www.osipi.org/task-force-2-3/).

