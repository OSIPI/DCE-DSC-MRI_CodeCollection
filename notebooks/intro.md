OSIPI DCE-DSC code
====================

This website publishes the results of the test results of the code in [OSIPI Code Collection](https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection).

### About OSIPI TF2.3 and the OSIPI code collection
The ISMRM Open-Source Initiative for Perfusion Imaging (OSIPI) aims to promote the sharing of perfusion imaging software in order to reduce duplication, improve reproducibility and speed up translation. 
The DCE-DSC-MRI_CodeCollection code library is an ongoing project maintained by Taskforce 2.3 of OSIPI and aims to collect, test and share open-source code for the analysis of dynamic contrast-enhanced (DCE-) and dynamic susceptibility-enhanced (DSC-) MRI for use in research and software development. 
Code contributions can include modules covering one or more steps within the image processing pipeline, for example T1 mapping, converting signal to concentration and pharmacokinetic modelling. 
A further aim of OSIPI is to develop a fully tested and harmonised code library, drawing on the contributions within this repository.


### Scientific validation
The goal of a testing framework was to verify and compare the output of contributions of a specific functionality, i.e. scientific validation. 
At this stage we were not concerned with testing non-scientific questions such as checking for valid parameters. 
This will become more relevant when a final DCE/DSC package will be developed.
A testing framework with pytest package and Github Actions was set up for automated testing of contributed source code. 
Test files were created in the test directory of the repository and structured per category. 
For each contribution a separate test file was created, but contributions within the same category used the same testing data and tolerances.

### Purpose of this website
The output of pytest is pass or fail dependent on the defined tolerances. 
This will identify contributions with large differences with the reference values. 
However, this does not provide enough detail to compare the accuracy of different contributions.
Therefore, the purpose of this website is to provide the end users of the contributed code with more detailed information about the degree of variation in the results.
In this way, users of the repository can make a more informed decision of which code snippet to use for their own analysis. 
As the choice may depend on the application, we are not giving recommendations on which code contribution is best. 
Instead we are showing the differences based on the test data that was currently used.

### Remarks
If you would like to contribute to this part of the OSIPI initiative please email the contacts listed on our [website](https://www.osipi.org/task-force-2-3/).

