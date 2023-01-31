# Overview of DCE-DSC-MRI_CodeCollection

[![build Actions Status](https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection/workflows/ci/badge.svg)](https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection/actions)
[![codecov](https://codecov.io/gh/OSIPI/DCE-DSC-MRI_CodeCollection/branch/develop/graph/badge.svg?token=ZR3RPV8Y0B)](https://codecov.io/gh/OSIPI/DCE-DSC-MRI_CodeCollection)



- [code database with links to source code](http://osipi.org/DCE-DSC-MRI_TestResults/contributors.html)
- [test-results website](http://osipi.org/DCE-DSC-MRI_TestResults)

The ISMRM Open Science Initiative for Perfusion Imaging ([OSIPI](https://www.osipi.org/)) aims to promote the sharing of perfusion imaging software in order to reduce duplication, improve reproducibility and speed up translation. This **DCE-DSC-MRI_CodeCollection** code library is maintained by [Taskforce 2.3](https://www.osipi.org/task-force-2-3/) of OSIPI and aims to collect, test and share open-source perfusion imaging code for use in research and software development. Code contributions can include modules covering one or more steps within the image processing pipeline, for example T1 mapping, converting signal to concentration and pharmacokinetic modelling. A further aim of OSIPI is to develop a fully tested and harmonised code library, drawing on the contributions within this repository.

A summary of the repository structure is shown below. More detailed information and documentation is located in the repository [wiki](https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection/wiki)

```
OSIPI/DCE-DSC-MRI_CodeCollection
├── doc
│    └── code_contributions_record.md  # Overview of code contributions
├── src
│   └── original                       # Original contributed code organized per contribution
│       └── DS_BW_VanderBiltUMC_USA
│       └── JBJA_GUSahlgrenskaSWE
│       └── ...
├── test                               # Python test files, organised by category
│   └── DCEmodels
│       └── data                       # data used for testing stored as csv files
│   └── DSCmodels
│       └── data
│   └── ...
│   └── results-meta.json              # stores meta-information about csv-results files, used by notebooks
├── notebooks                          # Jupyter notebooks and markdown files to build the test-results website
├── LICENCE                            # Apache version 2.0 licence
├── README.md
├── setup.py

Wiki                                   # Information and guidelines 
```

Click [here](doc/code_contributions_record.md) for 
database of code available with links to code. 

The results of the validated code are published on the [test-results website](http://osipi.org/DCE-DSC-MRI_TestResults), which is hosted via a separate [repository](https://github.com/OSIPI/DCE-DSC-MRI_TestResults).



This is an ongoing project, we welcome new contributions. If you would like to contribute to the OSIPI initiative please email the contacts listed on our [website](https://www.osipi.org/task-force-2-3/).