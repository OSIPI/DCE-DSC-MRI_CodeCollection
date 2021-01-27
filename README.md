# Overview of DCE-DSC-MRI_CodeCollection

[![build Actions Status](https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection/workflows/ci/badge.svg)](https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection/actions)
[![codecov](https://codecov.io/gh/OSIPI/DCE-DSC-MRI_CodeCollection/branch/Milestone-2/graph/badge.svg?token=ZR3RPV8Y0B)](https://codecov.io/gh/OSIPI/DCE-DSC-MRI_CodeCollection)

The ISMRM Open-Source Initiative for Perfusion Imaging ([OSIPI](https://www.osipi.org/)) aims to promote the sharing of perfusion imaging software in order to reduce duplication, improve reproducibility and speed up translation. This **DCE-DSC-MRI_CodeCollection** code library is maintained by [Taskforce 2.3](https://www.osipi.org/task-force-2-3/) of OSIPI and aims to collect, test and share open-source perfusion imaging code for use in research and software development. Code contributions can include modules covering one or more steps within the image processing pipeline, for example T1 mapping, converting signal to concentration and pharmacokinetic modelling. A further aim of OSIPI is to develop a fully tested and harmonised code library, drawing on the contributions within this repository.

If you would like to contribute to the OSIPI initiative please email the contacts listed on our website.

## Repository Organization

The main components of the repository comprise 3 folders and the configuration files for automated testing. 

The **doc** folder contains all documentation pertaining to the operation of TF 2.3.

*   [How to create a copy of the respository and contribute changes to the repository](doc/Create%20a%20local%20copy%20of%20repository.md)
*   [Guidelines for code contribution](doc/Guidelines%20for%20code%20contribution.md)
*   [Guidelines to creating a test file](doc/Creating%20a%20Python%20test%20file.md)   

The **src** folder contains the community contributed src code. Within **src**, the **original** folder contains the code to be tested, and the **tested** folder will house code that has been tested in the current or previous milestones. Within these folders, contributions are stored in Functionality/Initials_InstitutionCountry, e.g. src/original/T1/AB_YaleUSA.

The **test** folder contains the test files corresponding to the contributed code in **src**. Each contributed source code will have a corresponding test file in this folder. The directory structure does not necessarily have to mirror that of the **src** folder. 

## Installing Git
To install a fresh copy of Git or to upgrade to the latest version, please follow the instructions outlined in: [Installing Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

## View Testing Reports
**Github Actions** has been utilized for automated testing of the community contributed source code. The status of testing is shown at the beginning of this document by the **Github** status image denoted by **ci passing**. Clicking on it will take you to [Github Actions](https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection/actions) where a detailed report with past build information will be available. 

