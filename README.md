# Overview of TF2.3-CodeLibrary

[![Build Status](https://travis-ci.com/OSIPI/TF2.3-CodeLibrary.svg?token=gKUxg5qhLHFRNjKTZy8a&branch=Milestone-2)](https://travis-ci.com/OSIPI/TF2.3-CodeLibrary) 
[![codecov](https://codecov.io/gh/OSIPI/TF2.3-CodeLibrary/branch/Milestone-2/graph/badge.svg?token=ZR3RPV8Y0B)](undefined)

The **TF2.3-CodeLibrary** respository will house all contributed source code from the OSIPI and Perfusion MR community, that belong to the DCE and DSC pipelines. To learn more about the role of Task Force 2.3, please visit the [OSIPI website](https://www.osipi.org/task-force-2-3/).

## Repository Organization

The main components of the repository comprise of 3 folders and the configuration files for automated testing. 

The **doc** folder contains all documentation pertaining to the operation of TF 2.3. Some of the topics addressed as part of the documentation process for **Milestone-2** are:

*   How to create a copy of the respository and contribute changes to the repository
*   Guidelines to creating a test file
*   Guidelines for code contribution

The **src** folder contains the community contributed src code. Within **src**, the **original** folder contains the code to be tested, and the **tested** folder will house code that has been tested in the current or previous milestones.

The **test** folder contains the test files corresponding to the contributed code in **src**. Each contributed source code will have a corresponding test file in this folder. The directory structure does not necessarily have to mirror that of the **src** folder. 

## Installing Git
To install a fresh copy of Git or to upgrade to the latest version, please follow the instructions outlined in: [Installing Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

## View Testing Reports

**TRAVIS CI** has been utilized for automated testing of the community contributed source code. The status of testing is shown at the beginning of this document by the **TRAVIS** status image denoted by **build passing**. Clicking on it will take you to [TRAVIS CI](https://travis-ci.com/OSIPI/TF2.3-CodeLibrary) where a detailed report with past build information will be available. 