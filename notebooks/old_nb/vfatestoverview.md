# VFA T1 Mapping: Test Overview
**This document provides a rudimentary description of the unit tests for variable flip angle based T1 mapping.**

## Assertion Testing:
Test for specific inputs and expected outputs. Assertion tests determine if the code is providing the correct output. A case representing each of the following : simuated data (from MR signal equtions), DRO data, phantom data and invivo data, will be part of this test approach

**Current Status:** For the time being, simulated values without noise have been specified to test the fit. Additional test cases with varying degrees of noise should be included, from maybe a DRO.

## Test for valid input values :

This test case would check for validity of the input arguments passed to the VFA function. For example, if invalid flip angles are used i.e. flip angle < 0, then an exception will be raised.

**Current Status:** Check for negative flip angle is part of the test framework.

## Test for valid input data types:

Check if the correct data types are used for the variables


## Test for array dimensionality:

Check for array dimensionality mismatch which could result in an error while computing fit.
For example : len(signal array) not equal to len(flip angle array)
