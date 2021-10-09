# South Edinburgh Perfusion Analysis Library (SEPAL)
Python library for simulating and fitting DCE-MRI data. It permits arbitrary combinations of pulse sequence, pharmacokinetic model, water exchange model, etc.  
The code is a work-in-progress, has not been extensively tested and is not recommended or approved for use.

Created 28 September 2020
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

### Functionality:
- Enhancement-to-concentration conversion (assuming fast water exchange)
- Fit tissue concentration using pharmacokinetic model
- Fit signal enhancement using pharmacokinetic model
- Pharmacokinetic models: steady-state, Patlak, extended Tofts, Tofts, 2CXM, 2CUM
- AIFs: patient-specific (measured), Parker, bi-exponential Parker
- Fitting an AIF time delay
- Relaxivity models: linear
- Signal models: spoiled gradient echo
- Water exchange models: FXL, NXL, NXL_be
- T1 fitting using variable flip angle method, IR-SPGR and DESPOT1-HIFI

### Not yet implemented/limitations:
- Generally untested. Not optimised for speed or robustness.
- Additional pharmacokinetic models (add by inheriting from pk_model class)
- Additional relaxivity models (add by inheriting from c_to_r_model class)
- Additional AIF functions (add by inheriting from aif class)
- Additional water exchange models, e.g. 3S2X, 2S1X (add by inheriting from water_ex_model class)
- Additional signal models (add by inheriting from signal_model class)
- R2/R2* effects not included in fitting of enhancement curves (but is included for enhancement-to-concentration conversion)
- Compartment-specific relaxivity parameters/models
- Fitting free water exchange parameters
- Special model implementations, e.g. linear and graphical versions of Patlak model

TODO:
- replace minimize with least_squares (faster) for signal DCE fitting
- add check for vp+ve <= 1
- test update notebooks
- convert fitting functions to OO methods. Add image processing functions.