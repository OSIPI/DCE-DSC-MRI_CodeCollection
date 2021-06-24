original contribution can be found here:https://github.com/olegjo/Masteroppgave \
master thesis with more information: https://www.duo.uio.no/handle/10852/61659

@author: Ole Gunnar Johansen (current position at NordicImagingLab AS, ole.gunnar@nordicimaginglab.com, https://github.com/olegjo) \
@lab: Research Group Prof Bjornerud\
@institute: dept of Biophysics and Medical Physics, University of Oslo, Norway

Additional info on the data:
The thesis was about analyzing the effects on the accuracy of calculated parameters on various kinematic models used in DCE when changing uptake parameters. 
This was done by generating phantom data based on the two-compartment exchange model using an input-AIF with high temporal resolution. Uptake parameters were then systematically changed to look at the effect on the results, comparing to the known, "true", values.
 
The files in the folders ROOT/Simulations, ROOT/phantoms_json and ROOT/phantoms_py are basically the raw results and inputs to the analysis (the generated phantoms).
 
 
In the MRImageAnalysis folder, there is also a Data folder.
These are also used as input to the analysis where the files in AIF_BTT are all AIFs with different "bolus dispersion" or rates of CA injection. 
For the files in AIF_DCE we need to check what exactly they are. The file called Aorta.txt was the basis for all generated AIFs in AIF_BTT and AIF_DCE.
