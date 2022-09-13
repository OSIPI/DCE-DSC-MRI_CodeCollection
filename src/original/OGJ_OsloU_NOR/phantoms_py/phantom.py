import json
import sys
import MRImageAnalysis as mri
import numpy as np
import matplotlib.pyplot as plt


def _getData(phantomName):
	jsonData = open('phantoms_json/'+phantomName+'.json').read()
	jsonData      = json.loads(jsonData)

	if jsonData["AIF"] == "standard":
		t0, C_a0 = mri.DCE.AIF.loadStandard()
	elif type(jsonData["AIF"]) == int:
		t0, C_a0 = mri.DCE.AIF.loadStandard(i=jsonData["AIF"])
	else:
		t0, _ = mri.DCE.AIF.loadStandard()
		C_a0  = np.loadtxt(jsonData["AIF"])
		t0    = np.linspace(t0[0], t0[-1], len(C_a0))
	if jsonData["units"] == "min":
		t0 /= 60

	t   = t0
	C_a = C_a0
	if not "units" in jsonData:
		print('Units must be supplied in the phantom json file! (s or min). Aborting.')
		sys.exit()

	phantomArgs = {
		"C_a"                 : C_a,
		"t"                   : t,
		"firstParamaterRange" : jsonData["firstParamaterRange"],
		"secondParamaterRange": jsonData["secondParamaterRange"],
		"axisParamaterNames"  : jsonData["axisParamaterNames"],
		"AIFSection"          : jsonData["showAIFSection"]
	}
	for p in ['K_trans', 'k_ep', 'v_p', 'v_e', 'F_p', 'PS']:
		if p in jsonData:
			phantomArgs[p] = jsonData[p]
	phantomParams = {
		"noTimepoints"  : len(t),
		"noSlices"      : 1,
		"noRows"        : jsonData["noRows"],
		"noCols"        : jsonData["noCols"],
		"sectionHeight" : jsonData["sectionHeight"],
		"sectionWidth"  : jsonData["sectionWidth"],
		"sectionSpacing": jsonData["sectionSpacing"]
	}
	return phantomArgs, phantomParams, jsonData

def phantom(phantomName, save=False, saveNifti=False, C_a=None):
	phantomArgs, phantomParams, jsonData = _getData(phantomName)
	if C_a is not None:
		phantomArgs['C_a'] = C_a
		phantomArgs['t'] = np.linspace(phantomArgs['t'][0], phantomArgs['t'][-1], len(C_a))
		phantomParams['noTimepoints'] = len(C_a)
	phantom = mri.MRImage.PhantomImage(**phantomParams)
	phantom.createFromModel(jsonData["model"], **phantomArgs)
	if jsonData["noise"] != 0:
		if type(jsonData["noise"]) in [int, float]:
			phantom.addNoise(jsonData["noise"])
		else:
			print("Currently only int and float noise is supported")

	if jsonData["dt"] != phantomArgs['t'][1]-phantomArgs['t'][0] and jsonData["dt"] != 0:
		dt = jsonData["dt"]
		if jsonData["units"] == "min":
			dt /= 60

		# downsample the phantom and AIF
		# first the AIF
		new_t, new_C_a = mri.math.misc.downSampleAverage(phantom.time, phantom.C_a, dt)


		# make a new array for storing the voxel intensities
		shape          = list(phantom.voxelArray.shape)
		shape[1]       = len(new_t)+1
		new_voxelArray = np.zeros(shape)


		# downsample the signal
		pbar = mri.io.ProgressBar('Downsampling...', phantom.voxelArray[0,0].shape[0]*phantom.voxelArray[0,0].shape[1])
		for i in range(phantom.voxelArray[0,0].shape[0]):
			for j in range(phantom.voxelArray[0,0].shape[1]):
				new_t, new_signal = mri.math.misc.downSampleAverage(phantom.time, phantom.voxelArray[0,:,i,j], dt)
				new_voxelArray[0,1:,i,j] = new_signal
				pbar.update()
		pbar.finish()
		new_t = np.append(new_t, np.array([new_t[-1] + new_t[1] - new_t[0]]))
		new_C_a = np.append(np.zeros(1), new_C_a)

		# now update the phantom
		phantom.voxelArray = new_voxelArray
		phantom.C_a        = new_C_a
		phantom.time       = new_t


	data = {
		"k_ep"   : None,
		"K_trans": None,
		"v_e"    : None,
		"v_p"    : None,
		"dt"     : phantomArgs["t"][1]-phantomArgs["t"][0],
		"F_p"    : None,
		"PS"     : None,
		"noise"  : jsonData["noise"]
	}

	data[jsonData["axisParamaterNames"]["first"]]  = phantom.getParameterImage(0).voxelArray
	data[jsonData["axisParamaterNames"]["second"]] = phantom.getParameterImage(1).voxelArray
	for key in data:
		if data[key] is None and key in jsonData:
			data[key] = jsonData[key]

	stillNone = []
	for key in data:
		if data[key] is None:
			stillNone.append(key)
	
	while len(stillNone) > 0:
		for key in stillNone:
			if key == "k_ep":
				if not "K_trans" in stillNone and not "v_e" in stillNone:
					data["k_ep"] = mri.DCE.Models.Conversion.k_ep(K_trans=data["K_trans"], v_e=data["v_e"])
					stillNone.pop(stillNone.index(key))
				if not "PS" in stillNone and not "F_p" in stillNone and not "v_e" in stillNone:
					data["k_ep"] = mri.DCE.Models.Conversion.k_ep(PS=data["PS"], F_p=data["F_p"], v_e=data["v_e"])
					stillNone.pop(stillNone.index(key))

			if key == "K_trans":
				if not "PS" in stillNone and not "F_p" in stillNone:
					data["K_trans"] = mri.DCE.Models.Conversion.K_trans(PS=data["PS"], F_p=data["F_p"])
					stillNone.pop(stillNone.index(key))
				if not "k_ep" in stillNone and not "v_e" in stillNone:
					data["K_trans"] = mri.DCE.Models.Conversion.K_trans(k_ep=data["k_ep"], v_e=data["v_e"])
					stillNone.pop(stillNone.index(key))

			if key == "v_e":
				if not "K_trans" in stillNone and not "k_ep" in stillNone:
					data["v_e"] = mri.DCE.Models.Conversion.v_e(K_trans=data["K_trans"], k_ep=data["k_ep"])
					stillNone.pop(stillNone.index(key))

			if key == "PS":
				if not "F_p" in stillNone and not "K_trans" in stillNone:
					data["PS"] = mri.DCE.Models.Conversion.PS(F_p=data["F_p"], K_trans=data["K_trans"])
					stillNone.pop(stillNone.index(key))

			if key == "F_p":
				if not "PS" in stillNone and not "K_trans" in stillNone:
					data["F_p"] = mri.DCE.Models.Conversion.F_p(PS=data["PS"], K_trans=data["K_trans"])
					stillNone.pop(stillNone.index(key))

	for key in data:
		if type(data[key]) == np.ndarray:
			data[key] = phantom.getAverageTable(data[key])[0,0,:,:]
			shape = data[key].shape

	if save:
		for key in data:
			mri.io.saveAsTableText(data[key], shape, "../results/phantoms/"+phantomName+"/true/"+key+".txt")
	if saveNifti:
		phantom.toNifti(phantomName+'.nii')
		print('Saved to nifti.')

	return phantom, data, jsonData
	
if __name__ == '__main__':
	if 'nifti' in sys.argv:
		saveNifti = True
	else:
		saveNifti = False
	if 'nosave' in sys.argv:
		saveData = False
	else:
		saveData = True
	phantom, data, jsonData = phantom(sys.argv[1], saveData, saveNifti)

	if 'plot' in sys.argv:
		phantom.plot(showfig=True)






























