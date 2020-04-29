import autofit as af
import os
import matplotlib.pyplot as plt

import sys
sys.path.append('/disk2/sapple/VAMP/vamp_2.0')
from vamp_src.model import profile_models
from vamp_src.dataset.spectrum import *
import vamp_src.phase.phase as ph

workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
config_path = workspace_path + "config"
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output_params"
)

param_setting = 'default'
combo = 'e'
ncomp = 2
spectrum_dir = '/home/sarah/VAMP/vamp_2.0/vamp_workspace/runners/multinest_params/data/'
filename = spectrum_dir+'combo_' + combo + '_1_component.h5'

if ncomp == 1:
	phase = ph.Phase(
    	phase_name="phase_x1_combo_"+combo+'_'+param_setting,
    	profiles=af.CollectionPriorModel(
        	gaussian_0=profile_models.Gaussian
    	),
	)
elif ncomp == 2:
	phase = ph.Phase(
    	phase_name="phase_x2_combo_"+combo+'_'+param_setting,
    	profiles=af.CollectionPriorModel(
        	gaussian_0=profile_models.Gaussian, gaussian_1=profile_models.Gaussian
    	),
	)
elif ncomp == 3:
	phase = ph.Phase(
 	   phase_name="phase_x3_combo_"+combo+'_'+param_setting,
    	profiles=af.CollectionPriorModel(
        	gaussian_0=profile_models.Gaussian, gaussian_1=profile_models.Gaussian, gaussian_2=profile_models.Gaussian
    	),
	)
dataset = read_from_h5py(filename)
result = phase.run(dataset=dataset)

plt.plot(dataset.frequency, result.most_likely_model_spectrum)
plt.plot(dataset.frequency, dataset.flux)
plt.show()

result = phase.run(dataset=dataset)
model = result.most_likely_model_spectrum


print('\nBayesian evidence: ')
print(result.output.evidence)
print('\nMax Log Likelihood')
print(result.output.maximum_log_likelihood)
print('\nReduced chi squared')
print(result.analysis.get_reduced_chi_squared(model))

mp_instance = result.output.most_probable_instance

print("\nMost Probable Model:\n")
print("Centre = ", [i.center for i in mp_instance.profiles])
print("Intensity = ", [i.intensity for i in mp_instance.profiles])
print("Sigma = ", [i.sigma for i in mp_instance.profiles])