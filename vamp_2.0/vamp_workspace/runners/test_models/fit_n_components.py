import autofit as af
import os
import sys
sys.path.append('/disk2/sapple/VAMP/vamp_2.0')
from vamp_src.model import profile_models
from vamp_src.dataset.spectrum import Spectrum
import vamp_src.phase.phase as ph

ncomp = 1.

workspace_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))
config_path = workspace_path + "config"
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

from vamp_workspace.make_data import FakeGauss

fakeGaussA = FakeGauss(center=-1.0, sigma=2.0, intensity=0.5)
fakeGaussB = FakeGauss(center=1.5, sigma=1.0, intensity=1.0)
fakeGaussC = FakeGauss(center=3., sigma=0.5, intensity=0.7)

fakeGauss_3comp = fakeGaussB.gauss + fakeGaussA.gauss + fakeGaussC.gauss + fakeGaussA.noise
dataset = Spectrum(fakeGaussA.x, fakeGaussA.x, 1.0 - fakeGauss_2comp, fakeGaussA.noise)

if ncomp == 1:
	phase = ph.Phase(
    	phase_name="phase_x2_gaussians",
    	profiles=af.CollectionPriorModel(
        	gaussian_0=profile_models.Gaussian
    	),
	)
elif ncomp == 2:
	phase = ph.Phase(
    	phase_name="phase_x2_gaussians",
    	profiles=af.CollectionPriorModel(
        	gaussian_0=profile_models.Gaussian, gaussian_1=profile_models.Gaussian
    	),
	)
elif ncomp == 3:
	phase = ph.Phase(
 	   phase_name="phase_x2_gaussians",
    	profiles=af.CollectionPriorModel(
        	gaussian_0=profile_models.Gaussian, gaussian_1=profile_models.Gaussian, gaussian_2=profile_models.Gaussian
    	),
	)

result = phase.run(dataset=dataset)

plt.plot(dataset.frequency, result.most_likely_model_spectrum)
plt.plot(dataset.frequency, dataset.flux)
plt.show()

print(result.output.evidence)
mp_instance = result.output.most_probable_instance
print()
print("Most Probable Model:\n")
print("Centre = ", [i.center for i in mp_instance.profiles])
print("Intensity = ", [i.intensity for i in mp_instance.profiles])
print("Sigma = ", [i.sigma for i in mp_instance.profiles])