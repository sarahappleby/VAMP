import autofit as af
import os

import sys
sys.path.append('/disk2/sapple/VAMP/vamp_2.0')
from vamp_src.model import profiles
from vamp_src.dataset.spectrum import Spectrum
import vamp_src.phase.phase as ph

# TODO : Relative path makes our life easier.

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))

# Setup the path to the config folder, using the autolens_workspace path.
config_path = workspace_path + "config"

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

from vamp_workspace.make_data import FakeGauss

fakeGaussA = FakeGauss(center=-1.0, sigma=2.0, intensity=0.5)
fakeGaussB = FakeGauss(center=1.5, sigma=1.0, intensity=1.0)

fakeGauss_2comp = fakeGaussB.gauss + fakeGaussA.gauss + fakeGaussA.noise
dataset = Spectrum(fakeGaussA.x, fakeGaussA.x, 1.0 - fakeGauss_2comp, fakeGaussA.noise)

phase = ph.Phase(
    phase_name="phase_x2_gaussians",
    gaussians=af.CollectionPriorModel(
        gaussian_0=profiles.Gaussian, gaussian_1=profiles.Gaussian
    ),
)
result = phase.run(dataset=dataset)



# We also have an 'output' attribute, which in this case is a MultiNestOutput object:
print(result.output)
# This object acts as an interface between the MultiNest output results on your hard-disk and this Python code. For
# example, we can use it to get the evidence estimated by MultiNest.
print(result.output.evidence)
# We can also use it to get a model-instance of the "most probable" model, which is the model where each parameter is
# the value estimated from the probability distribution of parameter space.
mp_instance = result.output.most_probable_instance
print()
print("Most Probable Model:\n")
print("Centre = ", [i.center for i in mp_instance.gaussians])
print("Intensity = ", [i.intensity for i in mp_instance.gaussians])
print("Sigma = ", [i.sigma for i in mp_instance.gaussians])

# dataset_filename = '/disk2/sapple/VAMP/data/simple_gauss.h5'
# with h5py.File(dataset_filename, 'r') as f:
# 	waves = f['waves'][:]
# 	nu = f['nu'][:]
# 	flux = f['flux'][:]
# 	noise = f['noise'][:]
