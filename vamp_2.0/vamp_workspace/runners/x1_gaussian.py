import autofit as af
import os
import matplotlib.pyplot as plt

import sys
sys.path.append('/disk2/sapple/VAMP/vamp')
from vamp_src.model import profiles
from vamp_src.dataset.spectrum import Spectrum
import vamp_src.phase.phase as ph

# TODO : Relative path makes our life easier.

# Setup the path to the vamp_workspace, using a relative directory name.
workspace_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))

# Setup the path to the config folder, using the vamp_workspace path.
config_path = workspace_path + "config"

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

from vamp_workspace.make_data import FakeGauss

fakeGauss = FakeGauss()

phase = ph.Phase(phase_name="phase_x1_gaussians",
                 gaussians=af.CollectionPriorModel(gaussian_0=profiles.Gaussian))

dataset = Spectrum(
    fakeGauss.x, fakeGauss.x, 1.0 - fakeGauss.noisy_gauss, fakeGauss.noise
)
result = phase.run(dataset=dataset)

plt.plot(dataset.frequency, result.most_likely_model_spectrum)
plt.plot(dataset.frequency, dataset.flux)
plt.show()