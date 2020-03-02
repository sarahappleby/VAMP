import autofit as af 
from gaussian import Gaussian
from spectrum import Spectrum
import phase as ph 
import h5py

vamp_path = '/disk2/sapple/VAMP/vamp_autofit/'

af.conf.instance = af.conf.Config(
	config_path = vamp_path + '/config/',
	output_path = vamp_path + '/output/'
	)

dataset_filename = '/disk2/sapple/VAMP/data/simple_gauss.h5'
with h5py.File(dataset_filename, 'r') as f:
	waves = f['waves'][:]
	nu = f['nu'][:]
	flux = f['flux'][:]
	noise = f['noise'][:]

phase = ph.Phase(phase_name='phase_example', gaussian=af.PriorModel(Gaussian))
dataset = Spectrum(nu, waves, flux, noise)
result = phase.run(dataset=dataset)

