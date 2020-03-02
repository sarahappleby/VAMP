import autoarray as aa
import numpy as np
from spectrum import Spectrum

class Gaussian:

	def __init__(
		self, 
		center=0.0, 
		intensity=0.1, 
		sigma=0.01,
	):

		self.center = center
		self.intensity = intensity
		self.sigma = sigma

	def model_from_dataset(self, dataset: Spectrum):
		self.model = self.intensity * np.exp(-0.5 * ((dataset.frequency - self.center) / self.sigma) ** 2)
		
