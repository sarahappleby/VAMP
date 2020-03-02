import numpy as np
import autofit as af

class Spectrum():

	def __init__(
		self,
		frequency, 
		wavelength, 
		flux, 
		noise,
	):

		self.frequency = np.array(frequency)
		self.wavelength = np.array(wavelength)
		self.flux = np.array(flux)
		self.noise = np.array(noise)



