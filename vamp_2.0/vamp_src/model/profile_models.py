import numpy as np

import autofit as af

from astropy.modeling.models import Voigt1D


class Profile(af.ModelObject):

    def __init__(self, center, intensity):

        super(Profile, self).__init__()

        self.center = center
        self.intensity = intensity


class Gaussian(Profile):

    def __init__(self, center=0.0, intensity=0.1, sigma=0.01):

        super(Gaussian, self).__init__(center=center, intensity=intensity)
        #    self.center = center # Delete this, the super method passes center 'up' such that all Pofiles have a centre
        #    self.intensity = intensity # The same happens with the intensity.
        self.sigma = sigma

    def model_from_frequencies(self, frequencies):
        return 1.0 - self.intensity * np.exp(
            -0.5 * ((frequencies - self.center) / self.sigma) ** 2
        )


class Voigt(Profile):
    def __init__(self, center=0.0, intensity=0.1, fwhm_L=0.01, fwhm_G=0.01):

        super(Voigt, self).__init__(center=center, intensity=intensity)

        # TODO : You should rename these to all lower case and a more readable and dsecriptive name,
        # TODO : e.g. fwhm_L -> full_width_half_max_lower

        self.fwhm_L = fwhm_L
        self.fwhm_G = fwhm_G

    def model_from_frequencies(self, frequencies):

        v = Voigt1D(
            x_0=self.center,
            amplitude_L=self.intensity,
            fwhm_L=self.fwhm_L,
            fwhm_G=self.fwhm_G,
        )
        return 1. - v(frequencies)
