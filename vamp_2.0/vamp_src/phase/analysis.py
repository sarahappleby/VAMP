import autofit as af
from vamp_src.dataset.spectrum import Spectrum
from vamp_src.fit.fit import DatasetFit
from vamp_src.phase import visualizer 
import numpy as np


class Analysis(af.Analysis):
    def __init__(self, dataset: Spectrum, visualize_path=None):

        self.dataset = dataset

        self.visualizer = visualizer.Visualizer(
            dataset=self.dataset, visualize_path=visualize_path
        )

    def fit(self, instance):
        model_spectrum = self.model_spectrum_from_instance(instance=instance)
        spec_fit = self.fit_from_model_spectrum(model_spectrum=model_spectrum)
        return spec_fit.likelihood

    def model_spectrum_from_instance(self, instance):
        return sum(
            list(
                map(
                    lambda profile: profile.model_from_frequencies(
                        self.dataset.frequency
                    ),
                    instance.profiles,
                )
            )
        )

    def fit_from_model_spectrum(self, model_spectrum):
        return DatasetFit(
            data=self.dataset.flux,
            noise_map=self.dataset.noise,
            model_data=model_spectrum,
        )

    def get_dof(self, instance):
        return len(self.dataset.flux) - 'dimensionality' 

    def get_reduced_chi_squared(self, instance):
        #dof = self.get_dof(instance=instance)
        dof = 4
        fit = self.model_spectrum_from_instance(instance=instance)

        deviations = (self.dataset.flux - fit)**2.
        chi_squared = np.sum(deviations / (self.dataset.noise**2.))
        return chi_squared / dof

    def visualize(self, instance, during_analysis):

        # Visualization will be covered in tutorial 4.

        fit = self.model_spectrum_from_instance(instance=instance)
        n_components = len(instance.profiles)
        reduced_chi_squared = self.get_reduced_chi_squared(instance=instance)

        self.visualizer.visualize_fit(fit=fit, n_components=n_components, 
                                      reduced_chi_squared=reduced_chi_squared, 
                                      during_analysis=during_analysis)
