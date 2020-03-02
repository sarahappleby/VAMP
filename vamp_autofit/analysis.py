import autofit as af
from spectrum import Spectrum
from fit import DatasetFit

class Analysis(af.Analysis):
    def __init__(self, dataset: Spectrum):

        self.dataset = dataset

    def fit(self, instance):
        model_spectrum = instance.gaussian.model_from_dataset(self.dataset)
        fit = self.fit_from_model_spectrum(model_spectrum=model_spectrum)
        return fit.likelihood

    def fit_from_model_spectrum(self, model_spectrum):
        return DatasetFit(data=self.dataset, noise_map=self.dataset.noise, model_data=model_spectrum)

    def visualize(self, instance, during_analysis):

        # Visualization will be covered in tutorial 4.

        pass