import autofit as af
from vamp_src.dataset.spectrum import Spectrum
from vamp_src.fit.fit import DatasetFit


class Analysis(af.Analysis):
    def __init__(self, dataset: Spectrum):

        self.dataset = dataset

    def fit(self, instance):
        model_spectrum = self.model_spectrum_from_instance(instance=instance)
        spec_fit = self.fit_from_model_spectrum(model_spectrum=model_spectrum)
        return spec_fit.likelihood

    def model_spectrum_from_instance(self, instance):
        return sum(
            list(
                map(
                    lambda gaussian: gaussian.model_from_frequencies(
                        self.dataset.frequency
                    ),
                    instance.gaussians,
                )
            )
        )

    def fit_from_model_spectrum(self, model_spectrum):
        return DatasetFit(
            data=self.dataset.flux,
            noise_map=self.dataset.noise,
            model_data=model_spectrum,
        )

    def visualize(self, instance, during_analysis):

        # Visualization will be covered in tutorial 4.

        pass
