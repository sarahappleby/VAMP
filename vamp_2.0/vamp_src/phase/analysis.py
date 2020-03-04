import autofit as af
from vamp_src.dataset.spectrum import Spectrum
from vamp_src.fit.fit import DatasetFit
from vamp_src.phase import visualizer 


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

        fit = self.model_spectrum_from_instance(instance=instance)
        n_components = len(instance.gaussians)
        self.visualizer.visualize_fit(fit=fit, n_components=n_components, during_analysis=during_analysis)
