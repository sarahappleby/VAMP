import autofit as af

class Result(af.Result):
    def __init__(self, instance, likelihood, analysis):
        """
        The result of a non-linear search.

        Parameters
        ----------
        instance: autofit.mapper.model.ModelInstance
            A model instance comprising the model instances that gave the highest likelihood fit.
        figure_of_merit: float
            A value indicating the figure of merit (e.g. the likelihood) given by the highest likelihood fit.
        """
        self.instance = instance
        self.likelihood = likelihood
        self.analysis = analysis

    @property
    def most_likely_model_spectrum(self):
        return self.instance.gaussian.model_from_dataset(
            dataset=self.analysis.dataset
        )

    @property
    def most_likely_fit(self):
        return self.analysis.fit_from_model_spectrum(
            model_spectrum=self.most_likely_model_spectrum
        )