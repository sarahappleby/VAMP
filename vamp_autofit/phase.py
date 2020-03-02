import autofit as af
from spectrum import Spectrum
from analysis import Analysis
from result import Result


class Phase(af.AbstractPhase):

    gaussian = af.PhaseProperty("gaussian")

    Result = Result

    @af.convert_paths
    def __init__(
        self,
        paths,
        gaussian,
        optimizer_class=af.MultiNest,
    ):

        super().__init__(paths=paths, optimizer_class=optimizer_class)
        self.gaussian = gaussian

    def run(self, dataset: Spectrum):
        """
        Pass a dataset to the phase, running the phase and non-linear search.

        Parameters
        ----------
        dataset: aa.Imaging
            The dataset fitted by the phase, which in this case is a PyAutoArray imaging object.

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model.
        """

        analysis = self.make_analysis(dataset=dataset)

        result = self.run_analysis(analysis=analysis)

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, dataset):
        """
        Create an Analysis object, which creates the dataset and contains the functions which perform the fit.

        Parameters
        ----------
        dataset: aa.Imaging
            The dataset fitted by the phase, which in this case is a PyAutoArray imaging object.

        Returns
        -------
        analysis : Analysis
            An analysis object that the non-linear search calls to determine the fit likelihood for a given model
            instance.
        """
        return Analysis(dataset=dataset)

    def make_result(self, result, analysis):
        return self.Result(
            instance=result.instance,
            likelihood=result.likelihood,
            analysis=analysis,
        )