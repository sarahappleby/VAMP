import matplotlib.pyplot as plt

class Visualizer(): # do i need the abstract class?
	
	def __init__(self, dataset, visualize_path):

		self.dataset = dataset
		self.visualize_path = visualize_path

	def visualize_fit(self, fit, n_components, during_analysis):

		plt.plot(self.dataset.frequency, self.dataset.flux, c='k', label='Data')
		plt.plot(self.dataset.frequency, fit, c='b', ls='--', label='fit, %g components' % n_components,)
		plt.legend()
		plt.xlabel('Frequency')
		plt.ylabel('Flux')
		plt.savefig(self.visualize_path + 'fit.png')
		plt.clf()

