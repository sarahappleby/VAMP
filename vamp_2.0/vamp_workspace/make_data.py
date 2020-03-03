import numpy as np


class FakeGauss:
    def __init__(
        self,
        center=0.0,
        intensity=1.0,
        sigma=1.0,
        x_min=-5.0,
        x_max=5.0,
        n_points=100,
        snr=50,
    ):
        self.center = center
        self.intensity = intensity
        self.sigma = sigma
        self.x_max = x_max
        self.x_min = x_min
        self.n_points = n_points
        self.snr = snr

        # make the x array
        self.dx = np.abs(self.x_max - self.x_min) / self.n_points
        self.x = np.arange(self.x_min, self.x_max, self.dx)

        # make the gaussian
        self.gauss = self.intensity * np.exp(
            -0.5 * ((self.x - self.center) / self.sigma) ** 2
        )

        # generate noise
        self.noise = np.random.normal(0.0, 1.0 / self.snr, self.n_points)
        self.noisy_gauss = self.gauss + self.noise
