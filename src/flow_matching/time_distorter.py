import torch
import numpy as np
from scipy.stats import norm
from scipy.special import beta as beta_func, betaln
from scipy.stats import beta as sp_beta
from scipy.interpolate import interp1d


def beta_pdf(x, alpha, beta):
    """Beta distribution PDF."""
    # coeff = np.exp(betaln(alpha, beta))
    # return x ** (alpha - 1) * (1 - x) ** (beta - 1) / coeff
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / beta_func(alpha, beta)


def objective_function(alpha, beta, y, t):
    """Objective function to minimize (mean squared error)."""
    y_pred = beta_pdf(t, alpha, beta)
    regularization = (alpha + beta) + (1 / alpha + 1 / beta)
    error = np.mean((y - y_pred) ** 2)
    error = error + 0.0001 * regularization
    return error


class TimeDistorter:

    def __init__(
        self,
        train_distortion,
        sample_distortion,
        mu=0,
        sigma=1,
        alpha=1,
        beta=1,
    ):
        self.train_distortion = train_distortion  # used for sample_ft
        self.sample_distortion = sample_distortion  # used for get_ft
        self.alpha = alpha
        self.beta = beta
        print(
            f"TimeDistorter: train_distortion={train_distortion}, sample_distortion={sample_distortion}"
        )
        self.f_inv = None

    def train_ft(self, batch_size, device):
        t_uniform = torch.rand((batch_size, 1), device=device)
        t_distort = self.apply_distortion(t_uniform, self.train_distortion)

        return t_distort

    def sample_ft(self, t, sample_distortion):
        t_distort = self.apply_distortion(t, sample_distortion)
        return t_distort

    def fit(self, difficulty, t_array, learning_rate=0.01, iterations=1000):
        """Fit a beta distribution to data using the method of moments."""
        alpha, beta = self.alpha, self.beta
        t_array = t_array + 1e-6  # Avoid division by zero

        for _ in range(iterations):
            y_pred = beta_pdf(t_array, alpha, beta)

            # Numerical approximation of the gradients
            epsilon = 1e-5
            grad_alpha = (
                objective_function(alpha + epsilon, beta, difficulty, t_array)
                - objective_function(alpha - epsilon, beta, difficulty, t_array)
            ) / (2 * epsilon)
            grad_beta = (
                objective_function(alpha, beta + epsilon, difficulty, t_array)
                - objective_function(alpha, beta - epsilon, difficulty, t_array)
            ) / (2 * epsilon)

            # # Add regularization gradient components
            # grad_alpha += learning_rate * (1 - 1 / alpha**2)
            # grad_beta += learning_rate * (1 + 1 / beta**2)

            # Update parameters
            alpha -= learning_rate * grad_alpha
            beta -= learning_rate * grad_beta

            alpha = min(max(0.3, alpha), 3)
            beta = min(max(0.3, beta), 3)

        y_pred = beta_pdf(t_array, alpha, beta)
        self.approximate_f_inverse(alpha, beta)

        return y_pred, alpha, beta

    def approximate_f_inverse(self, alpha, beta):
        # Generate data points
        t_values = np.linspace(0, 1, 100000)
        f_values = sp_beta.cdf(t_values, alpha, beta)

        # Sort and remove duplicates
        sorted_indices = np.argsort(f_values)
        f_values_sorted = f_values[sorted_indices]
        t_values_sorted = t_values[sorted_indices]

        # Remove duplicates
        _, unique_indices = np.unique(f_values_sorted, return_index=True)
        f_values_unique = f_values_sorted[unique_indices]
        t_values_unique = t_values_sorted[unique_indices]

        # Create the interpolation function for the inverse
        f_inv = interp1d(
            f_values_unique,
            t_values_unique,
            bounds_error=False,
            fill_value="extrapolate",
        )

        self.f_inv = f_inv

    def apply_distortion(self, t, distortion_type):
        assert torch.all((t >= 0) & (t <= 1)), "t must be in the range (0, 1)"

        if distortion_type == "identity":
            ft = t
        elif distortion_type == "cos":
            ft = (1 - torch.cos(t * torch.pi)) / 2
        elif distortion_type == "revcos":
            ft = 2 * t - (1 - torch.cos(t * torch.pi)) / 2
        elif distortion_type == "polyinc":
            ft = t**2
        elif distortion_type == "polydec":
            ft = 2 * t - t**2
        elif distortion_type == "beta":
            raise ValueError(f"Unsupported for now: {distortion_type}")
        elif distortion_type == "logitnormal":
            raise ValueError(f"Unsupported for now: {distortion_type}")
        else:
            raise ValueError(f"Unknown distortion type: {distortion_type}")

        return ft
