import torch

from .base import BaseGenerator


class UniformGenerator(BaseGenerator):
    """Generate samples from a uniform distribution.

    Parameters
    ----------
    low : float
        Lower bound of the support.
    high : float
        Upper bound of the support.

    Raises
    ------
    ValueError
        If `high` is not greater than `low`.
    """

    def __init__(self, low: float, high: float):
        if high <= low:
            raise ValueError(f"high must be greater than low. Got {low} and {high}.")

        self.low = low
        self.high = high
        return

    def generate(self, n_samples: int) -> torch.Tensor:
        return self.low + (self.high - self.low) * torch.rand(n_samples)
