"""
Lattice filter implementation and conversion utilities.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Optional
from .filters import Filter, FIRFilter, IIRFilter
from .utils import FilterConversionError

class LatticeFilter(ABC):
    """
    Abstract base class for symmetric lattice filters (FIR/IIR).
    Handles common logic and validation.
    """
    def __init__(self, reflection_coeffs: Union[List[float], np.ndarray]):
        self.reflection_coeffs = np.asarray(reflection_coeffs, dtype=np.float64)
        self.order = len(self.reflection_coeffs)
        self._validate_coeffs()

    def _validate_coeffs(self):
        if self.order == 0:
            raise ValueError("At least one reflection coefficient is required.")
        if not np.any(self.reflection_coeffs != 0):
            raise ValueError("All reflection coefficients cannot be zero.")
        if np.any(np.abs(self.reflection_coeffs) >= 1):
            raise ValueError("Reflection coefficients must be in (-1, 1) for stability.")

    @abstractmethod
    def filter(self, x: np.ndarray) -> np.ndarray:
        pass

    def get_order(self) -> int:
        return self.order

    def __str__(self):
        return f"{self.__class__.__name__}(order={self.order}, k={self.reflection_coeffs})"

class FIRLatticeFilter(LatticeFilter):
    """
    Symmetric FIR lattice filter implementation.
    """
    def filter(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        m = self.order
        y_len = n + m  # Output length
        f = np.zeros((m + 1, y_len))
        b = np.zeros((m + 1, y_len))
        f[0, :n] = x  # e_0[n] = x[n]
        b[0, :n] = x  # b_0[n] = x[n]

        for i in range(1, m + 1):
            k = self.reflection_coeffs[i - 1]
            for t in range(y_len):
                prev_b = b[i - 1, t - 1] if t - 1 >= 0 else 0.0
                f[i, t] = f[i - 1, t] - k * prev_b
                b[i, t] = -k * f[i - 1, t] + prev_b

        return f[m]

class IIRLatticeFilter(LatticeFilter):
    """
    Symmetric IIR lattice filter implementation.
    """    
    def filter(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        m = self.order
        k = self.reflection_coeffs
        b = np.zeros(m + 1, dtype=np.float64)
        y= np.zeros(n, dtype=np.float64)

        for j in range(n):
            yt = x[j]
            for i in range(m - 1, -1, -1):
                yt = yt + b[i] * k[i]
                b[i + 1] = b[i] - yt * k[i] # Exremely inefficient, we are computing one extra term for every iteration on j
            b[0] = yt
            y[j] = yt
        return y

def tf2lattice(coeffs: Union[List[float], np.ndarray], type_of_filter: str="FIR") -> Union[FIRLatticeFilter, IIRLatticeFilter]:
    """
    This function converts a transfer function to a lattice filter.
        This is a direct implementation of the Levinson-Durbin recursion.
        It is assumed that the polynomial of the FIR filter in Z-domain has this form:
        1-a1*z^-1-a2*z^-2-...-aN*z^-N = A(z), i.e. vector a should be a0=1, a1=-a1, a2=-a2, ...
        Therefore, we do such a transformation to the vector a:
        a = [1, -a1, -a2, ..., -aN]
        and then we can use the Levinson-Durbin recursion to compute the reflection coefficients.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    n = len(coeffs) - 1
    # Transform the coefficients to the form of A(z)
    if coeffs[0] == 0:
        raise ValueError("The first coefficient of the FIR/IIR filter cannot be zero. If there is a pure delay, it should be handled separately.")
    alpha = -coeffs[1:]/coeffs[0]
    k = np.zeros(n)
    for i in range(n-1, -1, -1):
        k[i] = alpha[i]
        print(f"k[{i}] = {k[i]}")
        if i > 0:
            alpha_prev = alpha.copy()
            for j in range(0, i):
                alpha[j] = (alpha_prev[j] + k[i] * alpha_prev[i-j-1]) / (1 - k[i]**2)

    if type_of_filter == "FIR":
        return FIRLatticeFilter(k)
    elif type_of_filter == "IIR":
        return IIRLatticeFilter(k)
    else:
        raise ValueError("Invalid type of filter. Must be 'FIR' or 'IIR'.")

