"""
Lattice filter implementation and conversion utilities.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Optional
from .filters import Filter, FIRFilter, IIRFilter
from .utils import FilterConversionError
import warnings
import matplotlib.pyplot as plt

class LatticeFilter(ABC):
    """
    Abstract base class for symmetric lattice filters (FIR/IIR).
    Handles common logic and validation.
    """
    def __init__(self, reflection_coeffs: Union[List[float], np.ndarray], dtype: np.dtype = np.float64):
        self.reflection_coeffs = np.asarray(reflection_coeffs, dtype=dtype)
        self.order = len(self.reflection_coeffs)
        self._validate_coeffs()

    def _validate_coeffs(self):
        if self.order == 0:
            raise ValueError("At least one reflection coefficient is required.")
        if not np.any(self.reflection_coeffs != 0):
            raise ValueError("All reflection coefficients cannot be zero.")
        # Do not check |k| >= 1 here; handle in subclasses

    @abstractmethod
    def filter(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def tf(self, stage: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the transfer function coefficients.
        
        Args:
            stage: Optional stage number up to which to calculate the transfer function.
                  If None, uses all stages (order of the filter).
                  
        Returns:
            Tuple of (numerator, denominator) polynomials.
        """
        pass

    def get_order(self) -> int:
        return self.order

    def __str__(self):
        return f"{self.__class__.__name__}(order={self.order}, k={self.reflection_coeffs})"

    def plot(self):
        N = self.order
        k = self.reflection_coeffs
        fig, ax = plt.subplots(figsize=(2*N+2, 4))
        ax.axis('off')
        # Draw e-path (top)
        for i in range(N+1):
            ax.plot([i, i+1], [2, 2], 'k-', lw=2)
            ax.text(i+0.5, 2.15, f"e{i}[n]", ha='center', va='bottom', fontsize=12)
        ax.text(0, 2.3, 'x[n]', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(N+1, 2.3, 'y[n]', ha='center', va='bottom', fontsize=12, fontweight='bold')
        # Draw b-path (bottom)
        for i in range(N+1):
            ax.plot([i, i+1], [0, 0], 'k-', lw=2)
            ax.text(i+0.5, -0.15, f"b{i}[n]", ha='center', va='top', fontsize=12)
        # Draw verticals and diagonals
        for i in range(1, N+1):
            # Vertical from e to b
            ax.plot([i, i], [2, 0], 'k:', lw=1)
            # Diagonal from b[i-1] to e[i]
            ax.annotate('', xy=(i,2), xytext=(i-1,0), arrowprops=dict(arrowstyle='->', lw=1, color='tab:blue'))
            # Diagonal from e[i-1] to b[i]
            ax.annotate('', xy=(i,0), xytext=(i-1,2), arrowprops=dict(arrowstyle='->', lw=1, color='tab:red'))
            # Reflection coefficient (actual value)
            ax.text(i-0.5, 1, f"{k[i-1]:+.3f}", ha='center', va='center', fontsize=12, color='tab:blue')
            # Delay
            ax.text(i-0.5, -0.5, "z$^{-1}$", ha='center', va='center', fontsize=12, color='tab:gray')
        plt.ylim(-1, 3)
        plt.xlim(-0.5, N+1.5)
        plt.title('FIR Lattice Structure', fontsize=14)
        plt.show()

class FIRLatticeFilter(LatticeFilter):
    """
    Symmetric FIR lattice filter implementation.
    """
    def _validate_coeffs(self):
        if self.order == 0:
            raise ValueError("At least one reflection coefficient is required.")
        if not np.any(self.reflection_coeffs != 0):
            raise ValueError("All reflection coefficients cannot be zero.")
        # No |k| >= 1 check for FIR

    def filter(self, x: np.ndarray, stage: Optional[int] = None) -> np.ndarray:
        """
        Filter the input signal x using the FIR lattice filter.
        The outpus has length n + m, where n is the length of the input signal and m is the order of the filter.
        The stage parameter is the number of stages to apply the lattice filter to.
        If stage is not provided, the filter is applied to all stages.
        """
        if stage is None:
            stage = self.order
        n = len(x)
        m = stage
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

    def tf(self, stage: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the transfer function coefficients for the FIR lattice filter.
        
        Args:
            stage: Optional stage number up to which to calculate the transfer function.
                  If None, uses all stages (order of the filter).
                  
        Returns:
            Tuple of (numerator, denominator) polynomials.
            For FIR, denominator is always [1.0].
        """
        if stage is None:
            stage = self.order
        elif stage < 0 or stage > self.order:
            raise ValueError(f"Stage must be between 0 and {self.order}")
            
        n = stage
        k = self.reflection_coeffs[:n]  # Only use coefficients up to the specified stage
        
        # Initialize polynomials
        a = np.zeros(n + 1)
        a[0] = 1.0  # a[0] is always 1
        
        # Forward recursion to get transfer function coefficients
        for i in range(n):
            a_prev = a.copy()
            for j in range(1, i + 2):
                a[j] = a_prev[j] - k[i] * a_prev[i + 1 - j]
        
        return a, np.array([1.0])

    def plot(self, title: str = "FIR Lattice Structure"):
        N = self.order
        k = self.reflection_coeffs
        fig, ax = plt.subplots(figsize=(2*N+2, 4))
        ax.axis('off')
        # Draw e-path (top, right to left)
        for i in range(N, -1, -1):
            ax.plot([N-i, N-i+1], [2, 2], 'k-', lw=2)
            ax.text(N-i+0.5, 2.15, f"e{i}[n]", ha='center', va='bottom', fontsize=12)
        ax.text(0, 2.3, f'x[n]=e{N}[n]', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(N+1, 2.3, 'y[n]', ha='center', va='bottom', fontsize=12, fontweight='bold')
        # Draw b-path (bottom, right to left)
        for i in range(N, -1, -1):
            ax.plot([N-i, N-i+1], [0, 0], 'k-', lw=2)
            ax.text(N-i+0.5, -0.15, f"b{i}[n]", ha='center', va='top', fontsize=12)
        # Draw verticals and diagonals
        for i in range(N, 0, -1):
            idx = N-i+1
            # Vertical from e to b
            ax.plot([idx, idx], [2, 0], 'k:', lw=1)
            # Diagonal from b[i] to e[i-1]
            ax.annotate('', xy=(idx,2), xytext=(idx-1,0), arrowprops=dict(arrowstyle='->', lw=1, color='tab:blue'))
            # Diagonal from e[i] to b[i-1]
            ax.annotate('', xy=(idx,0), xytext=(idx-1,2), arrowprops=dict(arrowstyle='->', lw=1, color='tab:red'))
            # Reflection coefficients (actual value)
            ax.text(idx-0.5, 1.2, f"+{k[i-1]:.3f}", ha='center', va='center', fontsize=12, color='tab:blue')
            ax.text(idx-0.5, 0.8, f"-{k[i-1]:.3f}", ha='center', va='center', fontsize=12, color='tab:red')
            # Delay
            ax.text(idx-0.5, -0.5, "z$^{-1}$", ha='center', va='center', fontsize=12, color='tab:gray')
        plt.ylim(-1, 3)
        plt.xlim(-0.5, N+1.5)
        plt.title(title, fontsize=14)
        plt.show()

class IIRLatticeFilter(LatticeFilter):
    """
    Symmetric IIR lattice filter implementation.
    """    
    def _validate_coeffs(self):
        if self.order == 0:
            raise ValueError("At least one reflection coefficient is required.")
        if not np.any(self.reflection_coeffs != 0):
            raise ValueError("All reflection coefficients cannot be zero.")
        if np.any(np.abs(self.reflection_coeffs) >= 1):
            warnings.warn(
                "At least one reflection coefficient has |k| >= 1. The filter may be unstable (one or more poles may be outside the unit circle).",
                UserWarning
            )

    def filter(self, x: np.ndarray, stage: Optional[int] = None) -> np.ndarray:
        """
        Filter the input signal x using the IIR lattice filter.
        The outpus has length n, where n is the length of the input signal.
        The stage parameter is the number of stages to apply the lattice filter to.
        If stage is not provided, the filter is applied to all stages.
        """
        if stage is None:
            stage = self.order
        n = len(x)
        m = stage
        k = self.reflection_coeffs
        b = np.zeros(m + 1, dtype=np.float64)
        y = np.zeros(n, dtype=np.float64)

        for j in range(n):
            yt = x[j]
            for i in range(m - 1, -1, -1):
                yt = yt + b[i] * k[i]
                b[i + 1] = b[i] - yt * k[i]
            b[0] = yt
            y[j] = yt
        return y

    def tf(self, stage: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the transfer function coefficients for the IIR lattice filter.
        
        Args:
            stage: Optional stage number up to which to calculate the transfer function.
                  If None, uses all stages (order of the filter).
                  
        Returns:
            Tuple of (numerator, denominator) polynomials.
            For IIR, numerator is always [1.0].
        """
        if stage is None:
            stage = self.order
        elif stage < 0 or stage > self.order:
            raise ValueError(f"Stage must be between 0 and {self.order}")
            
        n = stage
        k = self.reflection_coeffs[:n]  # Only use coefficients up to the specified stage
        
        # Initialize polynomials
        a = np.zeros(n + 1)
        a[0] = 1.0  # a[0] is always 1
        
        # Forward recursion to get transfer function coefficients
        for i in range(n):
            a_prev = a.copy()
            for j in range(1, i + 2):
                a[j] = a_prev[j] - k[i] * a_prev[i + 1 - j]
        
        return np.array([1.0]), a

    def is_stable(self) -> bool:
        """
        Check if the IIR lattice filter is stable.
        """
        return np.all(np.abs(self.reflection_coeffs) < 1)

    def plot(self, title: str = "IIR Lattice Structure"):
        N = self.order
        k = self.reflection_coeffs
        fig, ax = plt.subplots(figsize=(2*N+2, 4))
        ax.axis('off')
        # Draw e-path (top, right to left)
        for i in range(N, -1, -1):
            ax.plot([N-i, N-i+1], [2, 2], 'k-', lw=2)
            ax.text(N-i+0.5, 2.15, f"e{i}[n]", ha='center', va='bottom', fontsize=12)
        ax.text(0, 2.3, f'x[n]=e{N}[n]', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(N+1, 2.3, 'y[n]', ha='center', va='bottom', fontsize=12, fontweight='bold')
        # Draw b-path (bottom, right to left)
        for i in range(N, -1, -1):
            ax.plot([N-i, N-i+1], [0, 0], 'k-', lw=2)
            ax.text(N-i+0.5, -0.15, f"b{i}[n]", ha='center', va='top', fontsize=12)
        # Draw verticals and diagonals
        for i in range(N, 0, -1):
            idx = N-i+1
            # Vertical from e to b
            ax.plot([idx, idx], [2, 0], 'k:', lw=1)
            # Diagonal from b[i] to e[i-1]
            ax.annotate('', xy=(idx,2), xytext=(idx-1,0), arrowprops=dict(arrowstyle='->', lw=1, color='tab:blue'))
            # Diagonal from e[i] to b[i-1]
            ax.annotate('', xy=(idx,0), xytext=(idx-1,2), arrowprops=dict(arrowstyle='->', lw=1, color='tab:red'))
            # Reflection coefficients (actual value)
            ax.text(idx-0.5, 1.2, f"+{k[i-1]:.3f}", ha='center', va='center', fontsize=12, color='tab:blue')
            ax.text(idx-0.5, 0.8, f"-{k[i-1]:.3f}", ha='center', va='center', fontsize=12, color='tab:red')
            # Delay
            ax.text(idx-0.5, -0.5, "z$^{-1}$", ha='center', va='center', fontsize=12, color='tab:gray')
        plt.ylim(-1, 3)
        plt.xlim(-0.5, N+1.5)
        plt.title(title, fontsize=14)
        plt.show()

def tf2lattice(coeffs: Union[List[float], np.ndarray], type_of_filter: str="FIR", dtype: np.dtype = np.float64) -> Union[FIRLatticeFilter, IIRLatticeFilter]:
    """
    This function converts a transfer function to a lattice filter.
        This is a direct implementation of the Levinson-Durbin recursion.
        It is assumed that the polynomial of the FIR filter in Z-domain has this form:
        1-a1*z^-1-a2*z^-2-...-aN*z^-N = A(z), i.e. vector a should be a0=1, a1=-a1, a2=-a2, ...
        Therefore, we do such a transformation to the vector a:
        a = [1, -a1, -a2, ..., -aN]
        and then we can use the Levinson-Durbin recursion to compute the reflection coefficients.
    """
    coeffs = np.asarray(coeffs, dtype=dtype)
    n = len(coeffs) - 1
    # Transform the coefficients to the form of A(z)
    if coeffs[0] == 0:
        raise ValueError("The first coefficient of the FIR/IIR filter cannot be zero. If there is a pure delay, it should be handled separately.")
    alpha = -coeffs[1:]/coeffs[0]
    k = np.zeros(n)
    for i in range(n-1, -1, -1):
        k[i] = alpha[i]
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

