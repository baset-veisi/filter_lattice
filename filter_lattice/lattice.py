"""
Lattice filter implementation and conversion utilities.
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from .filters import Filter, FIRFilter, IIRFilter
from .utils import FilterConversionError

class LatticeFilter:
    """
    Lattice filter structure implementation.
    
    This class implements both FIR and IIR lattice structures and provides
    methods for converting from standard filter forms to lattice structures.
    """
    
    def __init__(self, reflection_coeffs: Union[List[float], np.ndarray],
                 feedforward_coeffs: Optional[Union[List[float], np.ndarray]] = None):
        """
        Initialize a lattice filter with reflection coefficients.
        
        Args:
            reflection_coeffs: Reflection coefficients (k-parameters)
            feedforward_coeffs: Feedforward coefficients (for IIR lattice)
        """
        self.reflection_coeffs = np.asarray(reflection_coeffs, dtype=np.float64)
        self.feedforward_coeffs = (np.asarray(feedforward_coeffs, dtype=np.float64) 
                                 if feedforward_coeffs is not None else None)
        self._validate_coefficients()
    
    def _validate_coefficients(self) -> None:
        """Validate lattice filter coefficients."""
        if len(self.reflection_coeffs) == 0:
            raise ValueError("Lattice filter must have at least one reflection coefficient")
        if self.feedforward_coeffs is not None:
            if len(self.feedforward_coeffs) != len(self.reflection_coeffs):
                raise ValueError("Feedforward coefficients must match reflection coefficients length")
    
    def apply_fir_lattice(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply FIR lattice filter to input signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Filtered signal
        """
        if self.feedforward_coeffs is not None:
            raise ValueError("This is an IIR lattice filter. Use apply_iir_lattice instead.")
            
        n = len(signal)
        m = len(self.reflection_coeffs)
        
        # Initialize forward and backward prediction errors
        f = np.zeros((m + 1, n))
        b = np.zeros((m + 1, n))
        
        # Set initial forward prediction error to input signal
        f[0] = signal
        
        # Lattice filter computation
        for i in range(m):
            k = self.reflection_coeffs[i]
            f[i + 1] = f[i] - k * np.roll(b[i], 1)
            b[i + 1] = b[i] - k * np.roll(f[i], 1)
        
        # For FIR lattice, output is the last forward prediction error
        return f[-1]
    
    def apply_iir_lattice(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply IIR lattice filter to input signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Filtered signal
        """
        if self.feedforward_coeffs is None:
            raise ValueError("This is an FIR lattice filter. Use apply_fir_lattice instead.")
            
        n = len(signal)
        m = len(self.reflection_coeffs)
        
        # Initialize forward and backward prediction errors
        f = np.zeros((m + 1, n))
        b = np.zeros((m + 1, n))
        
        # Set initial forward prediction error to input signal
        f[0] = signal
        
        # Lattice filter computation
        for i in range(m):
            k = self.reflection_coeffs[i]
            f[i + 1] = f[i] - k * np.roll(b[i], 1)
            b[i + 1] = b[i] - k * np.roll(f[i], 1)
        
        # For IIR lattice, compute output using feedforward coefficients
        output = np.zeros(n)
        for i in range(m + 1):
            output += self.feedforward_coeffs[i] * b[i]
            
        return output
    
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply the lattice filter to input signal.
        Automatically selects FIR or IIR implementation based on filter type.
        
        Args:
            signal: Input signal array
            
        Returns:
            Filtered signal
        """
        if self.feedforward_coeffs is None:
            return self.apply_fir_lattice(signal)
        else:
            return self.apply_iir_lattice(signal)
    
    @classmethod
    def from_fir_filter(cls, fir_filter: FIRFilter) -> 'LatticeFilter':
        """
        Convert an FIR filter to a lattice structure.
        
        Args:
            fir_filter: FIR filter to convert
            
        Returns:
            LatticeFilter instance
        """
        if not isinstance(fir_filter, FIRFilter):
            raise FilterConversionError("Input must be an FIR filter")
        
        # Get filter coefficients
        coeffs = fir_filter.coefficients
        
        # Initialize arrays for reflection coefficients
        n = len(coeffs) - 1
        k = np.zeros(n)
        
        # Compute reflection coefficients using Levinson-Durbin recursion
        a = coeffs.copy()
        for i in range(n-1, -1, -1):
            k[i] = a[i+1]
            if i > 0:
                a_prev = a.copy()
                for j in range(1, i+1):
                    a[j] = (a_prev[j] - k[i] * a_prev[i-j+1]) / (1 - k[i]**2)
        
        return cls(k)
    
    @classmethod
    def from_iir_filter(cls, iir_filter: IIRFilter) -> 'LatticeFilter':
        """
        Convert an IIR filter to a lattice structure.
        
        Args:
            iir_filter: IIR filter to convert
            
        Returns:
            LatticeFilter instance
        """
        if not isinstance(iir_filter, IIRFilter):
            raise FilterConversionError("Input must be an IIR filter")
        
        # Get numerator and denominator coefficients
        num, den = iir_filter.get_transfer_function()
        
        # Normalize denominator
        den = den / den[0]
        
        # Compute reflection coefficients using Schur-Cohn algorithm
        n = len(den) - 1
        k = np.zeros(n)
        a = den.copy()
        
        for i in range(n-1, -1, -1):
            k[i] = a[i+1]
            if i > 0:
                a_prev = a.copy()
                for j in range(1, i+1):
                    a[j] = (a_prev[j] - k[i] * a_prev[i-j+1]) / (1 - k[i]**2)
        
        # Compute feedforward coefficients
        v = np.zeros(n+1)
        v[0] = num[0]
        for i in range(1, min(len(num), n+1)):
            v[i] = num[i]
            for j in range(i):
                v[i] -= v[j] * den[i-j]
        
        return cls(k, v)
    
    def get_order(self) -> int:
        """Get the lattice filter order."""
        return len(self.reflection_coeffs)
    
    def __str__(self) -> str:
        """String representation of the lattice filter."""
        return (f"LatticeFilter(order={self.get_order()}, "
                f"has_feedforward={self.feedforward_coeffs is not None})") 