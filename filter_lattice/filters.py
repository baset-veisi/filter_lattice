"""
Core filter classes for the Filter Lattice library.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Optional

class Filter(ABC):
    """Base abstract class for all digital filters."""
    
    def __init__(self, coefficients: Union[List[float], np.ndarray]):
        """
        Initialize a filter with its coefficients.
        
        Args:
            coefficients: Filter coefficients as a list or numpy array
        """
        # Convert to numpy array and remove trailing zeros
        coeffs = np.asarray(coefficients, dtype=np.float64)
        # Find the last non-zero coefficient
        last_nonzero = np.max(np.where(coeffs != 0)[0]) if np.any(coeffs != 0) else 0
        self.coefficients = coeffs[:last_nonzero + 1]
        self._validate_coefficients()
    
    @abstractmethod
    def _validate_coefficients(self) -> None:
        """Validate the filter coefficients."""
        pass
    
    @abstractmethod
    def get_order(self) -> int:
        """Get the filter order."""
        pass
    
    @abstractmethod
    def get_transfer_function(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the transfer function (numerator and denominator)."""
        pass
    
    def __str__(self) -> str:
        """String representation of the filter."""
        return f"{self.__class__.__name__}(order={self.get_order()})"

class FIRFilter(Filter):
    """Finite Impulse Response (FIR) filter implementation."""
    
    def _validate_coefficients(self) -> None:
        """Validate FIR filter coefficients."""
        if len(self.coefficients) == 0:
            raise ValueError("FIR filter must have at least one coefficient")
        if not np.any(self.coefficients != 0):
            raise ValueError("FIR filter coefficients cannot be all zero")
    
    def get_order(self) -> int:
        """Get the FIR filter order."""
        return len(self.coefficients) - 1
    
    def get_transfer_function(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the transfer function of the FIR filter.
        
        Returns:
            Tuple containing (numerator, denominator) of the transfer function
        """
        return self.coefficients, np.array([1.0])

class IIRFilter(Filter):
    """Infinite Impulse Response (IIR) filter implementation."""
    
    def _validate_coefficients(self) -> None:
        """Validate IIR filter coefficients."""
        if len(self.coefficients) == 0:
            raise ValueError("IIR filter must have at least one coefficient")
        if not np.any(self.coefficients != 0):
            raise ValueError("IIR filter coefficients cannot be all zero")
    
    def get_order(self) -> int:
        """Get the IIR filter order."""
        return len(self.coefficients) - 1
    
    def get_transfer_function(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the transfer function of the IIR filter.
        
        Returns:
            Tuple containing (numerator, denominator) of the transfer function
        """
        return np.array([1.0]), self.coefficients 