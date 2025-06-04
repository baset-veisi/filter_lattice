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
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
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
    
    def __init__(self, denominator: Union[List[float], np.ndarray]):
        """
        Initialize an IIR filter with denominator coefficients.
        
        Args:
            denominator: Denominator coefficients (A(z))
        """
        self.denominator = np.asarray(denominator, dtype=np.float64)
        super().__init__(self.denominator)  # Store denominator as main coefficients
        self._validate_coefficients()
    
    def _validate_coefficients(self) -> None:
        """Validate IIR filter coefficients."""
        if len(self.denominator) == 0:
            raise ValueError("IIR filter must have at least one denominator coefficient")
        if self.denominator[0] == 0:
            raise ValueError("First denominator coefficient cannot be zero")
    
    def get_order(self) -> int:
        """Get the IIR filter order."""
        return len(self.denominator) - 1
    
    def get_transfer_function(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the transfer function of the IIR filter.
        
        Returns:
            Tuple containing (numerator, denominator) of the transfer function
        """
        return np.array([1.0]), self.denominator 