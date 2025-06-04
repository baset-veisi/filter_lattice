"""
Utility functions and custom exceptions for the Filter Lattice library.
"""

class FilterConversionError(Exception):
    """Exception raised for errors during filter conversion."""
    pass

def normalize_coefficients(coefficients: list) -> list:
    """
    Normalize filter coefficients by dividing by the first coefficient.
    
    Args:
        coefficients: List of filter coefficients
        
    Returns:
        Normalized coefficients
    """
    if not coefficients:
        raise ValueError("Empty coefficient list")
    return [c / coefficients[0] for c in coefficients]

def check_stability(reflection_coeffs: list) -> bool:
    """
    Check if the lattice filter is stable by verifying reflection coefficients.
    
    Args:
        reflection_coeffs: List of reflection coefficients
        
    Returns:
        True if the filter is stable, False otherwise
    """
    return all(abs(k) < 1.0 for k in reflection_coeffs) 