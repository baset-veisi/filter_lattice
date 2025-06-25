"""
Filter Lattice - A Python library for converting digital filters to lattice structures.

This library provides tools for converting digital filters (FIR and IIR) to lattice structures,
with comprehensive testing and visualization capabilities. It implements efficient algorithms
for filter conversion and provides utilities for filter analysis and visualization.

Main Features:
    - FIR and IIR filter conversion to lattice structures
    - Efficient implementation using NumPy
    - Comprehensive test suite with algebraic verification
    - Visualization tools for impulse and frequency responses
    - Type hints and comprehensive documentation
"""

from typing import List, Type

# Version handling
__version__ = "0.1.0"

# Core filter classes
from .filters import Filter, IIRFilter, FIRFilter
from .lattice import LatticeFilter, FIRLatticeFilter, IIRLatticeFilter, tf2lattice, tf2ltc
from .utils import FilterConversionError

# Define public API
__all__: List[str] = [
    # Core filter classes
    'Filter',
    'IIRFilter',
    'FIRFilter',
    
    # Lattice filter classes
    'LatticeFilter',
    'FIRLatticeFilter',
    'IIRLatticeFilter',
    
    # Conversion functions
    'tf2lattice',
    'tf2ltc',
    
    # Exceptions
    'FilterConversionError',
    
    # Version
    '__version__',
]

# Type hints for public API
FilterType: Type[Filter] = Filter
IIRFilterType: Type[IIRFilter] = IIRFilter
FIRFilterType: Type[FIRFilter] = FIRFilter
LatticeFilterType: Type[LatticeFilter] = LatticeFilter
FIRLatticeFilterType: Type[FIRLatticeFilter] = FIRLatticeFilter
IIRLatticeFilterType: Type[IIRLatticeFilter] = IIRLatticeFilter 