"""
Filter Lattice - A Python library for converting digital filters to lattice structures.
"""

__version__ = '0.1.0'

from .filters import Filter, IIRFilter, FIRFilter
from .lattice import LatticeFilter
from .utils import FilterConversionError

__all__ = ['Filter', 'IIRFilter', 'FIRFilter', 'LatticeFilter', 'FilterConversionError'] 