"""
Filter Lattice - A Python library for converting digital filters to lattice structures.
"""

__version__ = '0.1.0'

from .filters import Filter, IIRFilter, FIRFilter
from .lattice import LatticeFilter, FIRLatticeFilter, IIRLatticeFilter, tf2lattice
from .utils import FilterConversionError

__all__ = [
    'Filter', 'IIRFilter', 'FIRFilter',
    'LatticeFilter', 'FIRLatticeFilter', 'IIRLatticeFilter',
    'tf2lattice',
    'FilterConversionError'
] 