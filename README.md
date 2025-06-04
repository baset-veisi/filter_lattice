# Filter Lattice

A Python library for converting digital filters (FIR and IIR) to lattice structures.

## Features

- Support for FIR and IIR filter conversion to lattice structures
- Efficient implementation using NumPy
- Type hints and comprehensive documentation
- Extensible design for future enhancements

## Installation

```bash
pip install filter-lattice
```

## Usage

### Converting an FIR Filter to Lattice Structure

```python
from filter_lattice import FIRFilter, LatticeFilter

# Create an FIR filter
fir_coeffs = [1.0, 0.5, 0.25, 0.125]
fir_filter = FIRFilter(fir_coeffs)

# Convert to lattice structure
lattice_filter = LatticeFilter.from_fir_filter(fir_filter)

# Access reflection coefficients
print(f"Reflection coefficients: {lattice_filter.reflection_coeffs}")
```

### Converting an IIR Filter to Lattice Structure

```python
from filter_lattice import IIRFilter, LatticeFilter

# Create an IIR filter
numerator = [1.0, 0.5]
denominator = [1.0, -0.5, 0.25]
iir_filter = IIRFilter(numerator, denominator)

# Convert to lattice structure
lattice_filter = LatticeFilter.from_iir_filter(iir_filter)

# Access reflection and feedforward coefficients
print(f"Reflection coefficients: {lattice_filter.reflection_coeffs}")
print(f"Feedforward coefficients: {lattice_filter.feedforward_coeffs}")
```

## Development

The library is structured as follows:

- `filter_lattice/`
  - `__init__.py`: Package initialization and version
  - `filters.py`: Base filter classes (Filter, FIRFilter, IIRFilter)
  - `lattice.py`: Lattice filter implementation and conversion methods
  - `utils.py`: Utility functions and custom exceptions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 