# Filter Lattice

A Python library for converting digital filters (FIR and IIR) to lattice structures, with comprehensive testing and visualization capabilities.

## Features

- Support for FIR and IIR filter conversion to lattice structures
- Efficient implementation using NumPy
- Comprehensive test suite with algebraic verification
- Visualization tools for impulse and frequency responses
- Type hints and comprehensive documentation
- Extensible design for future enhancements

## Installation

```bash
pip install filter-lattice
```

## Usage

### Converting an FIR Filter to Lattice Structure

```python
from filter_lattice import FIRFilter, FIRLatticeFilter, tf2lattice
import numpy as np

# Create an FIR filter
fir_coeffs = [1.0, 0.5, 0.25, 0.125]
fir_filter = FIRFilter(fir_coeffs)

# Convert to lattice structure using tf2lattice
lattice_filter = tf2lattice(fir_coeffs, type_of_filter="FIR")

# Access reflection coefficients
print(f"Reflection coefficients: {lattice_filter.reflection_coeffs}")

# Test with impulse response
impulse = np.zeros(1)
impulse[0] = 1.0
response = lattice_filter.filter(impulse)
print(f"Impulse response: {response}")
```

### Converting an IIR Filter to Lattice Structure

```python
from filter_lattice import IIRFilter, IIRLatticeFilter, tf2lattice

# Create an IIR filter
denominator = [1.0, -0.5, 0.25]  # Note: First coefficient must be non-zero
iir_filter = IIRFilter(denominator)

# Convert to lattice structure using tf2lattice
lattice_filter = tf2lattice(denominator, type_of_filter="IIR")

# Access reflection coefficients
print(f"Reflection coefficients: {lattice_filter.reflection_coeffs}")

# Test with impulse response
impulse = np.zeros(50)
impulse[0] = 1.0
response = lattice_filter.filter(impulse)
print(f"Impulse response: {response}")
```

### Visualizing Filter Responses

```python
import matplotlib.pyplot as plt
import numpy as np
from filter_lattice import tf2lattice

# Create and convert filter
fir_coeffs = [1.0, 0.5, 0.25, 0.125]
lattice_filter = tf2lattice(fir_coeffs, type_of_filter="FIR")

# Plot impulse response
plt.figure(figsize=(10, 4))
plt.stem(np.arange(len(fir_coeffs)), fir_coeffs, linefmt='-', markerfmt='o', basefmt=' ', label='FIR Direct')
plt.stem(np.arange(len(fir_coeffs)), lattice_filter.filter([1]), linefmt='-', markerfmt='x', basefmt=' ', label='FIR Lattice')
plt.title('FIR Impulse Response')
plt.legend()
plt.grid(True)
plt.show()

# Compare signals
t = np.linspace(0, 1, 1000)
signal_in = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
fir_out = np.convolve(signal_in, fir_coeffs, mode='full')
lattice_out = lattice_filter.filter(signal_in)
mse = np.mean((fir_out - lattice_out) ** 2)
print(f"MSE between FIR Direct and FIR Lattice: {mse:.2e}")
```

## Development

The library is structured as follows:

- `filter_lattice/`
  - `__init__.py`: Package initialization and version
  - `filters.py`: Base filter classes (Filter, FIRFilter, IIRFilter)
  - `lattice.py`: Lattice filter implementation (FIRLatticeFilter, IIRLatticeFilter) and tf2lattice conversion
  - `utils.py`: Utility functions and custom exceptions

### Key Components

1. **Base Filter Classes** (`filters.py`):
   - `Filter`: Abstract base class for all filters
   - `FIRFilter`: Implementation of FIR filters
   - `IIRFilter`: Implementation of IIR filters

2. **Lattice Implementation** (`lattice.py`):
   - `LatticeFilter`: Abstract base class for lattice filters
   - `FIRLatticeFilter`: Implementation of FIR lattice filters
   - `IIRLatticeFilter`: Implementation of IIR lattice filters
   - `tf2lattice`: Function to convert transfer function to lattice structure

3. **Utilities** (`utils.py`):
   - `FilterConversionError`: Custom exception for conversion errors
   - `normalize_coefficients`: Function to normalize filter coefficients
   - `check_stability`: Function to verify lattice filter stability

### Testing

The project includes two test files:
- `test_filter_lattice.py`: Comprehensive tests for both FIR and IIR filters, including visualization
- `simple_test.py`: Algebraic verification tests for lattice filter implementations

To run the tests:
```bash
python test_filter_lattice.py
python simple_test.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. When contributing, please ensure:
1. All tests pass
2. New features include appropriate test cases
3. Documentation is updated
4. Code follows the existing style

## License

This project is licensed under the MIT License - see the LICENSE file for details. 