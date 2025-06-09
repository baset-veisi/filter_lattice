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
from filter_lattice import FIRFilter, LatticeFilter
import numpy as np

# Create an FIR filter
fir_coeffs = [1.0, 0.5, 0.25, 0.125]
fir_filter = FIRFilter(fir_coeffs)

# Convert to lattice structure
lattice_filter = LatticeFilter.from_fir_filter(fir_filter)

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

# Test with impulse response
impulse = np.zeros(50)
impulse[0] = 1.0
response = lattice_filter.filter(impulse)
print(f"Impulse response: {response}")
```

### Visualizing Filter Responses

```python
import matplotlib.pyplot as plt
from filter_lattice import plot_impulse_response, compare_signals

# Plot impulse response
plt.figure(figsize=(10, 4))
plot_impulse_response(fir_coeffs, 'FIR Direct', len(fir_coeffs))
plot_impulse_response(lattice_filter.filter([1]), 'FIR Lattice', len(fir_coeffs))
plt.title('FIR Impulse Response')
plt.legend()
plt.grid(True)
plt.show()

# Compare signals
t = np.linspace(0, 1, 1000)
signal_in = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
fir_out = np.convolve(signal_in, fir_coeffs, mode='full')
lattice_out = lattice_filter.filter(signal_in)
mse = compare_signals(fir_out, lattice_out, 'FIR Direct', 'FIR Lattice')
```

## Development

The library is structured as follows:

- `filter_lattice/`
  - `__init__.py`: Package initialization and version
  - `filters.py`: Base filter classes (Filter, FIRFilter, IIRFilter)
  - `lattice.py`: Lattice filter implementation and conversion methods
  - `utils.py`: Utility functions and custom exceptions

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