import numpy as np
import matplotlib.pyplot as plt
from filter_lattice import FIRFilter, IIRFilter, LatticeFilter

def test_fir_lattice():
    # Create a simple FIR filter (moving average)
    fir_coeffs = [1.0, 0.5, 0.25, 0.125]
    fir_filter = FIRFilter(fir_coeffs)
    
    # Convert to lattice
    lattice_filter = LatticeFilter.from_fir_filter(fir_filter)
    
    print("FIR to Lattice Test:")
    print(f"Original FIR coefficients: {fir_coeffs}")
    print(f"Lattice reflection coefficients: {lattice_filter.reflection_coeffs}")
    
    # Generate test signal
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    
    # Apply both filters
    fir_output = np.convolve(signal, fir_coeffs, mode='same')
    lattice_output = lattice_filter.apply(signal)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, label='Input')
    plt.plot(t, fir_output, label='FIR Output')
    plt.plot(t, lattice_output, label='Lattice Output')
    plt.legend()
    plt.title('FIR Filter Test')
    plt.show()
    
    # Compare outputs
    mse = np.mean((fir_output - lattice_output) ** 2)
    print(f"Mean Square Error between FIR and Lattice outputs: {mse:.2e}")

def test_iir_lattice():
    # Create a simple IIR filter (lowpass)
    denominator = [1.0, -0.5, 0.25]
    iir_filter = IIRFilter(denominator)
    
    # Convert to lattice
    lattice_filter = LatticeFilter.from_iir_filter(iir_filter)
    
    print("\nIIR to Lattice Test:")
    print(f"Original IIR denominator: {denominator}")
    print(f"Lattice reflection coefficients: {lattice_filter.reflection_coeffs}")
    print(f"Lattice feedforward coefficients: {lattice_filter.feedforward_coeffs}")
    
    # Generate test signal
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    
    # Apply both filters
    from scipy import signal as scipy_signal
    iir_output = scipy_signal.lfilter([1.0], denominator, signal)
    lattice_output = lattice_filter.apply(signal)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, label='Input')
    plt.plot(t, iir_output, label='IIR Output')
    plt.plot(t, lattice_output, label='Lattice Output')
    plt.legend()
    plt.title('IIR Filter Test')
    plt.show()
    
    # Compare outputs
    mse = np.mean((iir_output - lattice_output) ** 2)
    print(f"Mean Square Error between IIR and Lattice outputs: {mse:.2e}")

if __name__ == "__main__":
    test_fir_lattice()
    test_iir_lattice() 