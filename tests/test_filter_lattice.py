import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from filter_lattice import (
    FIRFilter, IIRFilter, LatticeFilter, FIRLatticeFilter, IIRLatticeFilter,
    tf2lattice, tf2ltc
)

# Utility functions

def plot_impulse_response(response, label, n=4):
    # impulse = np.zeros(n)
    # impulse[0] = 1.0
    # response = filter_func(impulse)
    print(f"Impulse response: {response}")
    plt.stem(np.arange(n), response, linefmt='-', markerfmt='o', basefmt=' ', label=label)


def compare_signals(sig1, sig2, name1, name2):
    mse = np.mean((sig1 - sig2) ** 2)
    print(f"MSE between {name1} and {name2} for a sinusoid: {mse:.2e}")
    return mse


def test_fir_lattice():
    print("\n=== FIR Lattice Filter Test ===")
    fir_coeffs = [1, -2.55, -0.2, 0.3]
    fir_filter = FIRFilter(fir_coeffs)
    fir_lattice = tf2lattice(fir_coeffs, type_of_filter="FIR")

    print(f"FIR coefficients: {fir_coeffs}")
    print(f"Lattice reflection coefficients: {fir_lattice.reflection_coeffs}")

    # Test transfer function recovery
    recovered_num, recovered_den = fir_lattice.tf()
    print(f"Original coefficients: {fir_coeffs}")
    print(f"Recovered coefficients: {recovered_num}")
    print(f"Recovery MSE: {np.mean((fir_coeffs - recovered_num) ** 2):.2e}")

    # Test partial transfer function recovery
    for stage in range(1, len(fir_coeffs)):
        partial_num, partial_den = fir_lattice.tf(stage=stage)
        print(f"\nPartial recovery up to stage {stage}:")
        print(f"Original coefficients up to stage {stage}: {fir_coeffs[:stage+1]}")
        print(f"Recovered coefficients: {partial_num}")
        print(f"Recovery MSE: {np.mean((fir_coeffs[:stage+1] - partial_num) ** 2):.2e}")

    # Impulse response
    plt.figure(figsize=(10, 4))
    plot_impulse_response(fir_coeffs, 'FIR Direct', len(fir_coeffs))
    plot_impulse_response(fir_lattice.filter([1]), 'FIR Lattice', len(fir_coeffs))
    plt.title('FIR Impulse Response')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Frequency response
    plt.figure(figsize=(10, 4))
    # plot_frequency_response(fir_coeffs, [1.0], 'FIR Direct')
    # For lattice, get impulse response and use FFT
    impulse = np.zeros(1)
    impulse[0] = 1.0
    lattice_ir = fir_lattice.filter(impulse)
    print(f"LatticeIR: {lattice_ir}")
    w = np.linspace(0, np.pi, 512)
    lattice_fft = np.fft.fft(lattice_ir, 512)
    fir_fft = np.fft.fft(fir_coeffs, 512)
    plt.plot(w / np.pi, 20 * np.log10(np.abs(lattice_fft) + 1e-12), label='FIR Lattice')
    plt.plot(w / np.pi, 20 * np.log10(np.abs(fir_fft) + 1e-12), label='FIR Direct')
    plt.title('FIR Frequency Response')
    plt.xlabel('Normalized Frequency (×π rad/sample)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Signal test
    t = np.linspace(0, 1, 1000)
    signal_in = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    fir_out = np.convolve(signal_in, fir_coeffs, mode='full')
    lattice_out = fir_lattice.filter(signal_in)
    compare_signals(fir_out, lattice_out, 'FIR Direct', 'FIR Lattice')


def test_iir_lattice():
    print("\n=== IIR Lattice Filter Test ===")
    iir_coeffs = [1.0, -0.5, 0.25]
    iir_filter = IIRFilter(iir_coeffs)
    iir_lattice = tf2lattice(iir_coeffs, type_of_filter="IIR")

    print(f"IIR denominator coefficients: {iir_coeffs}")
    print(f"Lattice reflection coefficients: {iir_lattice.reflection_coeffs}")

    # Test transfer function recovery
    recovered_num, recovered_den = iir_lattice.tf()
    print(f"Original coefficients: {iir_coeffs}")
    print(f"Recovered coefficients: {recovered_den}")
    print(f"Recovery MSE: {np.mean((iir_coeffs - recovered_den) ** 2):.2e}")

    # Test partial transfer function recovery
    for stage in range(1, len(iir_coeffs)):
        partial_num, partial_den = iir_lattice.tf(stage=stage)
        print(f"\nPartial recovery up to stage {stage}:")
        print(f"Original coefficients up to stage {stage}: {iir_coeffs[:stage+1]}")
        print(f"Recovered coefficients: {partial_den}")
        print(f"Recovery MSE: {np.mean((iir_coeffs[:stage+1] - partial_den) ** 2):.2e}")

    # Impulse response
    # We should also plot the impulse response from the direct form using scipy.signal.lfilter simultaneously using the same plot
    plt.figure(figsize=(10, 4))
    x = np.array([1,0,0,0,0,0,0,0,0,0,0,0])
    direct_ir = scipy_signal.lfilter([1.0], iir_coeffs, x)
    plot_impulse_response(direct_ir, 'IIR Direct', n=len(x))
    plot_impulse_response(iir_lattice.filter(x), 'IIR Lattice', n=len(x))
    plt.title('IIR Impulse Response')
    plt.legend()
    plt.grid(True)
    plt.show()



    # Frequency response
    # We should also plot the frequency response from the direct form using scipy.signal.freqz
    plt.figure(figsize=(10, 4))
    x = np.zeros(512)
    x[0] = 1.0
    lattice_ir = iir_lattice.filter(x)
    w = np.linspace(0, np.pi, 512)
    lattice_fft = np.fft.fft(lattice_ir, 512)
    direct_fft = np.fft.fft(direct_ir, 512)
    plt.plot(w / np.pi, 20 * np.log10(np.abs(lattice_fft) + 1e-12), label='IIR Lattice')
    plt.plot(w / np.pi, 20 * np.log10(np.abs(direct_fft) + 1e-12), label='IIR Direct')
    plt.title('IIR Frequency Response')
    plt.xlabel('Normalized Frequency (×π rad/sample)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # # Signal test
    # t = np.linspace(0, 1, 1000)
    # signal_in = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    # iir_out = scipy_signal.lfilter([1.0], iir_coeffs, signal_in)
    # lattice_out = iir_lattice.filter(signal_in)
    # compare_signals(iir_out, lattice_out, 'IIR Direct', 'IIR Lattice')

if __name__ == "__main__":
    # test_fir_lattice()
    # test_iir_lattice()

    # # Visual test: plot lattice diagrams for FIR and IIR filters
    # print("\nPlotting FIR Lattice Structure...")
    # fir_coeffs = [1, 0.5, 0.25, 0.34, 0.6, -0.3, 0.2]
    # fir_lattice = tf2lattice(fir_coeffs, type_of_filter="FIR")
    # fir_lattice.plot()

    # print("\nPlotting IIR Lattice Structure...")
    # iir_coeffs = [1.0, -0.5, 0.25, 0.1, -0.2]
    # iir_lattice = tf2lattice(iir_coeffs, type_of_filter="IIR")
    # iir_lattice.plot() 

    # test case 4
    b = np.array([1, -0.5, 1])
    a = np.array([1.0, 0.5, 0.25])
    iir_coeffs = [1.0,  0.5, 0.25]
    iir_filter = IIRFilter(a)
    iir_lattice = tf2lattice(iir_coeffs, type_of_filter="IIR")
    kk = iir_lattice.reflection_coeffs
    print(f"k = {kk}")
    iir3 = tf2ltc((b,a))
    inp = np.array([1,1,2,1,0.5,0,0,0,0,0])
    yy = iir3.filter(inp)
    print("")
    print(f"yy = {yy}")

    ## Using a normal filter to authenticate the result
    signal = inp
    direct_out = scipy_signal.lfilter(b,a,signal)
    print(f"direct_out = {direct_out}")
    print(f"Recovery MSE: {np.mean((direct_out - yy) ** 2):.5e}")

    # # Test Case 3
    # b = np.array([0.129, 0.3867, 0.3869, 0.129])
    # a = np.array([1, -0.2971, 0.3564,-0.0276])
    # iir3 = tf2ltc((b,a))
    