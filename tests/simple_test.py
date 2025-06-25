import numpy as np
from filter_lattice import FIRLatticeFilter, IIRLatticeFilter

def expected_impulse_response_fir(k1, k2, k3):
    h = np.zeros(4)
    h[0] = 1
    h[1] = k3 * k2 - k1 * (1 - k2)
    h[2] = k3 * k1 * (1 - k2) - k2
    h[3] = -k3
    return h

def run_and_compare(fir_lattice, x, expected, test_name, k1, k2, k3):
    y = fir_lattice.filter(x)
    print(f"\n{test_name}")
    print(f"Testing with k1 = {k1:.6f}, k2 = {k2:.6f}, k3 = {k3:.6f}")
    print("Input:", x)
    print("Expected Output:", np.round(expected, 6))
    print("Actual Output:  ", np.round(y, 6))
    print("Match:", np.allclose(y, expected, atol=1e-6))

def algebraic_iir2(k1, k2, order):
    h = np.zeros(order)
    p1 = (k1*(1-k2)+np.sqrt(k1**2 * (1-k2)**2+4*k2))/2
    p2 = (k1*(1-k2)-np.sqrt(k1**2 * (1-k2)**2+4*k2))/2
    powers = np.arange(order)
    h = (p1*p1**powers - p2*p2**powers)/(p1-p2)
    return h

def algebraic_iir1(k1, order):
    h = np.zeros(order)
    powers = np.arange(order)
    h = (k1)**powers
    return h

def algebraic_iir3(k1, k2, k3):
    h = np.zeros(4)
    A1 = k1 - k2*(k1+k3)
    A2 = k2*(1+k1*k3)-k1*k3
    h[0] = 1
    h[1] = A1
    h[2] = A1**2 + A2
    h[3] = A1**3 + 2 * A1 * A2 + k3
    return h

def calculate_error_metrics(expected, actual):
    """Calculate MSE and NMSE between expected and actual outputs."""
    mse = np.mean((expected - actual) ** 2)
    nmse = mse / (np.mean(expected ** 2) + 1e-10)  # Add small epsilon to avoid division by zero
    return mse, nmse

def run_and_compare_iir_algebraic(iir_lattice, x, expected, test_name, coeffs):
    y = iir_lattice.filter(x)
    mse, nmse = calculate_error_metrics(expected, y[:len(expected)])
    print(f"\n{test_name}")
    if len(coeffs) == 1:
        print(f"Testing with k1 = {coeffs[0]:.6f}")
    else:
        print(f"Testing with k1 = {coeffs[0]:.6f}, k2 = {coeffs[1]:.6f}")
    print(f"MSE: {mse:.6e}")
    print(f"NMSE: {nmse:.6e}")
    print("Stable:", np.all(np.abs(iir_lattice.reflection_coeffs) < 1))

def run_and_compare_iir3(iir_lattice, x, expected, test_name, k1, k2, k3):
    y = iir_lattice.filter(x)
    print(f"\n{test_name}")
    print(f"Testing with k1 = {k1:.6f}, k2 = {k2:.6f}, k3 = {k3:.6f}")
    print("Expected Output:", np.round(expected, 6))
    print("Actual Output:  ", np.round(y[:len(expected)], 6))
    print("Match:", np.allclose(y[:len(expected)], expected, atol=1e-6))

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # FIR test cases
    # Test Case 1
    k1, k2, k3 = np.random.uniform(0, 1, 3)
    reflection_coeffs = [k1, k2, k3]
    fir_lattice = FIRLatticeFilter(reflection_coeffs)
    h = expected_impulse_response_fir(k1, k2, k3)
    impulse = np.zeros(1)
    impulse[0] = 1.0
    run_and_compare(fir_lattice, impulse, h, "FIR Impulse Response Test 1", k1, k2, k3)

    # Test Case 2
    k1, k2, k3 = [0.02, 0.1, -0.9]
    reflection_coeffs = [k1, k2, k3]
    fir_lattice = FIRLatticeFilter(reflection_coeffs)
    h = expected_impulse_response_fir(k1, k2, k3)
    run_and_compare(fir_lattice, impulse, h, "FIR Impulse Response Test 2", k1, k2, k3)

    # IIR1 test cases
    # Test Case 1
    k1s = np.random.uniform(0, 1)
    iir1 = IIRLatticeFilter([k1s])
    x = np.zeros(50)
    x[0] = 1.0  # Impulse at t=0
    order = len(x)
    h1 = algebraic_iir1(k1s, order)
    run_and_compare_iir_algebraic(iir1, x, h1, "IIR Algebraic Test 1 (1 coeff) - Case 1", [k1s])

    # Test Case 2
    k1s = np.random.uniform(0, 1)
    iir1 = IIRLatticeFilter([k1s])
    h1 = algebraic_iir1(k1s, order)
    run_and_compare_iir_algebraic(iir1, x, h1, "IIR Algebraic Test 1 (1 coeff) - Case 2", [k1s])

    # IIR2 test cases
    # Test Case 1
    k2s = np.random.uniform(0, 1, 2)
    x = np.zeros(50)
    x[0] = 1.0  # Impulse at t=0
    order = len(x)
    h2 = algebraic_iir2(*k2s, order)
    iir2 = IIRLatticeFilter(np.array(k2s))
    run_and_compare_iir_algebraic(iir2, x, h2, "IIR Algebraic Test 2 (2 coeffs) - Case 1", k2s)

    # Test Case 2
    k2s = np.random.uniform(0, 1, 2)
    h2 = algebraic_iir2(*k2s, order)
    iir2 = IIRLatticeFilter(np.array(k2s))
    run_and_compare_iir_algebraic(iir2, x, h2, "IIR Algebraic Test 2 (2 coeffs) - Case 2", k2s)

    # IIR3 test cases
    # Test Case 1
    k3s = np.random.uniform(0, 1, 3)
    x = [1, 0, 0, 0]
    h3 = algebraic_iir3(*k3s)
    iir3 = IIRLatticeFilter(np.array(k3s))    
    run_and_compare_iir3(iir3, x, h3, "IIR Algebraic Test 3 (3 coeffs) - Case 1", *k3s)

    # Test Case 2
    k3s = np.random.uniform(0, 1, 3)
    h3 = algebraic_iir3(*k3s)
    iir3 = IIRLatticeFilter(np.array(k3s))    
    run_and_compare_iir3(iir3, x, h3, "IIR Algebraic Test 3 (3 coeffs) - Case 2", *k3s)
    
    


if __name__ == "__main__":
    main() 