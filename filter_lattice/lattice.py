"""
Lattice filter implementation and conversion utilities.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Optional
from .filters import Filter, FIRFilter, IIRFilter
from .utils import FilterConversionError
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class LatticeFilter(ABC):
    """
    Abstract base class for symmetric lattice filters (FIR/IIR).
    Handles common logic and validation.
    """
    def __init__(self, reflection_coeffs: Union[List[float], np.ndarray], dtype: np.dtype = np.float64):
        self.reflection_coeffs = np.asarray(reflection_coeffs, dtype=dtype)
        self.order = len(self.reflection_coeffs)
        self._validate_coeffs()

    def _validate_coeffs(self):
        if self.order == 0:
            raise ValueError("At least one reflection coefficient is required.")
        if not np.any(self.reflection_coeffs != 0):
            raise ValueError("All reflection coefficients cannot be zero.")
        # Do not check |k| >= 1 here; handle in subclasses

    @abstractmethod
    def filter(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def tf(self, stage: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the transfer function coefficients.
        
        Args:
            stage: Optional stage number up to which to calculate the transfer function.
                  If None, uses all stages (order of the filter).
                  
        Returns:
            Tuple of (numerator, denominator) polynomials.
        """
        pass

    def get_order(self) -> int:
        return self.order

    def __str__(self):
        return f"{self.__class__.__name__}(order={self.order}, k={self.reflection_coeffs})"

    def plot(self):
        N = self.order
        k = self.reflection_coeffs
        fig, ax = plt.subplots(figsize=(2*N+2, 4))
        ax.axis('off')
        # Draw e-path (top)
        for i in range(N+1):
            ax.plot([i, i+1], [2, 2], 'k-', lw=2)
            ax.text(i+0.5, 2.15, f"e{i}[n]", ha='center', va='bottom', fontsize=12)
        ax.text(0, 2.3, 'x[n]', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(N+1, 2.3, 'y[n]', ha='center', va='bottom', fontsize=12, fontweight='bold')
        # Draw b-path (bottom)
        for i in range(N+1):
            ax.plot([i, i+1], [0, 0], 'k-', lw=2)
            ax.text(i+0.5, -0.15, f"b{i}[n]", ha='center', va='top', fontsize=12)
        # Draw verticals and diagonals
        for i in range(1, N+1):
            # Vertical from e to b
            ax.plot([i, i], [2, 0], 'k:', lw=1)
            # Diagonal from b[i-1] to e[i]
            ax.annotate('', xy=(i,2), xytext=(i-1,0), arrowprops=dict(arrowstyle='->', lw=1, color='tab:blue'))
            # Diagonal from e[i-1] to b[i]
            ax.annotate('', xy=(i,0), xytext=(i-1,2), arrowprops=dict(arrowstyle='->', lw=1, color='tab:red'))
            # Reflection coefficient (actual value)
            ax.text(i-0.5, 1, f"{k[i-1]:+.3f}", ha='center', va='center', fontsize=12, color='tab:blue')
            # Delay
            ax.text(i-0.5, -0.5, "z$^{-1}$", ha='center', va='center', fontsize=12, color='tab:gray')
        plt.ylim(-1, 3)
        plt.xlim(-0.5, N+1.5)
        plt.title('FIR Lattice Structure', fontsize=14)
        plt.show()

    
class FIRLatticeFilter(LatticeFilter):
    """
    Symmetric FIR lattice filter implementation.
    """
    def _validate_coeffs(self):
        if self.order == 0:
            raise ValueError("At least one reflection coefficient is required.")
        if not np.any(self.reflection_coeffs != 0):
            raise ValueError("All reflection coefficients cannot be zero.")
        # No |k| >= 1 check for FIR

    def filter(self, x: np.ndarray, stage: Optional[int] = None) -> np.ndarray:
        """
        Filter the input signal x using the FIR lattice filter.
        The outpus has length n + m, where n is the length of the input signal and m is the order of the filter.
        The stage parameter is the number of stages to apply the lattice filter to.
        If stage is not provided, the filter is applied to all stages.
        """
        if stage is None:
            stage = self.order
        n = len(x)
        m = stage
        y_len = n + m  # Output length
        f = np.zeros((m + 1, y_len))
        b = np.zeros((m + 1, y_len))
        f[0, :n] = x  # e_0[n] = x[n]
        b[0, :n] = x  # b_0[n] = x[n]

        for i in range(1, m + 1):
            k = self.reflection_coeffs[i - 1]
            for t in range(y_len):
                prev_b = b[i - 1, t - 1] if t - 1 >= 0 else 0.0
                f[i, t] = f[i - 1, t] - k * prev_b
                b[i, t] = -k * f[i - 1, t] + prev_b

        return f[m]

    def tf(self, stage: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the transfer function coefficients for the FIR lattice filter.
        
        Args:
            stage: Optional stage number up to which to calculate the transfer function.
                  If None, uses all stages (order of the filter).
                  
        Returns:
            Tuple of (numerator, denominator) polynomials.
            For FIR, denominator is always [1.0].
        """
        if stage is None:
            stage = self.order
        elif stage < 0 or stage > self.order:
            raise ValueError(f"Stage must be between 0 and {self.order}")
            
        n = stage
        k = self.reflection_coeffs[:n]  # Only use coefficients up to the specified stage
        
        # Initialize polynomials
        a = np.zeros(n + 1)
        a[0] = 1.0  # a[0] is always 1
        
        # Forward recursion to get transfer function coefficients
        for i in range(n):
            a_prev = a.copy()
            for j in range(1, i + 2):
                a[j] = a_prev[j] - k[i] * a_prev[i + 1 - j]
        
        return a, np.array([1.0])

    def plot(self, title: str = "FIR Lattice Structure"):
        N = self.order
        k = self.reflection_coeffs
        fig, ax = plt.subplots(figsize=(2*N+2, 4))
        ax.axis('off')
        # Draw e-path (top, right to left)
        for i in range(N, -1, -1):
            ax.plot([N-i, N-i+1], [2, 2], 'k-', lw=2)
            ax.text(N-i+0.5, 2.15, f"e{i}[n]", ha='center', va='bottom', fontsize=12)
        ax.text(0, 2.3, f'x[n]=e{N}[n]', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(N+1, 2.3, 'y[n]', ha='center', va='bottom', fontsize=12, fontweight='bold')
        # Draw b-path (bottom, right to left)
        for i in range(N, -1, -1):
            ax.plot([N-i, N-i+1], [0, 0], 'k-', lw=2)
            ax.text(N-i+0.5, -0.15, f"b{i}[n]", ha='center', va='top', fontsize=12)
        # Draw verticals and diagonals
        for i in range(N, 0, -1):
            idx = N-i+1
            # Vertical from e to b
            ax.plot([idx, idx], [2, 0], 'k:', lw=1)
            # Diagonal from b[i] to e[i-1]
            ax.annotate('', xy=(idx,2), xytext=(idx-1,0), arrowprops=dict(arrowstyle='->', lw=1, color='tab:blue'))
            # Diagonal from e[i] to b[i-1]
            ax.annotate('', xy=(idx,0), xytext=(idx-1,2), arrowprops=dict(arrowstyle='->', lw=1, color='tab:red'))
            # Reflection coefficients (actual value)
            ax.text(idx-0.5, 1.2, f"+{k[i-1]:.3f}", ha='center', va='center', fontsize=12, color='tab:blue')
            ax.text(idx-0.5, 0.8, f"-{k[i-1]:.3f}", ha='center', va='center', fontsize=12, color='tab:red')
            # Delay
            ax.text(idx-0.5, -0.5, "z$^{-1}$", ha='center', va='center', fontsize=12, color='tab:gray')
        plt.ylim(-1, 3)
        plt.xlim(-0.5, N+1.5)
        plt.title(title, fontsize=14)
        plt.show()

class IIRLatticeFilter(LatticeFilter):
    """
    Symmetric IIR lattice filter implementation.
    """    
    def _validate_coeffs(self):
        if self.order == 0:
            raise ValueError("At least one reflection coefficient is required.")
        if not np.any(self.reflection_coeffs != 0):
            raise ValueError("All reflection coefficients cannot be zero.")
        if np.any(np.abs(self.reflection_coeffs) >= 1):
            warnings.warn(
                "At least one reflection coefficient has |k| >= 1. The filter may be unstable (one or more poles may be outside the unit circle).",
                UserWarning
            )

    def filter(self, x: np.ndarray, stage: Optional[int] = None) -> np.ndarray:
        """
        Filter the input signal x using the IIR lattice filter.
        The outpus has length n, where n is the length of the input signal.
        The stage parameter is the number of stages to apply the lattice filter to.
        If stage is not provided, the filter is applied to all stages.
        """
        if stage is None:
            stage = self.order
        n = len(x)
        m = stage
        k = self.reflection_coeffs
        b = np.zeros(m + 1, dtype=np.float64)
        y = np.zeros(n, dtype=np.float64)

        for j in range(n):
            yt = x[j]
            for i in range(m - 1, -1, -1):
                yt = yt + b[i] * k[i]
                b[i + 1] = b[i] - yt * k[i]
            b[0] = yt
            y[j] = yt
        return y

    def tf(self, stage: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the transfer function coefficients for the IIR lattice filter.
        
        Args:
            stage: Optional stage number up to which to calculate the transfer function.
                  If None, uses all stages (order of the filter).
                  
        Returns:
            Tuple of (numerator, denominator) polynomials.
            For IIR, numerator is always [1.0].
        """
        if stage is None:
            stage = self.order
        elif stage < 0 or stage > self.order:
            raise ValueError(f"Stage must be between 0 and {self.order}")
            
        n = stage
        k = self.reflection_coeffs[:n]  # Only use coefficients up to the specified stage
        
        # Initialize polynomials
        a = np.zeros(n + 1)
        a[0] = 1.0  # a[0] is always 1
        
        # Forward recursion to get transfer function coefficients
        for i in range(n):
            a_prev = a.copy()
            for j in range(1, i + 2):
                a[j] = a_prev[j] - k[i] * a_prev[i + 1 - j]
        
        return np.array([1.0]), a

    def is_stable(self) -> bool:
        """
        Check if the IIR lattice filter is stable.
        """
        return np.all(np.abs(self.reflection_coeffs) < 1)

    def plot(self, title: str = "IIR Lattice Structure"):
        N = self.order
        k = self.reflection_coeffs
        fig, ax = plt.subplots(figsize=(2*N+2, 4))
        ax.axis('off')
        # Draw e-path (top, right to left)
        for i in range(N, -1, -1):
            ax.plot([N-i, N-i+1], [2, 2], 'k-', lw=2)
            ax.text(N-i+0.5, 2.15, f"e{i}[n]", ha='center', va='bottom', fontsize=12)
        ax.text(0, 2.3, f'x[n]=e{N}[n]', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(N+1, 2.3, 'y[n]', ha='center', va='bottom', fontsize=12, fontweight='bold')
        # Draw b-path (bottom, right to left)
        for i in range(N, -1, -1):
            ax.plot([N-i, N-i+1], [0, 0], 'k-', lw=2)
            ax.text(N-i+0.5, -0.15, f"b{i}[n]", ha='center', va='top', fontsize=12)
        # Draw verticals and diagonals
        for i in range(N, 0, -1):
            idx = N-i+1
            # Vertical from e to b
            ax.plot([idx, idx], [2, 0], 'k:', lw=1)
            # Diagonal from e[i] to b[i-1] (down)
            ax.annotate('', xy=(idx-1,0), xytext=(idx,2), arrowprops=dict(arrowstyle='->', lw=1, color='tab:blue'))
            # Diagonal from b[i] to e[i-1] (up)
            ax.annotate('', xy=(idx-1,2), xytext=(idx,0), arrowprops=dict(arrowstyle='->', lw=1, color='tab:red'))
            # Reflection coefficients (actual value)
            ax.text(idx-0.5, 1.2, f"+{k[i-1]:.3f}", ha='center', va='center', fontsize=12, color='tab:blue')
            ax.text(idx-0.5, 0.8, f"-{k[i-1]:.3f}", ha='center', va='center', fontsize=12, color='tab:red')
            # Delay
            ax.text(idx-0.5, -0.5, "z$^{-1}$", ha='center', va='center', fontsize=12, color='tab:gray')
        plt.ylim(-1, 3)
        plt.xlim(-0.5, N+1.5)
        plt.title(title, fontsize=14)
        plt.show()

class LatticeLadderFilter(LatticeFilter):
    """
    given a general iir filter with transfer function H(z) = B(z) / A(z),
    where A0 = 1,, and B(z) \ne constant, we impleemnt the structure of a lattice ladder filter.
    there will be reflection coefficcients, and now additionally, ladder coefficcients.
    """
    def __init__(self, reflection_coeffs: Union[List[float], np.ndarray], ladder_coeffs: Union[List[float], np.ndarray], dtype: np.dtype = np.float64):
        self.reflection_coeffs = np.asarray(reflection_coeffs, dtype=dtype)
        self.ladder_coeffs = np.asarray(ladder_coeffs, dtype=dtype)
        self.order = len(self.reflection_coeffs)
        self._validate_coeffs()

    def _validate_coeffs(self):
        if self.order == 0:
            raise ValueError("At least one reflection coefficient is required.")
        if not np.any(self.reflection_coeffs != 0):
            raise ValueError("All reflection coefficients cannot be zero.")
        if np.any(np.abs(self.reflection_coeffs) >= 1):
            warnings.warn(
                "At least one reflection coefficient has |k| >= 1. The filter may be unstable (one or more poles may be outside the unit circle).",
                UserWarning
            )

    def filter(self, x: np.ndarray, stage: Optional[int] = None) -> np.ndarray:
        """
        Filter the input signal x using the lattice ladder filter.
        The output has length m+n, where m is the order of the filter, and n is the length of the input signal.
        The stage parameter is the number of stages to apply the lattice ladder filter to.
        If stage is not provided, the filter is applied to all stages.
        """
        if stage is None:
            stage = self.order
        nx = len(x)
        M = stage # number of reflection coeffs
        y = np.zeros(nx)
        f = np.zeros((M + 1, nx), dtype=np.float64)
        g = np.zeros((M + 1, nx), dtype=np.float64)
        f[M,0] = x[0]
        k = self.reflection_coeffs
        c = self.ladder_coeffs
        #k = np.concatenate([k[:1], k[1:]])
        # print("In the .filter method the coeffs are:")
        # print(f"k = {k}")
        for i in range(0, nx):
            f[M,i] = x[i]
            for j in range(M-1, -1, -1):
                if i == 0:
                    f[j,i] = f[j+1,i]
                    g[j+1,i] = k[j] * f[j,i]
                else:
                    f[j,i] = f[j+1,i] - k[j] * g[j,i-1]
                    g[j+1,i] = k[j] * f[j,i] + g[j,i-1]
            g[0,i] = f[0,i]
            for l in range(len(c)):
                y[i] += c[l] * g[l,i]

        # printing for debugging
        # printing the full history, as f[j,i] = %number, g[j,i] = %number
        # print("f:")
        # for j in range(M+1):
        #     for i in range(nx):
        #         print(f"f[{j},{i}] = {f[j,i]:.3f}", end=" ")
        #     print()
        # print("g:")
        # for j in range(M+1):
        #     for i in range(nx):
        #         print(f"g[{j},{i}] = {g[j,i]:.3f}", end=" ")
        #     print()
        # print("y:")
        # for i in range(nx):
        #     print(f"y[{i}] = {y[i]:.3f}", end=" ")
        return y

    def tf(self, stage: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("The transfer function of the lattice ladder filter is not implemented.")

    def plot(self, title: str = "Lattice-Ladder Structure"):
        """Visualise the lattice-ladder filter.

        The drawing follows the same right-to-left convention used in
        `IIRLatticeFilter.plot`.  The lattice section occupies the
        y-coordinates 0 (backward path) and 2 (forward path).  A ladder
        section is appended at y = -1, fed from every backward path node
        `b_i[n]` through the ladder coefficients `c_i`.  The final sum of
        these ladder branches forms the output *y(n)*.
        """
        import matplotlib.pyplot as plt  # local import to avoid tkinter issues on headless

        N = self.order                         # number of reflection coeffs
        k = np.asarray(self.reflection_coeffs) # reflection coefficients (k_1 … k_N)
        c = np.asarray(self.ladder_coeffs)     # ladder coefficients (c_0 … c_N)

        fig, ax = plt.subplots(figsize=(2 * N + 4, 5))
        ax.axis("off")

        # ------------------------------------------------------------------
        # 1.  LATTICE SECTION  (identical style to IIRLatticeFilter.plot)
        # ------------------------------------------------------------------
        # Forward path  (f_i[n])  – drawn on y = 2   (right-to-left)
        for i in range(N, -1, -1):
            x0, x1 = N - i, N - i + 1
            ax.plot([x0, x1], [2, 2], "k-", lw=2)
            ax.text((x0 + x1) / 2, 2.15, f"f{i}(n)", ha="center", va="bottom", fontsize=11)

        ax.text(0, 2.35, f"x(n)=f{N}(n)", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.text(N + 1, 2.35, "f0(n)", ha="center", va="bottom", fontsize=12, fontweight="bold")

        # Backward path  (b_i[n]) – drawn on y = 0   (right-to-left)
        for i in range(N, -1, -1):
            x0, x1 = N - i, N - i + 1
            ax.plot([x0, x1], [0, 0], "k-", lw=2)
            ax.text(N-i+0.5, -0.15, f"b{i}(n)", ha='center', va='top', fontsize=12)
        # Draw adders at all lattice nodes (top & bottom)
        for node_x in range(N + 1):
            draw_adder(ax, node_x, 2)  # forward path adder
            draw_adder(ax, node_x, 0)  # backward path adder

        # Stage connections (vertical / diagonal lines) & reflection coeffs
        for stage in range(N, 0, -1):
            idx = N - stage + 1  # x-coordinate of current stage interface
            # vertical dashed helper (for visual aid only)
            ax.plot([idx, idx], [2, 0], "k:", lw=1)
            # Diagonal down (f_i -> b_{i-1})
            ax.annotate('', xy=(idx - 1 + 0.05, 0), xytext=(idx - 0.05, 2),
                        arrowprops=dict(arrowstyle='->', lw=1, color='tab:blue', shrinkB=10))
            # Diagonal up (b_i -> f_{i-1})
            ax.annotate('', xy=(idx - 1 + 0.05, 2), xytext=(idx - 0.05, 0),
                        arrowprops=dict(arrowstyle='->', lw=1, color='tab:red', shrinkB=10))
            # Reflection coefficient values (+k / -k)
            ax.text(idx - 0.5, 1.2, f"+{k[stage - 1]:.3f}", ha='center', va='center',
                    fontsize=11, color='tab:blue')
            ax.text(idx - 0.5, 0.8, f"-{k[stage - 1]:.3f}", ha='center', va='center',
                    fontsize=11, color='tab:red')
            # Delay annotation
            ax.text(idx - 0.5, -0.5, "z$^{-1}$", ha='center', va='center', fontsize=10, color='tab:gray')

        # ------------------------------------------------------------------
        # 2.  LADDER SECTION  (looks like a tapped-delay FIR)
        # ------------------------------------------------------------------
        ladder_y = -1  # y-coordinate of ladder signal
        left_x     = -0.5
        right_x    = N + 1.5

        # Main ladder signal path (left segment up to last tap)
        ax.plot([left_x, N], [ladder_y, ladder_y], 'k-', lw=2)

        # For each tap: vertical line + coefficient label + adder node on ladder path
        taps = min(len(c), N + 1)
        for tap in range(taps):
            # Bottom node for b_tap is located at integer x = N - tap
            source_x = N - tap  # exact node position on b-path
            dest_x   = source_x  # tap lands vertically below on ladder line

            # Solid dark-green arrow from b_i node down to ladder path
            ax.annotate('', xy=(dest_x, ladder_y + 0.05), xytext=(source_x, 0 - 0.05),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='forestgreen', shrinkB=10))

            # Adder (circle) on ladder path
            draw_adder(ax, dest_x, ladder_y, r=0.1)

        # Output arrow from last bottom adder to the right
        output_end_x = right_x
        ax.annotate('', xy=(output_end_x, ladder_y), xytext=(N, ladder_y),
                    arrowprops=dict(arrowstyle='->', lw=2))
        ax.text(output_end_x + 0.2, ladder_y, 'y(n)', ha='left', va='center', fontsize=12, fontweight='bold')

        # ------------------------------------------------------------------
        # 3.  Cosmetics
        # ------------------------------------------------------------------
        # Connect f0(n) and b0(n) with a solid vertical line
        ax.plot([N + 1, N + 1], [2, 0], 'k-', lw=2)

        ax.set_xlim(-0.8, output_end_x + 1.2)
        ax.set_ylim(ladder_y - 1.0, 3.0)
        plt.title(title, fontsize=14)
        plt.show()

def draw_adder(ax, x, y, r=0.12):
    """Draw a small adder (plus-in-circle) at (x, y)."""
    circ = Circle((x, y), r, fill=False, color='k', lw=1.3, zorder=3)
    ax.add_patch(circ)
    # horizontal and vertical lines
    ax.plot([x - r*0.6, x + r*0.6], [y, y], 'k-', lw=1, zorder=4)
    ax.plot([x, x], [y - r*0.6, y + r*0.6], 'k-', lw=1, zorder=4)

def tf2lattice(coeffs: Union[List[float], np.ndarray], type_of_filter: str="FIR", dtype: np.dtype = np.float64) -> Union[FIRLatticeFilter, IIRLatticeFilter]:
    """
    This function converts a transfer function to a lattice filter.
        This is a direct implementation of the Levinson-Durbin recursion.
        It is assumed that the polynomial of the FIR filter in Z-domain has this form:
        1-a1*z^-1-a2*z^-2-...-aN*z^-N = A(z), i.e. vector a should be a0=1, a1=-a1, a2=-a2, ...
        Therefore, we do such a transformation to the vector a:
        a = [1, -a1, -a2, ..., -aN]
        and then we can use the Levinson-Durbin recursion to compute the reflection coefficients.
    """
    alpha_matrix = calculate_lpc_coeffs(coeffs, dtype)
    k = np.diag(alpha_matrix)
    if type_of_filter == "FIR":
        return FIRLatticeFilter(k)
    elif type_of_filter == "IIR":
        return IIRLatticeFilter(k)
    else:
        raise ValueError("Invalid type of filter. Must be 'FIR' or 'IIR'.")

def tf2ltc(tf: Tuple[np.ndarray, np.ndarray], dtype: np.dtype = np.float64) -> LatticeLadderFilter:
    """
    This function converts a transfer function to a lattice ladder filter.
    """
    b,a = tf
    alpha = calculate_lpc_coeffs(a, dtype)
    # Except for the first row, negate the rest of the rows
    #alpha[1:,:] = -alpha[1:,:]
    alpha = -alpha
    # print("Inside tf2tlc the corrected alpha matrix is ")
    # print(alpha)
    # print("####################")
    M = len(a) 
    M_check = len(b)
    if M != M_check:
        raise ValueError("The order of the numerator and denominator of the transfer function must be the same. Unequal orders are not supported.")
    if M == 0:
        raise ValueError("The transfer function must be a non-trivial filter.")
    if M == 1:
        return FIRLatticeFilter(np.array([1]))

    k = np.diag(alpha)
    c = np.zeros(M)
    c[M-1] = b[M-1]
    for m in range(M-2, -1, -1):
        sum = 0
        # print("####################")
        # print("Inside Loop computations: ")
        for i in range(m+1, M):
            sum += alpha[i-1,i-m-1] * c[i]
        #     print(f"sum = sum + alpha[{i-1},{i-m-1}] *  {c[i]}")
        #     print("which is the same as numerically one by one:")
        #     print(f"{sum} = {sum} + {alpha[i-1,i-m-1]} * {c[i]} ")
        # print("####################")
        c[m] = b[m] - sum
        # print(f"c[{m}] = b[{m}] - sum")
        # print(f"which is the same as numerically one by one:")
        # print(f"{c[m]} = {b[m]} - {sum}")

    # print("inside tf2ltc, the reflection coeffs are:")
    # print(k)
    # print("the ladder coeffs are:")
    # print(c)
    # print("End of tf2ltc")
    
    return LatticeLadderFilter(k, c)
    




def calculate_lpc_coeffs(tf: Union[List[float], np.ndarray], dtype: np.dtype = np.float64) -> np.ndarray:
        """
        Calculate the reflection coefficients from the transfer function.
        """
        coeffs = np.asarray(tf, dtype=dtype)
        n = len(coeffs) - 1
        # Transform the coefficients to the form of A(z)
        if coeffs[0] == 0:
            raise ValueError("The first coefficient of the FIR/IIR filter cannot be zero. If there is a pure delay, it should be handled separately.")
        alpha_matrix = np.zeros((n, n))
        alpha_matrix[n-1,:] = -coeffs[1:]/coeffs[0]
        k = np.zeros(n)
        for i in range(n-1, -1, -1):
            k[i] = alpha_matrix[i,i]
            if i > 0:
                #alpha_prev = alpha.copy()
                for j in range(0, i):
                    alpha_matrix[i-1,j] = (alpha_matrix[i,j] + k[i] * alpha_matrix[i,i-j-1]) / (1 - k[i]**2)
        # print("We're at calc_lpc_coeffs:")
        # print(alpha_matrix)
        # print("End of calc_lpc_coeff")
        return alpha_matrix

