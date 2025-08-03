"""
Deconvolution-based Linear Combination Fitting - Clean Implementation

LCF using deconvolution methods for enhanced analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy import signal, ndimage
from tqdm import tqdm
import time
import multiprocessing as mp
from functools import partial

#  ARCTAN STEP + PEAK FITTING ADDITION   START 
from scipy.special import wofz  # For Voigt function
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
#  ARCTAN STEP + PEAK FITTING ADDITION  END 


def smooth_with_kernel_reflect(spectrum, kernel):
    """
    Apply convolution with reflective boundary conditions to avoid edge effects.
    
    This replaces signal.convolve(..., mode='same') which uses zero-padding
    and causes the fitted spectrum to drop to zero at high energies.
    
    Parameters:
    -----------
    spectrum : array
        Input spectrum data
    kernel : array  
        1D convolution kernel
        
    Returns:
    --------
    array : Smoothed spectrum with preserved boundary values
    """
    return ndimage.convolve1d(spectrum, kernel, mode='reflect')


def create_gaussian_kernel(size, sigma):
    """Create a Gaussian kernel manually."""
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / np.sum(kernel)


# start

def arctan_step_function(energy, E0, Gamma, A_step=1.0):
    """
    Arctan step function for XANES edge modeling.
    
    Parameters:
    -----------
    energy : array
        Energy values
    E0 : float
        Edge position (eV)
    Gamma : float
        Step width parameter (eV)
    A_step : float
        Step height (FIXED at 1.0 for normalization, not fitted as parameter)
    
    Returns:
    --------
    array : Arctan step function values (0 to A_step)
    
    Note:
    -----
    A_step is constrained to 1.0 to avoid systematic errors between 
    sample and reference spectra in LCF analysis.
    """
    return A_step * (np.arctan((energy - E0) / Gamma) / np.pi + 0.5)


def gaussian_peak(energy, A, E_center, sigma):
    """
    Gaussian peak function.
    
    Parameters:
    -----------
    energy : array
        Energy values
    A : float
        Peak amplitude
    E_center : float
        Peak center position (eV)
    sigma : float
        Gaussian width parameter (eV)
    
    Returns:
    --------
    array : Gaussian peak values
    """
    return A * np.exp(-0.5 * ((energy - E_center) / sigma) ** 2)


def voigt_peak(energy, A, E_center, sigma_G, gamma_L):
    """
    Voigt peak function (convolution of Gaussian and Lorentzian).
    
    Parameters:
    -----------
    energy : array
        Energy values
    A : float
        Peak amplitude
    E_center : float
        Peak center position (eV)
    sigma_G : float
        Gaussian width parameter (eV)
    gamma_L : float
        Lorentzian width parameter (eV)
    
    Returns:
    --------
    array : Voigt peak values
    """
    if sigma_G <= 0 or gamma_L <= 0:
        return np.zeros_like(energy)
    
    # Voigt profile using Faddeeva function
    z = ((energy - E_center) + 1j * gamma_L) / (sigma_G * np.sqrt(2))
    voigt_profile = np.real(wofz(z)) / (sigma_G * np.sqrt(2 * np.pi))
    return A * voigt_profile


def pseudo_voigt_peak(energy, A, E_center, fwhm, eta):
    """
    Pseudo-Voigt peak function (linear combination of Gaussian and Lorentzian).
    
    Parameters:
    -----------
    energy : array
        Energy values
    A : float
        Peak amplitude
    E_center : float
        Peak center position (eV)
    fwhm : float
        Full width at half maximum (eV)
    eta : float
        Mixing parameter (0=pure Gaussian, 1=pure Lorentzian)
    
    Returns:
    --------
    array : Pseudo-Voigt peak values
    """
    # Convert FWHM to sigma for Gaussian component
    sigma_G = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # Gaussian component
    gauss_component = np.exp(-0.5 * ((energy - E_center) / sigma_G) ** 2)
    
    # Lorentzian component
    gamma_L = fwhm / 2
    lorentz_component = 1 / (1 + ((energy - E_center) / gamma_L) ** 2)
    
    # Linear combination
    return A * ((1 - eta) * gauss_component + eta * lorentz_component)


def detect_peaks_from_derivative(energy, spectrum, prominence_threshold=0.1):
    """
    Detect peak positions from first derivative for initial parameter estimation.
    
    Parameters:
    -----------
    energy : array
        Energy values
    spectrum : array
        Spectrum data
    prominence_threshold : float
        Minimum peak prominence (relative to max)
    
    Returns:
    --------
    list : Peak center positions (eV)
    """
    # Calculate first derivative
    dy = np.gradient(spectrum, energy)
    
    # Find zero crossings of derivative (peaks)
    zero_crossings = []
    for i in range(1, len(dy) - 1):
        if dy[i-1] > 0 and dy[i+1] < 0:  # Peak (positive to negative)
            zero_crossings.append(i)
    
    # Filter by prominence
    peak_positions = []
    max_intensity = np.max(spectrum)
    
    for idx in zero_crossings:
        peak_height = spectrum[idx]
        if peak_height > prominence_threshold * max_intensity:
            peak_positions.append(energy[idx])
    
    return sorted(peak_positions)


def estimate_edge_position(energy, spectrum):
    """
    Estimate absorption edge position from maximum of first derivative.
    
    Parameters:
    -----------
    energy : array
        Energy values
    spectrum : array
        Spectrum data
    
    Returns:
    --------
    float : Estimated edge position (eV)
    """
    dy = np.gradient(spectrum, energy)
    edge_idx = np.argmax(dy)
    return energy[edge_idx]

# ARCTAN STEP + PEAK FITTING FUNCTIONS END


def parallel_deconvolution_worker(shift_chunk, energy, target_spectrum, reference_spectra, sigma=1.5):
    """
    Worker function for parallel deconvolution optimization.
    """
    try:
        best_r = float('inf')
        best_shifts = None
        
        # Create Gaussian kernel for deconvolution
        kernel = create_gaussian_kernel(21, sigma)
        
        for shifts in shift_chunk:
            # Apply energy shifts and deconvolution to references
            shifted_refs = {}
            for i, (ref_name, ref_spec) in enumerate(reference_spectra.items()):
                # Apply energy shift
                shifted_energy = energy + shifts[i]
                interp_func = interp1d(shifted_energy, ref_spec,
                                     kind='linear', bounds_error=False,
                                     fill_value=(ref_spec[0], ref_spec[-1]))
                shifted_ref = interp_func(energy)
                
                # Apply deconvolution
                deconvolved = signal.deconvolve(shifted_ref, kernel)[0]
                if len(deconvolved) != len(energy):
                    # Resize if needed
                    if len(deconvolved) > len(energy):
                        deconvolved = deconvolved[:len(energy)]
                    else:
                        deconvolved = np.pad(deconvolved, (0, len(energy) - len(deconvolved)), 'constant')
                
                shifted_refs[ref_name] = deconvolved
            
            # Optimize fractions for these shifts
            def objective(fractions):
                if len(fractions) != len(shifted_refs):
                    return float('inf')
                
                # Normalize fractions to sum to 1
                fractions = np.array(fractions)
                if np.sum(fractions) <= 0:
                    return float('inf')
                fractions = fractions / np.sum(fractions)
                
                # Calculate fitted spectrum
                fitted = np.zeros_like(target_spectrum)
                for j, (ref_name, ref_spec) in enumerate(shifted_refs.items()):
                    fitted += fractions[j] * ref_spec
                
                # Calculate R-factor using squared differences
                numerator = np.sum((target_spectrum - fitted) ** 2)
                denominator = np.sum(target_spectrum ** 2)
                return numerator / denominator if denominator > 0 else float('inf')
            
            # Initial guess: equal fractions
            n_refs = len(reference_spectra)
            x0 = np.ones(n_refs) / n_refs
            
            # Bounds: all fractions between 0 and 1
            bounds = [(0, 1) for _ in range(n_refs)]
            
            # Constraint: fractions sum to 1
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            
            try:
                result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
                if result.success:
                    r_factor = result.fun
                    if r_factor < best_r:
                        best_r = r_factor
                        best_shifts = shifts
            except:
                continue
        
        return best_r, best_shifts
    except Exception as e:
        return float('inf'), None


class DeconvolutionLCF:
    """
    Clean implementation of deconvolution-based LCF.
    
    Supports standard LCF optimization and enhanced arctan step + peak modeling:
    - Target spectra: 1-100 peaks for complex XANES features
    - Reference spectra: up to 3 peaks for improved modeling accuracy
    - Peak functions: Gaussian, Voigt, Pseudo-Voigt
    - Parameter constraints: A_step=1, b0≈0, b1≈0 for consistency
    """
    
    def __init__(self, r_factor_benchmark=0.04):
        self.energy = None
        self.target_spectrum = None
        self.reference_spectra = {}
        self.results = {}
        self.r_factor_benchmark = r_factor_benchmark
        
        # ARCTAN STEP + PEAK FITTING ADDITION START
        self.use_arctan_step = False
        self.peak_function_type = 'gaussian'  # 'gaussian' or 'voigt' or 'pseudo_voigt'
        self.fitted_models = {}  # Store fitted arctan+peak models for each spectrum
        self.edge_parameters = {}  # Store edge parameters for consistent modeling
        self.global_e0 = None  # Global E0 value for all reference fits
        self.e0_is_set = False  # Flag to track if global E0 has been set
        # ARCTAN STEP + PEAK FITTING ADDITION END
    
    def load_data(self, energy, target_spectrum, reference_spectra):
        """Load data for fitting."""
        self.energy = np.array(energy)
        self.target_spectrum = np.array(target_spectrum)
        self.reference_spectra = {name: np.array(spectrum) 
                                 for name, spectrum in reference_spectra.items()}
        
        print(f" Loaded target and {len(self.reference_spectra)} references")
    
    def apply_energy_shift(self, spectrum, shift):
        """Apply energy shift to spectrum."""
        shifted_energy = self.energy + shift
        interp_func = interp1d(shifted_energy, spectrum,
                              kind='linear', bounds_error=False,
                              fill_value=(spectrum[0], spectrum[-1]))
        return interp_func(self.energy)
    
    def calculate_r_factor(self, observed, calculated, mask=None):
        """
        Calculate R-factor goodness-of-fit using squared differences.
        
        Args:
            observed: Observed spectrum
            calculated: Calculated spectrum
            mask: Optional boolean mask to restrict calculation to specific energy range
        """
        if mask is not None:
            observed = observed[mask]
            calculated = calculated[mask]
        
        numerator = np.sum((observed - calculated) ** 2)
        denominator = np.sum(observed ** 2)
        return numerator / denominator if denominator > 0 else np.inf
    
    # ARCTAN STEP + PEAK FITTING METHODS START
    
    def enable_arctan_step_modeling(self, peak_function='gaussian'):
        """
        Enable arctan step + peak modeling for XANES analysis.
        
        Parameters:
        -----------
        peak_function : str
            Type of peak function ('gaussian', 'voigt', 'pseudo_voigt')
        """
        self.use_arctan_step = True
        self.peak_function_type = peak_function
        print(f" Enabled arctan step + {peak_function} peak modeling")
    
    def disable_arctan_step_modeling(self):
        """Disable arctan step + peak modeling, return to original deconvolution."""
        self.use_arctan_step = False
        self.fitted_models = {}
        self.edge_parameters = {}
        print(" Disabled arctan step + peak modeling")
    
    def interactive_peak_function_selection(self):
        """Interactive selection of peak function type."""
        print("\n Peak Function Selection:")
        print("1. Gaussian (fastest, good for noisy data)")
        print("2. Voigt (most accurate, slower)")
        print("3. Pseudo-Voigt (balanced accuracy/speed)")
        
        while True:
            try:
                choice = int(input("Select peak function (1-3): "))
                if choice == 1:
                    self.peak_function_type = 'gaussian'
                    break
                elif choice == 2:
                    self.peak_function_type = 'voigt'
                    break
                elif choice == 3:
                    self.peak_function_type = 'pseudo_voigt'
                    break
                else:
                    print("Please enter 1, 2, or 3")
            except ValueError:
                print("Please enter a valid number")
        
        print(f" Selected: {self.peak_function_type} peak function")
    
    def get_manual_edge_position(self, energy, spectrum, spectrum_name):
        """Get manual edge position from user or return global E0 if already set."""
        if self.e0_is_set:
            print(f" Using global E0: {self.global_e0:.2f} eV for {spectrum_name}")
            return self.global_e0
        
        # If this is the first reference, set the global E0
        return self.set_global_e0(energy, spectrum, spectrum_name)
    
    def calculate_r_squared(self, observed, fitted, show_plot=False, spectrum_name=""):
        """Calculate R-squared (coefficient of determination) with optional plot display."""
        ss_res = np.sum((observed - fitted) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        
        # Show plot if requested
        if show_plot and spectrum_name:
            self._plot_r_squared_fitting(observed, fitted, r_squared, spectrum_name)
        
        return r_squared
    
    def _plot_r_squared_fitting(self, observed, fitted, r_squared, spectrum_name):
        """Plot observed vs fitted data with R-squared information."""
        try:
            import matplotlib.pyplot as plt
            
            # Close any existing plots to prevent accumulation
            plt.close('all')
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Main plot: observed vs fitted
            if hasattr(self, 'energy') and self.energy is not None and len(self.energy) == len(observed):
                energy = self.energy
            else:
                energy = np.arange(len(observed))
            
            ax1.plot(energy, observed, 'ko-', markersize=3, linewidth=1, 
                    label='Observed', alpha=0.8)
            ax1.plot(energy, fitted, 'r-', linewidth=2, 
                    label=f'Fitted (R² = {r_squared:.6f})')
            ax1.set_xlabel('Energy (eV)')
            ax1.set_ylabel('Absorption')
            ax1.set_title(f'{spectrum_name} - Fitting Quality Assessment\n'
                         f'R² = {r_squared:.6f} (1.0 = perfect fit, 0.0 = no correlation)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Residuals plot
            residuals = observed - fitted
            ax2.plot(energy, residuals, 'g-', linewidth=1, label='Residuals')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Energy (eV)')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Fit Residuals (smaller = better)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Non-blocking display with automatic timeout
            plt.show(block=False)
            plt.draw()
            
            # Force display update
            fig.canvas.flush_events()
            
            # Small delay to ensure plot is visible
            import time
            time.sleep(0.1)
            
        except Exception as e:
            print(f"   (Plot display failed: {e})")
        
        finally:
            # Ensure we don't leave plots hanging
            try:
                plt.pause(0.1)  # Brief pause for display
            except:
                pass
    
    def interactive_peak_fitting_all_references(self, n_peaks_global):
        """Interactive peak fitting for ALL references using the SAME number of peaks."""
        print(f"\n Interactive Peak Fitting for ALL References")
        print("=" * 60)
        print(f" ALL references will be fitted with {n_peaks_global} peak(s)")
        print(f" This ensures consistent modeling across all references")
        print("=" * 60)
        
        if not hasattr(self, 'reference_spectra') or not self.reference_spectra:
            print(" ERROR: No reference spectra loaded")
            return False
        
        # Get global E0 first
        reference_names = list(self.reference_spectra.keys())
        first_ref_name = reference_names[0]
        first_ref_spectrum = self.reference_spectra[first_ref_name]
        
        if not self.e0_is_set:
            self.set_global_e0(self.energy, first_ref_spectrum, "all references")
        
        print(f" Global E0 set to: {self.global_e0:.2f} eV")
        print(f" This E0 will be used for ALL {len(reference_names)} reference(s)")
        
        successful_fits = 0
        
        # Fit each reference with the same number of peaks
        for i, ref_name in enumerate(reference_names):
            print(f"\n [{i+1}/{len(reference_names)}] Fitting {ref_name} with {n_peaks_global} peak(s)...")
            
            # Allow multiple attempts for each reference
            fit_accepted = False
            attempt = 1
            max_attempts = 3
            
            while not fit_accepted and attempt <= max_attempts:
                if attempt > 1:
                    print(f"   Attempt {attempt} for {ref_name}...")
                
                ref_spectrum = self.reference_spectra[ref_name]
                
                try:
                    fitted_result = self.fit_arctan_step_peaks_with_manual_e0(
                        ref_spectrum, ref_name, self.energy, 
                        n_peaks=n_peaks_global, e0_manual=self.global_e0
                    )
                    
                    if fitted_result is not None and fitted_result['success']:
                        fitted_spectrum = fitted_result['fitted_spectrum']
                        r_squared = self.calculate_r_squared(ref_spectrum, fitted_spectrum, 
                                                           show_plot=True, spectrum_name=ref_name)
                        
                        print(f" ✓ {ref_name}: R² = {r_squared:.6f} with {n_peaks_global} peak(s)")
                        
                        # User decision loop with more options
                        while True:
                            print(f"\n Options for {ref_name}:")
                            print(f"   1. ACCEPT this fit (R² = {r_squared:.6f})")
                            print(f"   2. REJECT and try again (attempt {attempt + 1}/{max_attempts})")
                            print(f"   3. SKIP this reference (use original data)")
                            print(f"   4. MODIFY peak count for this reference only")
                            
                            choice = input(f" Your choice (1-4): ").strip()
                            
                            if choice == '1':
                                # Accept fit
                                self.fitted_models[ref_name] = fitted_result
                                print(f" ✓ Confirmed fit for {ref_name}")
                                successful_fits += 1
                                fit_accepted = True
                                break
                                
                            elif choice == '2':
                                # Reject and retry
                                if attempt < max_attempts:
                                    print(f" ✗ Fit rejected. Will retry...")
                                    attempt += 1
                                    break
                                else:
                                    print(f" ✗ Maximum attempts reached. Using original data for {ref_name}")
                                    fit_accepted = True
                                    break
                                    
                            elif choice == '3':
                                # Skip - use original data
                                print(f" ○ Skipping fit for {ref_name}. Will use original data.")
                                fit_accepted = True
                                break
                                
                            elif choice == '4':
                                # Modify peak count for this reference
                                while True:
                                    try:
                                        new_peaks = int(input(f"   Enter new peak count for {ref_name} (1-50): "))
                                        if 1 <= new_peaks <= 50:
                                            print(f"   Trying {new_peaks} peaks for {ref_name}...")
                                            
                                            # Fit with new peak count
                                            new_fitted_result = self.fit_arctan_step_peaks_with_manual_e0(
                                                ref_spectrum, ref_name, self.energy, 
                                                n_peaks=new_peaks, e0_manual=self.global_e0
                                            )
                                            
                                            if new_fitted_result is not None and new_fitted_result['success']:
                                                new_fitted_spectrum = new_fitted_result['fitted_spectrum']
                                                new_r_squared = self.calculate_r_squared(
                                                    ref_spectrum, new_fitted_spectrum, 
                                                    show_plot=True, spectrum_name=f"{ref_name} ({new_peaks} peaks)"
                                                )
                                                
                                                print(f"   ✓ {ref_name}: R² = {new_r_squared:.6f} with {new_peaks} peak(s)")
                                                
                                                accept_new = input(f"   Accept this new fit? (y/n): ").strip().lower()
                                                if accept_new in ['y', 'yes']:
                                                    self.fitted_models[ref_name] = new_fitted_result
                                                    print(f"   ✓ Confirmed modified fit for {ref_name}")
                                                    successful_fits += 1
                                                    fit_accepted = True
                                                    break
                                                else:
                                                    print(f"   Continuing with original options...")
                                                    break
                                            else:
                                                print(f"   ✗ Failed to fit {ref_name} with {new_peaks} peaks")
                                                break
                                        else:
                                            print("   Please enter a number between 1 and 50")
                                    except ValueError:
                                        print("   Please enter a valid number")
                                break  # Back to main choice menu
                                
                            else:
                                print(" Please enter 1, 2, 3, or 4")
                    else:
                        print(f" ✗ Failed to fit {ref_name} with {n_peaks_global} peak(s)")
                        attempt += 1
                        
                except Exception as e:
                    print(f" ✗ Error fitting {ref_name}: {e}")
                    attempt += 1
            
            # Close any open plots to prevent blocking
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except:
                pass
        
        print(f"\n Fitting Summary:")
        print(f" Successfully fitted: {successful_fits}/{len(reference_names)} references")
        print(f" All successful fits used global E0 = {self.global_e0:.2f} eV")
        print(f" Ready for LCF analysis!")
        
        return successful_fits > 0
    
    def interactive_peak_fitting(self, spectrum_data, spectrum_name, energy=None):
        """Interactive peak fitting with R-squared feedback and user control."""
        if energy is None:
            energy = self.energy
        
        print(f"\n Interactive Peak Fitting for {spectrum_name}")
        print("=" * 50)
        
        # Get global E0 (set once for all references)
        e0_manual = self.get_manual_edge_position(energy, spectrum_data, spectrum_name)
        print(f" Using Global E0: {e0_manual:.2f} eV for {spectrum_name}")
        
        # Let user choose starting peak count
        print(f"\n Peak Count Selection:")
        print(f"You can start with any number of peaks (1-50)")
        while True:
            try:
                start_peaks = int(input(f"Enter starting peak count for {spectrum_name} (1-50): "))
                if 1 <= start_peaks <= 50:
                    break
                else:
                    print("Please enter a number between 1 and 50")
            except ValueError:
                print("Please enter a valid number")
        
        current_peaks = start_peaks
        max_peaks = 20
        best_r_squared = 0.0
        best_peaks = current_peaks
        best_result = None
        
        print(f" Starting with {start_peaks} peak(s) for {spectrum_name}")
        
        while current_peaks <= max_peaks:
            print(f"\n Fitting {spectrum_name} with {current_peaks} peak(s)...")
            
            try:
                # Fit with current number of peaks
                fitted_result = self.fit_arctan_step_peaks_with_manual_e0(
                    spectrum_data, spectrum_name, energy, 
                    n_peaks=current_peaks, e0_manual=e0_manual
                )
                
                if fitted_result is not None:
                    fitted_spectrum = fitted_result['fitted_spectrum']
                    r_squared = self.calculate_r_squared(spectrum_data, fitted_spectrum, 
                                                       show_plot=True, spectrum_name=f"{spectrum_name} ({current_peaks} peaks)")
                    
                    print(f" R-squared with {current_peaks} peak(s): {r_squared:.6f}")
                    
                    if r_squared > best_r_squared:
                        best_r_squared = r_squared
                        best_peaks = current_peaks
                        best_result = fitted_result
                    
                    # Enhanced user confirmation with flexible options
                    print(f"\n Fit Results for {current_peaks} peak(s):")
                    print(f"   Current R² = {r_squared:.6f}")
                    print(f"   Best R² so far = {best_r_squared:.6f} (with {best_peaks} peaks)")
                    print(f"\n What would you like to do?")
                    print(f"   1. STOP and use {current_peaks} peak(s) (R² = {r_squared:.6f})")
                    print(f"   2. CONTINUE and try {current_peaks + 1} peak(s)")
                    print(f"   3. STOP and use BEST result ({best_peaks} peaks, R² = {best_r_squared:.6f})")
                    print(f"   4. JUMP to specific peak count (1-{max_peaks})")
                    print(f"   5. GO BACK and try {current_peaks - 1} peak(s)" if current_peaks > 1 else "")
                    
                    valid_choices = ['1', '2', '3', '4']
                    if current_peaks > 1:
                        valid_choices.append('5')
                    
                    while True:
                        choice = input(f"Enter your choice ({'/'.join(valid_choices)}): ").strip()
                        if choice in valid_choices:
                            break
                        print(f"Please enter one of: {', '.join(valid_choices)}")
                    
                    if choice == '1':
                        # Use current result
                        self.fitted_models[spectrum_name] = fitted_result
                        print(f" CONFIRMED: Using {current_peaks} peak(s) for {spectrum_name}")
                        print(f" Final R² = {r_squared:.6f}")
                        return fitted_result
                    elif choice == '3':
                        # Use best result so far
                        if best_result is not None:
                            self.fitted_models[spectrum_name] = best_result
                            print(f" CONFIRMED: Using BEST result ({best_peaks} peaks) for {spectrum_name}")
                            print(f" Final R² = {best_r_squared:.6f}")
                            return best_result
                    elif choice == '4':
                        # Jump to specific peak count
                        while True:
                            try:
                                new_peaks = int(input(f"Enter peak count to try (1-{max_peaks}): "))
                                if 1 <= new_peaks <= max_peaks:
                                    current_peaks = new_peaks
                                    print(f" Jumping to {new_peaks} peaks...")
                                    break
                                else:
                                    print(f"Please enter a number between 1 and {max_peaks}")
                            except ValueError:
                                print("Please enter a valid number")
                        continue
                    elif choice == '5' and current_peaks > 1:
                        # Go back one peak
                        current_peaks -= 1
                        print(f" Going back to {current_peaks} peaks...")
                        continue
                    else:  # choice == '2'
                        # Continue to next peak count
                        if current_peaks >= max_peaks:
                            print(f" Maximum peaks ({max_peaks}) reached!")
                            break
                        current_peaks += 1
                        print(f" Continuing to {current_peaks} peaks...")
                        continue
                        
                else:
                    print(f" ERROR: Fitting failed with {current_peaks} peak(s)")
                    if current_peaks >= max_peaks:
                        break
                    current_peaks += 1
                    
            except Exception as e:
                print(f" ERROR: Exception with {current_peaks} peak(s): {e}")
                if current_peaks >= max_peaks:
                    break
                current_peaks += 1
        
        # If we reach here, use best result available
        if best_result is not None:
            self.fitted_models[spectrum_name] = best_result
            print(f"\n FINAL: Using best result ({best_peaks} peaks) for {spectrum_name}")
            print(f" Final R² = {best_r_squared:.6f}")
            return best_result
        else:
            print(f"\n ERROR: All fitting attempts failed for {spectrum_name}")
            return None
    
    def fit_arctan_step_peaks_with_manual_e0(self, spectrum_data, spectrum_name, energy, n_peaks=1, e0_manual=None):
        """Fit arctan step + peaks with manually set E0."""
        if e0_manual is None:
            e0_manual = energy[np.argmax(np.gradient(spectrum_data))]
        
        try:
            # Use the existing fitting method but with fixed E0
            return self.fit_arctan_step_peaks(spectrum_data, spectrum_name, energy, 
                                            n_peaks=n_peaks, manual_e0=e0_manual)
        except Exception as e:
            print(f"Fitting error: {e}")
            return None
    
    def fit_arctan_step_peaks(self, spectrum_data, spectrum_name, energy=None, 
                             n_peaks=3, fit_window=None, interactive=False, manual_e0=None):
        """
        Fit arctan step + multiple peaks to a spectrum with strict parameter constraints.
        
        IMPORTANT: This method is used ONLY for fitting REFERENCE spectra.
        LCF target data should NEVER be fitted or modified!
        
        Parameter Constraints (to avoid systematic errors in LCF):
        ----------------------------------------------------------
        - A_step: FIXED at 1.0 (not fitted, ensures consistent normalization)
        - b0: Constrained to [-0.005, +0.005] (minimal baseline drift)
        - b1: Constrained to [-1e-4, +1e-4] (minimal linear background)
        - E0, Gamma: Can be manually set or fitted for first spectrum, then fixed for consistency
        - Target peaks: 1-50 peaks allowed (flexible for complex spectra)
        - Reference peaks: Always 1 peak (simplified standard compounds)
        
        Parameters:
        -----------
        spectrum_data : array
            Spectrum intensity data
        spectrum_name : str
            Name of the spectrum (for storage)
        energy : array, optional
            Energy values (uses self.energy if None)
        n_peaks : int
            Number of peaks to fit
        fit_window : tuple, optional
            Energy range for fitting (E_min, E_max)
        interactive : bool
            Whether to show interactive plots for parameter adjustment
        manual_e0 : float, optional
            Manually set E0 value. If provided, this will be used instead of estimation
        
        Returns:
        --------
        dict : Fitted model parameters and components
        """
        if energy is None:
            energy = self.energy
        
        if fit_window is not None:
            mask = (energy >= fit_window[0]) & (energy <= fit_window[1])
            fit_energy = energy[mask]
            fit_spectrum = spectrum_data[mask]
        else:
            fit_energy = energy
            fit_spectrum = spectrum_data
        
        # Use manual E0 if provided, otherwise estimate
        if manual_e0 is not None:
            E0_est = manual_e0
            print(f" Using manual E0: {E0_est:.2f} eV")
        else:
            E0_est = estimate_edge_position(fit_energy, fit_spectrum)
            print(f" Estimated E0: {E0_est:.2f} eV")
        
        # Detect initial peak positions
        peak_positions = detect_peaks_from_derivative(fit_energy, fit_spectrum)
        
        # Ensure we have the requested number of peaks
        if len(peak_positions) < n_peaks:
            # Add peaks at reasonable intervals
            energy_range = fit_energy[-1] - fit_energy[0]
            for i in range(len(peak_positions), n_peaks):
                estimated_pos = E0_est + (i + 1) * energy_range / (n_peaks + 2)
                peak_positions.append(estimated_pos)
        elif len(peak_positions) > n_peaks:
            # Keep the most prominent peaks
            peak_positions = peak_positions[:n_peaks]
        
        # Use consistent edge parameters across all spectra if available and manual_e0 not provided
        if manual_e0 is None and 'E0' in self.edge_parameters and 'Gamma' in self.edge_parameters:
            E0_fixed = self.edge_parameters['E0']
            Gamma_fixed = self.edge_parameters['Gamma']
            use_fixed_edge = True
        else:
            E0_fixed = E0_est
            Gamma_fixed = 1.0  # Initial guess for step width
            use_fixed_edge = False
        
        # Define the complete model function
        def complete_model(energy_vals, *params):
            # Parse parameters
            if use_fixed_edge:
                E0, Gamma = E0_fixed, Gamma_fixed
                A_step = 1.0  # Fixed step height (always 1.0)
                param_idx = 0
            else:
                E0, Gamma = params[0], params[1]
                A_step = 1.0  # Fixed step height (always 1.0, not fitted)
                param_idx = 2
            
            # Background (linear) - constrained to minimal drift
            # b0: fixed ~0 (±0.005), b1: fixed ~0 (±1e-4)
            b0, b1 = params[param_idx], params[param_idx + 1]
            param_idx += 2
            
            # Start with background and step
            model = b0 + b1 * (energy_vals - E0) + arctan_step_function(energy_vals, E0, Gamma, A_step)
            
            # Add peaks
            for i in range(n_peaks):
                if self.peak_function_type == 'gaussian':
                    A, E_center, sigma = params[param_idx:param_idx + 3]
                    model += gaussian_peak(energy_vals, A, E_center, sigma)
                    param_idx += 3
                elif self.peak_function_type == 'voigt':
                    A, E_center, sigma_G, gamma_L = params[param_idx:param_idx + 4]
                    model += voigt_peak(energy_vals, A, E_center, sigma_G, gamma_L)
                    param_idx += 4
                elif self.peak_function_type == 'pseudo_voigt':
                    A, E_center, fwhm, eta = params[param_idx:param_idx + 4]
                    model += pseudo_voigt_peak(energy_vals, A, E_center, fwhm, eta)
                    param_idx += 4
            
            return model
        
        # Initial parameter estimation
        initial_params = []
        param_bounds = []
        
        # Edge parameters (if not fixed)
        if not use_fixed_edge:
            initial_params.extend([E0_est, 1.0])  # E0, Gamma
            param_bounds.extend([(E0_est - 2, E0_est + 2), (0.3, 3.0)])
        
        # Background parameters - UPDATED CONSTRAINTS
        # b0: Fixed at 0 with minimal drift allowed (±0.005)
        # b1: Fixed at 0 with very narrow range (-1e-4, 1e-4)
        initial_params.extend([0.0, 0.0])  # b0, b1 (both start at 0)
        param_bounds.extend([(-0.005, 0.005), (-1e-4, 1e-4)])  # Very tight constraints
        
        # Peak parameters
        max_intensity = np.max(fit_spectrum)
        for i, peak_pos in enumerate(peak_positions):
            if self.peak_function_type == 'gaussian':
                initial_params.extend([max_intensity * 0.5, peak_pos, 1.0])  # A, E_center, sigma
                param_bounds.extend([
                    (0, max_intensity * 2),
                    (peak_pos - 3, peak_pos + 3),
                    (0.3, 3.0)
                ])
            elif self.peak_function_type == 'voigt':
                initial_params.extend([max_intensity * 0.5, peak_pos, 1.0, 0.5])  # A, E_center, sigma_G, gamma_L
                param_bounds.extend([
                    (0, max_intensity * 2),
                    (peak_pos - 3, peak_pos + 3),
                    (0.3, 3.0),
                    (0.1, 2.0)
                ])
            elif self.peak_function_type == 'pseudo_voigt':
                initial_params.extend([max_intensity * 0.5, peak_pos, 1.5, 0.5])  # A, E_center, fwhm, eta
                param_bounds.extend([
                    (0, max_intensity * 2),
                    (peak_pos - 3, peak_pos + 3),
                    (0.5, 4.0),
                    (0, 1)
                ])
        
        # Perform fitting
        try:
            popt, pcov = curve_fit(complete_model, fit_energy, fit_spectrum, 
                                 p0=initial_params, bounds=list(zip(*param_bounds)),
                                 maxfev=5000)
            
            # Calculate fitted spectrum
            fitted_spectrum = complete_model(fit_energy, *popt)
            residuals = fit_spectrum - fitted_spectrum
            r_factor = self.calculate_r_factor(fit_spectrum, fitted_spectrum)
            
            # Store edge parameters for consistency across spectra
            if not use_fixed_edge:
                self.edge_parameters['E0'] = popt[0]
                self.edge_parameters['Gamma'] = popt[1]
            
            # Parse fitted parameters
            fitted_model = {
                'success': True,
                'r_factor': r_factor,
                'energy': fit_energy,
                'spectrum': fit_spectrum,
                'fitted_spectrum': fitted_spectrum,
                'residuals': residuals,
                'parameters': popt,
                'covariance': pcov,
                'edge_E0': E0_fixed if use_fixed_edge else popt[0],
                'edge_Gamma': Gamma_fixed if use_fixed_edge else popt[1],
                'peak_function_type': self.peak_function_type,
                'n_peaks': n_peaks
            }
            
            # Store the fitted model
            self.fitted_models[spectrum_name] = fitted_model
            
            if interactive:
                self._plot_fitted_model(spectrum_name)
            
            print(f" Fitted {spectrum_name}: R-factor = {r_factor:.6f}")
            return fitted_model
            
        except Exception as e:
            print(f" Fitting failed for {spectrum_name}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def apply_arctan_step_modeling_to_all_spectra(self, n_peaks=3, ref_peaks=3, fit_window=None):
        """
        Apply arctan step + peak modeling to reference spectra ONLY.
        
        CORRECTED LOGIC:
        - Target spectra: Keep ORIGINAL data (experimental data to be fitted)
        - Reference spectra: Apply arctan+peak modeling (clean reference models)
        
        Parameters:
        -----------
        n_peaks : int
            Number of peaks to fit for target spectrum (DEPRECATED - target not fitted)
        ref_peaks : int
            Number of peaks to fit for reference spectra (default: 3)
        fit_window : tuple, optional
            Energy range for fitting (E_min, E_max)
        
        Note:
        -----
        CORRECT APPROACH:
        - Target (spot) data: Use original experimental data - DO NOT FIT
        - Reference data: Apply arctan+peak modeling for clean references
        - LCF: Use fitted references to fit original target
        """
        if not self.use_arctan_step:
            print(" Arctan step modeling is not enabled. Use enable_arctan_step_modeling() first.")
            return
        
        print(f"\n Applying arctan step + {self.peak_function_type} peak modeling...")
        print(f"   CORRECTED LOGIC: Only fitting reference spectra, keeping target original!")
        
        #  FIRST: Fit ONE reference to establish edge parameters
        first_ref_name = list(self.reference_spectra.keys())[0]
        first_ref_spectrum = self.reference_spectra[first_ref_name]
        
        print(f" Fitting first reference: {first_ref_name} with {ref_peaks} peaks to establish edge parameters...")
        # ARCTAN STEP + PEAK FITTING INTEGRATION   START
        # Use fitted arctan+peak model if available, otherwise use original spectrum
        if self.use_arctan_step and first_ref_name in self.fitted_models:
            ref_spectrum = self.get_fitted_spectrum(first_ref_name)
            if ref_spectrum is None:
                ref_spectrum = first_ref_spectrum
        else:
            ref_spectrum = first_ref_spectrum
        # ARCTAN STEP + PEAK FITTING INTEGRATION END
        
        # Fit the first reference to establish edge parameters
        result = self.fit_arctan_step_peaks(ref_spectrum, first_ref_name, energy, 
                                           n_peaks=ref_peaks, interactive=False)
        
        if result['success']:
            print(f" Reference fitted successfully: {first_ref_name}")
        else:
            print(f" Reference fitting failed: {first_ref_name}")
        
        # Set global E0 based on the first reference fit
        self.set_global_e0(energy, ref_spectrum, first_ref_name)
        
        # Now apply the model to all references
        for ref_name in self.reference_spectra:
            if ref_name == first_ref_name:
                continue  # Skip the first reference, already fitted
            
            print(f"\n Applying arctan step + peak modeling to reference: {ref_name}...")
            # ARCTAN STEP + PEAK FITTING INTEGRATION START
            # Use fitted arctan+peak model if available, otherwise use original spectrum
            if self.use_arctan_step and ref_name in self.fitted_models:
                ref_spectrum = self.get_fitted_spectrum(ref_name)
                if ref_spectrum is None:
                    ref_spectrum = self.reference_spectra[ref_name]
            else:
                ref_spectrum = self.reference_spectra[ref_name]
            # ARCTAN STEP + PEAK FITTING INTEGRATION EN
            
            # Fit each reference with the global E0
            result = self.fit_arctan_step_peaks(ref_spectrum, ref_name, energy, 
                                               n_peaks=ref_peaks, interactive=False,
                                               manual_e0=self.global_e0)
            
            if result['success']:
                print(f" Reference fitted successfully: {ref_name}")
            else:
                print(f" Reference fitting failed: {ref_name}")
    
    def get_fitted_spectrum(self, spectrum_name):
        """
        Get the fitted arctan+peak model for a spectrum.
        
        CORRECTED LOGIC:
        - For target ('target'): ALWAYS return original experimental data
        - For references: Return fitted arctan+peak model if available
        
        Parameters:
        -----------
        spectrum_name : str
            Name of the spectrum
        
        Returns:
        --------
        array : Spectrum data (original for target, fitted for references)
        """
        # TARGET: Always use original data (experimental data to be fitted)
        if spectrum_name == 'target':
            return self.target_spectrum
        
        # REFERENCES: Use fitted models if available
        if spectrum_name in self.fitted_models and self.fitted_models[spectrum_name]['success']:
            return self.fitted_models[spectrum_name]['fitted_spectrum']
        else:
            # Fallback to original reference spectrum
            if spectrum_name in self.reference_spectra:
                print(f"   Using ORIGINAL reference data for {spectrum_name} (fitting failed)")
                return self.reference_spectra[spectrum_name]
            else:
                return None
    
    def _plot_fitted_model(self, spectrum_name):
        """Plot the fitted arctan step + peak model for a spectrum."""
        if spectrum_name not in self.fitted_models:
            print(f"No fitted model found for {spectrum_name}")
            return
        
        model = self.fitted_models[spectrum_name]
        if not model['success']:
            print(f"Fitting failed for {spectrum_name}")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot original and fitted spectra
        plt.subplot(2, 1, 1)
        plt.plot(model['energy'], model['spectrum'], 'ko-', markersize=2, 
                label='Original', alpha=0.7)
        plt.plot(model['energy'], model['fitted_spectrum'], 'r-', linewidth=2, 
                label=f'Fitted ({self.peak_function_type})')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Absorption')
        plt.title(f'Arctan Step + {self.peak_function_type.title()} Peak Fit: {spectrum_name}\n'
                 f'R-factor = {model["r_factor"]:.6f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot residuals
        plt.subplot(2, 1, 2)
        plt.plot(model['energy'], model['residuals'], 'g-', linewidth=1)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Energy (eV)')
        plt.ylabel('Residuals')
        plt.title('Fit Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    #  ARCTAN STEP + PEAK FITTING METHODS  END

    def fit(self, max_shift=7.5, max_iterations=5):
        """
        Fit the target spectrum by calling the brute-force method.
        
        Parameters:
        -----------
        max_shift : float
            Maximum energy shift allowed (eV). Can be overridden by user input.
        max_iterations : int
            Maximum number of iterations
        """
        print(" Starting comprehensive brute-force fitting for deconvolution...")
        return self.brute_force_fit(max_shift=max_shift)

    def brute_force_fit(self, max_shift=7.5, target_time_minutes=27):
        """
        Perform a true brute-force deconvolution LCF fitting by simultaneously 
        searching the entire parameter space of fractions, shifts, and deconvolution
        parameters.

        This method uses a multi-level, randomized sampling strategy to explore
        the combined parameter space, crucial for the complex interplay between
        deconvolution and fitting.
        
        Parameters:
        -----------
        max_shift : float
            Maximum energy shift allowed (eV). Can be overridden by user input.
        target_time_minutes : float
            Target computation time in minutes.
            
        Returns:
        --------
        dict : Fitting results.
        """
        print(f"\n Starting True Brute-Force Deconvolution LCF (3-Stage)...")
        print(f"   Strategy: 1. Global Randomized Search -> 2. Refined Search -> 3. Gradient Optimization.")
        print(f"   Max energy shift: ±{max_shift} eV")
        print(f"   Target time: ~{target_time_minutes} minutes")
        
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        if n_refs < 2:
            return {'success': False, 'error': 'Need at least 2 references'}

        # Check which references have fitted models
        fitted_refs = []
        original_refs = []
        for ref_name in ref_names:
            if self.use_arctan_step and ref_name in self.fitted_models:
                fitted_refs.append(ref_name)
            else:
                original_refs.append(ref_name)
        
        print(f"\n Reference Status for LCF:")
        if fitted_refs:
            print(f"   Using FITTED arctan+peak models: {fitted_refs}")
            print(f"   ✓ FITTED models will be used AS-IS (no additional smoothing)")
        if original_refs:
            print(f"   Using ORIGINAL spectra: {original_refs}")
            print(f"   ✓ ORIGINAL spectra will have deconvolution applied")
        print(f"   Total references for LCF: {n_refs}")
        print(f"   LCF will optimize fractions and shifts for ALL references")
        print("")

        total_start_time = time.time()
        target_time_seconds = target_time_minutes * 60

        #  Stage 1 & 2: Randomized Global and Refined Search 
        print("\n Stage 1 & 2: Randomized global and refined search...")

        # Deconvolution adds complexity, so sample count is lower than interpolation
        # for the same timeframe. Estimate ~0.2ms per sample.
        n_samples = int(target_time_seconds / 0.0002)
        print(f"   Estimated samples for target time: {n_samples:,}")

        best_r_factor = np.inf
        best_params = None

        # Deconvolution parameters to search
        sigma_range = (0.5, 5.0) # Range of Gaussian sigma for deconvolution

        try:
            with tqdm(total=n_samples, desc="Brute-Force Deconv", unit="combo") as pbar:
                
                for i in range(n_samples):

                    #  Parameter Generation 
                    # Generate a full set of random parameters: fractions, shifts, and sigma
                    
                    # Generate random fractions that sum to 1
                    fractions = np.random.random(n_refs)
                    fractions /= np.sum(fractions)

                    # Generate random shifts and sigma
                    if best_params and i > n_samples * 0.1:
                        # Refine search around the best known parameters
                        refinement_factor = 1.0 - (i / n_samples)
                        
                        best_fracs = best_params['fractions']
                        best_shifts = best_params['shifts']
                        best_sigma = best_params['sigma']

                        # Perturb and re-normalize
                        fractions = np.abs(best_fracs + np.random.normal(0, 0.1 * refinement_factor, n_refs))
                        fractions /= np.sum(fractions)
                        shifts = best_shifts + np.random.normal(0, max_shift * 0.1 * refinement_factor, n_refs)
                        shifts = np.clip(shifts, -max_shift, max_shift)
                        sigma = best_sigma + np.random.normal(0, (sigma_range[1] - sigma_range[0]) * 0.1 * refinement_factor)
                        sigma = np.clip(sigma, sigma_range[0], sigma_range[1])
                    else:
                        # Pure random search
                        shifts = np.random.uniform(-max_shift, max_shift, n_refs)
                        sigma = np.random.uniform(sigma_range[0], sigma_range[1])

                    #  Evaluation 
                    try:
                        fitted_spectrum = np.zeros_like(self.target_spectrum)
                        
                        # Create deconvolution kernel
                        kernel_size = max(3, int(6 * sigma))
                        if kernel_size % 2 == 0: kernel_size += 1
                        kernel = create_gaussian_kernel(kernel_size, sigma)

                        for j, ref_name in enumerate(ref_names):
                            #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
                            # Use fitted arctan+peak model if available, otherwise use original spectrum
                            if self.use_arctan_step and ref_name in self.fitted_models:
                                ref_spectrum = self.get_fitted_spectrum(ref_name)
                                if ref_spectrum is None:
                                    ref_spectrum = self.reference_spectra[ref_name]
                                
                                #  CRITICAL FIX: Use fitted models as-is, without additional smoothing
                                shifted_spectrum = self.apply_energy_shift(ref_spectrum, shifts[j])
                                # Skip deconvolution for fitted models to preserve fitted peak shapes
                                fitted_spectrum += fractions[j] * shifted_spectrum
                            else:
                                ref_spectrum = self.reference_spectra[ref_name]
                                # Apply deconvolution only to original reference spectra
                                shifted_spectrum = self.apply_energy_shift(ref_spectrum, shifts[j])
                                deconvolved_spectrum = smooth_with_kernel_reflect(shifted_spectrum, kernel)
                                fitted_spectrum += fractions[j] * deconvolved_spectrum
                            #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
                        
                        #  TARGET: Always use original experimental data
                        target_for_comparison = self.target_spectrum
                        
                        r_factor = self.calculate_r_factor(target_for_comparison, fitted_spectrum)

                        if r_factor < best_r_factor:
                            best_r_factor = r_factor
                            best_params = {'fractions': np.copy(fractions), 'shifts': np.copy(shifts), 'sigma': sigma}
                            pbar.set_postfix({'Best R': f'{best_r_factor:.6f}'})
                    
                    except Exception:
                        # Ignore errors in single steps, common in wide searches
                        pass
                    
                    pbar.update(1)
        except KeyboardInterrupt:
            print("\n User interrupted the search. Moving to final optimization...")

        if not best_params:
            return {'success': False, 'error': 'Brute force search did not find any valid solution.'}

        print(f"   Best R-factor after random search: {best_r_factor:.6f}")

        #  Stage 3: Gradient-based Local Optimization 
        print("\n Stage 3: Final gradient-based local optimization...")

        def objective(params):
            """Objective function for the final optimizer."""
            fractions = params[:n_refs]
            shifts = params[n_refs:2*n_refs]
            sigma = params[-1]
            
            fitted = np.zeros_like(self.target_spectrum)
            
            try:
                kernel_size = max(3, int(6 * sigma))
                if kernel_size % 2 == 0: kernel_size += 1
                kernel = create_gaussian_kernel(kernel_size, sigma)

                for i, ref_name in enumerate(ref_names):
                    #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
                    # Use fitted arctan+peak model if available
                    if self.use_arctan_step and ref_name in self.fitted_models:
                        ref_spectrum = self.get_fitted_spectrum(ref_name)
                        if ref_spectrum is None:
                            ref_spectrum = self.reference_spectra[ref_name]
                        
                        #  CRITICAL FIX: Use fitted models as-is, without additional smoothing
                        shifted_spectrum = self.apply_energy_shift(ref_spectrum, shifts[i])
                        # Skip deconvolution for fitted models to preserve fitted peak shapes
                        fitted += fractions[i] * shifted_spectrum
                    else:
                        ref_spectrum = self.reference_spectra[ref_name]
                        # Apply deconvolution only to original reference spectra
                        shifted_spectrum = self.apply_energy_shift(ref_spectrum, shifts[i])
                        deconvolved_spectrum = smooth_with_kernel_reflect(shifted_spectrum, kernel)
                        fitted += fractions[i] * deconvolved_spectrum
                    #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
                
                #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
                #  TARGET: Always use original experimental data
                target_for_comparison = self.target_spectrum
                #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
                
                return self.calculate_r_factor(target_for_comparison, fitted)
            except:
                return np.inf

        initial_params_for_optimizer = np.concatenate([
            best_params['fractions'], 
            best_params['shifts'], 
            [best_params['sigma']]
        ])
        
        bounds = ([(0, 1)] * n_refs + 
                  [(-max_shift, max_shift)] * n_refs + 
                  [(sigma_range[0], sigma_range[1])])
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:n_refs]) - 1}]

        optimizer_result = minimize(objective, 
                                    initial_params_for_optimizer,
                                    method='SLSQP', 
                                    bounds=bounds,
                                    constraints=constraints,
                                    options={'maxiter': 2000, 'ftol': 1e-10})

        if optimizer_result.success:
            print(" Gradient optimization successful.")
            final_params = optimizer_result.x
            final_fractions = final_params[:n_refs]
            final_shifts = final_params[n_refs:2*n_refs]
            final_sigma = final_params[-1]
            # Normalize fractions just in case
            final_fractions = np.abs(final_fractions)
            final_fractions /= np.sum(final_fractions)
        else:
            print(" Gradient optimization failed. Using best result from random search.")
            final_fractions = best_params['fractions']
            final_shifts = best_params['shifts']
            final_sigma = best_params['sigma']

        #  Finalize and Return Results 
        final_fractions_dict = dict(zip(ref_names, final_fractions))
        final_shifts_dict = dict(zip(ref_names, final_shifts))

        # Recalculate final spectrum with best parameters
        fitted_spectrum = np.zeros_like(self.target_spectrum)
        kernel_size = max(3, int(6 * final_sigma))
        if kernel_size % 2 == 0: kernel_size += 1
        kernel = create_gaussian_kernel(kernel_size, final_sigma)
        for i, ref_name in enumerate(ref_names):
            #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
            if self.use_arctan_step and ref_name in self.fitted_models:
                ref_spectrum = self.get_fitted_spectrum(ref_name)
                if ref_spectrum is None:
                    ref_spectrum = self.reference_spectra[ref_name]
                
                #  CRITICAL FIX: Use fitted models as-is, without additional smoothing
                shifted_spectrum = self.apply_energy_shift(ref_spectrum, final_shifts_dict[ref_name])
                # Skip deconvolution for fitted models to preserve fitted peak shapes
                fitted_spectrum += final_fractions_dict[ref_name] * shifted_spectrum
            else:
                ref_spectrum = self.reference_spectra[ref_name]
                # Apply deconvolution only to original reference spectra
                shifted_spectrum = self.apply_energy_shift(ref_spectrum, final_shifts_dict[ref_name])
                deconvolved_spectrum = smooth_with_kernel_reflect(shifted_spectrum, kernel)
                fitted_spectrum += final_fractions_dict[ref_name] * deconvolved_spectrum
            #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 

        #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
        target_for_comparison = self.target_spectrum
        #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
        
        final_r_factor = self.calculate_r_factor(target_for_comparison, fitted_spectrum)
        total_time = time.time() - total_start_time
        print(f" Brute force finished in {total_time:.2f} seconds.")

        # Store results
        self.results = {
            'success': True,
            'method': 'brute_force_deconvolution',
            'r_factor': final_r_factor,
            'fractions': final_fractions_dict,
            'shifts': final_shifts_dict,
            'fitted_spectrum': fitted_spectrum,
            'deconvolution_sigma': final_sigma,
            'residuals': target_for_comparison - fitted_spectrum,
            'computation_time': total_time,
            #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
            'uses_arctan_modeling': self.use_arctan_step,
            'peak_function_type': self.peak_function_type if self.use_arctan_step else None
            #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
        }
        
        print(f"   Best R-factor found: {final_r_factor:.6f}")
        print(f"   Benchmark: ≤ {self.r_factor_benchmark:.3f}")
        if final_r_factor <= self.r_factor_benchmark:
            print(f"    Benchmark met!")
        else:
            print(f"    Benchmark NOT met.")
            
        print(f"\n   Component fractions:")
        for name, fraction in final_fractions_dict.items():
            shift = final_shifts_dict[name]
            if self.use_arctan_step and name in self.fitted_models:
                ref_type = "FITTED (no smoothing)"
            else:
                ref_type = "ORIGINAL (with deconvolution)"
            print(f"     {name}: {fraction:.3f} ({fraction*100:.1f}%) [shift: {shift:+.2f} eV] [{ref_type}]")
            
        return self.results

    def plot_results(self):
        """Plot fitting results."""
        if not self.results or not self.results['success']:
            print("No valid results to plot. Please check the fitting process.")
            return
        
        ref_names = list(self.results['fractions'].keys())
        fractions = list(self.results['fractions'].values())
        
        # Create subplots: 2 rows, 2 columns
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
        # Use fitted target if available for comparison
        target_spectrum = self.target_spectrum
        if self.use_arctan_step and 'target' in self.fitted_models:
            fitted_target = self.get_fitted_spectrum('target')
            if fitted_target is not None:
                target_spectrum = fitted_target
        #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
        
        # Plot 1: Target vs Fitted spectrum
        axes[0, 0].plot(self.energy, target_spectrum, 'ko-', 
                       markersize=3, linewidth=1, label='Target', alpha=0.8)
        axes[0, 0].plot(self.energy, self.results['fitted_spectrum'], 'r-',
                       linewidth=2, label='Fitted')
        axes[0, 0].set_xlabel('Energy (eV)')
        axes[0, 0].set_ylabel('Absorption')
        #  TARGET: Always use original experimental data for comparison
        target_spectrum = self.target_spectrum
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        axes[0, 1].plot(self.energy, self.results['residuals'], 'g-', linewidth=1)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Energy (eV)')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Fit Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Component contributions
        axes[1, 0].plot(self.energy, target_spectrum, 'k-',
                       linewidth=2, label='Target', alpha=0.7)
        
        import matplotlib.cm as cm
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(self.reference_spectra)))
        
        # Create deconvolution kernel from results
        sigma = self.results.get('deconvolution_sigma', 1.0)
        kernel_size = max(3, int(6 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = create_gaussian_kernel(kernel_size, sigma)
            
        for i, (ref_name, ref_spectrum) in enumerate(self.reference_spectra.items()):
            fraction = self.results['fractions'][ref_name]
            
            if fraction > 0.01:  # Only show significant components
                shift = self.results['shifts'][ref_name]
                
                #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
                #  CRITICAL FIX: Match LCF calculation logic for plotting
                if self.use_arctan_step and ref_name in self.fitted_models:
                    # Use fitted models as-is, without additional smoothing
                    spectrum_to_use = self.get_fitted_spectrum(ref_name)
                    if spectrum_to_use is None:
                        spectrum_to_use = ref_spectrum
                    
                    shifted_spectrum = self.apply_energy_shift(spectrum_to_use, shift)
                    # Skip deconvolution for fitted models to preserve fitted peak shapes
                    deconvolved = shifted_spectrum
                else:
                    # Apply deconvolution only to original reference spectra
                    spectrum_to_use = ref_spectrum
                    shifted_spectrum = self.apply_energy_shift(spectrum_to_use, shift)
                    deconvolved = smooth_with_kernel_reflect(shifted_spectrum, kernel)
                #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
                
                weighted_spectrum = fraction * deconvolved
                
                axes[1, 0].plot(self.energy, weighted_spectrum, '--',
                               color=colors[i], linewidth=1.5,
                               label=f'{ref_name} ({fraction:.2f})')
        
        axes[1, 0].set_xlabel('Energy (eV)')
        axes[1, 0].set_ylabel('Absorption')
        axes[1, 0].set_title('Component Contributions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Component fractions
        axes[1, 1].bar(range(len(ref_names)), fractions,
                       color=colors[:len(ref_names)], alpha=0.7)
        axes[1, 1].set_xticks(range(len(ref_names)))
        axes[1, 1].set_xticklabels(ref_names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Fraction')
        axes[1, 1].set_title('Component Fractions')
        axes[1, 1].set_ylim(0, max(1, max(fractions) * 1.1))
        
        # Add value labels
        bars = axes[1, 1].containers[0]
        for bar, fraction in zip(bars, fractions):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{fraction:.3f}', ha='center', va='bottom')
        
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
        # If arctan modeling is enabled, show additional plots
        if self.use_arctan_step and self.fitted_models:
            self._plot_arctan_modeling_summary()
        #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
    
    #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
    def _plot_arctan_modeling_summary(self):
        """Plot summary of arctan step + peak modeling results."""
        fitted_spectra = {name: model for name, model in self.fitted_models.items() 
                          if model['success']}
        
        if not fitted_spectra:
            return
        
        n_spectra = len(fitted_spectra)
        if n_spectra == 0:
            return
        
        # Calculate grid layout
        n_cols = min(3, n_spectra)
        n_rows = (n_spectra + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_spectra == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (spectrum_name, model) in enumerate(fitted_spectra.items()):
            ax = axes[idx] if n_spectra > 1 else axes[0]
            
            # Plot original and fitted
            ax.plot(model['energy'], model['spectrum'], 'ko-', markersize=2, 
                   label='Original', alpha=0.7)
            ax.plot(model['energy'], model['fitted_spectrum'], 'r-', linewidth=2, 
                   label=f'Fitted ({self.peak_function_type})')
            
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Absorption')
            ax.set_title(f'{spectrum_name}\nR = {model["r_factor"]:.6f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_spectra, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Arctan Step + {self.peak_function_type.title()} Peak Fitting Results', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
    
    def get_results(self):
        """Get fitting results."""
        return self.results

    def _optimize_fractions_for_shifts_deconv(self, shifts, sigma=1.5):
        """Helper to optimize fractions for given shifts and deconvolution."""
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        # Create deconvolution kernel
        kernel_size = max(3, int(6 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = create_gaussian_kernel(kernel_size, sigma)
        
        shifted_deconvolved_spectra = []
        for i, name in enumerate(ref_names):
            #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
            #  CRITICAL FIX: Match LCF calculation logic
            if self.use_arctan_step and name in self.fitted_models:
                spectrum_to_use = self.get_fitted_spectrum(name)
                if spectrum_to_use is None:
                    spectrum_to_use = self.reference_spectra[name]
                # Use fitted models as-is, without additional smoothing
                shifted = self.apply_energy_shift(spectrum_to_use, shifts[i])
                deconvolved = shifted  # No deconvolution for fitted models
            else:
                spectrum_to_use = self.reference_spectra[name]
                # Apply deconvolution only to original reference spectra
                shifted = self.apply_energy_shift(spectrum_to_use, shifts[i])
                deconvolved = smooth_with_kernel_reflect(shifted, kernel)
            #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
            
            shifted_deconvolved_spectra.append(deconvolved)
            
        shifted_deconvolved_spectra = np.array(shifted_deconvolved_spectra).T
        
        #  TARGET: Always use original experimental data  
        target_spectrum = self.target_spectrum
        
        def objective(fractions):
            fractions = np.abs(fractions)
            total = np.sum(fractions)
            if total > 0:
                fractions = fractions / total
            
            fitted = np.dot(shifted_deconvolved_spectra, fractions)
            return self.calculate_r_factor(target_spectrum, fitted)
        
        initial_fractions = np.ones(n_refs) / n_refs
        bounds = [(0, 1)] * n_refs
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(objective, initial_fractions,
                        method='SLSQP', bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 200, 'ftol': 1e-7})
        
        if result.success:
            fractions = result.x / np.sum(result.x)
            fitted_spectrum = np.dot(shifted_deconvolved_spectra, fractions)
            return self.calculate_r_factor(target_spectrum, fitted_spectrum)
        
        return np.inf

    def _optimize_fractions_for_shifts_deconv_fast(self, shifts, sigma=1.5):
        """Faster version of fraction optimization for deconvolution."""
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        kernel_size = max(3, int(6 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = create_gaussian_kernel(kernel_size, sigma)
        
        shifted_deconvolved_spectra = []
        for i, name in enumerate(ref_names):
            #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
            if self.use_arctan_step and name in self.fitted_models:
                spectrum_to_use = self.get_fitted_spectrum(name)
                if spectrum_to_use is None:
                    spectrum_to_use = self.reference_spectra[name]
            else:
                spectrum_to_use = self.reference_spectra[name]
            #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
            
            deconvolved = smooth_with_kernel_reflect(self.apply_energy_shift(spectrum_to_use, shifts[i]), kernel)
            shifted_deconvolved_spectra.append(deconvolved)
        
        shifted_deconvolved_spectra = np.array(shifted_deconvolved_spectra).T
        
        #  TARGET: Always use original experimental data
        target_spectrum = self.target_spectrum
        
        def objective(fractions):
            fractions = np.abs(fractions)
            total = np.sum(fractions)
            if total > 0:
                fractions = fractions / total
            
            fitted = np.dot(shifted_deconvolved_spectra, fractions)
            return self.calculate_r_factor(target_spectrum, fitted)
        
        initial_fractions = np.ones(n_refs) / n_refs
        bounds = [(0, 1)] * n_refs
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(objective, initial_fractions,
                        method='SLSQP', bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 100, 'ftol': 1e-6}) # Faster options
        
        if result.success:
            fractions = result.x / np.sum(result.x)
            fitted_spectrum = np.dot(shifted_deconvolved_spectra, fractions)
            return self.calculate_r_factor(target_spectrum, fitted_spectrum)
        
        return np.inf

    def _optimize_fractions_for_shifts_deconv_ultrafast(self, shifts, sigma=1.5):
        """Ultra-fast NNLS-based fraction optimization for deconvolution."""
        from scipy.optimize import nnls

        ref_names = list(self.reference_spectra.keys())
        
        kernel_size = max(3, int(6 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = create_gaussian_kernel(kernel_size, sigma)

        shifted_deconvolved_matrix = []
        for i, name in enumerate(ref_names):
            #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
            if self.use_arctan_step and name in self.fitted_models:
                spectrum_to_use = self.get_fitted_spectrum(name)
                if spectrum_to_use is None:
                    spectrum_to_use = self.reference_spectra[name]
            else:
                spectrum_to_use = self.reference_spectra[name]
            #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
            
            deconvolved = smooth_with_kernel_reflect(self.apply_energy_shift(spectrum_to_use, shifts[i]), kernel)
            shifted_deconvolved_matrix.append(deconvolved)

        shifted_deconvolved_matrix = np.array(shifted_deconvolved_matrix).T

        #  TARGET: Always use original experimental data
        target_spectrum = self.target_spectrum

        fractions, _ = nnls(shifted_deconvolved_matrix, target_spectrum)
        
        total_fraction = np.sum(fractions)
        if total_fraction > 0:
            fractions /= total_fraction
        else:
            return np.inf

        fitted_spectrum = np.dot(shifted_deconvolved_matrix, fractions)
        return self.calculate_r_factor(target_spectrum, fitted_spectrum)

    def _calculate_final_result_deconv(self, best_shifts, method='brute_force_deconvolution'):
        """Calculate final results with best shifts for deconvolution."""
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        #  TARGET: Always use original experimental data
        target_spectrum = self.target_spectrum
        
        def objective_fractions(fractions):
            fractions = np.abs(fractions)
            total = np.sum(fractions)
            if total > 0:
                fractions = fractions / total
            
            fitted = np.zeros_like(target_spectrum)
            for i, ref_name in enumerate(ref_names):
                #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
                if self.use_arctan_step and ref_name in self.fitted_models:
                    ref_spectrum = self.get_fitted_spectrum(ref_name)
                    if ref_spectrum is None:
                        ref_spectrum = self.reference_spectra[ref_name]
                else:
                    ref_spectrum = self.reference_spectra[ref_name]
                #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
                
                shifted_spectrum = self.apply_energy_shift(ref_spectrum, best_shifts[i])
                fitted += fractions[i] * shifted_spectrum
            
            return self.calculate_r_factor(target_spectrum, fitted)
        
        initial_fractions = np.ones(n_refs) / n_refs
        bounds = [(0, 1)] * n_refs
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(objective_fractions, initial_fractions,
                         method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000})
        
        final_fractions = result.x
        final_fractions = np.abs(final_fractions)
        total = np.sum(final_fractions)
        if total > 0:
            final_fractions = final_fractions / total
        
        # Calculate final fitted spectrum
        fitted_spectrum = np.zeros_like(target_spectrum)
        for i, ref_name in enumerate(ref_names):
            #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
            if self.use_arctan_step and ref_name in self.fitted_models:
                ref_spectrum = self.get_fitted_spectrum(ref_name)
                if ref_spectrum is None:
                    ref_spectrum = self.reference_spectra[ref_name]
            else:
                ref_spectrum = self.reference_spectra[ref_name]
            #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
            
            shifted_spectrum = self.apply_energy_shift(ref_spectrum, best_shifts[i])
            fitted_spectrum += final_fractions[i] * shifted_spectrum
        
        r_factor = self.calculate_r_factor(target_spectrum, fitted_spectrum)
        
        return {
            'success': True,
            'method': method,
            'r_factor': r_factor,
            'fractions': dict(zip(ref_names, final_fractions)),
            'shifts': dict(zip(ref_names, best_shifts)),
            'fitted_spectrum': fitted_spectrum,
            'residuals': target_spectrum - fitted_spectrum,
            #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
            'uses_arctan_modeling': self.use_arctan_step,
            'peak_function_type': self.peak_function_type if self.use_arctan_step else None
            #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
        }
    
    def _print_brute_force_results(self, results, computation_time):
        """Print detailed brute force results."""
        r_factor = results['r_factor']
        
        print(f"\n BRUTE FORCE RESULTS:")
        print(f"    Final R-factor: {r_factor:.6f}")
        print(f"    Benchmark: ≤ {self.r_factor_benchmark:.4f}")
        
        #  ARCTAN STEP + PEAK FITTING INTEGRATION   START 
        if self.use_arctan_step:
            print(f"    Arctan step modeling: Enabled ({self.peak_function_type})")
        #  ARCTAN STEP + PEAK FITTING INTEGRATION  END 
        
        if r_factor <= self.r_factor_benchmark:
            print(f"    EXCELLENT! Supreme precision benchmark met!")
        elif r_factor < 0.01:
            print(f"    Excellent supreme precision fit!")
        elif r_factor < 0.02:
            print(f"    Very good precision fit!")
        elif r_factor < 0.05:
            print(f"    Good fit")
        else:
            print(f"     Acceptable fit")
        
        print(f"     Computation time: {computation_time:.1f}s")
        
        print(f"\n    Component fractions:")
        for name, fraction in results['fractions'].items():
            print(f"      {name}: {fraction:.3f} ({fraction*100:.1f}%)")
        
        print(f"\n    Energy shifts (eV):")
        for name, shift in results['shifts'].items():
            print(f"      {name}: {shift:+.2f}")
    
    def set_global_e0(self, energy, spectrum_data=None, spectrum_name="first reference"):
        """Set global E0 value that will be used for all reference fits."""
        if self.e0_is_set:
            print(f" Global E0 already set to: {self.global_e0:.2f} eV")
            return self.global_e0
        
        print(f"\n Setting Global E0 (Arctan Midpoint) for ALL References")
        print("=" * 60)
        print(f"This E0 value will be used for ALL reference spectrum fits.")
        print(f"Energy range: {energy.min():.1f} - {energy.max():.1f} eV")
        
        while True:
            try:
                e0_manual = float(input(f"Enter Global E0 (arctan midpoint) for ALL references (eV): "))
                if energy.min() <= e0_manual <= energy.max():
                    self.global_e0 = e0_manual
                    self.e0_is_set = True
                    print(f" Global E0 set to: {self.global_e0:.2f} eV")
                    print(f" This value will be used for ALL subsequent reference fits.")
                    return self.global_e0
                else:
                    print(f"E0 must be within energy range {energy.min():.1f} - {energy.max():.1f} eV")
            except ValueError:
                print("Please enter a valid number")
    
    def get_global_e0(self):
        """Get the global E0 value."""
        if not self.e0_is_set:
            raise ValueError("Global E0 has not been set yet. Call set_global_e0() first.")
        return self.global_e0
    
    def reset_global_e0(self):
        """Reset global E0 setting (for testing or re-analysis)."""
        self.global_e0 = None
        self.e0_is_set = False
        print(" Global E0 reset. Will need to be set again for next analysis.")

def main():
    """Test the deconvolution LCF."""
    print("Deconvolution LCF module ready for use.")
    print("New features:")
    print("- Arctan step + peak modeling for XANES analysis")
    print("- Interactive peak function selection (Gaussian/Voigt/Pseudo-Voigt)")
    print("- Consistent edge modeling across all spectra")


if __name__ == "__main__":
    main()
