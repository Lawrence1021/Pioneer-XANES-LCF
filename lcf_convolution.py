"""
Convolution-based Linear Combination Fitting - Clean Implementation

Most comprehensive LCF method with instrumental broadening effects.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy import signal, ndimage
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


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


def parallel_convolution_worker(shift_chunk, energy, target_spectrum, reference_spectra, kernel_type='gaussian', kernel_width=1.0):
    """
    Worker function for parallel convolution optimization.
    """
    try:
        best_r = float('inf')
        best_shifts = None
        
        # Create convolution kernel
        if kernel_type == 'gaussian':
            sigma = kernel_width / (2 * np.sqrt(2 * np.log(2)))
            x = np.linspace(-3*sigma, 3*sigma, 51)
            kernel = np.exp(-x**2 / (2 * sigma**2))
        else:  # lorentzian
            gamma = kernel_width / 2
            x = np.linspace(-5*gamma, 5*gamma, 51)
            kernel = gamma / (np.pi * (x**2 + gamma**2))
        
        kernel = kernel / np.sum(kernel)
        
        for shifts in shift_chunk:
            # Apply energy shifts and convolution to references
            shifted_refs = {}
            for i, (ref_name, ref_spec) in enumerate(reference_spectra.items()):
                # Apply energy shift
                shifted_energy = energy + shifts[i]
                interp_func = interp1d(shifted_energy, ref_spec,
                                     kind='linear', bounds_error=False,
                                     fill_value=(ref_spec[0], ref_spec[-1]))
                shifted_ref = interp_func(energy)
                
                # Apply convolution
                convolved = smooth_with_kernel_reflect(shifted_ref, kernel)
                shifted_refs[ref_name] = convolved
            
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
                
                # Calculate R-factor
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


class ConvolutionLCF:
    """
    Clean implementation of convolution-based LCF.
    """
    
    def __init__(self, r_factor_benchmark=0.04):
        self.energy = None
        self.target_spectrum = None
        self.reference_spectra = {}
        self.results = {}
        self.r_factor_benchmark = r_factor_benchmark
    
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
    
    def gaussian_kernel(self, fwhm, size=51):
        """Create Gaussian convolution kernel."""
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        x = np.linspace(-3*sigma, 3*sigma, size)
        kernel = np.exp(-x**2 / (2 * sigma**2))
        return kernel / np.sum(kernel)
    
    def lorentzian_kernel(self, fwhm, size=51):
        """Create Lorentzian convolution kernel."""
        gamma = fwhm / 2
        x = np.linspace(-5*gamma, 5*gamma, size)
        kernel = gamma / (np.pi * (x**2 + gamma**2))
        return kernel / np.sum(kernel)

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
    
    def fit(self, max_shift=7.5, kernel_type='gaussian'):
        """
        Fit the target spectrum by calling the brute-force method.
        
        Parameters:
        -----------
        max_shift : float
            Maximum energy shift allowed (eV). Can be overridden by user input.
        kernel_type : str
            Type of convolution kernel
        """
        print(" Starting comprehensive brute-force fitting for convolution...")
        return self.brute_force_fit(max_shift=max_shift, kernel_type=kernel_type)

    def brute_force_fit(self, max_shift=7.5, kernel_type='gaussian', target_time_minutes=29):
        """
        Perform a true brute-force convolution LCF fitting by simultaneously 
        searching the entire parameter space of fractions, shifts, and kernel width.

        This method uses a multi-level, randomized sampling strategy to explore
        the combined parameter space, which is essential for the complex interplay
        between convolution and fitting parameters.
        
        Parameters:
        -----------
        max_shift : float
            Maximum energy shift allowed (eV). Can be overridden by user input.
        kernel_type : str
            Type of convolution kernel ('gaussian' or 'lorentzian').
        target_time_minutes : float
            Target computation time in minutes.
            
        Returns:
        --------
        dict : Fitting results.
        """
        print(f"\n Starting True Brute-Force Convolution LCF (3-Stage)...")
        print(f"   Strategy: 1. Global Randomized Search -> 2. Refined Search -> 3. Gradient Optimization.")
        print(f"   Max energy shift: ±{max_shift} eV")
        print(f"   Kernel type: {kernel_type}")
        print(f"   Target time: ~{target_time_minutes} minutes")
        
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        if n_refs < 2:
            return {'success': False, 'error': 'Need at least 2 references'}

        import time
        from tqdm import tqdm

        total_start_time = time.time()
        target_time_seconds = target_time_minutes * 60

        #  Stage 1 & 2: Randomized Global and Refined Search ---
        print("\n Stage 1 & 2: Randomized global and refined search...")

        # Convolution is computationally heavier. Estimate ~0.25ms per sample.
        n_samples = int(target_time_seconds / 0.00025)
        print(f"   Estimated samples for target time: {n_samples:,}")

        best_r_factor = np.inf
        best_params = None

        # Convolution parameters to search
        kernel_width_range = (0.1, 5.0) # FWHM for Gaussian/Lorentzian

        # Set up parallel processing for convolution
        n_cores = mp.cpu_count()
        print(f"   Using {n_cores} CPU cores for parallel convolution processing")
        
        # Generate shift combinations for parallel processing
        n_samples_parallel = min(1500, n_samples // 2)  # Use parallel for first half
        all_shift_combinations = []
        
        for i in range(n_samples_parallel):
            shifts = [np.random.uniform(-max_shift, max_shift) for _ in range(n_refs)]
            all_shift_combinations.append(shifts)
        
        # Split combinations into chunks for parallel processing
        chunk_size = max(1, n_samples_parallel // n_cores)
        shift_chunks = [all_shift_combinations[i:i + chunk_size] 
                       for i in range(0, len(all_shift_combinations), chunk_size)]
        
        # Process chunks in parallel
        print(f"   Processing {len(shift_chunks)} convolution chunks in parallel...")
        try:
            with mp.Pool(n_cores) as pool:
                # Create partial function with fixed arguments
                worker_func = partial(parallel_convolution_worker, 
                                    energy=self.energy, 
                                    target_spectrum=self.target_spectrum,
                                    reference_spectra=self.reference_spectra,
                                    kernel_type=kernel_type,
                                    kernel_width=1.0)
                
                # Process chunks in parallel
                results = pool.map(worker_func, shift_chunks)
            
            # Find best result from parallel processing
            best_r_parallel = float('inf')
            best_shifts_parallel = None
            
            for r_factor, shifts in results:
                if r_factor < best_r_parallel:
                    best_r_parallel = r_factor
                    best_shifts_parallel = shifts
            
            print(f"   Parallel convolution completed: Best R-factor = {best_r_parallel:.6f}")
            
            # Update best overall result if parallel processing found better solution
            if best_shifts_parallel is not None:
                best_overall_r = best_r_parallel
                best_overall_result = best_shifts_parallel
        except Exception as e:
            print(f"   Parallel processing failed: {e}, continuing with sequential processing")
        
        # Continue with remaining sequential processing for refinement

        try:
            with tqdm(total=n_samples, desc="Brute-Force Conv", unit="combo") as pbar:
                
                for i in range(n_samples):

                    #  Parameter Generation 
                    fractions = np.random.random(n_refs)
                    fractions /= np.sum(fractions)

                    if best_params and i > n_samples * 0.1:
                        # Refine search around the best known parameters
                        refinement_factor = 1.0 - (i / n_samples)
                        
                        best_fracs = best_params['fractions']
                        best_shifts = best_params['shifts']
                        best_kw = best_params['kernel_width']

                        # Perturb and re-normalize
                        fractions = np.abs(best_fracs + np.random.normal(0, 0.1 * refinement_factor, n_refs))
                        fractions /= np.sum(fractions)
                        shifts = best_shifts + np.random.normal(0, max_shift * 0.1 * refinement_factor, n_refs)
                        shifts = np.clip(shifts, -max_shift, max_shift)
                        kernel_width = best_kw + np.random.normal(0, (kernel_width_range[1] - kernel_width_range[0]) * 0.1 * refinement_factor)
                        kernel_width = np.clip(kernel_width, kernel_width_range[0], kernel_width_range[1])
                    else:
                        # Pure random search
                        shifts = np.random.uniform(-max_shift, max_shift, n_refs)
                        kernel_width = np.random.uniform(kernel_width_range[0], kernel_width_range[1])

                    #  Evaluation 
                    try:
                        fitted_spectrum = np.zeros_like(self.target_spectrum)
                        
                        if kernel_type == 'gaussian':
                            kernel = self.gaussian_kernel(kernel_width)
                        else: # lorentzian
                            kernel = self.lorentzian_kernel(kernel_width)

                        for j, ref_name in enumerate(ref_names):
                            ref_spectrum = self.reference_spectra[ref_name]
                            shifted_spectrum = self.apply_energy_shift(ref_spectrum, shifts[j])
                            convolved_spectrum = smooth_with_kernel_reflect(shifted_spectrum, kernel)
                            fitted_spectrum += fractions[j] * convolved_spectrum
                        
                        r_factor = self.calculate_r_factor(self.target_spectrum, fitted_spectrum)

                        if r_factor < best_r_factor:
                            best_r_factor = r_factor
                            best_params = {'fractions': np.copy(fractions), 'shifts': np.copy(shifts), 'kernel_width': kernel_width}
                            pbar.set_postfix({'Best R': f'{best_r_factor:.6f}'})
                    
                    except Exception:
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
            kernel_width = params[-1]
            
            try:
                if kernel_type == 'gaussian':
                    kernel = self.gaussian_kernel(kernel_width)
                else:
                    kernel = self.lorentzian_kernel(kernel_width)
                
                fitted = np.zeros_like(self.target_spectrum)
                for i, ref_name in enumerate(ref_names):
                    ref_spectrum = self.reference_spectra[ref_name]
                    shifted_spectrum = self.apply_energy_shift(ref_spectrum, shifts[i])
                    convolved_spectrum = smooth_with_kernel_reflect(shifted_spectrum, kernel)
                    fitted += fractions[i] * convolved_spectrum
                
                return self.calculate_r_factor(self.target_spectrum, fitted)
            except Exception:
                return np.inf

        initial_params_for_optimizer = np.concatenate([
            best_params['fractions'], 
            best_params['shifts'], 
            [best_params['kernel_width']]
        ])
        
        bounds = ([(0, 1)] * n_refs + 
                  [(-max_shift, max_shift)] * n_refs + 
                  [(kernel_width_range[0], kernel_width_range[1])])
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
            final_kernel_width = final_params[-1]
            # Normalize fractions just in case
            final_fractions = np.abs(final_fractions)
            final_fractions /= np.sum(final_fractions)
        else:
            print(" Gradient optimization failed. Using best result from random search.")
            final_fractions = best_params['fractions']
            final_shifts = best_params['shifts']
            final_kernel_width = best_params['kernel_width']

        #  Finalize and Return Results 
        final_fractions_dict = dict(zip(ref_names, final_fractions))
        final_shifts_dict = dict(zip(ref_names, final_shifts))

        # Recalculate final spectrum with best parameters
        fitted_spectrum = np.zeros_like(self.target_spectrum)
        if kernel_type == 'gaussian':
            final_kernel = self.gaussian_kernel(final_kernel_width)
        else:
            final_kernel = self.lorentzian_kernel(final_kernel_width)

        for i, ref_name in enumerate(ref_names):
            ref_spectrum = self.reference_spectra[ref_name]
            shifted_spectrum = self.apply_energy_shift(ref_spectrum, final_shifts_dict[ref_name])
            convolved_spectrum = smooth_with_kernel_reflect(shifted_spectrum, final_kernel)
            fitted_spectrum += final_fractions_dict[ref_name] * convolved_spectrum

        final_r_factor = self.calculate_r_factor(self.target_spectrum, fitted_spectrum)
        total_time = time.time() - total_start_time
        print(f" Brute force finished in {total_time:.2f} seconds.")
        
        self.results = {
            'success': True,
            'method': 'convolution_brute_force',
            'r_factor': final_r_factor,
            'fractions': final_fractions_dict,
            'shifts': final_shifts_dict,
            'kernel_type': kernel_type,
            'kernel_width': final_kernel_width,
            'fitted_spectrum': fitted_spectrum,
            'residuals': self.target_spectrum - fitted_spectrum,
            'computation_time': total_time
        }
        
        print(f"   Best R-factor found: {final_r_factor:.6f}")
        print(f"   Benchmark: ≤ {self.r_factor_benchmark:.3f}")
        if final_r_factor <= self.r_factor_benchmark:
            print(f"   Benchmark met!")
        else:
            print(f"   Benchmark NOT met.")
            
        print(f"\n   Component fractions:")
        for name, fraction in self.results['fractions'].items():
            print(f"     {name}: {fraction:.3f} ({fraction*100:.1f}%)")
            
        return self.results

    def plot_results(self):
        """Plot fitting results."""
        if not self.results or not self.results['success']:
            print(" No valid results to plot. Please check fitting status.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Target vs Fitted
        axes[0, 0].plot(self.energy, self.target_spectrum, 'ko-', 
                       markersize=3, linewidth=1, label='Target', alpha=0.8)
        axes[0, 0].plot(self.energy, self.results['fitted_spectrum'], 'r-',
                       linewidth=2, label='Fitted')
        axes[0, 0].set_xlabel('Energy (eV)')
        axes[0, 0].set_ylabel('Absorption')
        axes[0, 0].set_title(f'Convolution LCF (R = {self.results["r_factor"]:.4f})')
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
        axes[1, 0].plot(self.energy, self.target_spectrum, 'k-',
                       linewidth=2, label='Target', alpha=0.7)
        
        import matplotlib.cm as cm
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(self.reference_spectra)))
        
        # Create kernel from results
        kernel_width = self.results.get('kernel_width', 1.0)
        kernel_type = self.results.get('kernel_type', 'gaussian')
        if kernel_type == 'gaussian':
            kernel = self.gaussian_kernel(kernel_width)
        else:
            kernel = self.lorentzian_kernel(kernel_width)

        for i, (ref_name, ref_spectrum) in enumerate(self.reference_spectra.items()):
            fraction = self.results['fractions'][ref_name]
            
            if fraction > 0.01:  # Only show significant components
                shift = self.results['shifts'][ref_name]
                shifted_spectrum = self.apply_energy_shift(ref_spectrum, shift)
                
                # Apply convolution
                convolved_spectrum = smooth_with_kernel_reflect(shifted_spectrum, kernel)
                
                weighted_spectrum = fraction * convolved_spectrum
                
                axes[1, 0].plot(self.energy, weighted_spectrum, '--',
                               color=colors[i], linewidth=1.5,
                               label=f'{ref_name} ({fraction:.2f})')
        
        axes[1, 0].set_xlabel('Energy (eV)')
        axes[1, 0].set_ylabel('Absorption')
        axes[1, 0].set_title('Component Contributions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Fraction bar chart
        ref_names = list(self.results['fractions'].keys())
        fractions = list(self.results['fractions'].values())
        
        bars = axes[1, 1].bar(range(len(ref_names)), fractions,
                             color=colors[:len(ref_names)], alpha=0.7)
        axes[1, 1].set_xticks(range(len(ref_names)))
        axes[1, 1].set_xticklabels(ref_names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Fraction')
        axes[1, 1].set_title('Component Fractions')
        axes[1, 1].set_ylim(0, max(1, max(fractions) * 1.1))
        
        # Add value labels
        for bar, fraction in zip(bars, fractions):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{fraction:.3f}', ha='center', va='bottom')
        
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def get_results(self):
        """Get fitting results."""
        return self.results

    def _optimize_fractions_for_shifts_conv(self, shifts, kernel_type, kernel_width=1.0):
        """Helper to optimize fractions for given shifts and convolution."""
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        if kernel_type == 'gaussian':
            kernel = self.gaussian_kernel(kernel_width)
        else:
            kernel = self.lorentzian_kernel(kernel_width)
        
        convolved_spectra = []
        for i, name in enumerate(ref_names):
            shifted = self.apply_energy_shift(self.reference_spectra[name], shifts[i])
            convolved = smooth_with_kernel_reflect(shifted, kernel)
            convolved_spectra.append(convolved)
            
        convolved_spectra = np.array(convolved_spectra).T
        
        def objective(fractions):
            fractions = np.abs(fractions)
            total = np.sum(fractions)
            if total > 0:
                fractions = fractions / total
            
            fitted = np.dot(convolved_spectra, fractions)
            return self.calculate_r_factor(self.target_spectrum, fitted)
        
        initial_fractions = np.ones(n_refs) / n_refs
        bounds = [(0, 1)] * n_refs
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(objective, initial_fractions,
                        method='SLSQP', bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 200, 'ftol': 1e-7})
        
        if result.success:
            fractions = result.x / np.sum(result.x)
            fitted_spectrum = np.dot(convolved_spectra, fractions)
            return self.calculate_r_factor(self.target_spectrum, fitted_spectrum)
        
        return np.inf

    def _optimize_fractions_for_shifts_conv_fast(self, shifts, kernel_type, kernel_width=1.0):
        """Faster version of fraction optimization for convolution."""
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        if kernel_type == 'gaussian':
            kernel = self.gaussian_kernel(kernel_width)
        else:
            kernel = self.lorentzian_kernel(kernel_width)
        
        convolved_spectra = np.array([
            smooth_with_kernel_reflect(self.apply_energy_shift(self.reference_spectra[name], shifts[i]), kernel)
            for i, name in enumerate(ref_names)
        ]).T
        
        def objective(fractions):
            fractions = np.abs(fractions)
            total = np.sum(fractions)
            if total > 0:
                fractions = fractions / total
            
            fitted = np.dot(convolved_spectra, fractions)
            return self.calculate_r_factor(self.target_spectrum, fitted)
        
        initial_fractions = np.ones(n_refs) / n_refs
        bounds = [(0, 1)] * n_refs
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(objective, initial_fractions,
                        method='SLSQP', bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 100, 'ftol': 1e-6}) # Faster options
        
        if result.success:
            fractions = result.x / np.sum(result.x)
            fitted_spectrum = np.dot(convolved_spectra, fractions)
            return self.calculate_r_factor(self.target_spectrum, fitted_spectrum)
        
        return np.inf

    def _optimize_fractions_for_shifts_conv_ultrafast(self, shifts, kernel_type, kernel_width=1.0):
        """Ultra-fast NNLS-based fraction optimization for convolution."""
        from scipy.optimize import nnls

        ref_names = list(self.reference_spectra.keys())
        
        if kernel_type == 'gaussian':
            kernel = self.gaussian_kernel(kernel_width)
        else:
            kernel = self.lorentzian_kernel(kernel_width)

        convolved_matrix = np.array([
            smooth_with_kernel_reflect(self.apply_energy_shift(self.reference_spectra[name], shifts[i]), kernel)
            for i, name in enumerate(ref_names)
        ]).T

        fractions, _ = nnls(convolved_matrix, self.target_spectrum)
        
        total_fraction = np.sum(fractions)
        if total_fraction > 0:
            fractions /= total_fraction
        else:
            return np.inf

        fitted_spectrum = np.dot(convolved_matrix, fractions)
        return self.calculate_r_factor(self.target_spectrum, fitted_spectrum)

    def _gradient_helper_function(self, current_shifts, n_refs, max_shift, kernel_type):
        """Helper function for gradient calculation."""
        # Calculate numerical gradient with extremely small epsilon
        gradient = np.zeros(n_refs)
        eps = 0.01  # Extremely small epsilon for supreme precision gradient
        
        for i in range(n_refs):
            # Forward difference
            shifts_plus = current_shifts.copy()
            shifts_plus[i] = np.clip(shifts_plus[i] + eps, -max_shift, max_shift)
            r_plus = self._optimize_fractions_for_shifts_conv(shifts_plus, kernel_type)
            
            # Backward difference
            shifts_minus = current_shifts.copy()
            shifts_minus[i] = np.clip(shifts_minus[i] - eps, -max_shift, max_shift)
            r_minus = self._optimize_fractions_for_shifts_conv(shifts_minus, kernel_type)
            
            # Numerical gradient
            gradient[i] = (r_plus - r_minus) / (2 * eps)
        
        # Return the gradient for now
        return gradient

    def _gradient_update_helper(self, new_shifts, kernel_type, prev_r, best_local_r, best_local_result, current_shifts, learning_rate):
        """Helper function for gradient update."""
        new_r = self._optimize_fractions_for_shifts_conv(new_shifts, kernel_type)
        
        # Accept if better
        if new_r < prev_r:
            current_shifts = new_shifts
            prev_r = new_r
            
            if new_r < best_local_r:
                best_local_r = new_r
                best_local_result = current_shifts.copy()
        else:
            # Reduce learning rate if no improvement
            learning_rate *= 0.98  # Very slow reduction for supreme precision
            if learning_rate < 0.001:  # Extremely small threshold
                return "break"
                
                # Accept if better
                if new_r < prev_r:
                    current_shifts = new_shifts
                    prev_r = new_r
                    
                    if new_r < best_local_r:
                        best_local_r = new_r
                        best_local_result = current_shifts.copy()
                else:
                    # Reduce learning rate if no improvement
                    learning_rate *= 0.95  # Slower reduction for ultra-extreme precision
                    if learning_rate < 0.002:  # Much smaller threshold
                        return "break"
        
        return current_shifts, prev_r, best_local_r, best_local_result, learning_rate

    def _finalize_results_helper(self, best_local_r, best_overall_r, best_local_result, total_start_time, kernel_type):
        """Helper function to finalize results."""
        total_time = time.time() - total_start_time
        
        print(f"    Supreme precision brute force completed in {total_time/60:.1f} min")
        print(f"    Final R-factor: {best_local_r:.6f}")
        print(f"    Improvement: {best_overall_r:.6f} → {best_local_r:.6f}")
        
        # Calculate final results with best shifts
        final_result = self._calculate_final_result_conv(best_local_result, kernel_type, method='supreme_precision_brute_force_convolution')
        
        # Print detailed results
        self._print_brute_force_results(final_result, total_time)
        
        return final_result
    
    def _optimize_fractions_for_shifts_conv(self, shifts, kernel_type):
        """Optimize fractions for given energy shifts using convolution."""
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        def objective_fractions(params):
            fractions = params[:n_refs]
            kernel_width = params[n_refs]
            
            fractions = np.abs(fractions)
            total = np.sum(fractions)
            if total > 0:
                fractions = fractions / total
            
            kernel_width = max(0.1, min(kernel_width, 5.0))
            
            fitted = np.zeros_like(self.target_spectrum)
            for i, ref_name in enumerate(ref_names):
                ref_spectrum = self.reference_spectra[ref_name]
                shifted_spectrum = self.apply_energy_shift(ref_spectrum, shifts[i])
                
                if kernel_type == 'gaussian':
                    kernel = self.gaussian_kernel(kernel_width)
                else:  # lorentzian
                    kernel = self.lorentzian_kernel(kernel_width)
                
                convolved_spectrum = smooth_with_kernel_reflect(shifted_spectrum, kernel)
                fitted += fractions[i] * convolved_spectrum
            
            return self.calculate_r_factor(self.target_spectrum, fitted)
        
        initial_params = np.concatenate([np.ones(n_refs) / n_refs, [1.0]])
        bounds = [(0, 1)] * n_refs + [(0.1, 5.0)]  # Kernel width bounds
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:n_refs]) - 1}]
        
        try:
            result = minimize(objective_fractions, initial_params,
                            method='SLSQP', bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000, 'disp': False})
            return result.fun if result.success else np.inf
        except:
            return np.inf
    
    def _optimize_fractions_for_shifts_conv_fast(self, shifts, kernel_type):
        """Fast optimization for convolution."""
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        def objective_fractions(params):
            fractions = params[:n_refs]
            kernel_width = params[n_refs]
            
            fractions = np.abs(fractions)
            total = np.sum(fractions)
            if total > 0:
                fractions = fractions / total
            
            kernel_width = max(0.1, min(kernel_width, 5.0))
            
            fitted = np.zeros_like(self.target_spectrum)
            for i, ref_name in enumerate(ref_names):
                ref_spectrum = self.reference_spectra[ref_name]
                shifted_spectrum = self.apply_energy_shift(ref_spectrum, shifts[i])
                
                if kernel_type == 'gaussian':
                    kernel = self.gaussian_kernel(kernel_width)
                else:  # lorentzian
                    kernel = self.lorentzian_kernel(kernel_width)
                
                convolved_spectrum = smooth_with_kernel_reflect(shifted_spectrum, kernel)
                fitted += fractions[i] * convolved_spectrum
            
            return self.calculate_r_factor(self.target_spectrum, fitted)
        
        initial_params = np.concatenate([np.ones(n_refs) / n_refs, [1.0]])
        bounds = [(0, 1)] * n_refs + [(0.1, 5.0)]
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:n_refs]) - 1}]
        
        try:
            result = minimize(objective_fractions, initial_params,
                            method='SLSQP', bounds=bounds, constraints=constraints,
                            options={'maxiter': 50, 'disp': False})
            return result.fun if result.success else np.inf
        except:
            return np.inf
    
    def _optimize_fractions_for_shifts_conv_ultrafast(self, shifts, kernel_type):
        """Ultra-fast optimization for convolution."""
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        def objective_fractions(fractions):
            # Simplified: fixed kernel width for speed
            fractions = np.abs(fractions)
            total = np.sum(fractions)
            if total > 0:
                fractions = fractions / total
            
            kernel_width = 1.0  # Fixed kernel width for speed
            
            fitted = np.zeros_like(self.target_spectrum)
            for i, ref_name in enumerate(ref_names):
                ref_spectrum = self.reference_spectra[ref_name]
                shifted_spectrum = self.apply_energy_shift(ref_spectrum, shifts[i])
                
                if kernel_type == 'gaussian':
                    kernel = self.gaussian_kernel(kernel_width)
                else:  # lorentzian
                    kernel = self.lorentzian_kernel(kernel_width)
                
                convolved_spectrum = smooth_with_kernel_reflect(shifted_spectrum, kernel)
                fitted += fractions[i] * convolved_spectrum
            
            return self.calculate_r_factor(self.target_spectrum, fitted)
        
        initial_fractions = np.ones(n_refs) / n_refs
        bounds = [(0, 1)] * n_refs
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        try:
            result = minimize(objective_fractions, initial_fractions,
                            method='SLSQP', bounds=bounds, constraints=constraints,
                            options={'maxiter': 50, 'disp': False})
            return result.fun if result.success else np.inf
        except:
            return np.inf
    
    def _calculate_final_result_conv(self, best_shifts, kernel_type, method='brute_force_convolution'):
        """Calculate final results with best shifts for convolution."""
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        def objective_params(params):
            fractions = params[:n_refs]
            kernel_width = params[n_refs]
            
            fractions = np.abs(fractions)
            total = np.sum(fractions)
            if total > 0:
                fractions = fractions / total
            
            kernel_width = max(0.1, min(kernel_width, 5.0))
            
            fitted = np.zeros_like(self.target_spectrum)
            for i, ref_name in enumerate(ref_names):
                ref_spectrum = self.reference_spectra[ref_name]
                shifted_spectrum = self.apply_energy_shift(ref_spectrum, best_shifts[i])
                
                if kernel_type == 'gaussian':
                    kernel = self.gaussian_kernel(kernel_width)
                else:
                    kernel = self.lorentzian_kernel(kernel_width)
                
                convolved_spectrum = smooth_with_kernel_reflect(shifted_spectrum, kernel)
                fitted += fractions[i] * convolved_spectrum
            
            return self.calculate_r_factor(self.target_spectrum, fitted)
        
        initial_params = np.concatenate([np.ones(n_refs) / n_refs, [1.0]])
        bounds = [(0, 1)] * n_refs + [(0.1, 5.0)]
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:n_refs]) - 1}]
        
        result = minimize(objective_params, initial_params,
                         method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000})
        
        final_fractions = result.x[:n_refs]
        final_kernel_width = result.x[n_refs]
        
        final_fractions = np.abs(final_fractions)
        total = np.sum(final_fractions)
        if total > 0:
            final_fractions = final_fractions / total
        
        # Calculate final fitted spectrum
        fitted_spectrum = np.zeros_like(self.target_spectrum)
        for i, ref_name in enumerate(ref_names):
            ref_spectrum = self.reference_spectra[ref_name]
            shifted_spectrum = self.apply_energy_shift(ref_spectrum, best_shifts[i])
            
            if kernel_type == 'gaussian':
                kernel = self.gaussian_kernel(final_kernel_width)
            else:
                kernel = self.lorentzian_kernel(final_kernel_width)
            
            convolved_spectrum = smooth_with_kernel_reflect(shifted_spectrum, kernel)
            fitted_spectrum += final_fractions[i] * convolved_spectrum
        
        r_factor = self.calculate_r_factor(self.target_spectrum, fitted_spectrum)
        
        return {
            'success': True,
            'method': method,
            'r_factor': r_factor,
            'fractions': dict(zip(ref_names, final_fractions)),
            'shifts': dict(zip(ref_names, best_shifts)),
            'kernel_width': final_kernel_width,
            'fitted_spectrum': fitted_spectrum,
            'residuals': self.target_spectrum - fitted_spectrum
        }
    
    def _print_brute_force_results(self, results, computation_time):
        """Print detailed brute force results."""
        r_factor = results['r_factor']
        
        print(f"\n BRUTE FORCE RESULTS:")
        print(f"    Final R-factor: {r_factor:.6f}")
        print(f"    Benchmark: ≤ {self.r_factor_benchmark:.4f}")
        print(f"    Kernel width: {results['kernel_width']:.3f}")
        
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

    def get_results(self):
        """Get fitting results."""
        return self.results


def main():
    """Test the convolution LCF."""
    print("Convolution LCF module ready for use.")


if __name__ == "__main__":
    main()
