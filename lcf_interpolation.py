"""
Interpolation-based Linear Combination Fitting - Clean Implementation

Fastest LCF method using interpolation for energy alignment.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from itertools import product
from tqdm import tqdm
from scipy.interpolate import interp1d
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def parallel_optimization_worker(shift_chunk, energy, target_spectrum, reference_spectra):
    """
    Worker function for parallel optimization of energy shifts.
    """
    try:
        best_r = float('inf')
        best_shifts = None
        
        for shifts in shift_chunk:
            # Apply energy shifts to references
            shifted_refs = {}
            for i, (ref_name, ref_spec) in enumerate(reference_spectra.items()):
                shifted_energy = energy + shifts[i]
                interp_func = interp1d(shifted_energy, ref_spec,
                                     kind='linear', bounds_error=False,
                                     fill_value=(ref_spec[0], ref_spec[-1]))
                shifted_refs[ref_name] = interp_func(energy)
            
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


class InterpolationLCF:
    """
    Clean implementation of interpolation-based LCF.
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
        """Apply energy shift to spectrum using interpolation."""
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
    
    def fit(self, max_shift=7.5):
        """
        Fit the target spectrum by calling the brute-force method.
        
        Note: max_shift default value can be overridden by user input in main program.
        """
        print(" Starting comprehensive brute-force fitting...")
        return self.brute_force_fit(max_shift=max_shift)

    def plot_results(self):
        """Plot fitting results."""
        if not self.results or not self.results['success']:
            print(" No successful fitting results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Target vs Fitted
        axes[0, 0].plot(self.energy, self.target_spectrum, 'ko-', 
                       markersize=3, linewidth=1, label='Target', alpha=0.8)
        axes[0, 0].plot(self.energy, self.results['fitted_spectrum'], 'r-',
                       linewidth=2, label='Fitted')
        axes[0, 0].set_xlabel('Energy (eV)')
        axes[0, 0].set_ylabel('Absorption')
        axes[0, 0].set_title(f'Interpolation LCF (R = {self.results["r_factor"]:.4f})')
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
        for i, (ref_name, ref_spectrum) in enumerate(self.reference_spectra.items()):
            fraction = self.results['fractions'][ref_name]
            
            if fraction > 0.01:  # Only show significant components
                shift = self.results['shifts'][ref_name]
                shifted_spectrum = self.apply_energy_shift(ref_spectrum, shift)
                weighted_spectrum = fraction * shifted_spectrum
                
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
    
    def brute_force_fit(self, max_shift=7.5, target_time_minutes=25):
        """
        Perform a true brute-force LCF fitting by simultaneously searching 
        the entire parameter space of fractions and shifts.

        This method uses a multi-level, randomized sampling strategy to explore
        the combined parameter space, avoiding local minima by not decoupling
        the optimization of shifts and fractions.
        
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
        print(f"\n Starting True Brute-Force Interpolation LCF (3-Stage)...")
        print(f"   Strategy: 1. Global Randomized Search -> 2. Refined Search -> 3. Gradient Optimization.")
        print(f"   Max energy shift: ±{max_shift} eV")
        print(f"   Target time: ~{target_time_minutes} minutes")
        
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        if n_refs < 2:
            return {'success': False, 'error': 'Need at least 2 references'}

        total_start_time = time.time()
        target_time_seconds = target_time_minutes * 60

        # --- Stage 1 & 2: Randomized Global and Refined Search ---
        print("\n Stage 1 & 2: Randomized global and refined search...")
        
        # Estimate number of samples based on target time
        # A single evaluation is fast for interpolation.
        # Let's estimate ~0.05ms per sample.
        n_samples = int(target_time_seconds / 0.00005)
        print(f"   Estimated samples for target time: {n_samples:,}")

        best_r_factor = np.inf
        best_params = None

        # Pre-calculate shifted spectra for a discrete grid to speed up initial search
        # This helps find promising regions faster.
        shift_grid = np.linspace(-max_shift, max_shift, 21) # 21 steps for shifts
        precomputed_shifted_spectra = {
            name: {s: self.apply_energy_shift(spec, s) for s in shift_grid}
            for name, spec in self.reference_spectra.items()
        }

        try:
            with tqdm(total=n_samples, desc="Brute-Force Search", unit="combo") as pbar:
                
                for i in range(n_samples):

                    # --- Parameter Generation ---
                    # Generate a full set of random parameters (fractions and shifts)
                    
                    # Generate random fractions that sum to 1
                    fractions = np.random.random(n_refs)
                    fractions /= np.sum(fractions)

                    # Generate random shifts
                    if best_params and i > n_samples * 0.1: # Start refining after 10% of search
                        # Refine search around the best known parameters
                        refinement_factor = 1.0 - (i / n_samples) # Shrinks over time
                        
                        best_fracs = best_params['fractions']
                        best_shifts = best_params['shifts']

                        # Perturb fractions and re-normalize
                        fractions = np.abs(best_fracs + np.random.normal(0, 0.1 * refinement_factor, n_refs))
                        fractions /= np.sum(fractions)

                        # Perturb shifts
                        shifts = best_shifts + np.random.normal(0, max_shift * 0.1 * refinement_factor, n_refs)
                        shifts = np.clip(shifts, -max_shift, max_shift)
                    else:
                        # Pure random search for the initial phase
                        shifts = np.random.uniform(-max_shift, max_shift, n_refs)

                    # --- Evaluation ---
                    # Calculate the R-factor for the generated parameter set
                    
                    fitted_spectrum = np.zeros_like(self.target_spectrum)
                    for j, ref_name in enumerate(ref_names):
                        # Use pre-computed for coarse search, then interpolate for fine search
                        if i < n_samples * 0.1:
                            # Find closest pre-computed shift
                            closest_shift = shift_grid[np.argmin(np.abs(shift_grid - shifts[j]))]
                            shifted_spec = precomputed_shifted_spectra[ref_name][closest_shift]
                        else:
                            shifted_spec = self.apply_energy_shift(self.reference_spectra[ref_name], shifts[j])
                        
                        fitted_spectrum += fractions[j] * shifted_spec
                    
                    r_factor = self.calculate_r_factor(self.target_spectrum, fitted_spectrum)

                    # --- Update Best Result ---
                    if r_factor < best_r_factor:
                        best_r_factor = r_factor
                        best_params = {'fractions': np.copy(fractions), 'shifts': np.copy(shifts)}
                        pbar.set_postfix({'Best R': f'{best_r_factor:.6f}'})
                    
                    pbar.update(1)
        except KeyboardInterrupt:
            print("\n User interrupted the search. Moving to final optimization...")

        if not best_params:
            return {'success': False, 'error': 'Brute force search did not find any valid solution.'}

        print(f"   Best R-factor after random search: {best_r_factor:.6f}")

        # --- Stage 3: Gradient-based Local Optimization ---
        print("\n Stage 3: Final gradient-based local optimization...")

        def objective(params):
            """Objective function for the final optimizer."""
            fractions = params[:n_refs]
            shifts = params[n_refs:]
            
            # Fractions are constrained to sum to 1 and be non-negative by the optimizer
            
            fitted = np.zeros_like(self.target_spectrum)
            for i, ref_name in enumerate(ref_names):
                ref_spectrum = self.reference_spectra[ref_name]
                shifted_spectrum = self.apply_energy_shift(ref_spectrum, shifts[i])
                fitted += fractions[i] * shifted_spectrum
            
            return self.calculate_r_factor(self.target_spectrum, fitted)

        # Use the best parameters from the brute-force search as the starting point
        initial_params_for_optimizer = np.concatenate([best_params['fractions'], best_params['shifts']])
        
        bounds = ([(0, 1)] * n_refs + [(-max_shift, max_shift)] * n_refs)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:n_refs]) - 1}]

        optimizer_result = minimize(objective, 
                                    initial_params_for_optimizer,
                                    method='SLSQP', 
                                    bounds=bounds,
                                    constraints=constraints,
                                    options={'maxiter': 2000, 'ftol': 1e-10}) # Tighter tolerance

        if optimizer_result.success:
            print(" Gradient optimization successful.")
            final_params = optimizer_result.x
            final_fractions = final_params[:n_refs]
            final_shifts = final_params[n_refs:]
            # Normalize fractions just in case
            final_fractions = np.abs(final_fractions)
            final_fractions /= np.sum(final_fractions)
        else:
            print(" Gradient optimization failed. Using best result from random search.")
            final_fractions = best_params['fractions']
            final_shifts = best_params['shifts']


        # --- Finalize and Return Results ---
        final_fractions_dict = dict(zip(ref_names, final_fractions))
        final_shifts_dict = dict(zip(ref_names, final_shifts))

        # Calculate final fitted spectrum with the best parameters found
        fitted_spectrum = np.zeros_like(self.target_spectrum)
        for i, ref_name in enumerate(ref_names):
            ref_spectrum = self.reference_spectra[ref_name]
            shifted_spectrum = self.apply_energy_shift(ref_spectrum, final_shifts_dict[ref_name])
            fitted_spectrum += final_fractions_dict[ref_name] * shifted_spectrum
        
        final_r_factor = self.calculate_r_factor(self.target_spectrum, fitted_spectrum)

        total_time = time.time() - total_start_time
        print(f" Brute force finished in {total_time:.2f} seconds.")
        
        self.results = {
            'success': True,
            'method': 'interpolation_brute_force',
            'r_factor': final_r_factor,
            'fractions': final_fractions_dict,
            'shifts': final_shifts_dict,
            'fitted_spectrum': fitted_spectrum,
            'residuals': self.target_spectrum - fitted_spectrum,
            'computation_time': total_time
        }
        
        print(f"   Best R-factor found: {final_r_factor:.6f}")
        print(f"   Benchmark: ≤ {self.r_factor_benchmark:.3f}")
        if final_r_factor <= self.r_factor_benchmark:
            print(f"    Benchmark met!")
        else:
            print(f"    Benchmark NOT met.")
            
        print(f"\n   Component fractions:")
        for name, fraction in self.results['fractions'].items():
            print(f"     {name}: {fraction:.3f} ({fraction*100:.1f}%)")
            
        return self.results

    def _optimize_fractions_for_shifts(self, shifts):
        """
        Optimize fractions for given energy shifts.
        
        Parameters:
        -----------
        shifts : list
            Energy shifts for each reference
            
        Returns:
        --------
        float : Best R-factor for these shifts
        """
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        def objective_fractions(fractions):
            """Objective function for fraction optimization."""
            # Normalize fractions
            fractions = np.abs(fractions)
            total = np.sum(fractions)
            if total > 0:
                fractions = fractions / total
            
            # Calculate fitted spectrum
            fitted = np.zeros_like(self.target_spectrum)
            
            for i, ref_name in enumerate(ref_names):
                ref_spectrum = self.reference_spectra[ref_name]
                shifted_spectrum = self.apply_energy_shift(ref_spectrum, shifts[i])
                fitted += fractions[i] * shifted_spectrum
            
            return self.calculate_r_factor(self.target_spectrum, fitted)
        
        # Initial fractions
        initial_fractions = np.ones(n_refs) / n_refs
        
        # Bounds and constraints
        bounds = [(0, 1)] * n_refs
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        try:
            result = minimize(objective_fractions, initial_fractions,
                            method='SLSQP', bounds=bounds, constraints=constraints,
                            options={'maxiter': 500, 'disp': False})
            
            if result.success:
                return result.fun
            else:
                return np.inf
                
        except:
            return np.inf
    
    def _optimize_fractions_for_shifts_fast(self, shifts):
        """
        Fast optimization of fractions for given energy shifts (reduced precision).
        
        Parameters:
        -----------
        shifts : list
            Energy shifts for each reference
            
        Returns:
        --------
        float : Best R-factor for these shifts
        """
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        def objective_fractions(fractions):
            """Objective function for fraction optimization."""
            # Normalize fractions
            fractions = np.abs(fractions)
            total = np.sum(fractions)
            if total > 0:
                fractions = fractions / total
            
            # Calculate fitted spectrum
            fitted = np.zeros_like(self.target_spectrum)
            
            for i, ref_name in enumerate(ref_names):
                ref_spectrum = self.reference_spectra[ref_name]
                shifted_spectrum = self.apply_energy_shift(ref_spectrum, shifts[i])
                fitted += fractions[i] * shifted_spectrum
            
            return self.calculate_r_factor(self.target_spectrum, fitted)
        
        # Initial fractions
        initial_fractions = np.ones(n_refs) / n_refs
        
        # Bounds and constraints
        bounds = [(0, 1)] * n_refs
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        try:
            result = minimize(objective_fractions, initial_fractions,
                            method='SLSQP', bounds=bounds, constraints=constraints,
                            options={'maxiter': 100, 'disp': False})  # Reduced iterations for speed
            
            if result.success:
                return result.fun
            else:
                return np.inf
                
        except:
            return np.inf
    
    def _optimize_fractions_for_shifts_ultrafast(self, shifts):
        """
        Ultra-fast optimization of fractions for given energy shifts (minimal precision).
        
        Parameters:
        -----------
        shifts : list
            Energy shifts for each reference
            
        Returns:
        --------
        float : Best R-factor for these shifts
        """
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        def objective_fractions(fractions):
            """Objective function for fraction optimization."""
            # Normalize fractions
            fractions = np.abs(fractions)
            total = np.sum(fractions)
            if total > 0:
                fractions = fractions / total
            
            # Calculate fitted spectrum
            fitted = np.zeros_like(self.target_spectrum)
            
            for i, ref_name in enumerate(ref_names):
                ref_spectrum = self.reference_spectra[ref_name]
                shifted_spectrum = self.apply_energy_shift(ref_spectrum, shifts[i])
                fitted += fractions[i] * shifted_spectrum
            
            return self.calculate_r_factor(self.target_spectrum, fitted)
        
        # Initial fractions
        initial_fractions = np.ones(n_refs) / n_refs
        
        # Bounds and constraints
        bounds = [(0, 1)] * n_refs
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        try:
            result = minimize(objective_fractions, initial_fractions,
                            method='SLSQP', bounds=bounds, constraints=constraints,
                            options={'maxiter': 30, 'disp': False})  # Minimal iterations for speed
            
            if result.success:
                return result.fun
            else:
                return np.inf
                
        except:
            return np.inf

    def _calculate_final_result(self, best_shifts, method='brute_force_interpolation'):
        """Calculate final results with best shifts."""
        ref_names = list(self.reference_spectra.keys())
        n_refs = len(ref_names)
        
        # Optimize fractions one more time for final result
        def objective_fractions(fractions):
            fractions = np.abs(fractions)
            total = np.sum(fractions)
            if total > 0:
                fractions = fractions / total
            
            fitted = np.zeros_like(self.target_spectrum)
            for i, ref_name in enumerate(ref_names):
                ref_spectrum = self.reference_spectra[ref_name]
                shifted_spectrum = self.apply_energy_shift(ref_spectrum, best_shifts[i])
                fitted += fractions[i] * shifted_spectrum
            
            return self.calculate_r_factor(self.target_spectrum, fitted)
        
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
        fitted_spectrum = np.zeros_like(self.target_spectrum)
        for i, ref_name in enumerate(ref_names):
            ref_spectrum = self.reference_spectra[ref_name]
            shifted_spectrum = self.apply_energy_shift(ref_spectrum, best_shifts[i])
            fitted_spectrum += final_fractions[i] * shifted_spectrum
        
        r_factor = self.calculate_r_factor(self.target_spectrum, fitted_spectrum)
        
        return {
            'success': True,
            'method': method,
            'r_factor': r_factor,
            'fractions': dict(zip(ref_names, final_fractions)),
            'shifts': dict(zip(ref_names, best_shifts)),
            'fitted_spectrum': fitted_spectrum,
            'residuals': self.target_spectrum - fitted_spectrum
        }
    
    def _print_brute_force_results(self, results, computation_time):
        """Print detailed brute force results."""
        r_factor = results['r_factor']
        
        print(f"\n BRUTE FORCE RESULTS:")
        print(f"    Final R-factor: {r_factor:.6f}")
        print(f"    Benchmark: ≤ {self.r_factor_benchmark:.3f}")
        
        if r_factor <= self.r_factor_benchmark:
            print(f"    EXCELLENT! Benchmark met!")
        elif r_factor < 0.02:
            print(f"    Excellent fit!")
        elif r_factor < 0.05:
            print(f"    Good fit")
        else:
            print(f"     Acceptable fit")
        
        print(f"    Computation time: {computation_time:.1f}s")
        
        print(f"\n    Component fractions:")
        for name, fraction in results['fractions'].items():
            print(f"      {name}: {fraction:.3f} ({fraction*100:.1f}%)")
        
        print(f"\n    Energy shifts (eV):")
        for name, shift in results['shifts'].items():
            print(f"      {name}: {shift:+.2f}")

def main():
    """Test the interpolation LCF."""
    print("Interpolation LCF module ready for use.")


if __name__ == "__main__":
    main()
