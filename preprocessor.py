"""
XANES Data Preprocessor - Clean Implementation

User-controlled preprocessing that preserves data integrity.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


class XANESPreprocessor:
    """
    Clean, user-controlled XANES data preprocessor.
    """
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.processed_data = {}
        self.common_energy = None
        
    def interactive_preprocessing(self, targets, references):
        """
        Interactive preprocessing with user control.
        
        Parameters:
        -----------
        targets : list
            Target spectrum names
        references : list
            Reference spectrum names
            
        Returns:
        --------
        dict : Processed data ready for LCF
        """
        all_spectra = targets + references
        
        print("\n" + "="*60)
        print(" PREPROCESSING")
        print("="*60)
        
        # Step 1: Choose preprocessing option
        print("\nPreprocessing options:")
        print("1. Skip preprocessing (use original data)")
        print("2. Apply preprocessing to selected spectra")
        print("3. View raw data first")
        
        choice = self._get_user_choice("Enter your choice (1-3): ", ['1', '2', '3'])
        
        if choice == '3':
            self._show_raw_data(all_spectra)
            choice = self._get_user_choice("After viewing, enter choice (1 or 2): ", ['1', '2'])
        
        if choice == '1':
            print(" Using original data without preprocessing")
            return self._prepare_original_data(targets, references)
        
        # Step 2: Select spectra to preprocess
        print("\nSelect spectra to preprocess:")
        print("1. All spectra")
        print("2. Only targets")
        print("3. Only references")
        print("4. Custom selection")
        
        spec_choice = self._get_user_choice("Enter choice (1-4): ", ['1', '2', '3', '4'])
        
        if spec_choice == '1':
            selected_spectra = all_spectra
        elif spec_choice == '2':
            selected_spectra = targets
        elif spec_choice == '3':
            selected_spectra = references
        else:  # Custom
            selected_spectra = self._custom_spectrum_selection(all_spectra)
        
        print(f" Selected for preprocessing: {selected_spectra}")
        
        # Step 3: Select preprocessing steps
        print("\nSelect preprocessing steps:")
        print("1. Background removal only")
        print("2. Normalization only")
        print("3. Smoothing only")
        print("4. Background + Normalization")
        print("5. All steps (Background + Normalization + Smoothing)")
        print("6. Custom combination")
        
        step_choice = self._get_user_choice("Enter choice (1-6): ", ['1', '2', '3', '4', '5', '6'])
        
        if step_choice == '1':
            steps = ['background']
        elif step_choice == '2':
            steps = ['normalize']
        elif step_choice == '3':
            steps = ['smooth']
        elif step_choice == '4':
            steps = ['background', 'normalize']
        elif step_choice == '5':
            steps = ['background', 'normalize', 'smooth']
        else:  # Custom
            steps = self._custom_step_selection()
        
        print(f" Selected steps: {steps}")
        
        # Step 4: Apply preprocessing
        return self._apply_preprocessing(targets, references, selected_spectra, steps)
    
    def _get_user_choice(self, prompt, valid_choices):
        """Get validated user input."""
        while True:
            choice = input(prompt).strip()
            if choice in valid_choices:
                return choice
            print(f" Please enter one of: {valid_choices}")
    
    def _show_raw_data(self, spectrum_names):
        """Show raw data plots."""
        print(" Showing raw data...")
        self.data_loader.plot_raw_spectra(spectrum_names)
    
    def _custom_spectrum_selection(self, all_spectra):
        """Custom spectrum selection."""
        print(f"\nAvailable spectra:")
        for i, name in enumerate(all_spectra, 1):
            print(f"{i}. {name}")
        
        while True:
            try:
                choice = input("Enter spectrum numbers (e.g., '1,3,5'): ").strip()
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                if all(0 <= i < len(all_spectra) for i in indices):
                    return [all_spectra[i] for i in indices]
                print(" Invalid selection")
            except ValueError:
                print(" Please enter numbers separated by commas")
    
    def _custom_step_selection(self):
        """Custom preprocessing step selection."""
        available_steps = ['background', 'normalize', 'smooth']
        print(f"\nAvailable steps:")
        for i, step in enumerate(available_steps, 1):
            print(f"{i}. {step}")
        
        while True:
            try:
                choice = input("Enter step numbers (e.g., '1,2'): ").strip()
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                if all(0 <= i < len(available_steps) for i in indices):
                    return [available_steps[i] for i in indices]
                print(" Invalid selection")
            except ValueError:
                print(" Please enter numbers separated by commas")
    
    def _prepare_original_data(self, targets, references):
        """
        Prepare original data for LCF (no preprocessing).
        
        This creates a common energy grid by interpolation only when necessary for fitting.
        """
        all_spectra = targets + references
        
        # Find the energy grid with most overlap
        energy_ranges = []
        for name in all_spectra:
            energy, _ = self.data_loader.get_spectrum_data(name)
            energy_ranges.append((energy.min(), energy.max(), len(energy), name))
        
        # Sort by number of points (descending)
        energy_ranges.sort(key=lambda x: x[2], reverse=True)
        
        # Use energy grid from spectrum with most points
        reference_name = energy_ranges[0][3]
        self.common_energy, _ = self.data_loader.get_spectrum_data(reference_name)
        
        print(f" Using energy grid from '{reference_name}': {len(self.common_energy)} points")
        print(f"   Energy range: {self.common_energy.min():.1f} - {self.common_energy.max():.1f} eV")
        
        # Prepare data dictionary
        data = {'energy': self.common_energy}
        
        for name in all_spectra:
            original_energy, original_absorption = self.data_loader.get_spectrum_data(name)

            if original_energy is None or original_absorption is None:
                print(f"   Warning: Could not retrieve data for {name}. Skipping.")
                continue
            
            if len(original_energy) == len(self.common_energy) and np.allclose(original_energy, self.common_energy):
                # Same energy grid - use original data
                data[name] = original_absorption
                print(f"    {name}: using original data ({len(original_absorption)} points)")
            else:
                # Different energy grid - interpolate
                interp_func = interp1d(original_energy, original_absorption, 
                                     kind='linear', bounds_error=False,
                                     fill_value=(original_absorption[0], original_absorption[-1]))
                data[name] = interp_func(self.common_energy)
                print(f"   {name}: interpolated {len(original_absorption)} -> {len(self.common_energy)} points")
        
        self.processed_data = data
        return data
    
    def _apply_preprocessing(self, targets, references, selected_spectra, steps):
        """Apply selected preprocessing steps."""
        all_spectra = targets + references
        
        print(f"\n Applying preprocessing...")
        
        # Start with original data
        data = self._prepare_original_data(targets, references)
        
        # Apply preprocessing to selected spectra
        for name in selected_spectra:
            print(f"\n   Processing {name}...")
            
            energy = data['energy']
            spectrum = data[name].copy()
            original_spectrum = spectrum.copy()
            
            # Apply each selected step
            for step in steps:
                if step == 'background':
                    spectrum = self._remove_background(spectrum, energy)
                    print(f"       Background removed")
                elif step == 'normalize':
                    spectrum = self._normalize(spectrum, energy)
                    print(f"       Normalized")
                elif step == 'smooth':
                    spectrum = self._smooth(spectrum)
                    print(f"       Smoothed")
            
            # Update data
            data[name] = spectrum
            
            # Show comparison
            improvement = self._calculate_improvement(original_spectrum, spectrum)
            print(f"      Processing improvement: {improvement:.2%}")
        
        self.processed_data = data
        return data
    
    def _remove_background(self, spectrum, energy):
        """Remove linear background using pre-edge region."""
        # Use first 20% for pre-edge
        n_pre = max(5, int(len(spectrum) * 0.2))
        pre_energy = energy[:n_pre]
        pre_spectrum = spectrum[:n_pre]
        
        # Fit linear background
        coeffs = np.polyfit(pre_energy, pre_spectrum, 1)
        background = np.polyval(coeffs, energy)
        
        return spectrum - background
    
    def _normalize(self, spectrum, energy):
        """Normalize using post-edge region."""
        # Use last 30% for post-edge
        n_post = max(5, int(len(spectrum) * 0.3))
        post_spectrum = spectrum[-n_post:]
        post_edge_value = np.mean(post_spectrum)
        
        if post_edge_value > 0:
            return spectrum / post_edge_value
        else:
            print("      Warning: Invalid post-edge value, skipping normalization")
            return spectrum
    
    def _smooth(self, spectrum):
        """Apply Savitzky-Golay smoothing."""
        window_length = min(11, len(spectrum) // 4)
        if window_length < 5:
            window_length = 5
        if window_length % 2 == 0:
            window_length += 1
        
        try:
            return savgol_filter(spectrum, window_length, 3)
        except:
            print("      Warning: Smoothing failed, using original data")
            return spectrum
    
    def _calculate_improvement(self, original, processed):
        """Calculate processing improvement metric."""
        original_noise = np.std(np.diff(original))
        processed_noise = np.std(np.diff(processed))
        
        if original_noise > 0:
            return max(0.0, (original_noise - processed_noise) / original_noise)
        return 0
    
    def get_processed_data(self):
        """Get the processed data."""
        return self.processed_data
    
    def plot_preprocessing_results(self, spectrum_names=None):
        """Plot preprocessing results comparison."""
        if not self.processed_data:
            print(" No processed data available")
            return
        
        if spectrum_names is None:
            spectrum_names = [k for k in self.processed_data.keys() if k != 'energy']
        
        energy = self.processed_data['energy']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, name in enumerate(spectrum_names[:4]):  # Show up to 4 spectra
            if name in self.processed_data:
                # Get original data
                orig_energy, orig_spectrum = self.data_loader.get_spectrum_data(name)
                proc_spectrum = self.processed_data[name]
                
                axes[i].plot(orig_energy, orig_spectrum, 'k-', alpha=0.7, label='Original')
                axes[i].plot(energy, proc_spectrum, 'r-', linewidth=2, label='Processed')
                axes[i].set_title(f'{name}')
                axes[i].set_xlabel('Energy (eV)')
                axes[i].set_ylabel('Absorption')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Test the preprocessor."""
    from data_loader import XANESDataLoader
    from pathlib import Path
    
    loader = XANESDataLoader()
    test_file = "/Users/CanaanWangJiaNan/Desktop/spot1.xlsx"
    
    if Path(test_file).exists():
        data = loader.load_excel_file(test_file)
        if data:
            targets, references = loader.identify_targets_and_references()
            
            preprocessor = XANESPreprocessor(loader)
            processed_data = preprocessor.interactive_preprocessing(targets, references)
            
            print(f"\n Preprocessing complete!")
            print(f"   Energy points: {len(processed_data['energy'])}")
            print(f"   Processed spectra: {len(processed_data)-1}")
    else:
        print(f"Test file not found: {test_file}")


if __name__ == "__main__":
    main()
