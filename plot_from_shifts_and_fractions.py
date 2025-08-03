#!/usr/bin/env python3
"""
Independent LCF Plot Generator

This tool creates LCF fit plots using only shift values and component percentages.
It loads original data and applies the shifts to recreate the fit visualization.

Usage:
    python plot_from_shifts_and_fractions.py

Requirements:
    - Excel data file with target and reference spectra
    - Known shift values for each reference
    - Known component fractions/percentages
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.interpolate import interp1d

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from data_loader import XANESDataLoader


class LCFPlotGenerator:
    """
    Generate LCF fit plots from shift values and component fractions.
    """
    
    def __init__(self):
        self.energy = None
        self.target_spectrum = None
        self.reference_spectra = {}
        self.target_name = None
        self.reference_names = []
        
    def load_data_from_excel(self, file_path, target_name, reference_names):
        """
        Load spectral data from Excel file.
        
        Parameters:
        -----------
        file_path : str
            Path to Excel file
        target_name : str
            Name of target spectrum sheet
        reference_names : list
            List of reference spectrum sheet names
        """
        print(f"Loading data from: {file_path}")
        
        loader = XANESDataLoader()
        raw_data = loader.load_excel_file(file_path)
        
        if not raw_data:
            raise ValueError("Failed to load Excel data")
        
        # Extract target spectrum
        if target_name not in raw_data:
            raise ValueError(f"Target '{target_name}' not found in Excel file")
        
        # Get target energy and spectrum from the loader's data structure
        target_energy = loader.energy_grids[target_name]
        target_spectrum = raw_data[target_name]
        
        self.energy = target_energy
        self.target_spectrum = target_spectrum
        self.target_name = target_name
        
        # Extract reference spectra
        self.reference_spectra = {}
        for ref_name in reference_names:
            if ref_name not in raw_data:
                print(f"Warning: Reference '{ref_name}' not found, skipping...")
                continue
            
            # Get reference energy and spectrum
            ref_energy = loader.energy_grids[ref_name]
            ref_spectrum = raw_data[ref_name]
            
            # Interpolate reference to target energy grid
            interp_func = interp1d(ref_energy, ref_spectrum,
                                 kind='linear', bounds_error=False,
                                 fill_value='extrapolate')
            self.reference_spectra[ref_name] = interp_func(self.energy)
        
        self.reference_names = list(self.reference_spectra.keys())
        print(f" Loaded target: {target_name}")
        print(f" Loaded {len(self.reference_names)} references: {self.reference_names}")
    
    def apply_energy_shifts(self, shift_values):
        """
        Apply energy shifts to reference spectra.
        
        Parameters:
        -----------
        shift_values : dict
            Dictionary mapping reference names to shift values (eV)
            
        Returns:
        --------
        dict : Shifted reference spectra
        """
        shifted_refs = {}
        
        for ref_name, shift in shift_values.items():
            if ref_name not in self.reference_spectra:
                print(f"Warning: Reference '{ref_name}' not found, skipping...")
                continue
            
            # Apply energy shift
            shifted_energy = self.energy + shift
            original_spectrum = self.reference_spectra[ref_name]
            
            # Create interpolation function for shifted spectrum
            interp_func = interp1d(shifted_energy, original_spectrum,
                                 kind='linear', bounds_error=False,
                                 fill_value=(original_spectrum[0], original_spectrum[-1]))
            
            # Interpolate back to original energy grid
            shifted_refs[ref_name] = interp_func(self.energy)
        
        return shifted_refs
    
    def calculate_fitted_spectrum(self, shifted_references, fractions):
        """
        Calculate fitted spectrum from shifted references and fractions.
        
        Parameters:
        -----------
        shifted_references : dict
            Dictionary of shifted reference spectra
        fractions : dict
            Dictionary mapping reference names to fractions
            
        Returns:
        --------
        array : Fitted spectrum
        """
        fitted_spectrum = np.zeros_like(self.energy)
        
        for ref_name, fraction in fractions.items():
            if ref_name in shifted_references:
                fitted_spectrum += fraction * shifted_references[ref_name]
        
        return fitted_spectrum
    
    def calculate_r_factor(self, target, fitted, mask=None):
        """
        Calculate R-factor between target and fitted spectra using squared differences.
        
        Args:
            target: Target spectrum
            fitted: Fitted spectrum  
            mask: Optional boolean mask to restrict calculation to specific energy range
        """
        if mask is not None:
            target = target[mask]
            fitted = fitted[mask]
            
        numerator = np.sum((target - fitted) ** 2)
        denominator = np.sum(target ** 2)
        return numerator / denominator if denominator > 0 else float('inf')
    
    def plot_fit_results(self, shift_values, fractions, energy_range=None, 
                        save_path=None, method_name="LCF", show_residuals=True):
        """
        Generate comprehensive LCF fit plot.
        
        Parameters:
        -----------
        shift_values : dict
            Energy shifts for each reference (eV)
        fractions : dict
            Component fractions for each reference
        energy_range : tuple, optional
            Energy range for plotting (min_eV, max_eV)
        save_path : str, optional
            Path to save the plot
        method_name : str
            Name of LCF method used
        show_residuals : bool
            Whether to show residuals subplot
        """
        if self.energy is None:
            raise ValueError("No data loaded. Use load_data_from_excel() first.")
        
        # Apply shifts and calculate fitted spectrum
        shifted_refs = self.apply_energy_shifts(shift_values)
        fitted_spectrum = self.calculate_fitted_spectrum(shifted_refs, fractions)
        
        # Calculate R-factor within the selected energy range
        if energy_range:
            r_factor_mask = (self.energy >= energy_range[0]) & (self.energy <= energy_range[1])
            r_factor = self.calculate_r_factor(self.target_spectrum, fitted_spectrum, mask=r_factor_mask)
        else:
            r_factor = self.calculate_r_factor(self.target_spectrum, fitted_spectrum)
        
        # Select energy range for plotting
        if energy_range:
            mask = (self.energy >= energy_range[0]) & (self.energy <= energy_range[1])
            plot_energy = self.energy[mask]
            plot_target = self.target_spectrum[mask]
            plot_fitted = fitted_spectrum[mask]
        else:
            plot_energy = self.energy
            plot_target = self.target_spectrum
            plot_fitted = fitted_spectrum
        
        # Create figure
        if show_residuals:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Target vs Fitted
        ax1.plot(plot_energy, plot_target, 'ko-', markersize=3, linewidth=1.5, 
                label='Target', alpha=0.8)
        ax1.plot(plot_energy, plot_fitted, 'r-', linewidth=2.5, label='Fitted')
        
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Absorption')
        ax1.set_title(f'{self.target_name} - {method_name} Fit\n'
                     f'R-factor: {r_factor:.6f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Component Fractions
        ref_names = list(fractions.keys())
        fraction_values = list(fractions.values())
        shift_labels = [f'{name}\n(Shift: {shift_values.get(name, 0.0):+.2f} eV)' 
                       for name in ref_names]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(ref_names)))
        bars = ax2.bar(range(len(ref_names)), fraction_values, color=colors, alpha=0.7)
        
        ax2.set_xticks(range(len(ref_names)))
        ax2.set_xticklabels(shift_labels, rotation=45, ha='right')
        ax2.set_ylabel('Fraction')
        ax2.set_title('Component Fractions with Energy Shifts')
        ax2.set_ylim(0, max(fraction_values) * 1.1)
        
        # Add value labels on bars
        for bar, fraction in zip(bars, fraction_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{fraction:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Residuals (if requested)
        if show_residuals:
            residuals = plot_target - plot_fitted
            ax3.plot(plot_energy, residuals, 'g-', linewidth=1.5)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Energy (eV)')
            ax3.set_ylabel('Residuals')
            ax3.set_title('Fit Residuals')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Plot saved to: {save_path}")
        
        plt.show()
        
        # Print summary
        print(f"\n LCF Fit Summary for {self.target_name}")
        print("=" * 50)
        print(f"Method: {method_name}")
        print(f"R-factor: {r_factor:.6f}")
        print(f"\nComponent Analysis:")
        for ref_name in ref_names:
            shift = shift_values.get(ref_name, 0.0)
            fraction = fractions[ref_name]
            print(f"  {ref_name:<20} {fraction:>6.1%}  (shift: {shift:+6.2f} eV)")
        
        return {
            'r_factor': r_factor,
            'fitted_spectrum': fitted_spectrum,
            'residuals': self.target_spectrum - fitted_spectrum,
            'fractions': fractions,
            'shifts': shift_values
        }


def auto_detect_targets_and_references(file_path):
    """
    Automatically detect target and reference spectra from Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to Excel file
        
    Returns:
    --------
    tuple : (targets, references) lists
    """
    loader = XANESDataLoader()
    raw_data = loader.load_excel_file(file_path)
    
    if not raw_data:
        return [], []
    
    all_sheets = list(raw_data.keys())
    targets = []
    references = []
    
    # Auto-detect logic (similar to main.py)
    for sheet_name in all_sheets:
        name_lower = sheet_name.lower()
        
        # Check for target patterns
        if ('spot' in name_lower or 
            'sample' in name_lower or 
            'target' in name_lower or
            'unknown' in name_lower):
            targets.append(sheet_name)
        else:
            # Everything else is considered a reference
            references.append(sheet_name)
    
    # If no targets found using patterns, ask user to select
    if not targets and all_sheets:
        print("\n🔍 No target patterns found. Available sheets:")
        for i, sheet in enumerate(all_sheets, 1):
            print(f"  {i}. {sheet}")
        
        target_choice = input("\nSelect target sheet number: ").strip()
        try:
            target_idx = int(target_choice) - 1
            if 0 <= target_idx < len(all_sheets):
                targets = [all_sheets[target_idx]]
                references = [s for s in all_sheets if s != targets[0]]
        except ValueError:
            pass
    
    return targets, references


def interactive_plot_generator():
    """Interactive interface for generating LCF plots with auto-detection."""
    print("=" * 70)
    print("          LCF Plot Generator from Shifts and Fractions")
    print("=" * 70)
    
    plotter = LCFPlotGenerator()
    
    # Get Excel file path
    default_file = "/Users/CanaanWangJiaNan/Desktop/spot1.xlsx"
    file_path = input(f"Enter Excel file path [default: {default_file}]: ").strip()
    if not file_path:
        file_path = default_file
    
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return
    
    # Auto-detect targets and references
    print("\n Auto-detecting targets and references...")
    targets, references = auto_detect_targets_and_references(file_path)
    
    if not targets:
        print(" No targets detected")
        return
    
    if len(references) < 2:
        print(" Need at least 2 references")
        return
    
    print(f"\n Detected targets: {targets}")
    print(f" Detected references: {references}")
    
    # Get maximum energy shift setting
    print(f"\nMaximum Energy Shift Setting:")
    print(f"Default: ±7.5 eV (typical for XANES analysis)")
    
    use_custom_shift = input("Use custom maximum shift? (y/n, default=n): ").strip().lower()
    if use_custom_shift in ['y', 'yes']:
        while True:
            try:
                max_shift = float(input("Enter maximum energy shift in eV (1.0-20.0): "))
                if 1.0 <= max_shift <= 20.0:
                    break
                else:
                    print(" Please enter a value between 1.0 and 20.0 eV")
            except ValueError:
                print(" Please enter a valid number")
    else:
        max_shift = 7.5
    
    print(f" Using maximum energy shift: ±{max_shift} eV")
    
    # Select target if multiple
    if len(targets) > 1:
        print(f"\nMultiple targets found. Select one:")
        for i, target in enumerate(targets, 1):
            print(f"  {i}. {target}")
        
        while True:
            try:
                choice = int(input("Enter choice: ")) - 1
                if 0 <= choice < len(targets):
                    target_name = targets[choice]
                    break
                print(f"Please enter 1-{len(targets)}")
            except ValueError:
                print("Please enter a valid number")
    else:
        target_name = targets[0]
    
    # Allow user to modify reference selection
    print(f"\n📊 References to use (currently {len(references)} detected):")
    for i, ref in enumerate(references, 1):
        print(f"  {i}. {ref}")
    
    modify_refs = input("\nModify reference selection? (y/n) [default: n]: ").strip().lower()
    if modify_refs in ['y', 'yes']:
        print("Enter reference numbers to use (e.g., 1,3,4):")
        ref_input = input("References: ").strip()
        try:
            ref_indices = [int(x.strip()) - 1 for x in ref_input.split(',')]
            selected_refs = [references[i] for i in ref_indices if 0 <= i < len(references)]
            if len(selected_refs) >= 2:
                references = selected_refs
            else:
                print("  Need at least 2 references, using all detected references")
        except (ValueError, IndexError):
            print("  Invalid input, using all detected references")
    
    try:
        # Load data
        plotter.load_data_from_excel(file_path, target_name, references)
        
        # Get shift values with option for zero defaults
        print(f"\n Enter energy shift values for each reference:")
        print(f"(Press Enter for 0.0 eV shift, valid range: ±{max_shift} eV)")
        shift_values = {}
        for ref_name in plotter.reference_names:
            while True:
                try:
                    shift_input = input(f"  {ref_name} shift (eV) [default: 0.0]: ").strip()
                    shift = float(shift_input) if shift_input else 0.0
                    if -max_shift <= shift <= max_shift:
                        shift_values[ref_name] = shift
                        break
                    else:
                        print(f"    Shift must be between ±{max_shift} eV")
                except ValueError:
                    print("    Please enter a valid number or press Enter for 0.0")
        
        # Get fractions with equal defaults
        print(f"\n Enter component fractions for each reference:")
        print(f"(Press Enter for equal fractions: {1.0/len(plotter.reference_names):.3f} each)")
        fractions = {}
        total_fraction = 0
        use_equal = True
        
        for ref_name in plotter.reference_names:
            while True:
                try:
                    default_fraction = 1.0 / len(plotter.reference_names)
                    fraction_input = input(f"  {ref_name} fraction (0-1) [default: {default_fraction:.3f}]: ").strip()
                    
                    if fraction_input:
                        fraction = float(fraction_input)
                        use_equal = False
                    else:
                        fraction = default_fraction
                    
                    if 0 <= fraction <= 1:
                        fractions[ref_name] = fraction
                        total_fraction += fraction
                        break
                    else:
                        print("    Fraction must be between 0 and 1")
                except ValueError:
                    print("    Please enter a valid number or press Enter for default")
        
        # Normalize fractions if needed (unless using equal defaults)
        if not use_equal and abs(total_fraction - 1.0) > 0.01:
            print(f"\n  Total fraction is {total_fraction:.3f}, normalizing to 1.0...")
            for ref_name in fractions:
                fractions[ref_name] /= total_fraction
        
        # Get energy range for plotting
        if plotter.energy is not None:
            energy_min, energy_max = plotter.energy.min(), plotter.energy.max()
            print(f"\nEnergy range: {energy_min:.1f} - {energy_max:.1f} eV")
            use_range = input("Specify custom energy range? (y/n) [default: n]: ").strip().lower()
            
            energy_range = None
            if use_range in ['y', 'yes']:
                try:
                    min_e = float(input(f"  Minimum energy (eV) [current: {energy_min:.1f}]: "))
                    max_e = float(input(f"  Maximum energy (eV) [current: {energy_max:.1f}]: "))
                    energy_range = (min_e, max_e)
                except ValueError:
                    print("  Invalid input, using full range")
        else:
            energy_range = None
        
        # Ask about saving
        save_path = None
        save_plot = input("\nSave plot to file? (y/n) [default: y]: ").strip().lower()
        if save_plot != 'n':
            default_name = f"LCF_Plot_{target_name}"
            save_path = input(f"Enter save path [default: {default_name}]: ").strip()
            if not save_path:
                save_path = default_name
            if not save_path.endswith('.png'):
                save_path += '.png'
        
        # Generate plot
        print(f"\n🎨 Generating LCF plot...")
        results = plotter.plot_fit_results(
            shift_values=shift_values,
            fractions=fractions,
            energy_range=energy_range,
            save_path=save_path,
            method_name="Custom"
        )
        
        print(f"\n Plot generation completed!")
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()


def quick_demo_mode(file_path):
    """Quick demo mode with automatic defaults."""
    print("\n Quick Demo Mode - Using Default Values")
    print("="*50)
    
    # Auto-detect
    targets, references = auto_detect_targets_and_references(file_path)
    
    if not targets or len(references) < 2:
        print(" Insufficient data for demo")
        return False
    
    target_name = targets[0]
    print(f" Target: {target_name}")
    print(f" References: {references}")
    
    # Create plotter and load data
    plotter = LCFPlotGenerator()
    plotter.load_data_from_excel(file_path, target_name, references)
    
    # Use defaults
    shift_values = {ref: 0.0 for ref in plotter.reference_names}
    fractions = {ref: 1.0/len(plotter.reference_names) for ref in plotter.reference_names}
    
    print(f" Using defaults: zero shifts, equal fractions")
    
    # Generate plot
    results = plotter.plot_fit_results(
        shift_values=shift_values,
        fractions=fractions,
        save_path=f"Quick_Demo_{target_name}.png",
        method_name="Quick Demo"
    )
    
    print(f" Demo completed! Check the plot file.")
    return True


if __name__ == "__main__":
    try:
        print("=" * 70)
        print("          LCF Plot Generator from Shifts and Fractions")
        print("=" * 70)
        print("1. Interactive mode (full control)")
        print("2. Quick demo mode (auto-detect + defaults)")
        
        mode = input("\nSelect mode (1 or 2) [default: 1]: ").strip()
        
        if mode == '2':
            # Quick demo mode
            default_file = "/Users/CanaanWangJiaNan/Desktop/spot1.xlsx"
            file_path = input(f"Enter Excel file path [default: {default_file}]: ").strip()
            if not file_path:
                file_path = default_file
            
            if os.path.exists(file_path):
                quick_demo_mode(file_path)
            else:
                print(f" File not found: {file_path}")
        else:
            # Interactive mode
            interactive_plot_generator()
            
    except KeyboardInterrupt:
        print(f"\n\n Plot generator interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()
