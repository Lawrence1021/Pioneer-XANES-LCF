"""
XANES Data Loader - Simple and Clean Implementation

Ensures data integrity and exact matching with Excel source.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class XANESDataLoader:
    """
    Simple, reliable XANES data loader that preserves original data integrity.
    """
    
    def __init__(self):
        self.raw_data = {}
        self.energy_grids = {}  # Store original energy for each spectrum
        
    def load_excel_file(self, file_path):
        """
        Load XANES data from Excel file.
        
        Parameters:
        -----------
        file_path : str
            Path to Excel file
            
        Returns:
        --------
        dict : Raw data with exact Excel values preserved
        """
        print(f"Loading data from: {file_path}")
        
        try:
            # Read all sheet names
            xl_file = pd.ExcelFile(file_path)
            sheet_names = xl_file.sheet_names
            print(f"Found {len(sheet_names)} sheets: {sheet_names}")
            
            data = {}
            energy_grids = {}
            
            for sheet_name in sheet_names:
                try:
                    # Read each sheet
                    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
                    
                    if df.shape[1] >= 2:
                        # Extract energy and absorption
                        energy = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
                        absorption = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna()
                        
                        # Ensure same length
                        min_len = min(len(energy), len(absorption))
                        energy = energy.iloc[:min_len].values
                        absorption = absorption.iloc[:min_len].values
                        
                        if len(energy) > 10:  # Minimum data points
                            data[sheet_name] = absorption
                            energy_grids[sheet_name] = energy
                            print(f" {sheet_name}: {len(energy)} points, "
                                  f"range {energy.min():.1f}-{energy.max():.1f} eV")
                        else:
                            print(f"  {sheet_name}: insufficient data ({len(energy)} points)")
                    else:
                        print(f"  {sheet_name}: insufficient columns")
                        
                except Exception as e:
                    print(f" Error reading {sheet_name}: {e}")
            
            if data:
                self.raw_data = data
                self.energy_grids = energy_grids
                print(f" Successfully loaded {len(data)} spectra")
                return data
            else:
                print(" No valid spectra found")
                return None
                
        except Exception as e:
            print(f" Error loading Excel file: {e}")
            return None
    
    def identify_targets_and_references(self, auto_detect=True):
        """
        Identify target and reference spectra.
        
        Parameters:
        -----------
        auto_detect : bool
            If True, auto-detect based on naming convention
            
        Returns:
        --------
        tuple : (targets, references)
        """
        if not self.raw_data:
            print(" No data loaded")
            return [], []
        
        all_spectra = list(self.raw_data.keys())
        
        if auto_detect:
            # Auto-detect: spectra starting with 'spot' are targets
            targets = [name for name in all_spectra if name.lower().startswith('spot')]
            references = [name for name in all_spectra if not name.lower().startswith('spot')]
        else:
            # Manual selection
            print(f"\nAvailable spectra: {all_spectra}")
            print("\nSelect target spectra (to be fitted):")
            targets = self._get_user_selection(all_spectra, "targets")
            
            remaining = [s for s in all_spectra if s not in targets]
            print(f"\nSelect reference spectra from: {remaining}")
            references = self._get_user_selection(remaining, "references")
        
        print(f"\n Targets: {targets}")
        print(f" References: {references}")
        
        return targets, references
    
    def _get_user_selection(self, options, selection_type):
        """Helper for manual spectrum selection."""
        selected = []
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        
        while True:
            try:
                choice = input(f"\nEnter {selection_type} numbers (e.g., '1,2,3'): ").strip()
                if choice:
                    indices = [int(x.strip()) - 1 for x in choice.split(',')]
                    if all(0 <= i < len(options) for i in indices):
                        selected = [options[i] for i in indices]
                        break
                print(" Invalid selection, try again")
            except ValueError:
                print(" Please enter numbers separated by commas")
        
        return selected
    
    def get_spectrum_data(self, spectrum_name):
        """
        Get original energy and absorption data for a spectrum.
        
        Parameters:
        -----------
        spectrum_name : str
            Name of the spectrum
            
        Returns:
        --------
        tuple : (energy, absorption) arrays
        """
        if spectrum_name in self.raw_data and spectrum_name in self.energy_grids:
            return self.energy_grids[spectrum_name], self.raw_data[spectrum_name]
        else:
            print(f" Spectrum '{spectrum_name}' not found")
            return None, None
    
    def plot_raw_spectra(self, spectrum_names=None):
        """Plot raw spectra for visualization."""
        if spectrum_names is None:
            spectrum_names = list(self.raw_data.keys())
        
        plt.figure(figsize=(12, 8))
        
        for name in spectrum_names:
            if name in self.raw_data:
                energy, absorption = self.get_spectrum_data(name)
                if energy is not None and absorption is not None:
                    plt.plot(energy, absorption, label=name, alpha=0.8)
        
        plt.xlabel('Energy (eV)')
        plt.ylabel('Absorption')
        plt.title('Raw XANES Spectra')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_data_summary(self):
        """Get summary of loaded data."""
        if not self.raw_data:
            return "No data loaded"
        
        summary = []
        summary.append(f"Data Summary:")
        summary.append(f"   Total spectra: {len(self.raw_data)}")
        
        for name in self.raw_data.keys():
            energy, absorption = self.get_spectrum_data(name)
            if energy is not None:
                summary.append(f"   {name}: {len(energy)} points, "
                              f"{energy.min():.1f}-{energy.max():.1f} eV")
        
        return "\n".join(summary)


def main():
    """Test the data loader."""
    loader = XANESDataLoader()
    
    # Test with sample file
    test_file = "/Users/CanaanWangJiaNan/Desktop/spot1.xlsx"
    
    if Path(test_file).exists():
        data = loader.load_excel_file(test_file)
        if data:
            print(loader.get_data_summary())
            targets, references = loader.identify_targets_and_references()
            
            # Show first target spectrum
            if targets:
                energy, absorption = loader.get_spectrum_data(targets[0])
                print(f"\n Sample {targets[0]} data:")
                print(f"   First 5 energy points: {energy[:5]}")
                print(f"   First 5 absorption points: {absorption[:5]}")
    else:
        print(f"Test file not found: {test_file}")


if __name__ == "__main__":
    main()
