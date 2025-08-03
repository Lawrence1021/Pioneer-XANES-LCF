#!/usr/bin/env python3
"""
XANES Linear Combination Fitting (LCF) Analysis Program - Clean Version

Th    # Get R-factor benchmark from user
    print(f"\n R-factor Benchmark Setting:")
    print(f"Current be                r_factor = result['r_factor']
                
                # Check benchmark status
                benchmark_val = "0.02"  # Updated benchmark
                if r_factor <= 0.02:
                    status = " Met"
                else:
                    status = " Not Met" ≤ 0.0015 (Ultra-extreme precision - 10x higher than standard)")
    
    use_custom = input("Use custom R-factor benchmark? (y/n, default=n): ").strip().lower()
    if use_custom in ['y', 'yes']:
        while True:
            try:
                benchmark = float(input("Enter R-factor benchmark (0.001-0.10): "))
                if 0.001 <= benchmark <= 0.10:
                    break
                else:
                    print(" Please enter a value between 0.001 and 0.10")
            except ValueError:
                print(" Please enter a valid number")
    else:
        benchmark = 0.0015y rewritten, clean implementation that ensures:
1. Data integrity (output matches Excel exactly when no preprocessing)
2. User-controlled preprocessing
3. Three reliable LCF methods
4. Clear visualization

All code is in English with comprehensive comments.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

# Import clean modules
from data_loader import XANESDataLoader
from preprocessor import XANESPreprocessor
from lcf_interpolation import InterpolationLCF
from lcf_deconvolution import DeconvolutionLCF
from lcf_convolution import ConvolutionLCF


def get_user_choice(prompt, options):
    """Get validated user choice."""
    while True:
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        
        try:
            choice = int(input("\nEnter your choice: ").strip())
            if 1 <= choice <= len(options):
                return choice
            print(f" Please enter a number between 1 and {len(options)}")
        except ValueError:
            print(" Please enter a valid number")


def select_energy_range(energy):
    """Allow user to select energy range for fitting."""
    print(f"\nEnergy range selection:")
    print(f"Full range: {energy.min():.1f} - {energy.max():.1f} eV ({len(energy)} points)")
    
    options = [
        "Use full energy range",
        "Select custom range",
        "Use overlap region (recommended)"
    ]
    
    choice = get_user_choice("Select energy range for fitting:", options)
    
    if choice == 1:
        # Full range
        return np.ones(len(energy), dtype=bool)
    
    elif choice == 2:
        # Custom range
        while True:
            try:
                min_energy = float(input(f"Enter minimum energy ({energy.min():.1f}): "))
                max_energy = float(input(f"Enter maximum energy ({energy.max():.1f}): "))
                
                if min_energy < max_energy:
                    mask = (energy >= min_energy) & (energy <= max_energy)
                    if np.sum(mask) > 10:
                        print(f" Selected range: {min_energy} - {max_energy} eV ({np.sum(mask)} points)")
                        return mask
                    else:
                        print(" Range too narrow, need at least 10 points")
                else:
                    print(" Minimum must be less than maximum")
            except ValueError:
                print(" Please enter valid numbers")
    
    else:
        # Overlap region (middle 70%)
        n_points = len(energy)
        start_idx = int(0.15 * n_points)
        end_idx = int(0.85 * n_points)
        
        mask = np.zeros(len(energy), dtype=bool)
        mask[start_idx:end_idx] = True
        
        print(f" Using overlap region: {energy[start_idx]:.1f} - {energy[end_idx]:.1f} eV ({np.sum(mask)} points)")
        return mask


def run_lcf_analysis(energy, target_spectrum, reference_spectra, target_name, max_shift=7.5):
    """Run LCF analysis with all three methods."""
    print(f"\n" + "="*60)
    print(f"LCF ANALYSIS FOR {target_name.upper()}")
    print("="*60)
    
    # Get R-factor benchmark from user
    print(f"\nR-factor Benchmark Setting:")
    print(f"Current benchmark: ≤ 0.02 (High precision with 10x+ sampling for superior accuracy)")
    
    use_custom = input("Use custom R-factor benchmark? (y/n, default=n): ").strip().lower()
    if use_custom in ['y', 'yes']:
        while True:
            try:
                benchmark = float(input("Enter R-factor benchmark (0.001-0.10): "))
                if 0.001 <= benchmark <= 0.10:
                    break
                else:
                    print(" Please enter a value between 0.001 and 0.10")
            except ValueError:
                print(" Please enter a valid number")
    else:
        benchmark = 0.02
    
    # Select methods to run
    methods = ["Interpolation LCF", "Deconvolution LCF", "Convolution LCF"]
    
    print("\nSelect LCF methods to run:")
    print("1. Run all methods")
    print("2. Select individual methods")
    
    method_choice = input("Enter choice (1 or 2): ").strip()
    
    if method_choice == '2':
        # Individual selection
        selected_methods = []
        for i, method in enumerate(methods, 1):
            run_method = input(f"Run {method}? (y/n): ").strip().lower()
            if run_method in ['y', 'yes']:
                selected_methods.append(i-1)
    else:
        # Run all
        selected_methods = [0, 1, 2]
    
    results = {}
    
    # Run selected methods
    for method_idx in selected_methods:
        method_name = methods[method_idx]
        print(f"\n{'='*40}")
        print(f"Running {method_name}...")
        print('='*40)
        
        try:
            if method_idx == 0:  # Interpolation
                lcf = InterpolationLCF(r_factor_benchmark=benchmark)
                lcf.load_data(energy, target_spectrum, reference_spectra)
                result = lcf.fit(max_shift=max_shift)
                results['interpolation'] = result
                print(f" {method_name} completed!")
            
            elif method_idx == 1:  # Deconvolution
                lcf = DeconvolutionLCF(r_factor_benchmark=benchmark)
                lcf.load_data(energy, target_spectrum, reference_spectra)
                
                # Ask user about arctan step + peak modeling
                print(f"\n Enhanced XANES Analysis Options for {method_name}:")
                print("1. Standard deconvolution (original method)")
                print("2. Enhanced with arctan step + peak modeling (recommended for XANES)")
                
                enhance_choice = input("Select option (1 or 2): ").strip()
                if enhance_choice == '2':
                    # Enable arctan step modeling with interactive peak function selection
                    lcf.enable_arctan_step_modeling()
                    lcf.interactive_peak_function_selection()
                    
                    # CRITICAL: Only fit reference spectra, NEVER target spectra (LCF data)
                    print(f"\n IMPORTANT: Interactive fitting for REFERENCE spectra only")
                    print(f"           Target spectra (LCF data) will remain UNCHANGED")
                    print(f"=" * 60)
                    
                    # STEP 1: Set global E0 for ALL reference spectra
                    print(f"\n STEP 1: Setting Global E0 for ALL Reference Spectra")
                    print(f"This E0 value will be used for ALL reference fits to ensure consistency.")
                    reference_names = list(reference_spectra.keys())
                    if reference_names:
                        first_ref_name = reference_names[0]
                        first_ref_spectrum = reference_spectra[first_ref_name]
                        lcf.set_global_e0(energy, first_ref_spectrum, "all references")
                    
                    # STEP 2: Set global peak count for ALL references
                    print(f"\n STEP 2: Setting Peak Count for ALL Reference Spectra")
                    print(f"All references will use the SAME number of peaks for consistency.")
                    print(f"Available range: 1-50 peaks")
                    
                    while True:
                        try:
                            global_n_peaks = int(input("Enter number of peaks for ALL references (1-50): "))
                            if 1 <= global_n_peaks <= 50:
                                break
                            else:
                                print("Please enter a number between 1 and 50")
                        except ValueError:
                            print("Please enter a valid number")
                    
                    print(f" All {len(reference_names)} references will be fitted with {global_n_peaks} peak(s)")
                    print(f" Global E0 = {lcf.get_global_e0():.2f} eV will be used for all")
                    
                    # STEP 3: Apply unified fitting to ALL reference spectra (ONE TIME ONLY)
                    print(f"\n STEP 3: One-Time Arctan+Peak Fitting for References")
                    print(f"=" * 60)
                    print(f"Note: Once fitted, these models will be used directly for LCF")
                    print(f"      No further peak modifications during LCF optimization")
                    
                    success = lcf.interactive_peak_fitting_all_references(global_n_peaks)
                    
                    if success:
                        print(f"\n ✓ Reference arctan+peak modeling completed!")
                        print(f" Fitted references will be used directly for LCF analysis")
                        print(f" No further peak fitting during LCF optimization")
                        print(f" Target spectrum remains ORIGINAL experimental data")
                    else:
                        print(f"\n ⚠ Some reference fits failed. Proceeding with available fits...")
                    
                    # Close any remaining plots to prevent blocking
                    try:
                        import matplotlib.pyplot as plt
                        plt.close('all')
                    except:
                        pass
                
                print(f"\n Proceeding to LCF optimization...")
                print(f"Using fitted reference models (no further peak changes)")
                
                result = lcf.fit(max_shift=max_shift)
                results['deconvolution'] = result
                print(f" {method_name} completed!")
            
            elif method_idx == 2:  # Convolution
                lcf = ConvolutionLCF(r_factor_benchmark=benchmark)
                lcf.load_data(energy, target_spectrum, reference_spectra)
                
                # CRITICAL: Convolution method should NOT modify target spectra (LCF data)
                print(f"\n IMPORTANT: Convolution applied to REFERENCE spectra only")
                print(f"           Target spectra (LCF data) will remain UNCHANGED")
                
                result = lcf.fit(max_shift=max_shift)
                results['convolution'] = result
                print(f" {method_name} completed!")
        
        except Exception as e:
            print(f" {method_name} failed with error: {e}")
    
    return results


def display_results_summary(all_results, targets):
    """Display comprehensive results summary with shift values and save option."""
    print("\n" + "="*90)
    print("                    COMPREHENSIVE RESULTS SUMMARY")
    print("="*90)
    
    # Basic summary table
    print(f"\n{'Target':<12} {'Method':<15} {'R-factor':<10} {'Benchmark':<12} {'Status':<10} {'Best Component':<20} {'Fraction':<10}")
    print("-" * 100)
    
    detailed_results = []
    
    for target_name in targets:
        if target_name in all_results:
            target_results = all_results[target_name]
            
            for method_name, result in target_results.items():
                if not result['success']:
                    continue
                
                r_factor = result['r_factor']
                
                # Check benchmark status
                benchmark_val = "0.02"  # Updated benchmark
                if r_factor <= 0.02:
                    status = " Met"
                else:
                    status = " Not Met"
                
                # Find best component
                fractions = result['fractions']
                best_component = max(fractions.items(), key=lambda x: x[1])
                
                print(f"{target_name:<12} {method_name:<15} {r_factor:<10.6f} "
                      f"≤{benchmark_val:<11} {status:<10} {best_component[0]:<20} {best_component[1]:<10.3f}")
                
                # Store detailed results for later display and saving
                detailed_results.append({
                    'target': target_name,
                    'method': method_name,
                    'r_factor': r_factor,
                    'fractions': fractions,
                    'shifts': result.get('shifts', {}),
                    'result': result
                })
    
    print("-" * 100)
    
    # Detailed results with shift values
    print("\n" + "="*90)
    print("                    DETAILED RESULTS WITH ENERGY SHIFTS")
    print("="*90)
    
    for detail in detailed_results:
        print(f"\n🎯 {detail['target'].upper()} - {detail['method'].upper()}")
        print(f"   R-factor: {detail['r_factor']:.6f}")
        print(f"   Component Analysis:")
        
        fractions = detail['fractions']
        shifts = detail['shifts']
        
        for ref_name in fractions.keys():
            fraction = fractions[ref_name]
            shift = shifts.get(ref_name, 0.0)
            print(f"     {ref_name:<25} {fraction:>6.1%}  (shift: {shift:+7.2f} eV)")
    
    # Ask about saving results
    print(f"\n" + "="*60)
    save_results = input("Save detailed results to file? (y/n) [default: y]: ").strip().lower()
    if save_results != 'n':
        save_detailed_results(detailed_results, targets)


def save_detailed_results(detailed_results, targets):
    """Save detailed LCF results to files."""
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    results_dir = Path("LCF_Results")
    results_dir.mkdir(exist_ok=True)
    
    # Save JSON summary
    json_file = results_dir / f"LCF_Summary_{timestamp}.json"
    
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'targets': targets,
        'results': {}
    }
    
    for detail in detailed_results:
        target = detail['target']
        method = detail['method']
        
        if target not in json_data['results']:
            json_data['results'][target] = {}
        
        json_data['results'][target][method] = {
            'r_factor': detail['r_factor'],
            'fractions': detail['fractions'],
            'shifts': detail['shifts'],
            'success': detail['result']['success'],
            'method_type': detail['result'].get('method', 'unknown')
        }
        
        # Add arctan modeling info if available
        if 'uses_arctan_modeling' in detail['result']:
            json_data['results'][target][method]['uses_arctan_modeling'] = detail['result']['uses_arctan_modeling']
            json_data['results'][target][method]['peak_function_type'] = detail['result'].get('peak_function_type')
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save human-readable text summary
    txt_file = results_dir / f"LCF_Summary_{timestamp}.txt"
    
    with open(txt_file, 'w') as f:
        f.write("XANES Linear Combination Fitting (LCF) Analysis Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Targets: {', '.join(targets)}\n\n")
        
        for detail in detailed_results:
            f.write(f"Target: {detail['target']}\n")
            f.write(f"Method: {detail['method']}\n")
            f.write(f"R-factor: {detail['r_factor']:.6f}\n")
            f.write(f"Component Analysis:\n")
            
            fractions = detail['fractions']
            shifts = detail['shifts']
            
            for ref_name in fractions.keys():
                fraction = fractions[ref_name]
                shift = shifts.get(ref_name, 0.0)
                f.write(f"  {ref_name:<25} {fraction:>6.1%}  (shift: {shift:+7.2f} eV)\n")
            
            # Add arctan modeling info if available
            result = detail['result']
            if result.get('uses_arctan_modeling', False):
                f.write(f"  Enhanced with arctan step + {result.get('peak_function_type', 'unknown')} peak modeling\n")
            
            f.write("\n" + "-" * 40 + "\n\n")
    
    # Save CSV for easy import to Excel
    csv_file = results_dir / f"LCF_Summary_{timestamp}.csv"
    
    import csv
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Target', 'Method', 'R-factor', 'Reference', 'Fraction', 'Energy_Shift_eV', 'Arctan_Modeling', 'Peak_Function'])
        
        # Data rows
        for detail in detailed_results:
            target = detail['target']
            method = detail['method']
            r_factor = detail['r_factor']
            
            result = detail['result']
            uses_arctan = result.get('uses_arctan_modeling', False)
            peak_function = result.get('peak_function_type', '')
            
            fractions = detail['fractions']
            shifts = detail['shifts']
            
            for ref_name in fractions.keys():
                fraction = fractions[ref_name]
                shift = shifts.get(ref_name, 0.0)
                writer.writerow([target, method, f"{r_factor:.6f}", ref_name, f"{fraction:.4f}", f"{shift:.2f}", uses_arctan, peak_function])
    
    print(f"\n Results saved to:")
    print(f"    JSON: {json_file}")
    print(f"    Text: {txt_file}")
    print(f"    CSV:  {csv_file}")
    print(f"    Directory: {results_dir.absolute()}")


def plot_all_results(all_results, processed_data, targets, fit_energy, fit_mask):
    """Plot results for all targets and methods."""
    print("\nGenerating comprehensive plots...")
    
    for target_name in targets:
        if target_name not in all_results:
            continue
        
        target_results = all_results[target_name]
        successful_methods = [method for method, result in target_results.items() 
                            if result['success']]
        
        if not successful_methods:
            print(f"  No successful results for {target_name}")
            continue
        
        n_methods = len(successful_methods)
        
        # Create plots
        fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
        if n_methods == 1:
            axes = axes.reshape(2, 1)
        
        energy = processed_data['energy']
        target_spectrum = processed_data[target_name]
        
        # Use fit energy range for plotting
        fit_energy_for_plot = energy[fit_mask]
        fit_target_for_plot = target_spectrum[fit_mask]
        
        for i, method_name in enumerate(successful_methods):
            result = target_results[method_name]
            
            # Top row: Target vs Fitted
            axes[0, i].plot(fit_energy_for_plot, fit_target_for_plot, 'ko-', markersize=2, 
                           linewidth=1, label='Target', alpha=0.8)
            axes[0, i].plot(fit_energy_for_plot, result['fitted_spectrum'], 'r-',
                           linewidth=2, label='Fitted')
            axes[0, i].set_title(f'{target_name} - {method_name.title()}\n'
                                f'R-factor: {result["r_factor"]:.4f}')
            axes[0, i].set_xlabel('Energy (eV)')
            axes[0, i].set_ylabel('Absorption')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Bottom row: Component fractions
            ref_names = list(result['fractions'].keys())
            fractions = list(result['fractions'].values())
            
            import matplotlib.cm as cm
            colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(ref_names)))
            bars = axes[1, i].bar(range(len(ref_names)), fractions,
                                 color=colors, alpha=0.7)
            axes[1, i].set_xticks(range(len(ref_names)))
            axes[1, i].set_xticklabels(ref_names, rotation=45, ha='right')
            axes[1, i].set_ylabel('Fraction')
            axes[1, i].set_title('Component Fractions')
            axes[1, i].set_ylim(0, 1)
            
            # Add value labels
            for bar, fraction in zip(bars, fractions):
                height = bar.get_height()
                axes[1, i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{fraction:.2f}', ha='center', va='bottom')
            
            axes[1, i].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main program function."""
    print("="*80)
    print("    XANES Linear Combination Fitting (LCF) Analysis Program")
    print("                           Clean Version")
    print("="*80)
    print("This program ensures exact data matching with Excel when no preprocessing is applied.")
    print("All code and outputs are in English with comprehensive documentation.")
    
    # Step 1: Load data
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING")
    print("="*60)
    
    loader = XANESDataLoader()
    
    # Get file path
    default_file = "/Users/CanaanWangJiaNan/Desktop/spot1.xlsx"
    file_path = input(f"Enter Excel file path [default: {default_file}]: ").strip()
    if not file_path:
        file_path = default_file
    
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return
    
    # Load data
    raw_data = loader.load_excel_file(file_path)
    if not raw_data:
        print(" Failed to load data")
        return
    
    print(f"\n{loader.get_data_summary()}")
    
    # Step 2: Identify targets and references
    print("\n" + "="*60)
    print("STEP 2: TARGET AND REFERENCE IDENTIFICATION")
    print("="*60)
    
    auto_detect = input("Auto-detect targets and references? (y/n) [default: y]: ").strip().lower()
    auto_detect = auto_detect != 'n'
    
    targets, references = loader.identify_targets_and_references(auto_detect)
    
    if not targets or not references:
        print(" Need at least one target and two references")
        return
    
    # Step 3: Preprocessing
    print("\n" + "="*60)
    print("STEP 3: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = XANESPreprocessor(loader)
    processed_data = preprocessor.interactive_preprocessing(targets, references)
    
    if not processed_data:
        print(" Preprocessing failed")
        return
    
    print(" Data preprocessing completed!")
    
    # Step 3.5: Global Analysis Parameters
    print("\n" + "="*60)
    print("STEP 3.5: GLOBAL ANALYSIS PARAMETERS")
    print("="*60)
    
    # Get maximum energy shift setting
    print(f"\nMaximum Energy Shift Setting:")
    print(f"Default: ±7.5 eV (typical for XANES analysis)")
    
    use_custom_shift = input("Use custom maximum shift? (y/n, default=n): ").strip().lower()
    if use_custom_shift in ['y', 'yes']:
        while True:
            try:
                global_max_shift = float(input("Enter maximum energy shift in eV (1.0-20.0): "))
                if 1.0 <= global_max_shift <= 20.0:
                    break
                else:
                    print(" Please enter a value between 1.0 and 20.0 eV")
            except ValueError:
                print(" Please enter a valid number")
    else:
        global_max_shift = 7.5
    
    print(f" Using maximum energy shift: ±{global_max_shift} eV")
    
    # Step 4: Energy range selection
    print("\n" + "="*60)
    print("STEP 4: ENERGY RANGE SELECTION")
    print("="*60)
    
    energy = processed_data['energy']
    fit_mask = select_energy_range(energy)
    fit_energy = energy[fit_mask]
    
    # Step 5: LCF Analysis
    print("\n" + "="*60)
    print("STEP 5: LINEAR COMBINATION FITTING")
    print("="*60)
    
    all_results = {}
    
    for target_name in targets:
        # Get target and reference data for fitting
        target_spectrum = processed_data[target_name][fit_mask]
        reference_spectra = {ref: processed_data[ref][fit_mask] for ref in references}
        
        # Run LCF analysis
        target_results = run_lcf_analysis(fit_energy, target_spectrum, 
                                        reference_spectra, target_name, global_max_shift)
        
        if target_results:
            all_results[target_name] = target_results
    
    # Step 6: Results Summary and Visualization
    print("\n" + "="*60)
    print("STEP 6: RESULTS AND VISUALIZATION")
    print("="*60)
    
    if all_results:
        display_results_summary(all_results, targets)
        
        # Ask about plotting
        plot_choice = input("\nGenerate comprehensive plots? (y/n) [default: y]: ").strip().lower()
        if plot_choice != 'n':
            plot_all_results(all_results, processed_data, targets, fit_energy, fit_mask)
        
        print("\n Analysis completed successfully!")
        print("\nKey points:")
        print("- All original data integrity preserved")
        print("- Preprocessing was user-controlled") 
        print("- Energy shifts optimized within ±{:.1f} eV".format(global_max_shift))
        print("- R-factor minimized (target < 0.02 with 10x+ sampling for superior accuracy)")
        print("- Optimized for 2470-2485 eV region without preprocessing")
        
    else:
        print(" No successful fitting results obtained")
    
    print("\n" + "="*80)
    print("Thank you for using the XANES LCF Analysis Program!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check your data and try again.")
