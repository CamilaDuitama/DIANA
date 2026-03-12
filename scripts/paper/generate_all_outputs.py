#!/usr/bin/env python3
"""
Generate all paper figures and tables in one command.

Runs all paper generation scripts in sequence:
1. Data split validation figure (sup_03)
2. Feature importance figure (main_03)
3. Memory vs data size figure (supp_04)
4. Runtime and memory scalability figure (main_04)
5. All 6 LaTeX tables

Usage:
    python scripts/paper/generate_all_outputs.py
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path: str, description: str) -> bool:
    """Run a Python script and report success/failure."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        print(f"\n✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FAILED")
        print(f"   Error code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ {description} - SCRIPT NOT FOUND: {script_path}")
        return False

def main():
    print("\n" + "="*70)
    print("Generating All Paper Outputs (Figures + Tables)")
    print("="*70)
    
    scripts_dir = Path(__file__).parent
    
    scripts = [
        (scripts_dir / "01_data_split_figure.py", "Data Split Validation Figure (sup_03)"),
        (scripts_dir / "02_generate_feature_importance_figure.py", "Feature Importance Figure (main_03)"),
        (scripts_dir / "03_memory_vs_data_size.py", "Memory vs Data Size Figure (supp_04)"),
        (scripts_dir / "05_runtime_memory_scalability.py", "Runtime & Memory Scalability Figure (main_04)"),
        (scripts_dir / "06_generate_latex_tables.py", "All 6 LaTeX Tables"),
    ]
    
    results = []
    for script_path, description in scripts:
        success = run_script(str(script_path), description)
        results.append((description, success))
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for description, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status:12} - {description}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n" + "="*70)
        print("✅ All paper outputs generated successfully!")
        print("="*70)
        print("\nOutputs:")
        print("  Figures: paper/figures/")
        print("    - main_03_feature_importance.png")
        print("    - main_04_runtime_memory_scalability.png")
        print("    - sup_03_data_split_validation.png")
        print("    - supp_04_memory_vs_datasize.png")
        print("\n  Tables: paper/tables/")
        print("    - table1_performance.tex")
        print("    - supp_table1_resources.tex")
        print("    - supp_table2_distribution.tex")
        print("    - supp_table3_unseen.tex")
        print("    - supp_table4_perclass.tex")
        print("    - supp_table5_hyperparams.tex")
        print("="*70 + "\n")
        return 0
    else:
        print("\n" + "="*70)
        print("⚠️  Some scripts failed. Check output above for details.")
        print("="*70 + "\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
