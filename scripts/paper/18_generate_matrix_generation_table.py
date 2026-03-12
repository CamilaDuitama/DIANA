#!/usr/bin/env python3
"""
Generate Feature Matrix Generation Parameters Table (Supplementary Table 8)

PURPOSE:
    Create LaTeX table documenting the muset/kmtricks pipeline parameters
    and computational resources used to build the unitig feature matrix.

OUTPUTS:
    - paper/tables/final/sup_table_08_matrix_generation.tex

HARDCODED VALUES:
    All values reflect the actual muset run on maestro cluster (node maestro-2499).
    Update these if the matrix is regenerated with different parameters.

USAGE:
    python scripts/paper/18_generate_matrix_generation_table.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS


def generate_matrix_generation_table(output_path: Path) -> None:
    """Generate the feature matrix generation parameters table."""
    lines = []
    lines.append("\\centering")
    lines.append("\\caption{Feature matrix generation parameters and computational resources"
                 "\\label{tab:matrix_generation}}")
    lines.append("\\addcontentsline{toc}{subsection}{Supplementary Table 8: Feature matrix generation parameters}")
    lines.append("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}llr@{\\extracolsep{\\fill}}}")
    lines.append("\\toprule")
    lines.append("Category & Parameter & Value \\\\")
    lines.append("\\midrule")
    lines.append("\\multirow{2}{*}{Input} & Samples & 3,070 \\\\")
    lines.append(" & Data source & Logan unitigs \\\\")
    lines.append("\\addlinespace")
    lines.append("\\multirow{7}{*}{k-mer Filtering} & k-mer size & 31 \\\\")
    lines.append(" & Minimum abundance & 2 \\\\")
    lines.append(" & Minimizer size & 15 \\\\")
    lines.append(" & Min. fraction present (-F) & 0.1 \\\\")
    lines.append(" & Max. fraction absent (-f) & 0.1 \\\\")
    lines.append(" & Initial k-mers & 436,456,132,058 \\\\")
    lines.append(" & Retained k-mers & 18,953,119 \\\\")
    lines.append("\\addlinespace")
    lines.append("\\multirow{3}{*}{Unitig Assembly} & Minimum length & 61 bp \\\\")
    lines.append(" & Total unitigs assembled & 2,755,241 \\\\")
    lines.append(" & Final unitigs & 107,480 \\\\")
    lines.append("\\addlinespace")
    lines.append("\\multirow{4}{*}{Computational} & Tool & muset v0.6.1 \\\\")
    lines.append(" & CPU cores & 32 \\\\")
    lines.append(" & Memory & 500 GB \\\\")
    lines.append(" & Runtime & 3d 4h 45m \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  ✓ {output_path.name}")


def main():
    print("=" * 80)
    print("GENERATING MATRIX GENERATION TABLE (SUPPLEMENTARY TABLE 8)")
    print("=" * 80)

    output_path = Path(PATHS["tables_dir"]) / "sup_table_08_matrix_generation.tex"
    generate_matrix_generation_table(output_path)

    print("\n" + "=" * 80)
    print("✓ COMPLETE")


if __name__ == "__main__":
    main()
