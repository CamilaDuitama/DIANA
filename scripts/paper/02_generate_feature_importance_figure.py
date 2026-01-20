#!/usr/bin/env python3
"""
Generate feature importance figure for main paper.

Single panel showing:
- Taxonomic composition of top 100 discriminant features by task

Output: paper/figures/Figure_feature_importance.png
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def create_feature_importance_figure(
    taxonomy_path: Path,
    output_path: Path
):
    """
    Create single-panel figure showing taxonomic composition.
    
    Args:
        taxonomy_path: Path to taxonomic composition PNG
        output_path: Path to save figure
    """
    # Read image
    taxonomy_img = mpimg.imread(taxonomy_path)
    
    # Create figure with single panel
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    
    # Display taxonomic composition
    ax.imshow(taxonomy_img)
    ax.axis('off')
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature importance figure to {output_path}")
    print(f"  Source: Taxonomic composition of top 100 features by task")
    print(f"  Dimensions: {taxonomy_img.shape[1]}x{taxonomy_img.shape[0]} pixels")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate feature importance figure (taxonomic composition only)"
    )
    parser.add_argument(
        '--taxonomy',
        type=Path,
        default=Path('paper/figures/feature_analysis/taxonomic_composition_by_task.png'),
        help='Path to taxonomic composition PNG'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('paper/figures/main_03_feature_importance.png'),
        help='Output path for figure'
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not args.taxonomy.exists():
        raise FileNotFoundError(f"Taxonomy plot not found: {args.taxonomy}")
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate figure
    create_feature_importance_figure(args.taxonomy, args.output)


if __name__ == "__main__":
    main()
