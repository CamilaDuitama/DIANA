#!/usr/bin/env python3
"""
Generate Data Split Validation Figures (Supplementary Figure 2)

PURPOSE:
    Visualize train/test/validation split distributions across multiple variables
    to demonstrate proper data partitioning and avoid data leakage.

INPUTS (from config.py):
    - PATHS['train_metadata']: Training set metadata
    - PATHS['test_metadata']: Test set metadata
    - PATHS['validation_metadata']: Validation set metadata
    - paper/metadata/country_coords.csv: Geographic coordinates (optional)
    - data/validation/raw/{sample}/*.fastq*: FASTQ files for size calculation

OUTPUTS:
    - paper/figures/final/sup_02_data_split_sample_type.png
    - paper/figures/final/sup_02_data_split_projects.png
    - paper/figures/final/sup_02_data_split_community_type.png
    - paper/figures/final/sup_02_data_split_publication_year.png
    - paper/figures/final/sup_02_data_split_file_size.png
    - paper/figures/final/sup_02_data_split_geographic.png (HTML only)
    - Corresponding .html interactive versions

PROCESS:
    Generate 6 separate figures showing distribution comparisons:
    1. Sample type % distribution (ancient vs modern)
    2. Top 10 project names % distribution
    3. Community type % distribution
    4. Publication year histogram (%)
    5. Input file size box plots (GB)
    6. Geographic world map (scattergeo)

CONFIGURATION:
    All styling, paths, and constants imported from config.py:
    - PATHS: File locations for metadata
    - PLOT_CONFIG: Colors (Train=Blue, Test=Orange, Validation=Red)
    - SAMPLE_TYPE_MAP: Normalization

HARDCODED VALUES:
    - Top N projects: 10
    - Publication year bins: 20
    - File size estimation: read_count × 250 bytes for train/test
    - Dataset colors: Train=Blue, Test=Orange, Validation=Red

DEPENDENCIES:
    - pandas, numpy, plotly, pathlib, os
    - config.py (same directory)

USAGE:
    python scripts/paper/generate_data_split.py
    
AUTHOR: Generated via refactoring of 06_compare_predictions.py
"""

import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add script directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, PLOT_CONFIG, SAMPLE_TYPE_MAP


# ============================================================================
# HARDCODED PARAMETERS
# ============================================================================

TOP_N_PROJECTS = 10
PUBLICATION_YEAR_BINS = 20


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def load_metadata():
    """Load all three metadata files."""
    train_meta = pd.read_csv(PATHS['train_metadata'], sep='\t')
    test_meta = pd.read_csv(PATHS['test_metadata'], sep='\t')
    val_meta = pd.read_csv(PATHS['validation_metadata'], sep='\t')
    
    # Normalize sample_type
    for df in [train_meta, test_meta, val_meta]:
        if 'sample_type' in df.columns:
            df['sample_type'] = df['sample_type'].map(SAMPLE_TYPE_MAP).fillna(df['sample_type'])
    
    print(f"✓ Loaded metadata:")
    print(f"  Train: {len(train_meta)} samples")
    print(f"  Test: {len(test_meta)} samples")
    print(f"  Validation: {len(val_meta)} samples")
    
    return train_meta, test_meta, val_meta


def plot_sample_type(train_meta, test_meta, val_meta, output_dir):
    """Figure 1: Sample type distribution."""
    colors = [PLOT_CONFIG['colors']['train'], PLOT_CONFIG['colors']['test'], PLOT_CONFIG['colors']['validation']]
    
    if 'sample_type' not in train_meta.columns:
        print("  ⚠ No sample_type column, skipping")
        return
    
    # Get all sample types
    all_sample_types = sorted(set(
        list(train_meta['sample_type'].dropna().unique()) +
        list(test_meta['sample_type'].dropna().unique()) +
        list(val_meta['sample_type'].dropna().unique())
    ))
    
    fig = go.Figure()
    
    for split, df, color in [('Train', train_meta, colors[0]), 
                              ('Test', test_meta, colors[1]), 
                              ('Validation', val_meta, colors[2])]:
        counts = df['sample_type'].value_counts()
        total = len(df['sample_type'].dropna())
        y_values = [(counts.get(st, 0) / total * 100) if total > 0 else 0 for st in all_sample_types]
        
        fig.add_trace(go.Bar(
            x=all_sample_types,
            y=y_values,
            name=split,
            marker=dict(
                color=color,
                opacity=PLOT_CONFIG['fill_opacity'],
                line=dict(color=PLOT_CONFIG['border_color'], width=PLOT_CONFIG['line_width'])
            ),
            text=[f"{v:.1f}%" for v in y_values],
            textposition='outside'
        ))
    
    fig.update_layout(
        title="Sample Type Distribution Across Datasets",
        xaxis_title="Sample Type",
        yaxis_title="Percentage of Samples in Split (%)",
        template=PLOT_CONFIG['template'],
        font=dict(size=PLOT_CONFIG['font_size']),
        height=600,
        width=800,
        barmode='group'
    )
    
    output_file = output_dir / "sup_02_data_split_sample_type.png"
    fig.write_html(str(output_file.with_suffix('.html')))
    fig.write_image(str(output_file), width=800, height=600, scale=2)
    print(f"  ✓ {output_file.name}")


def plot_projects(train_meta, test_meta, val_meta, output_dir):
    """Figure 2: Top 10 project names distribution."""
    colors = [PLOT_CONFIG['colors']['train'], PLOT_CONFIG['colors']['test'], PLOT_CONFIG['colors']['validation']]
    
    if 'project_name' not in train_meta.columns:
        print("  ⚠ No project_name column, skipping")
        return
    
    # Get top 10 projects
    all_projects = pd.concat([
        train_meta['project_name'].dropna(),
        test_meta['project_name'].dropna(),
        val_meta['project_name'].dropna()
    ])
    top_projects = all_projects.value_counts().head(TOP_N_PROJECTS).index.tolist()
    
    fig = go.Figure()
    
    for split, df, color in [('Train', train_meta, colors[0]), 
                              ('Test', test_meta, colors[1]), 
                              ('Validation', val_meta, colors[2])]:
        counts = df['project_name'].value_counts()
        total = len(df['project_name'].dropna())
        y_values = [(counts.get(proj, 0) / total * 100) if total > 0 else 0 for proj in top_projects]
        
        fig.add_trace(go.Bar(
            x=top_projects,
            y=y_values,
            name=split,
            marker=dict(
                color=color,
                opacity=PLOT_CONFIG['fill_opacity'],
                line=dict(color=PLOT_CONFIG['border_color'], width=PLOT_CONFIG['line_width'])
            )
        ))
    
    fig.update_layout(
        title=f"Top {TOP_N_PROJECTS} Project Names Distribution",
        xaxis_title="Project Name",
        yaxis_title="Percentage of Samples in Split (%)",
        template=PLOT_CONFIG['template'],
        font=dict(size=PLOT_CONFIG['font_size']),
        height=600,
        width=1000,
        barmode='group',
        xaxis=dict(tickangle=-45)
    )
    
    output_file = output_dir / "sup_02_data_split_projects.png"
    fig.write_html(str(output_file.with_suffix('.html')))
    fig.write_image(str(output_file), width=1000, height=600, scale=2)
    print(f"  ✓ {output_file.name}")


def plot_community_type(train_meta, test_meta, val_meta, output_dir):
    """Figure 3: Community type distribution."""
    colors = [PLOT_CONFIG['colors']['train'], PLOT_CONFIG['colors']['test'], PLOT_CONFIG['colors']['validation']]
    
    if 'community_type' not in train_meta.columns:
        print("  ⚠ No community_type column, skipping")
        return
    
    # Get all community types
    all_community_types = sorted(set(
        list(train_meta['community_type'].dropna().unique()) +
        list(test_meta['community_type'].dropna().unique()) +
        list(val_meta['community_type'].dropna().unique())
    ))
    
    fig = go.Figure()
    
    for split, df, color in [('Train', train_meta, colors[0]), 
                              ('Test', test_meta, colors[1]), 
                              ('Validation', val_meta, colors[2])]:
        counts = df['community_type'].value_counts()
        total = len(df['community_type'].dropna())
        y_values = [(counts.get(ct, 0) / total * 100) if total > 0 else 0 for ct in all_community_types]
        
        fig.add_trace(go.Bar(
            x=all_community_types,
            y=y_values,
            name=split,
            marker=dict(
                color=color,
                opacity=PLOT_CONFIG['fill_opacity'],
                line=dict(color=PLOT_CONFIG['border_color'], width=PLOT_CONFIG['line_width'])
            )
        ))
    
    fig.update_layout(
        title="Community Type Distribution Across Datasets",
        xaxis_title="Community Type",
        yaxis_title="Percentage of Samples in Split (%)",
        template=PLOT_CONFIG['template'],
        font=dict(size=PLOT_CONFIG['font_size']),
        height=600,
        width=1200,
        barmode='group',
        xaxis=dict(tickangle=-45)
    )
    
    output_file = output_dir / "sup_02_data_split_community_type.png"
    fig.write_html(str(output_file.with_suffix('.html')))
    fig.write_image(str(output_file), width=1200, height=600, scale=2)
    print(f"  ✓ {output_file.name}")


def plot_publication_year(train_meta, test_meta, val_meta, output_dir):
    """Figure 4: Publication year distribution."""
    colors = [PLOT_CONFIG['colors']['train'], PLOT_CONFIG['colors']['test'], PLOT_CONFIG['colors']['validation']]
    
    year_col = 'Publication_year' if 'Publication_year' in train_meta.columns else 'publication_year'
    
    if year_col not in train_meta.columns:
        print("  ⚠ No publication year column, skipping")
        return
    
    fig = go.Figure()
    
    for split, df, color in [('Train', train_meta, colors[0]), 
                              ('Test', test_meta, colors[1]), 
                              ('Validation', val_meta, colors[2])]:
        if year_col in df.columns:
            # Convert to numeric first (handles "Not applicable" and other text)
            years = pd.to_numeric(df[year_col], errors='coerce')
            # Then drop NaN values (from empty strings and non-numeric text)
            years = years.dropna()
            # Only add trace if there are valid years to plot
            if len(years) > 0:
                fig.add_trace(go.Histogram(
                    x=years,
                    name=f"{split} (n={len(years)})",
                    histnorm='percent',
                    marker=dict(
                        color=color,
                        opacity=PLOT_CONFIG['fill_opacity'],
                        line=dict(color=PLOT_CONFIG['border_color'], width=PLOT_CONFIG['line_width'])
                    ),
                    nbinsx=PUBLICATION_YEAR_BINS
                ))
    
    fig.update_layout(
        title="Publication Year Distribution",
        xaxis_title="Publication Year",
        yaxis_title="Percentage of Samples in Split (%)",
        template=PLOT_CONFIG['template'],
        font=dict(size=PLOT_CONFIG['font_size']),
        height=600,
        width=900,
        barmode='group'
    )
    
    output_file = output_dir / "sup_02_data_split_publication_year.png"
    fig.write_html(str(output_file.with_suffix('.html')))
    fig.write_image(str(output_file), width=900, height=600, scale=2)
    print(f"  ✓ {output_file.name}")


def plot_file_size(train_meta, test_meta, val_meta, output_dir):
    """Figure 5: File size distribution."""
    colors = [PLOT_CONFIG['colors']['train'], PLOT_CONFIG['colors']['test'], PLOT_CONFIG['colors']['validation']]
    
    fig = go.Figure()
    
    for split, df, color in [('Train', train_meta, colors[0]), 
                              ('Test', test_meta, colors[1]), 
                              ('Validation', val_meta, colors[2])]:
        sizes_gb = []
        
        if split == 'Validation':
            # Calculate actual FASTQ sizes
            for _, row in df.iterrows():
                sample_id = row['Run_accession']
                fastq_dir = Path(f"data/validation/raw/{sample_id}")
                if fastq_dir.exists():
                    fastq_size_gb = sum(os.path.getsize(fq) / (1024**3) 
                                       for fq in fastq_dir.glob("*.fastq*"))
                    if fastq_size_gb > 0:
                        sizes_gb.append(fastq_size_gb)
        else:
            # Estimate from read counts
            if 'Avg_num_reads' in df.columns:
                reads = pd.to_numeric(df['Avg_num_reads'], errors='coerce').fillna(0)
                sizes_gb = [(r * 250 / 1e9) for r in reads if r > 0]
        
        if sizes_gb:
            fig.add_trace(go.Box(
                y=sizes_gb,
                name=split,
                marker=dict(
                    color=color,
                    opacity=PLOT_CONFIG['fill_opacity'],
                    line=dict(color=PLOT_CONFIG['border_color'], width=PLOT_CONFIG['line_width'])
                ),
                fillcolor=color
            ))
    
    fig.update_layout(
        title="Input File Size Distribution",
        xaxis_title="Dataset Split",
        yaxis_title="File Size (GB)",
        template=PLOT_CONFIG['template'],
        font=dict(size=PLOT_CONFIG['font_size']),
        height=600,
        width=800
    )
    
    output_file = output_dir / "sup_02_data_split_file_size.png"
    fig.write_html(str(output_file.with_suffix('.html')))
    fig.write_image(str(output_file), width=800, height=600, scale=2)
    print(f"  ✓ {output_file.name}")


def plot_geographic(train_meta, test_meta, val_meta, output_dir):
    """Figure 6: Geographic distribution (HTML only)."""
    colors = [PLOT_CONFIG['colors']['train'], PLOT_CONFIG['colors']['test'], PLOT_CONFIG['colors']['validation']]
    
    coords_file = Path('paper/metadata/country_coords.csv')
    if not coords_file.exists():
        print("  ⚠ No country_coords.csv, skipping geographic map")
        return
    
    coords_df = pd.read_csv(coords_file)
    
    # Extract country data
    geo_data = []
    for df, split in [(train_meta, 'Train'), (test_meta, 'Test'), (val_meta, 'Validation')]:
        if 'geo_loc_name' in df.columns:
            for country in df['geo_loc_name'].dropna():
                country_name = str(country).split(':')[0].strip()
                geo_data.append({'country': country_name, 'split': split})
    
    if not geo_data:
        print("  ⚠ No geographic data available")
        return
    
    geo_df = pd.DataFrame(geo_data)
    country_counts = geo_df.groupby(['country', 'split']).size().reset_index(name='count')
    country_counts = country_counts.merge(
        coords_df[['COUNTRY', 'latitude', 'longitude']],
        left_on='country',
        right_on='COUNTRY',
        how='left'
    ).dropna(subset=['latitude', 'longitude'])
    
    fig = go.Figure()
    
    for split, color in [('Train', colors[0]), ('Validation', colors[2]), ('Test', colors[1])]:
        split_data = country_counts[country_counts['split'] == split]
        
        if len(split_data) > 0:
            fig.add_trace(go.Scattergeo(
                lon=split_data['longitude'],
                lat=split_data['latitude'],
                text=split_data['country'] + '<br>' + split_data['count'].astype(str) + ' samples',
                marker=dict(
                    size=split_data['count'],
                    sizemode='area',
                    sizeref=2.*max(country_counts['count'])/(40.**2),
                    sizemin=4,
                    color=color,
                    opacity=0.8,
                    line=dict(width=1.5, color='white')
                ),
                name=split
            ))
    
    fig.update_geos(
        resolution=50,
        scope='world',
        projection_type='natural earth',
        showcountries=True,
        countrycolor="lightgray"
    )
    
    fig.update_layout(
        title="Geographic Sample Distribution",
        template=PLOT_CONFIG['template'],
        font=dict(size=PLOT_CONFIG['font_size']),
        height=700,
        width=1200
    )
    
    output_file = output_dir / "sup_02_data_split_geographic.html"
    fig.write_html(str(output_file))
    print(f"  ✓ {output_file.name} (interactive HTML only - geo maps don't export to PNG)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATING DATA SPLIT VALIDATION FIGURES")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(PATHS['figures_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("\n[1/7] Loading metadata...")
    train_meta, test_meta, val_meta = load_metadata()
    
    # Generate figures
    print("\n[2/7] Generating sample type distribution...")
    plot_sample_type(train_meta, test_meta, val_meta, output_dir)
    
    print("\n[3/7] Generating project names distribution...")
    plot_projects(train_meta, test_meta, val_meta, output_dir)
    
    print("\n[4/7] Generating community type distribution...")
    plot_community_type(train_meta, test_meta, val_meta, output_dir)
    
    print("\n[5/7] Generating publication year distribution...")
    plot_publication_year(train_meta, test_meta, val_meta, output_dir)
    
    print("\n[6/7] Generating file size distribution...")
    plot_file_size(train_meta, test_meta, val_meta, output_dir)
    
    print("\n[7/7] Generating geographic distribution...")
    plot_geographic(train_meta, test_meta, val_meta, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE - Data split validation figures generated (6 plots)")
    print("=" * 80)


if __name__ == '__main__':
    main()
