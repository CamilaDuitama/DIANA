#!/usr/bin/env python3
"""
Generate world map PNG using matplotlib scatter plot.
Simple approach with no external GIS dependencies.

Output: paper/figures/data_distribution/splits_world_map.png
"""
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Plotly vivid color palette
COLORS = {'Train': '#636EFA', 'Test': '#EF553B'}

# Approximate country coordinates (latitude, longitude)
COUNTRY_COORDS = {
    # Europe
    "United Kingdom": (54, -2),
    "Germany": (51, 10),
    "Italy": (42, 12),
    "France": (46, 2),
    "Spain": (40, -4),
    "Netherlands": (52, 5),
    "Belgium": (50, 4),
    "Switzerland": (47, 8),
    "Austria": (47, 13),
    "Denmark": (56, 9),
    "Sweden": (62, 15),
    "Norway": (62, 10),
    "Finland": (64, 26),
    "Poland": (52, 20),
    "Czech Republic": (49, 15),
    "Czechia": (49, 15),
    "Hungary": (47, 19),
    "Romania": (46, 25),
    "Bulgaria": (43, 25),
    "Greece": (39, 22),
    "Portugal": (39, -8),
    "Ireland": (53, -8),
    "Croatia": (45, 16),
    "Serbia": (44, 21),
    "Slovakia": (48, 19),
    "Slovenia": (46, 15),
    "Estonia": (59, 26),
    "Latvia": (57, 25),
    "Lithuania": (55, 24),
    "Iceland": (65, -18),
    "Ukraine": (49, 32),
    
    # Americas
    "United States": (38, -97),
    "USA": (38, -97),
    "Canada": (60, -95),
    "Mexico": (23, -102),
    "Brazil": (-10, -55),
    "Argentina": (-34, -64),
    "Chile": (-30, -71),
    "Peru": (-9, -76),
    "Colombia": (4, -72),
    "Venezuela": (7, -66),
    "Ecuador": (-2, -77),
    "Bolivia": (-17, -65),
    
    # Asia
    "China": (35, 105),
    "Japan": (36, 138),
    "India": (20, 77),
    "Russia": (60, 100),
    "Russian Federation": (60, 100),
    "South Korea": (37, 127),
    "Turkey": (39, 35),
    "Israel": (31, 35),
    "Iran": (32, 53),
    "Kazakhstan": (48, 68),
    "Mongolia": (46, 105),
    "Thailand": (15, 100),
    "Vietnam": (16, 108),
    "Philippines": (13, 122),
    "Indonesia": (-5, 120),
    "Malaysia": (2, 112),
    "Pakistan": (30, 70),
    "Afghanistan": (33, 65),
    
    # Middle East & Africa  
    "Egypt": (26, 30),
    "South Africa": (-29, 24),
    "Kenya": (0, 38),
    "Nigeria": (10, 8),
    "Morocco": (32, -5),
    "Ethiopia": (8, 38),
    "Tanzania": (-6, 35),
    
    # Oceania
    "Australia": (-25, 133),
    "New Zealand": (-41, 174),
}

def extract_country(location_str):
    """Extract country name from geo_loc_name field."""
    if not location_str or location_str == "Unavailable data":
        return None
    # Format is typically "Country: Region" or just "Country"
    parts = location_str.split(":")
    country = parts[0].strip()
    return country if country else None

def main():
    logger.info("Loading train/test metadata...")
    
    # Load splits
    train_df = pl.read_csv("data/splits/train_metadata.tsv", separator='\t')
    test_df = pl.read_csv("data/splits/test_metadata.tsv", separator='\t')
    
    # Extract countries
    train_countries = [extract_country(loc) for loc in train_df['geo_loc_name'].to_list()]
    train_countries = [c for c in train_countries if c and c in COUNTRY_COORDS]
    
    test_countries = [extract_country(loc) for loc in test_df['geo_loc_name'].to_list()]
    test_countries = [c for c in test_countries if c and c in COUNTRY_COORDS]
    
    # Count samples per country
    train_counts = Counter(train_countries)
    test_counts = Counter(test_countries)
    
    logger.info(f"Mapped {len(train_counts)} countries in train set")
    logger.info(f"Mapped {len(test_counts)} countries in test set")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot train samples (larger circles for more samples)
    if train_counts:
        train_lats = [COUNTRY_COORDS[c][0] for c in train_counts.keys()]
        train_lons = [COUNTRY_COORDS[c][1] for c in train_counts.keys()]
        train_sizes = [train_counts[c] * 15 for c in train_counts.keys()]
        
        ax.scatter(train_lons, train_lats, s=train_sizes, c=COLORS['Train'],
                   alpha=0.7, edgecolors='black', linewidth=1.0, label='Train', zorder=3)
    
    # Plot test samples
    if test_counts:
        test_lats = [COUNTRY_COORDS[c][0] for c in test_counts.keys()]
        test_lons = [COUNTRY_COORDS[c][1] for c in test_counts.keys()]
        test_sizes = [test_counts[c] * 15 for c in test_counts.keys()]
        
        ax.scatter(test_lons, test_lats, s=test_sizes, c=COLORS['Test'],
                   alpha=0.7, edgecolors='black', linewidth=1.0, label='Test', zorder=3)
    
    # Style the map
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude', fontsize=16, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=16, fontweight='bold')
    ax.set_title('Geographic Distribution of Train and Test Samples', 
                 fontsize=20, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=14, loc='lower left', framealpha=0.9)
    ax.set_facecolor('#e6f2ff')  # Light blue background
    
    # Add reference lines (equator and prime meridian)
    ax.axhline(y=0, color='gray', linewidth=1, alpha=0.4, linestyle=':')
    ax.axvline(x=0, color='gray', linewidth=1, alpha=0.4, linestyle=':')
    
    # Add continent labels for context
    continents = [
        ("North America", 45, -100),
        ("South America", -15, -60),
        ("Europe", 50, 10),
        ("Africa", 0, 20),
        ("Asia", 30, 90),
        ("Australia", -25, 135),
    ]
    for name, lat, lon in continents:
        ax.text(lon, lat, name, fontsize=11, alpha=0.4, style='italic',
                ha='center', va='center', color='gray')
    
    # Save figure
    output_dir = Path("paper/figures/data_distribution")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "splits_world_map.png"
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved world map: {output_path}")
    logger.info(f"  Size: {output_path.stat().st_size / 1024:.0f} KB")

if __name__ == '__main__':
    main()
