#!/usr/bin/env python3
"""
Generate world map PNG using matplotlib with Natural Earth coastlines.
Simple approach with no external GIS dependencies.

Output: paper/figures/data_distribution/splits_world_map.png
"""
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from collections import Counter
import urllib.request
import json

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

def download_world_geojson():
    """Download simplified world coastlines from Natural Earth."""
    cache_file = Path("paper/figures/.world_coastlines.json")
    
    if cache_file.exists():
        logger.info("Using cached world coastlines")
        with open(cache_file) as f:
            return json.load(f)
    
    logger.info("Downloading world coastlines from Natural Earth...")
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
        
        # Cache for future use
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        
        logger.info("✓ Downloaded and cached world coastlines")
        return data
    except Exception as e:
        logger.warning(f"Failed to download coastlines: {e}")
        return None

def plot_world_borders(ax, geojson_data):
    """Plot country borders from GeoJSON data."""
    if not geojson_data:
        return
    
    for feature in geojson_data.get('features', []):
        geom = feature.get('geometry', {})
        geom_type = geom.get('type')
        coords = geom.get('coordinates', [])
        
        if geom_type == 'Polygon':
            for polygon in coords:
                xs = [pt[0] for pt in polygon]
                ys = [pt[1] for pt in polygon]
                ax.plot(xs, ys, color='#666666', linewidth=0.5, alpha=0.6, zorder=1)
        
        elif geom_type == 'MultiPolygon':
            for multi_polygon in coords:
                for polygon in multi_polygon:
                    xs = [pt[0] for pt in polygon]
                    ys = [pt[1] for pt in polygon]
                    ax.plot(xs, ys, color='#666666', linewidth=0.5, alpha=0.6, zorder=1)

def main():
    logger.info("Loading train/test metadata...")
    
    # Download world borders
    world_data = download_world_geojson()
    
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
    
    # Log sample distribution to verify size variation
    if train_counts:
        train_min, train_max = min(train_counts.values()), max(train_counts.values())
        logger.info(f"Train samples per country: {train_min} to {train_max}")
    if test_counts:
        test_min, test_max = min(test_counts.values()), max(test_counts.values())
        logger.info(f"Test samples per country: {test_min} to {test_max}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Set background and limits first
    ax.set_facecolor('#d4e6f1')  # Light blue ocean
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    
    # Plot world borders
    plot_world_borders(ax, world_data)
    
    # Plot train samples (marker size proportional to sample count)
    if train_counts:
        train_lats = [COUNTRY_COORDS[c][0] for c in train_counts.keys()]
        train_lons = [COUNTRY_COORDS[c][1] for c in train_counts.keys()]
        # Scale markers: smaller multiplier (20) to avoid overlap
        train_sizes = [train_counts[c] * 20 for c in train_counts.keys()]
        
        ax.scatter(train_lons, train_lats, s=train_sizes, c=COLORS['Train'],
                   alpha=0.7, edgecolors='black', linewidth=0.8, label='Train', zorder=3)
    
    # Plot test samples (marker size proportional to sample count)
    if test_counts:
        test_lats = [COUNTRY_COORDS[c][0] for c in test_counts.keys()]
        test_lons = [COUNTRY_COORDS[c][1] for c in test_counts.keys()]
        # Scale markers: smaller multiplier (20) to avoid overlap
        test_sizes = [test_counts[c] * 20 for c in test_counts.keys()]
        
        ax.scatter(test_lons, test_lats, s=test_sizes, c=COLORS['Test'],
                   alpha=0.7, edgecolors='black', linewidth=0.8, label='Test', zorder=3)
    
    # Style the map
    ax.set_xlabel('Longitude', fontsize=16, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=16, fontweight='bold')
    ax.set_title('Geographic Distribution of Train and Test Samples', 
                 fontsize=20, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, zorder=2)
    ax.legend(fontsize=16, loc='upper left', framealpha=0.95, 
              edgecolor='black', fancybox=False, shadow=False, markerscale=0.4)
    
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
