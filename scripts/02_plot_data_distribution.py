#!/usr/bin/env python3
"""Plot data distributions for full dataset and splits using Plotly and Polars."""

import sys
import argparse
import logging
import polars as pl
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diana.data.loader import MetadataLoader
from diana.data.splitter import StratifiedSplitter
from diana.evaluation.plotting import plot_class_distribution, plot_split_distributions
from diana.utils.config import load_config

import plotly.express as px

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_country_from_location(location: str) -> str:
    """
    Extract country name from geo_loc_name.
    Handles formats like "USA:California" or "Germany" or "United Kingdom:England"
    """
    if not location or location == "Unavailable data":
        return "Unknown"
    
    # Split by colon and take the first part (country)
    parts = location.split(":")
    country = parts[0].strip()
    
    # Normalize some common country names
    country_mapping = {
        "USA": "United States",
        "UK": "United Kingdom",
        "Czech Republic": "Czechia",
    }
    
    return country_mapping.get(country, country)

def main():
    parser = argparse.ArgumentParser(description='Plot data distributions')
    parser.add_argument('--config', default='configs/data_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.config))
    
    # Load metadata
    logger.info("Loading metadata...")
    metadata_path = Path(config["metadata_path"])
    if not metadata_path.exists():
        # Try relative to workspace root
        metadata_path = Path.cwd() / config["metadata_path"]
        
    metadata_loader = MetadataLoader(metadata_path)
    df = metadata_loader.load()
    
    # Parse ReleaseDate to extract year
    if "ReleaseDate" in df.columns:
        try:
            # Try to parse as date and extract year, or extract first 4 characters as year
            df = df.with_columns(
                pl.col("ReleaseDate").str.slice(0, 4).alias("ReleaseYear")
            )
            logger.info("Extracted ReleaseYear from ReleaseDate")
        except Exception as e:
            logger.warning(f"Could not extract year from ReleaseDate: {e}")
    
    # Cast numeric columns explicitly
    numeric_cols = [
        "Avg_num_reads", "Avg_read_len", "unitig_size_gb",
        "total_runs_for_sample", "available_runs_for_sample", "unavailable_runs_for_sample",
        "seqstats_contigs_n50", "seqstats_contigs_nbseq", "seqstats_contigs_maxlen", "seqstats_contigs_sumlen",
        "seqstats_unitigs_n50", "seqstats_unitigs_maxlen", "seqstats_unitigs_sumlen",
        "size_contigs_after_compression", "size_contigs_before_compression",
        "size_unitigs_before_compression", "size_unitigs_after_compression"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            try:
                # Cast to Float64, turning errors into nulls
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
            except Exception as e:
                logger.warning(f"Could not cast {col} to float: {e}")

    # Replace NaN/Null with "Unavailable data" for categorical columns
    for col in df.columns:
        if col not in numeric_cols:
            try:
                df = df.with_columns(pl.col(col).fill_null("Unavailable data"))
            except Exception as e:
                logger.warning(f"Could not fill nulls in {col}: {e}")

    logger.info(f"Total samples: {len(df)}")
    
    # Output directory for figures
    figures_dir = Path("paper/figures/data_distribution")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot full dataset distributions
    logger.info("Plotting full dataset distributions...")
    
    # Identify columns to plot
    # Plot ALL columns except Run_accession, archive_accession, and original ReleaseDate (use ReleaseYear instead)
    columns_to_plot = [col for col in df.columns if col not in ["Run_accession", "archive_accession", "ReleaseDate"]]
            
    logger.info(f"Plotting distributions for {len(columns_to_plot)} columns...")
    
    for col in columns_to_plot:
        try:
            plot_class_distribution(
                df, 
                col, 
                f"Distribution of {col} (Full Dataset)", 
                figures_dir / f"full_distribution_{col}.png"
            )
        except Exception as e:
            logger.warning(f"Could not plot {col}: {e}")
    
    # Create geographic world map
    logger.info("Creating world map for geographic distribution...")
    if "geo_loc_name" in df.columns:
        # For full dataset: aggregate by location
        location_counts = df.group_by("geo_loc_name").len().rename({"len": "count"})
        location_counts = location_counts.filter(pl.col("geo_loc_name") != "Unavailable data")
        location_counts = location_counts.sort("count", descending=True)
        
        logger.info(f"Samples from {len(location_counts)} unique locations")
        
        # Extract country for location matching
        location_counts = location_counts.with_columns(
            pl.col("geo_loc_name").map_elements(
                extract_country_from_location, 
                return_dtype=pl.Utf8
            ).alias("country")
        )
        
        # Create scatter geo map
        fig = px.scatter_geo(
            location_counts.to_pandas(),
            locations="country",
            locationmode="country names",
            size="count",
            hover_name="geo_loc_name",
            hover_data={"count": True, "geo_loc_name": False, "country": False},
            color_discrete_sequence=["#636EFA"],
            title="Geographic Distribution of Samples (Full Dataset)",
            labels={"count": "Number of Samples"},
            size_max=50
        )
        
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth',
                showcountries=True,
                countrycolor="lightgray"
            ),
            height=600,
            width=1400
        )
        
        # Save
        output_path = figures_dir / "full_dataset_world_map.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved interactive world map to {output_path}")
        
        try:
            fig.write_image(str(figures_dir / "full_dataset_world_map.png"))
            logger.info(f"Saved static world map to {figures_dir / 'full_dataset_world_map.png'}")
        except Exception as e:
            logger.warning(f"Could not save world map PNG: {e}")
            
    # Load splits
    splits_dir = Path(config["splits_dir"])
    if not (splits_dir / "train_ids.txt").exists():
        logger.warning(f"Splits not found in {splits_dir}. Run 01_create_splits.py first.")
        return
        
    logger.info("Loading splits...")
    train_ids, val_ids, test_ids = StratifiedSplitter.load_splits(splits_dir)
    
    # Subset data
    id_col = "Run_accession" 
    
    train_df = df.filter(pl.col(id_col).is_in(train_ids))
    val_df = df.filter(pl.col(id_col).is_in(val_ids))
    test_df = df.filter(pl.col(id_col).is_in(test_ids))
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create geographic world map comparison for splits
    logger.info("Creating world map comparison for splits...")
    if "geo_loc_name" in df.columns:
        # Add split label
        df_with_split = df.with_columns(
            pl.when(pl.col(id_col).is_in(train_ids))
            .then(pl.lit("Train"))
            .when(pl.col(id_col).is_in(test_ids))
            .then(pl.lit("Test"))
            .otherwise(pl.lit("Val"))
            .alias("split")
        )
        
        # Count per location per split
        split_location_counts = df_with_split.group_by(["geo_loc_name", "split"]).len().rename({"len": "count"})
        split_location_counts = split_location_counts.filter(pl.col("geo_loc_name") != "Unavailable data")
        split_location_counts = split_location_counts.sort("count", descending=True)
        
        # Extract country for location matching
        split_location_counts = split_location_counts.with_columns(
            pl.col("geo_loc_name").map_elements(
                extract_country_from_location, 
                return_dtype=pl.Utf8
            ).alias("country")
        )
        
        # Create scatter geo map with color by split
        fig = px.scatter_geo(
            split_location_counts.to_pandas(),
            locations="country",
            locationmode="country names",
            size="count",
            color="split",
            hover_name="geo_loc_name",
            hover_data={"count": True, "geo_loc_name": False, "country": False, "split": True},
            color_discrete_map={"Train": "#636EFA", "Test": "#EF553B", "Val": "#00CC96"},
            title="Geographic Distribution of Samples (Train vs Test)",
            labels={"count": "Number of Samples"},
            size_max=50
        )
        
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth',
                showcountries=True,
                countrycolor="lightgray"
            ),
            height=600,
            width=1400
        )
        
        # Save
        output_path = figures_dir / "splits_world_map.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved interactive split comparison world map to {output_path}")
        
        try:
            fig.write_image(str(figures_dir / "splits_world_map.png"))
            logger.info(f"Saved static split comparison world map to {figures_dir / 'splits_world_map.png'}")
        except Exception as e:
            logger.warning(f"Could not save split world map PNG: {e}")
    
    # Plot split comparisons for ALL columns
    logger.info("Plotting split comparisons...")
    plot_split_distributions(
        train_df,
        val_df,
        test_df,
        columns_to_plot,
        figures_dir
    )
    
    logger.info(f"All plots saved to {figures_dir}")

if __name__ == "__main__":
    main()
