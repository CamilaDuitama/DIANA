"""Plotting utilities for data analysis and evaluation using Plotly and Polars."""

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from pathlib import Path
from typing import List, Optional, Dict, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)

def save_plot(fig, output_path: Path):
    """Save plotly figure to static file and html."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save HTML
    html_path = output_path.with_suffix('.html')
    try:
        fig.write_html(str(html_path))
        logger.info(f"Saved interactive plot to {html_path}")
    except Exception as e:
        logger.error(f"Failed to save html plot to {html_path}: {e}")

    # Save PNG
    try:
        # Use kaleido for static image export
        fig.write_image(str(output_path), engine="kaleido", scale=2)
        logger.info(f"Saved plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot to {output_path}: {e}")

def plot_class_distribution(df: pl.DataFrame, 
                          target_col: str, 
                          title: str, 
                          output_path: Path):
    """
    Plot distribution for a specific target column.
    Uses Violin/Box plots for numerical data and Bar plots for categorical.
    
    Args:
        df: Polars DataFrame containing the data
        target_col: Column name to plot
        title: Plot title
        output_path: Path to save the plot
    """
    # Check if column is numerical
    is_numeric = df[target_col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]
    
    # If numeric but low cardinality (e.g. < 10 unique values), treat as categorical
    if is_numeric and df[target_col].n_unique() < 10:
        is_numeric = False

    if is_numeric:
        if target_col == "unitig_size_gb":
             fig = px.box(
                 df.to_pandas(),
                 y=target_col,
                 points="all",
                 title=title,
                 template="plotly_white"
             )
             fig.update_traces(boxmean=True)
        else:
            # Violin plot with box
            fig = px.violin(
                df.to_pandas(), # Plotly express works best with pandas or dicts
                y=target_col,
                box=True,
                points="all",
                title=title,
                template="plotly_white"
            )
            fig.update_traces(meanline_visible=True)
        fig.update_layout(yaxis_title=target_col)
    else:
        # Bar plot of counts
        counts = df[target_col].value_counts().sort("count", descending=True)
        
        # Filter logic
        title_suffix = ""
        if target_col in ["BioSample", "Center Name", "BioProject", "ReleaseYear", "geo_loc_name", "project_name"]:
            # User request: "Dont plot the top 20... Plot them ALL."
            pass
        elif len(counts) > 50:
            counts = counts.head(50)
            title_suffix = " (Top 50)"
            
        fig = px.bar(
            counts.to_pandas(),
            x=target_col,
            y="count",
            title=title + title_suffix,
            template="plotly_white",
            text="count"
        )
        fig.update_layout(xaxis_title=target_col, yaxis_title="Count")
        fig.update_traces(textposition='outside')
        
        # User request: "make the image wider for the names to fit. The xaxis fontsize should be smaller and completely vertical"
        if target_col in ["BioSample", "Center Name", "BioProject", "geo_loc_name", "project_name"]:
            fig.update_layout(width=2500, height=600)
            fig.update_xaxes(tickfont=dict(size=7), tickangle=90)
        elif target_col == "ReleaseYear":
            # Year plot should be more compact but still readable
            fig.update_xaxes(tickangle=45)
        
    save_plot(fig, output_path)

def plot_split_distributions(train_df: pl.DataFrame,
                           val_df: pl.DataFrame,
                           test_df: pl.DataFrame,
                           target_cols: List[str],
                           output_dir: Path):
    """
    Plot distributions for multiple targets across splits.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        target_cols: List of columns to plot
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    
    # Add split label
    train_df = train_df.with_columns(pl.lit("Train").alias("Split"))
    val_df = val_df.with_columns(pl.lit("Val").alias("Split"))
    test_df = test_df.with_columns(pl.lit("Test").alias("Split"))
    
    # Combine
    combined_df = pl.concat([train_df, val_df, test_df])
    
    # Convert to pandas once for plotly
    combined_pd = combined_df.to_pandas()
    
    for col in target_cols:
        if col not in combined_df.columns:
            logger.warning(f"Column {col} not found in data")
            continue
            
        # Check if column is numerical
        is_numeric = combined_df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]
        
        if is_numeric and combined_df[col].n_unique() < 10:
            is_numeric = False
            
        if is_numeric:
            # User request: "For boxplot and violin plots always include the outlier points and the indicators of mean and median"
            
            if col == "unitig_size_gb":
                 # User request: "Unitig size plot should be a boxplot"
                 fig = px.box(
                     combined_pd, 
                     y=col, 
                     x="Split", 
                     color="Split", 
                     points="outliers", 
                     title=f"Distribution of {col} across Splits", 
                     template="plotly_white"
                 )
                 fig.update_traces(boxmean=True) # Shows mean as dashed line
            elif col in ["Avg_num_reads", "Avg_read_len"]:
                 # User request: "Avg_num_reads should be a Boxplot or violinplot"
                 # User request: "Avg_read_len should be a boxplot or violinplot"
                 # Using Violin with box inside as it shows distribution density too
                 fig = px.violin(
                     combined_pd, 
                     y=col, 
                     x="Split", 
                     color="Split", 
                     box=True, 
                     points="outliers", 
                     title=f"Distribution of {col} across Splits", 
                     template="plotly_white"
                 )
                 fig.update_traces(meanline_visible=True)
            else:
                 # Default numeric
                 fig = px.violin(
                     combined_pd, 
                     y=col, 
                     x="Split", 
                     color="Split", 
                     box=True, 
                     points="outliers", 
                     title=f"Distribution of {col} across Splits", 
                     template="plotly_white"
                 )
                 fig.update_traces(meanline_visible=True)
        else:
            # Calculate percentages within each split
            # We need to do this manually for a nice grouped bar chart
            
            # Group by Split and Col
            counts = combined_df.group_by(["Split", col]).len().rename({"len": "count"})
            
            # Calculate totals per split
            totals = combined_df.group_by("Split").len().rename({"len": "total"})
            
            # Join and calculate percentage
            counts = counts.join(totals, on="Split")
            counts = counts.with_columns((pl.col("count") / pl.col("total") * 100).alias("percentage"))
            
            # Sort for better visualization
            counts = counts.sort(["Split", "percentage"], descending=[False, True])
            
            # Round percentage for display
            counts = counts.with_columns(pl.col("percentage").round(1).alias("percentage_text"))
            
            # Filter logic
            title_suffix = ""
            if col in ["BioSample", "Center Name", "BioProject", "ReleaseYear", "geo_loc_name", "project_name"]:
                # User request: "Dont plot the top 20... Plot them ALL."
                pass
            elif combined_df[col].n_unique() > 20:
                # Get top 20 categories overall
                top_cats = combined_df[col].value_counts().sort("count", descending=True).head(20)[col]
                counts = counts.filter(pl.col(col).is_in(top_cats))
                title_suffix = " (Top 20)"

            fig = px.bar(
                counts.to_pandas(),
                x=col,
                y="percentage",
                color="Split",
                barmode="group",
                title=f"Distribution of {col} across Splits{title_suffix}",
                template="plotly_white",
                text="percentage_text"
            )
            fig.update_layout(yaxis_title="Percentage (%)")
            
            # User request: "make the image wider for the names to fit. The xaxis fontsize should be smaller and completely vertical"
            if col in ["BioSample", "Center Name", "BioProject", "geo_loc_name", "project_name"]:
                fig.update_layout(width=2500, height=600)
                fig.update_xaxes(tickfont=dict(size=7), tickangle=90)
            elif col == "ReleaseYear":
                # Year plot should be more compact but still readable
                fig.update_xaxes(tickangle=45)
                fig.update_xaxes(tickfont=dict(size=8))
            
        save_plot(fig, output_dir / f"distribution_{col}_splits.png")
