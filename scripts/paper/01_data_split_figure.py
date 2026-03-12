#!/usr/bin/env python3
"""
Generate data split validation figure (2x2 grid, 3 data panels + world map).

Output: paper/figures/sup_03_data_split_validation.png
"""
import logging
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    fdir = Path("paper/figures/data_distribution")
    output = Path("paper/figures/sup_03_data_split_validation.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load all 4 panels (no labels as per user request)
    a = Image.open(fdir / "distribution_community_type_splits.png")
    b = Image.open(fdir / "distribution_publication_year_splits.png")
    c = Image.open(fdir / "distribution_unitig_size_gb_splits.png")
    d = Image.open(fdir / "splits_world_map.png")
    
    # Resize to same size
    mw = max(a.width, b.width, c.width, d.width)
    mh = max(a.height, b.height, c.height, d.height)
    a = a.resize((mw, mh), Image.Resampling.LANCZOS)
    b = b.resize((mw, mh), Image.Resampling.LANCZOS)
    c = c.resize((mw, mh), Image.Resampling.LANCZOS)
    d = d.resize((mw, mh), Image.Resampling.LANCZOS)
    
    # Create 2x2 grid
    grid = Image.new('RGB', (mw*2, mh*2), (255, 255, 255))
    grid.paste(a, (0, 0))
    grid.paste(b, (mw, 0))
    grid.paste(c, (0, mh))
    grid.paste(d, (mw, mh))
    
    grid.save(output, dpi=(300, 300))
    logger.info(f"✓ Saved: {output} ({grid.width}x{grid.height})")

if __name__ == "__main__":
    main()
