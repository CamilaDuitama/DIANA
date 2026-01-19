#!/usr/bin/env python3
"""
Generate main paper figures for DIANA multi-task classifier.

Creates 4 main figures:
1. Multi-task performance comparison (Accuracy + Balanced Accuracy)
2. 2x2 grid of confusion matrices (all 4 tasks)
3. 2x2 grid of ROC curves (all 4 tasks)
4. 2x2 grid of Precision-Recall curves (all 4 tasks)

Also creates validation resource statistics table.
"""

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import subprocess

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
FIGURES_DIR = BASE_DIR / 'paper' / 'figures'
TABLES_DIR = BASE_DIR / 'paper' / 'tables'
VALIDATION_DIR = TABLES_DIR / 'validation'

# Color palette - using colorblind-friendly palette (Okabe-Ito)
import plotly.express as px
# Okabe-Ito colorblind-safe palette
COLORS = {
    'train': '#0173B2',      # Blue
    'test': '#DE8F05',       # Orange  
    'validation': '#029E73', # Green
    'accuracy': '#CC78BC',   # Purple
    'balanced_acc': '#CA9161', # Tan
    'seen': '#56B4E9',       # Sky blue
    'all': '#ECE133',        # Yellow
}

# Task display names
TASK_NAMES = {
    'sample_type': 'Sample Type',
    'community_type': 'Community Type',
    'sample_host': 'Sample Host',
    'material': 'Material'
}


def hex_to_rgba(hex_color, opacity=0.5):
    """Convert hex color to rgba string with opacity."""
    # Remove # if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    return f'rgba({r},{g},{b},{opacity})'


def load_performance_data():
    """Load performance metrics from final table."""
    df = pd.read_csv(TABLES_DIR / 'final_model_performance.csv')
    
    # Restructure data - the CSV has Train_*, Test_*, Val_* columns
    data = {}
    for _, row in df.iterrows():
        task = row['Task'].lower().replace(' (ancient/modern)', '').replace(' (species)', '').replace(' ', '_')
        
        data[task] = {
            'train': {
                'accuracy': row['Train_Acc'],
                'balanced_acc': row['Train_Bal_Acc']
            },
            'test': {
                'accuracy': row['Test_Acc'],
                'balanced_acc': row['Test_Bal_Acc']
            },
            'validation': {
                'accuracy': row['Val_Acc'],
                'balanced_acc': row['Val_Bal_Acc']
            }
        }
    
    return data


def create_figure1_performance_bars():
    """Create Figure 1: Multi-task performance comparison."""
    print("Creating Figure 1: Multi-task performance comparison...")
    
    data = load_performance_data()
    
    # Create subplots - 2 metrics
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Accuracy', 'Balanced Accuracy'),
        horizontal_spacing=0.12
    )
    
    tasks = ['sample_type', 'community_type', 'sample_host', 'material']
    task_labels = [TASK_NAMES[t] for t in tasks]
    datasets = ['train', 'test', 'validation']
    dataset_labels = ['Training', 'Test', 'Validation (seen)']
    
    # Accuracy subplot
    for i, (dataset, label) in enumerate(zip(datasets, dataset_labels)):
        values = [data[task][dataset]['accuracy'] if dataset in data[task] else None 
                  for task in tasks]
        
        fig.add_trace(
            go.Bar(
                name=label,
                x=task_labels,
                y=values,
                marker=dict(
                    color=hex_to_rgba(COLORS[dataset], 0.5),  # Semi-transparent fill
                    line=dict(color=COLORS[dataset], width=2)  # Solid border
                ),
                showlegend=True,
                legendgroup=dataset,
            ),
            row=1, col=1
        )
    
    # Balanced Accuracy subplot
    for i, (dataset, label) in enumerate(zip(datasets, dataset_labels)):
        values = [data[task][dataset]['balanced_acc'] if dataset in data[task] else None 
                  for task in tasks]
        
        fig.add_trace(
            go.Bar(
                name=label,
                x=task_labels,
                y=values,
                marker=dict(
                    color=hex_to_rgba(COLORS[dataset], 0.5),  # Semi-transparent fill
                    line=dict(color=COLORS[dataset], width=2)  # Solid border
                ),
                showlegend=False,
                legendgroup=dataset,
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    fig.update_yaxes(title_text='Performance (%)', range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text='Performance (%)', range=[0, 105], row=1, col=2)
    
    fig.update_layout(
        title_text='Multi-Task Model Performance Across Datasets',
        height=600,
        width=1400,
        barmode='group',
        font=dict(size=14),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save
    output_path = FIGURES_DIR / 'figure1_multitask_performance.png'
    fig.write_image(str(output_path), width=1400, height=600, scale=2)
    fig.write_html(str(output_path.with_suffix('.html')))
    print(f"  ✓ Saved to {output_path}")
    
    return fig


def load_confusion_matrix(task):
    """Load confusion matrix data from validation comparison JSON."""
    with open(VALIDATION_DIR / 'validation_comparison.json', 'r') as f:
        data = json.load(f)
    
    task_data = data.get(task, {})
    confusion = task_data.get('all', {}).get('confusion_matrix', {})
    accuracy = task_data.get('all', {}).get('accuracy', 0) * 100
    
    # Parse confusion matrix
    labels = set()
    for key in confusion.keys():
        true_label, pred_label = key.split('_to_')
        labels.add(true_label)
        labels.add(pred_label)
    
    labels = sorted(labels)
    
    # Build matrix
    matrix = np.zeros((len(labels), len(labels)))
    for key, value in confusion.items():
        true_label, pred_label = key.split('_to_')
        i = labels.index(true_label)
        j = labels.index(pred_label)
        matrix[i, j] = value
    
    return matrix, labels, accuracy


def create_figure2_confusion_matrices():
    """Create Figure 2: 2x2 grid of confusion matrices."""
    print("Creating Figure 2: Confusion matrices (2x2 grid)...")
    
    tasks = ['sample_type', 'community_type', 'sample_host', 'material']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'({chr(65+i)}) {TASK_NAMES[t]}' for i, t in enumerate(tasks)],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
               [{'type': 'heatmap'}, {'type': 'heatmap'}]]
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for task, (row, col) in zip(tasks, positions):
        matrix, labels, accuracy = load_confusion_matrix(task)
        
        # Truncate long labels
        display_labels = [l[:20] + '...' if len(l) > 20 else l for l in labels]
        
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=display_labels,
                y=display_labels,
                colorscale='Blues',
                showscale=(col == 2),  # Only show colorbar on right side
                text=matrix.astype(int),
                texttemplate='%{text}',
                textfont=dict(size=10 if len(labels) <= 6 else 8),
                hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>',
            ),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text='Predicted', tickangle=45, row=row, col=col)
        fig.update_yaxes(title_text='True', row=row, col=col)
    
    fig.update_layout(
        title_text='Confusion Matrices for All Tasks (Validation Set)',
        height=1400,
        width=1600,
        font=dict(size=12),
    )
    
    # Save
    output_path = FIGURES_DIR / 'figure2_confusion_matrices.png'
    fig.write_image(str(output_path), width=1600, height=1400, scale=2)
    fig.write_html(str(output_path.with_suffix('.html')))
    print(f"  ✓ Saved to {output_path}")
    
    return fig


def load_roc_data(task):
    """Load ROC curve data from validation metrics."""
    # Try to load from validation metrics if available
    # For now, we'll note that this data should come from the validation script
    # The validation script already generates ROC curves, so we'll use placeholder
    # In practice, you'd save the ROC data during validation
    
    return {
        'note': f'ROC data for {task} - using existing generated figures',
        'path': FIGURES_DIR / 'validation' / f'roc_curves_{task}.png'
    }


def create_figure3_roc_curves():
    """Create Figure 3: ROC curves for all tasks (2x2 grid layout)."""
    print("Creating Figure 3: ROC curves for all tasks...")
    
    from PIL import Image, ImageDraw, ImageFont
    
    # All tasks now have ROC curves (sample_type too!)
    tasks = ['sample_type', 'community_type', 'sample_host', 'material']
    task_labels = ['(A) Sample Type', '(B) Community Type', '(C) Sample Host', '(D) Material']
    
    # Load existing ROC curve images
    images = []
    for task in tasks:
        img_path = FIGURES_DIR / 'validation' / f'roc_curves_{task}.png'
        if img_path.exists():
            images.append(Image.open(img_path))
        else:
            print(f"  ⚠ Warning: ROC curve not found for {task}")
            # Create placeholder
            placeholder = Image.new('RGB', (800, 600), (255, 255, 255))
            draw = ImageDraw.Draw(placeholder)
            text = f"ROC curve not available\nfor {TASK_NAMES[task]}"
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font = None
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (800 - text_width) / 2
            y = (600 - text_height) / 2
            draw.text((x, y), text, fill=(100, 100, 100), font=font, align='center')
            images.append(placeholder)
    
    if len(images) == 4:
        # Create 2x2 grid
        widths, heights = zip(*(i.size for i in images))
        max_width = max(widths)
        max_height = max(heights)
        
        # Create 2x2 grid with padding
        padding = 30
        total_width = max_width * 2 + padding * 3
        total_height = max_height * 2 + padding * 3
        
        composite = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        
        # Position images in 2x2 grid
        positions = [
            (padding, padding),  # top-left (sample_type)
            (max_width + padding * 2, padding),  # top-right (community_type)
            (padding, max_height + padding * 2),  # bottom-left (sample_host)
            (max_width + padding * 2, max_height + padding * 2)  # bottom-right (material)
        ]
        
        for img, pos in zip(images, positions):
            composite.paste(img, pos)
        
        output_path = FIGURES_DIR / 'figure3_roc_curves_combined.png'
        composite.save(output_path, dpi=(300, 300))
        print(f"  ✓ Saved to {output_path}")
    else:
        print(f"  ⚠ Could not create ROC grid ({len(images)}/4 images). Run validation first.")
    
    return None


def create_figure4_pr_curves():
    """Create Figure 4: Precision-Recall curves for all tasks (2x2 grid layout)."""
    print("Creating Figure 4: Precision-Recall curves for all tasks...")
    
    from PIL import Image, ImageDraw, ImageFont
    
    # All tasks have PR curves
    tasks = ['sample_type', 'community_type', 'sample_host', 'material']
    task_labels = ['(A) Sample Type', '(B) Community Type', '(C) Sample Host', '(D) Material']
    
    # Load existing PR curve images
    images = []
    for task in tasks:
        img_path = FIGURES_DIR / 'validation' / f'pr_curves_{task}.png'
        if img_path.exists():
            images.append(Image.open(img_path))
        else:
            print(f"  ⚠ Warning: PR curve not found for {task}")
            # Create placeholder
            placeholder = Image.new('RGB', (800, 600), (255, 255, 255))
            draw = ImageDraw.Draw(placeholder)
            text = f"PR curve not available\nfor {TASK_NAMES[task]}"
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font = None
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (800 - text_width) / 2
            y = (600 - text_height) / 2
            draw.text((x, y), text, fill=(100, 100, 100), font=font, align='center')
            images.append(placeholder)
    
    if len(images) == 4:
        # Create 2x2 grid
        widths, heights = zip(*(i.size for i in images))
        max_width = max(widths)
        max_height = max(heights)
        
        # Create 2x2 grid with padding
        padding = 30
        total_width = max_width * 2 + padding * 3
        total_height = max_height * 2 + padding * 3
        
        composite = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        
        # Position images in 2x2 grid
        positions = [
            (padding, padding),  # top-left (sample_type)
            (max_width + padding * 2, padding),  # top-right (community_type)
            (padding, max_height + padding * 2),  # bottom-left (sample_host)
            (max_width + padding * 2, max_height + padding * 2)  # bottom-right (material)
        ]
        
        for img, pos in zip(images, positions):
            composite.paste(img, pos)
        
        output_path = FIGURES_DIR / 'figure4_pr_curves_combined.png'
        composite.save(output_path, dpi=(300, 300))
        print(f"  ✓ Saved to {output_path}")
    else:
        print(f"  ⚠ Could not create PR grid ({len(images)}/4 images). Run validation first.")
    
    return None


def create_validation_resource_table():
    """Create LaTeX table with resource statistics for successful validation samples."""
    print("Creating validation resource statistics table...")
    
    # Directories
    predictions_dir = BASE_DIR / 'results' / 'validation_predictions'
    metadata_file = BASE_DIR / 'paper' / 'metadata' / 'validation_metadata.tsv'
    
    # Load metadata to get file sizes
    metadata = pd.read_csv(metadata_file, sep='\t')
    # Convert fastq_bytes to numeric (handle strings like "123;456")
    metadata['fastq_bytes'] = pd.to_numeric(
        metadata['fastq_bytes'].astype(str).str.split(';').str[0],
        errors='coerce'
    )
    metadata['file_size_gb'] = metadata['fastq_bytes'] / (1024**3)  # Convert to GB
    
    # Collect resource statistics from successful predictions
    data = []
    
    for sample_dir in sorted(predictions_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        
        jobinfo_file = sample_dir / '.jobinfo'
        if not jobinfo_file.exists():
            continue
        
        # Read jobinfo
        with open(jobinfo_file) as f:
            jobinfo = json.load(f)
        
        # Only include successful predictions
        if jobinfo.get('status') != 'SUCCESS':
            continue
        
        run_accession = jobinfo.get('run_accession', sample_dir.name)
        job_id = jobinfo.get('job_id', 'N/A')
        task_id = jobinfo.get('task_id', 'N/A')
        memory_mb = jobinfo.get('memory_mb', 0)
        cpus = jobinfo.get('cpus', 0)
        elapsed_sec = jobinfo.get('elapsed_seconds', 0)
        node = jobinfo.get('node', 'unknown')
        
        # Get file size from metadata
        sample_meta = metadata[metadata['run_accession'] == run_accession]
        if len(sample_meta) > 0:
            file_size_gb = sample_meta['file_size_gb'].values[0]
        else:
            file_size_gb = 0
        
        # Get efficiency from SLURM if job_id is available
        mem_efficiency = None
        cpu_efficiency = None
        
        if job_id != 'N/A' and task_id != 'N/A':
            try:
                # Query sacct for efficiency data
                cmd = [
                    'sacct', '-j', f'{job_id}.{task_id}',
                    '--format=JobID,MaxRSS,TotalCPU,CPUTime,Elapsed',
                    '--parsable2', '--noheader'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    # Get the first line (main job step)
                    if lines:
                        fields = lines[0].split('|')
                        if len(fields) >= 5:
                            max_rss = fields[1]  # e.g., "1234567K"
                            total_cpu = fields[2]  # e.g., "00:05:30"
                            cpu_time = fields[3]  # e.g., "00:30:00"
                            
                            # Parse MaxRSS (remove K suffix and convert to MB)
                            if max_rss.endswith('K'):
                                max_rss_mb = float(max_rss[:-1]) / 1024
                            elif max_rss.endswith('M'):
                                max_rss_mb = float(max_rss[:-1])
                            elif max_rss.endswith('G'):
                                max_rss_mb = float(max_rss[:-1]) * 1024
                            else:
                                max_rss_mb = 0
                            
                            # Calculate memory efficiency
                            if memory_mb > 0 and max_rss_mb > 0:
                                mem_efficiency = (max_rss_mb / memory_mb) * 100
                            
                            # Parse CPU times (HH:MM:SS format)
                            def parse_time(time_str):
                                if not time_str or time_str == '':
                                    return 0
                                parts = time_str.split(':')
                                if len(parts) == 3:
                                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                                return 0
                            
                            total_cpu_sec = parse_time(total_cpu)
                            cpu_time_sec = parse_time(cpu_time)
                            
                            # Calculate CPU efficiency
                            if cpu_time_sec > 0:
                                cpu_efficiency = (total_cpu_sec / cpu_time_sec) * 100
            except Exception as e:
                # If sacct fails, leave efficiency as None
                pass
        
        data.append({
            'run_accession': run_accession,
            'memory_gb': memory_mb / 1024,
            'cpus': cpus,
            'elapsed_min': elapsed_sec / 60,
            'mem_efficiency': mem_efficiency,
            'cpu_efficiency': cpu_efficiency,
            'file_size_gb': file_size_gb,
            'node': node
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("  ⚠ No successful predictions found")
        return
    
    # Sort by memory allocation
    df = df.sort_values('memory_gb')
    
    # Calculate summary statistics
    print(f"  Total samples: {len(df)}")
    print(f"  Memory range: {df['memory_gb'].min():.1f} - {df['memory_gb'].max():.1f} GB")
    print(f"  File size range: {df['file_size_gb'].min():.2f} - {df['file_size_gb'].max():.2f} GB")
    
    # Create LaTeX table
    VALIDATION_DIR.mkdir(exist_ok=True, parents=True)
    output_file = VALIDATION_DIR / 'resource_statistics.tex'
    
    with open(output_file, 'w') as f:
        # Write table header
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Resource usage statistics for validation predictions}\n")
        f.write("\\label{tab:validation_resources}\n")
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Sample & Memory & CPUs & Mem Eff. & CPU Eff. & File Size \\\\\n")
        f.write("       & (GB)   &      & (\\%)     & (\\%)     & (GB) \\\\\n")
        f.write("\\midrule\n")
        
        # Write table rows (first 50 samples)
        for i, row in df.head(50).iterrows():
            mem_eff_str = f"{row['mem_efficiency']:.1f}" if pd.notna(row['mem_efficiency']) else "---"
            cpu_eff_str = f"{row['cpu_efficiency']:.1f}" if pd.notna(row['cpu_efficiency']) else "---"
            
            f.write(f"{row['run_accession']} & "
                   f"{row['memory_gb']:.0f} & "
                   f"{row['cpus']:.0f} & "
                   f"{mem_eff_str} & "
                   f"{cpu_eff_str} & "
                   f"{row['file_size_gb']:.2f} \\\\\n")
        
        if len(df) > 50:
            f.write("\\midrule\n")
            f.write(f"\\multicolumn{{6}}{{c}}{{... {len(df) - 50} more samples ...}} \\\\\n")
        
        f.write("\\midrule\n")
        
        # Summary statistics
        f.write(f"\\textbf{{Mean}} & "
               f"{df['memory_gb'].mean():.1f} & "
               f"{df['cpus'].mean():.1f} & "
               f"{df['mem_efficiency'].mean():.1f} & "
               f"{df['cpu_efficiency'].mean():.1f} & "
               f"{df['file_size_gb'].mean():.2f} \\\\\n")
        
        f.write(f"\\textbf{{Median}} & "
               f"{df['memory_gb'].median():.1f} & "
               f"{df['cpus'].median():.1f} & "
               f"{df['mem_efficiency'].median():.1f} & "
               f"{df['cpu_efficiency'].median():.1f} & "
               f"{df['file_size_gb'].median():.2f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"  ✓ Saved to {output_file}")
    
    # Also save as CSV for reference
    csv_file = VALIDATION_DIR / 'resource_statistics.csv'
    df.to_csv(csv_file, index=False)
    print(f"  ✓ CSV saved to {csv_file}")
    
    return df


def main():
    """Generate all paper figures."""
    print("="*80)
    print("DIANA Paper Figures Generation")
    print("="*80)
    print()
    
    # Ensure output directory exists
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    
    # Generate figures
    try:
        create_figure1_performance_bars()
        print()
        create_figure2_confusion_matrices()
        print()
        create_figure3_roc_curves()
        print()
        create_figure4_pr_curves()
        print()
        create_validation_resource_table()
        print()
        print("="*80)
        print("✅ All figures and tables generated successfully!")
        print(f"Output directory: {FIGURES_DIR}")
        print(f"Tables directory: {VALIDATION_DIR}")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
