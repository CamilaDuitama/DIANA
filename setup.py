from setuptools import setup, find_packages

setup(
    name="diana",
    version="0.1.0",
    description="Ancient DNA sample classifier - Multi-task classification trained on the largest aDNA dataset",
    author="Camila Duitama",
    author_email="camiladuitama@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "polars>=0.19.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tensorboard",
        "tqdm",
        "h5py",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort"],
        "optim": ["optuna>=3.0.0"],  # For hyperparameter optimization
    },
    entry_points={
        "console_scripts": [
            "diana-train=diana.cli.train:main",
            "diana-test=diana.cli.test:main",
            "diana-predict=diana.cli.predict:main",
            # Inference pipeline scripts
            "00_extract_reference_kmers.sh=diana.scripts.inference:extract_reference_kmers",
            "01_count_kmers.sh=diana.scripts.inference:count_kmers",
            "02_aggregate_to_unitigs.sh=diana.scripts.inference:aggregate_to_unitigs",
            "03_run_inference.py=diana.scripts.inference:run_inference",
            "04_plot_results.py=diana.scripts.inference:plot_results",
        ],
    },
    # Include shell scripts as data files and make them executable
    scripts=[
        "scripts/inference/00_extract_reference_kmers.sh",
        "scripts/inference/01_count_kmers.sh",
        "scripts/inference/02_aggregate_to_unitigs.sh",
        "scripts/inference/03_run_inference.py",
        "scripts/inference/04_plot_results.py",
    ],
)
