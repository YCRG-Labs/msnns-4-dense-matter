from setuptools import setup, find_packages

setup(
    name="multiscale-neural-network",
    version="0.1.0",
    description="Multi-Scale Neural Networks for Discovering Collective Modes in Dense Matter",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "torchdiffeq>=0.2.3",
        "numpy>=1.24.0",
        "h5py>=3.8.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "tensorboard>=2.12.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "hypothesis>=6.75.0",
        ],
    },
)
