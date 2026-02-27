from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README_PATH = ROOT / "README.md"


setup(
    name="neurophorm",
    version="1.0.0",
    author="Mohaddeseh Mozaffari",
    author_email="mohaddeseh.mozaffarii@gmail.com",
    description="A Python package for topological brain network analysis using persistent homology",
    long_description=README_PATH.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/MohiMozaffari/NeuroPHorm",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23.5",
        "pandas>=1.5.3",
        "scipy>=1.10.1",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "scikit-learn>=1.2.2",
        "giotto-tda>=0.3.1",
        "pillow>=9.3.0",
        "statsmodels>=0.14.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
