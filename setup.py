from setuptools import setup, find_packages

with open("README.md", "r") as f:
    readme = f.read()

requirements = [
    "numpy>=1.26.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "scipy>=1.11.0",
    "matplotlib>=3.7.0",
    "nibabel>=5.0.0",
    "joblib>=1.2.0",
]

setup(
    name="brainsmash",
    version="0.12.0",
    author="Joshua Burt",
    author_email="joshua.burt@yale.edu",
    include_package_data=True,
    description="Brain Surrogate Maps with Autocorrelated Spatial Heterogeneity.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/murraylab/brainsmash",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.10',
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
