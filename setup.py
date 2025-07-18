from setuptools import setup, find_packages

setup(
    name="filter_lattice",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",  # For visualization
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.0.0',
            'black>=21.0.0',
            'flake8>=3.9.0',
            'mypy>=0.910',
        ],
    },
    author="baset veisi",
    author_email="basetveisy@gmail.com",  # Fixed the double .com
    description="A Python library for converting digital filters to lattice structures",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/baset-veisi/filter_lattice",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
) 