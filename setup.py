# setup.py
"""
Setup script for Federated Medical QA System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt", 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="federated-medical-qa",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Federated Learning System for Medical Question Answering with Split LLM Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/federated-medical-qa",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "metrics": [
            "rouge-score>=0.1.2",
            "nltk>=3.8.1",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fedmed-train=main:main",
            "fedmed-server=server.federated_server:main",
            "fedmed-client=client.federated_client:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml", "*.yml"],
    },
    keywords=[
        "federated learning",
        "medical AI",
        "privacy preserving",
        "transformer",
        "LLM",
        "healthcare",
        "question answering",
        "deep learning",
        "distributed training"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/federated-medical-qa/issues",
        "Source": "https://github.com/your-username/federated-medical-qa",
        "Documentation": "https://github.com/your-username/federated-medical-qa#readme",
    },
)
