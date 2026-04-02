from setuptools import setup, find_packages

setup(
    name="prometheus-ebm",
    version="0.1.0",
    author="PROMETHEUS-EBM Team",
    description="Epistemological Benchmark for Metacognition — evaluate how AI models handle epistemic uncertainty",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/prometheus-ebm/prometheus-ebm",
    packages=find_packages(),
    package_data={
        "prometheus_ebm": ["data/*.json"],
    },
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov"],
        "report": ["matplotlib", "jinja2"],
    },
    entry_points={
        "console_scripts": [
            "prometheus-ebm=prometheus_ebm.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
