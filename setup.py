from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="rate-eval",
    version="2.0.0",
    author="Yala Lab @ UC Berkeley and UCSF",
    author_email="yala@berkeley.edu",
    description="A comprehensive evaluation pipeline for Vision-Language Models on medical imaging tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/rate_eval",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="medical-imaging, vision-language-models, evaluation, deep-learning, radiology",
    entry_points={
        "console_scripts": [
            "rate-extract=rate_eval.cli.extract:main",
            "rate-evaluate=rate_eval.cli.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rate_eval": ["configs/*.yaml", "configs/**/*.yaml"],
    },
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
        ],
    },
    zip_safe=False,
)
