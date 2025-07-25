from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="yajph",
    version="0.2.0",  # Bumped version for epistemic features
    author="Your Name",
    author_email="your.email@example.com",
    description="The Anti-Black-Box Engine - GitHub's First Explainable Decision Framework with Epistemic Pull Requests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Maykon-fernanado/YAJPH",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",  # Upgraded from Alpha
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "PyYAML>=6.0",
        "dataclasses-json>=0.5.0",  # For better dataclass serialization
        "click>=8.0.0",  # For enhanced CLI
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "isort>=5.0",
            "pre-commit>=2.0",
        ],
        "ai": [
            "openai>=1.0.0",
            "anthropic>=0.3.0",
            # Add other AI service clients as needed
        ],
        "full": [
            "pytest>=7.0",
            "black>=22.0",
            "isort>=5.0",
            "pre-commit>=2.0",
            "openai>=1.0.0",
            "anthropic>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "yajph=yajph:main",
            "yajph-setup=yajph.setup:main",  # New setup command
        ],
    },
    keywords="ai explainable decisions yaml json transparency audit epistemic code-review git-hooks ci-cd",
    project_urls={
        "Bug Reports": "https://github.com/Maykon-fernanado/YAJPH/issues",
        "Source": "https://github.com/Maykon-fernanado/YAJPH",
        "Documentation": "https://yajph.readthedocs.io/",
        "Changelog": "https://github.com/Maykon-fernanado/YAJPH/blob/main/CHANGELOG.md",
    },
    include_package_data=True,
    package_data={
        "yajph": [
            "templates/*.yaml",
            "templates/*.yml", 
            "templates/*.md",
            "templates/*.sh",
        ],
    },
)
