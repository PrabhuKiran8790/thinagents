from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="thinagents",
    version="0.0.4",
    author="Prabhu Kiran Konda",
    description="A lightweight AI Agent framework",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/PrabhuKiran8790/thinagents",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.11",
    install_requires=[
        "litellm>=1.70.0",
        "graphviz>=0.20.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="AI LLM Agentic AI AI Agents",
)
