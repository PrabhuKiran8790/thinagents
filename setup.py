from setuptools import setup, find_packages

setup(
    name="thinagents",
    version="0.0.2",
    author="Prabhu Kiran Konda",
    description="A lightweight AI Agent framework",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.11",
    install_requires=[
        "litellm>=1.70.0",
        "graphviz>=0.20.0"
    ],
    setup_requires=[
        "litellm>=1.70.0",
        "graphviz>=0.20.0"
    ],
)
