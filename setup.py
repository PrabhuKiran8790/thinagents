from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="thinagents",
    author="Prabhu Kiran Konda",
    description="A lightweight AI Agent framework",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/PrabhuKiran8790/thinagents",
    license="MIT",
    packages=find_packages(exclude=[
        "tests", "tests.*", 
        "examples", "examples.*",
        "scripts", "scripts.*",
        "thinagents.frontend", "thinagents.frontend.*"
    ]),
    package_data={
        "thinagents.web": ["ui/build/**/*"],
    },
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "litellm>=1.70.0",
        "graphviz>=0.20.0"
    ],
    extras_require={
        "web": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="AI LLM Agentic AI AI Agents",
    use_scm_version=True,
    setup_requires=["setuptools-scm"],
)
