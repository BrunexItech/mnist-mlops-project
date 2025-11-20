from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="mnist-mlops",
    version="0.1.0",
    author="Bruno",
    author_email="bruno@example.com",
    description="End-to-end MLOps pipeline for MNIST classification",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
)