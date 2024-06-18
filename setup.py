from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="synthetic-gymnax",
    version="0.0.1",
    url="https://github.com/keraJLi/synthetic-gymnax",
    author="Jarek Liesen",
    description="Synthetic gymnax environments",
    packages=find_packages(),
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "chex",
        "gymnax",
        "brax",
        "flax",
        "distrax",
        "rejax",
    ],
)
