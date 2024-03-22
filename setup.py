from setuptools import setup, find_packages

setup(
    name="v1_depth_analysis",
    version="v0.1",
    packages=find_packages(),
    url="https://github.com/znamlab/cottage_analysis",
    license="MIT",
    author="Antonin Blot, Yiran He, Petr Znamenskiy",
    author_email="yiran.he@crick.ac.uk",
    description="Analysis of the V1 depth dataset",
    install_requires=[
        "seaborn",
        "cottage_analysis @ git+ssh://git@github.com/znamlab/cottage_analysis.git@dev",
        "roifile",
    ],
)
