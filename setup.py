from setuptools import setup, find_packages

setup(
    name="v1_depth_map",
    version="v1.0",
    packages=find_packages(),
    url="https://github.com/znamlab/cottage_analysis",
    license="MIT",
    author="Antonin Blot, Yiran He, Petr Znamenskiy",
    author_email="yiran.he@crick.ac.uk",
    description="Analysis for V1 depth map project",
    install_requires=[
        "seaborn",
        "cottage_analysis @ git+ssh://git@github.com/znamlab/cottage_analysis.git@dev",
        "wayla @ git+ssh://git@github.com/znamlab/wayla.git@dev",
        "roifile",
        "plotly",
        "kaleido",
    ],
)
