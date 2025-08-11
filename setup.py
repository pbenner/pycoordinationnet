from setuptools import setup

setup(
    name="CoordinationNet",
    version="0.0.1",
    long_description="file: README.md",
    license="BSD-3",
    classifiers=[
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    include_package_data=True,
    packages=["coordinationnet"],
    install_requires=[
        "numpy",
        "pymatgen",
        "torch",
        "pytorch-lightning",
        "dill",
        "torch_geometric",
    ],
    python_requires=">=3.9",
)
