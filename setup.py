from setuptools import setup, find_packages
import pathlib

packages = find_packages(where="src")
print(packages)
setup(

    name="craftsman",
    version="0.1.0",
    description="An image to 3D generator based on CraftsMan",
    url="https://github.com/charmed-ai/craftsman-geometry-algorithm",
    package_dir={"": "src"},
    packages=packages,
    python_requires=">=3.10, <3.11",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "craftsman_generate_geometry=craftsman.apps.generate:main"
        ],
    },
)