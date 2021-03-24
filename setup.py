from setuptools import setup, find_packages
import os
from typing import List


PATH_ROOT = os.path.dirname(__file__)


def _load_requirements(path_dir: str, file_name: str) -> List[str]:
    with open(os.path.join(path_dir, file_name), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = list()
    for line in lines:
        if line.startswith("#"):
            continue
        else:
            reqs.append(line)
    return reqs


setup(
    name="washing-learning",
    version="1.0.0",
    author="Lucas Robinet",
    author_email="lucas.robinet@yahoo.com",
    description="Machine Learning Toolbox",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="",
    license="Apache-2.0",
    packages=find_packages(exclude=["examples", "examples.*", "tests", "tests.*"]),
    install_requires=_load_requirements(
        path_dir=os.path.join(PATH_ROOT), file_name="requirements.txt"
    ),
    extras_requires={
        "dev": _load_requirements(
            path_dir=os.path.join(PATH_ROOT), file_name="requirements-dev.txt"
        ),
    },
)
