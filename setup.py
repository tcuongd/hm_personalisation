from setuptools import setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="hm_personalisation",
    version="0.1",
    description="Prepare data, train models, make predictions for H&M Personalisation challenge.",
    author="Cuong Duong",
    author_email="cuong.duong242@gmail.com",
    packages=["hm_personalisation"],
    install_requires=requirements,
)
