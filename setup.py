import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="src",
    version="0.0.1",
    author="sabina14",
    author_email="sabinachellan@gmail.com",
    description="A small package for ANN implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sabina14/ANN_keras",
    project_urls={
        "Bug Tracker": "https://github.com/sabina14/ANN_keras/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
   
    packages=["src"],
    python_requires=">=3.7",
    install_requires=["numpy","matplotlib","pandas","seaborn", "tensorflow"]
)