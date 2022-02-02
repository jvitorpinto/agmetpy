import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="agmetpy",
    version="0.0.2",
    author="João Vitor de Nóvoa Pinto",
    author_email="jvitorpinto@gmail.com",
    description="A python package for agricultural forecasting and crop modelling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jvitorpinto/agmetpy",
    project_urls={
        "Bug Tracker": "https://github.com/jvitorpinto/agmetpy",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=['numpy', 'pandas', 'matplotlib'],
)
