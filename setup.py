import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pycudacov",  # Replace with your own username
    version="0.1.2",
    author="Ivan Reyes",
    author_email="ivanrs297@gmail.com",
    description="A PyCuda Covariance Matrix Parallel Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ivanrs297/pycuda-covariance-matrix",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

# python3 setup.py sdist bdist_wheel
# python3 -m twine upload --skip-existing --repository testpypi dist/*
# python3 -m twine upload --repository testpypi dist/*

# python3 -m twine upload --skip-existing  dist/*

