[![Travis-CI Build Status][travis-image]][travis-url] [![Requirements Status][requires-image]][requires-url]

[travis-image]: https://travis-ci.org/LxMLS/lxmls-toolkit.svg?branch=master
[travis-url]: https://travis-ci.org/LxMLS/lxmls-toolkit

[requires-image]: https://requires.io/github/LxMLS/lxmls-toolkit/requirements.svg?branch=master
[requires-url]: https://requires.io/github/LxMLS/lxmls-toolkit/requirements/?branch=master

# LxMLS 2023

Machine learning toolkit for natural language processing. Written for Lisbon Machine Learning Summer School (lxmls.it.pt). This covers

* Scientific Python and Mathematical background
* Linear Classifiers
* Sequence Models
* Structured Prediction
* Syntax and Parsing
* Feed-forward models in deep learning
* Sequence models in deep learning
* Reinforcement Learning

Machine learning toolkit for natural language processing. Written for [LxMLS - Lisbon Machine Learning Summer School](http://lxmls.it.pt)

## Instructions for Students

* Use the [student branch](https://github.com/LxMLS/lxmls-toolkit/tree/student) **not** this one!

## Install with Anaconda or pip

If you are new to Python, the simplest method is to use `Anaconda`to handle your packages, just go to

    https://www.anaconda.com/download/

and follow the instructions. We strongly recommend using at least Python 3.

If you prefer `pip` to Anaconda you can install the toolkit in a way that does
not interfere with your existing installation. For this you can use a virtual
environment as follows 

    virtualenv venv
    source venv/bin/activate (on Windows: .\venv\Scripts\activate)
    pip install pip setuptools --upgrade
    pip install --editable . 

This will install the toolkit in a way that is modifiable. If you want to also
virtualize you Python version (e.g. you are stuck with Python2 on your system),
have a look at `pyenv`.

Bear in mind that the main purpose of the toolkit is educative. You may resort
to other toolboxes if you are looking for efficient implementations of the
algorithms described.

### Running

* Run from the project root directory. If an importing error occurs, try first adding the current path to the `PYTHONPATH` environment variable, e.g.:
  * `export PYTHONPATH=.`

### Development

To run the all tests install `tox` and `pytest` 

    pip install tox pytest

and run

    tox

Note, to combine the coverage data from all the tox environments run:

* Windows
    ```
    set PYTEST_ADDOPTS=--cov-append
    tox
    ```
* Other
    ```
    PYTEST_ADDOPTS=--cov-append tox
    ```
