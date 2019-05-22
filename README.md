[![Travis-CI Build Status][travis-image]][travis-url] [![Requirements Status][requires-image]][requires-url]

[travis-image]: https://travis-ci.org/LxMLS/lxmls-toolkit.svg?branch=master
[travis-url]: https://travis-ci.org/LxMLS/lxmls-toolkit

[requires-image]: https://requires.io/github/LxMLS/lxmls-toolkit/requirements.svg?branch=master
[requires-url]: https://requires.io/github/LxMLS/lxmls-toolkit/requirements/?branch=master

# Summary

Machine learning toolkit for natural language processing. Written for Lisbon Machine Learning Summer School (lxmls.it.pt). This covers

* Scientific Python and Mathematical background
* Linear Classifiers
* Sequence Models
* Structured Prediction
* Syntax and Parsing
* Feed-forward models in deep learning
* Sequence models in deep learning

Machine learning toolkit for natural language processing. Written for [LxMLS - Lisbon Machine Learning Summer School](http://lxmls.it.pt)

## Instructions for Students

* Use the [student branch](https://github.com/LxMLS/lxmls-toolkit/tree/student) **not** this one!

## Install with Anaconda or pip

If you are new to Python, the simplest method is to use `Anaconda`to handle your packages, just go to

    https://www.anaconda.com/download/

and follow the instructions. If you prefer `pip`, install the toolkit modules in a virtual environment

    virtualenv venv
    source venv/bin/activate
    pip install pip setuptools --upgrade
    pip install -r requirements.txt

In both cases, you will need to get a `pip` or `conda` command for your platform for pytorch from
`http://pytorch.org/` and apply them. 

Finally call

    python setup.py develop

to install the toolkit in a way that is modifiable.

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
