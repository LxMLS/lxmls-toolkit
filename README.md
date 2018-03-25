[![Travis-CI Build Status][travis-image]][travis-url] [![Requirements Status][requires-image]][requires-url]
[![Coverage Status][codecov-image]][codecov-url] [![Code Quality Status][landscape-image]][landscape-url]
[![Scrutinizer Status][scrutinizer-image]][scrutinizer-url] [![Codacy Code Quality Status][codacy-image]][codacy-url]

[travis-image]: https://travis-ci.org/LxMLS/lxmls-toolkit.svg?branch=master
[travis-url]: https://travis-ci.org/LxMLS/lxmls-toolkit

[requires-image]: https://requires.io/github/LxMLS/lxmls-toolkit/requirements.svg?branch=master
[requires-url]: https://requires.io/github/LxMLS/lxmls-toolkit/requirements/?branch=master

[codecov-image]: https://codecov.io/github/LxMLS/lxmls-toolkit/coverage.svg?branch=master
[codecov-url]: https://codecov.io/github/LxMLS/lxmls-toolkit

[landscape-image]: https://landscape.io/github/LxMLS/lxmls-toolkit/master/landscape.svg?style=flat
[landscape-url]: https://landscape.io/github/LxMLS/lxmls-toolkit/master

[codacy-image]: https://img.shields.io/codacy/REPLACE_WITH_PROJECT_ID.svg?style=flat
[codacy-url]: https://www.codacy.com/app/LxMLS/lxmls-toolkit

[version-image]: https://img.shields.io/pypi/v/lxmls.svg?style=flat
[version-url]: https://pypi.python.org/pypi/lxmls

[downloads-image]: https://img.shields.io/pypi/dm/lxmls.svg?style=flat
[downloads-url]: https://pypi.python.org/pypi/lxmls

[wheel-image]: https://img.shields.io/pypi/wheel/lxmls.svg?style=flat
[wheel-url]: https://pypi.python.org/pypi/lxmls

[supported-versions-image]: https://img.shields.io/pypi/pyversions/lxmls.svg?style=flat
[supported-versions-url]: https://pypi.python.org/pypi/lxmls

[supported-implementations-image]: https://img.shields.io/pypi/implementation/lxmls.svg?style=flat
[supported-implementations-url]: https://pypi.python.org/pypi/lxmls

[scrutinizer-image]: https://img.shields.io/scrutinizer/g/LxMLS/lxmls-toolkit/master.svg?style=flat
[scrutinizer-url]: https://scrutinizer-ci.com/g/LxMLS/lxmls-toolkit/

# Summary

Machine learning toolkit for natural language processing. Written for Lisbon Machine Learning Summer School (lxmls.it.pt). This covers

* Scientific Python and Mathematical background
* Linear Classifiers
* Sequence Models
* Structured Prediction
* Syntax and Parsing
* Feed-forward models in deep learning
* Sequence models in deep learning

Machine learning toolkit for natural language processing. Written for [LxMLS - Lisbon Machine Learning Summer School](lxmls.it.pt)

## Instructions for Students

* Use the [student branch](https://github.com/LxMLS/lxmls-toolkit/tree/student) **not** this one!

## Install with Anaconda

The simplest method is to use `Anaconda`to handle your packages as described on
`Day 0` of the lxmls-guide.

## Alternative install with pip and virtualenv

If you like `pip`, install the toolkit modules

    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt

Then get the right `pip install` command for your platform for pytorch from
`http://pytorch.org/` and apply them. Finally call

    python setup.py develop

to instal the toolkit in a way that is modifiable.

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
