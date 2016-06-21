# Summary

[![Build Status][travis-image]][travis-url]

[travis-image]: https://travis-ci.org/LxMLS/lxmls-toolkit.svg?branch=master
[travis-url]: https://travis-ci.org/LxMLS/lxmls-toolkit

Machine learning toolkit for natural language processing. Written for [LxMLS - Lisbon Machine Learning Summer School](lxmls.it.pt)

## Instructions for Students

* Use the student branch, not this one!

* You do not need to run setup.py or pip install, read instructions in the Day 0 chapter of the [LxMLS guide](https://github.com/LxMLS/lxmls_guide).


## Instructions for non Students

Feel free to install the toolkit with
```
    pip install .
```
or in development mode (and in the local user home) with
```
python setup.py develop --user
```
(in this last case, you can use the toolkit's package by calling `import lxmls` wherever you want, and any code changes in the `lxmls` directory
   will be reflected when the package is imported/reloaded).

Bear in mind that the main purpose of the toolkit is educative. You may resort
to other toolboxes if you are looking for efficient implementations of the
algorithms described.

### Running

* Run from the project root directory. If an importing error occurs, try first adding the current path to the `PYTHONPATH` environment variable, e.g.:
  * `export PYTHONPATH=.`

### Development

To run the all tests run:

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
