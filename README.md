[![Travis-CI Build Status][travis-image]][travis-url] [![Requirements Status][requires-image]][requires-url]

[travis-image]: https://travis-ci.org/LxMLS/lxmls-toolkit.svg?branch=master
[travis-url]: https://travis-ci.org/LxMLS/lxmls-toolkit

[requires-image]: https://requires.io/github/LxMLS/lxmls-toolkit/requirements.svg?branch=master
[requires-url]: https://requires.io/github/LxMLS/lxmls-toolkit/requirements/?branch=master

# LxMLS 2024

Machine learning toolkit for natural language processing. Written for [LxMLS - Lisbon Machine Learning Summer School](http://lxmls.it.pt)

* Scientific Python and Mathematical background
* Linear Classifiers (Gradient Descent)
* Feed-forward models in deep learning (Backpropagation)
* Sequence models in deep learning
* Attention Models (Transformers)

Bear in mind that the main purpose of the toolkit is educational. You may resort
to other toolboxes if you are looking for efficient implementations of the
algorithms described.

## Instructions for Students

* Use the [student branch](https://github.com/LxMLS/lxmls-toolkit/tree/student) **not** this one 🚨!

Download the code. If you are used to git just clone the student branch. For
example from the command line in do

    git clone git@github.com:LxMLS/lxmls-toolkit.git lxmls-toolkit-student

If you do not have a pyhon installation, install miniconda. Go to

    https://docs.conda.io/en/latest/miniconda.html

and follow the instructions for installation using Python 3.

After setting up the anaconda:

	use your favorite git tool to create a clone of this repository
	navigate to the folder where the repository resides
	
	install anaconda (see instruction)
	conda create --name lxmls_new
	conda activate lxmls_new
	conda install pip
	pip install --editable . 


and follow the instructions for your platform (Windows, Linux, OSX). We reccomend
that you create your virtual environment with a recent python version i.e.

    cd lxmls-toolkit-student
    conda create -y -p ./lxmls2023 python=3.9 -y
    conda activate ./lxmls2023

Note the `./` in `./lxmls2023` -- this will install the virtual environment
locally, so if you delete `lxmls-toolkit-student` you will also remove the
environment.

Then install the toolkit, just to be sure upgrade your pip (always good)

    pip install pip setuptools --upgrade
    pip install -r requirements.txt

This will install the toolkit in a way that is modifiable. Remember to run scripts from the root directory `lxmls-toolkit-student`

### Running

* Run from the project root directory. If an importing error occurs, try first adding the current path to the `PYTHONPATH` environment variable, e.g.:
  * `export PYTHONPATH=.`
```
### Development

To run the all tests install `tox` and `pytest`

    pip install tox pytest

and run

    tox

Note, to combine the coverage data from all the tox environments run:


* Windows
    set PYTEST_ADDOPTS=--cov-append
	
* Other
    PYTEST_ADDOPTS=--cov-append tox
```