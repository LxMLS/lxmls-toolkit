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

If you do not have a pyhon installation, install `Anaconda`. Go to

    https://www.anaconda.com/download/

and follow the instructions for installation using Python 3. After setting up the anaconda:

	use your favorite git tool to create a clone of this repository
	navigate to the folder where the repository resides
	
	install anaconda (see instruction)
	conda create --name lxmls_new
	conda activate lxmls_new
	conda install pip
	pip install --editable . 


If you prefer `pip` to Anaconda you can install the toolkit in a way that does not interfere with your existing installation. 
For this you can use a virtual environment as follows

	virtualenv venv
	source venv/bin/activate (on Windows: .\venv\Scripts\activate)
	pip install pip setuptools --upgrade
	pip install --editable .
	


This will install the toolkit in a way that is modifiable. Remember to run scripts from the root directory `lxmls-toolkit-student`

If you want to also virtualize you Python version (e.g. you are stuck with Python2 on your system), have a look at `pyenv`.


### Running

* Run from the project root directory. If an importing error occurs, try first adding the current path to the `PYTHONPATH` environment variable, e.g.:
  * `export PYTHONPATH=.`
