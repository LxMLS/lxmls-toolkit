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

Machine learning toolkit for natural language processing. Written for [LxMLS - Lisbon Machine Learning Summer School](http://lxmls.it.pt/)

## Get the lxmls code-base

If you are not familiar with `Git`, just download the zip available in the `Clone or Download` button. **Important**: Use the student version. It should be the one in the page displaying this README. Then unzip and enter the main folder. This will be your working folder

    cd lxmls-toolkit-student

If you feel comfortable with `Git` you may instead clone the repo and checkout the student branch

    git clone https://github.com/LxMLS/lxmls-toolkit.git
    cd lxmls-toolkit/
    git checkout student

## Install modules with Anaconda or pip

If you are new to Python, the simplest method is to use `Anaconda`to handle your packages, just go to

    https://www.anaconda.com/download/

and follow the instructions. If you prefer `pip`, install the toolkit modules in a virtual environment. It is safer to update `pip` and `setuptools` first

    virtualenv venv
    source venv/bin/activate
    pip install pip setuptools --upgrade
    pip install -r requirements.txt

This will not interfere with your existing installation.  In both cases, you will need to get a `pip` or `conda` command for your platform for pytorch from
`http://pytorch.org/` and apply them. 

Finally call

    python setup.py develop

to instal the toolkit in a way that is modifiable.

Bear in mind that the main purpose of the toolkit is educative. You may resort
to other toolboxes if you are looking for efficient implementations of the
algorithms described.

## Solving Exercises

Some day will require you to complete code from previous days. If you have not completed the exercises you can allways use the `solve.py` command as for example

    python solve.py sequence_models

**Important**: This will delete your code on the correspondig file!. Save it before. To undo solving (this wont return your code) do

    python solve.py --undo sequence_models
