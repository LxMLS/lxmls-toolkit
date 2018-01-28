### Summary

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

* Use the student branch, not this one!

* You do not need to run setup.py or pip install, read instructions in the Day 0 chapter of the [LxMLS guide](https://github.com/LxMLS/lxmls_guide).

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
