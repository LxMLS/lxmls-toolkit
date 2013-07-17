import os
from setuptools import setup, find_packages
try:
    from pip.req import parse_requirements
    # parse_requirements() returns generator of pip.req.InstallRequirement objects
    install_reqs = parse_requirements("pip-requirements.txt")
    # install_requires is a list of requirement
    install_requires = [str(ir.req) for ir in install_reqs]
except:
    # This is a bit of an ugly hack, but pip is not installed on EMR
    install_requires = []


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

package_data = {
    'lxmls.readers' : ['*.map']
}

setup(
    name="LXMLS_Toolkit",
    version="0.0.1",
    author="LXMLS-team",
    author_email="lxmls-2013-org@googlegroups.com",
    description=("Machine Learning and Natural Language toolkit"),
    license="MIT",
    keywords="machine learning",
    url="https://github.com/gracaninja/lxmls-toolkit",
    long_description=read('README.txt'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
    packages=find_packages(exclude=("labs", "labs.*")),
    install_requires=install_requires,
    package_data=package_data,
)
