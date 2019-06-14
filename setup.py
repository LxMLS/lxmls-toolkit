import os
# import sys
from setuptools import setup, find_packages
try:  # for pip >=12
    from pip._internal.req import parse_requirements
    from pip._internal import download
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
    from pip import download

VERSION = '0.0.2'

# parse_requirements() returns generator of pip.req.InstallRequirement
# objects
install_reqs = parse_requirements(
    "requirements.txt", session=download.PipSession()
)
# install_requires is a list of requirement
install_requires = [str(ir.req) for ir in install_reqs]


def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


package_data = {
    'lxmls.readers': ['*.map']
}

setup(
    name='LxMLS Toolkit',
    version=VERSION,
    author='LxMLS team',
    description='Machine Learning and Natural Language toolkit',
    long_description=read('README.md'),
    license='MIT',
    keywords='machine learning',
    url='https://github.com/LxMLS/lxmls-toolkit',
    py_modules=['lxmls'],
    # test_suite='tests',
    # See: http://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
    ],
    packages=find_packages(exclude=("labs", "labs.*")),
    install_requires=install_requires,
    package_data=package_data,
)
