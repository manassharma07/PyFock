# To publish on PyPi, run the following
# python setup.py sdist
# twine upload dist/* 
import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# Read the requirements
# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "requirements.txt")) as f:
    requirements = f.read().splitlines()

setup(
    name = "pyfock",
    version = "0.0.9", # DONT FORGET TO CHANGE THE VERSION IN __INIT__.py
    author = "Manas Sharma",
    author_email = "feedback@bragitoff.com",
    description = ("A simplistic and efficient pure-python quantum chemistry library from Phys Whiz."),
    license = "MIT",
    keywords = ["dft", "pure python", "numba dft", "density functional theory", "manas sharma", "bragitoff","quantum chemistry", "pyfock", "molecular integrals"],
    url = "https://github.com/manassharma07/pyfock",
    download_url = '',
    packages=find_packages(),#['pyfock'],
    include_package_data=True,
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        'Programming Language :: Python :: 3.0',      #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',      #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
    ],
)
