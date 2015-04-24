#!/usr/bin/env python

from distutils.core import setup

setup(name='PyAI',
      version='0.91',
      description='Python Machine Learning Framework',
      author='Alex Shmakov',
      author_email='Alexanders101@gmail.com',
      url='https://github.com/Alexanders101/PyAI',
      packages=['PyAI'],
      requires=['numpy', 'decorator', 'scipy', 'matplotlib']
      )
