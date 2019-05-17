# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools import find_packages

setup(name='Multimodal Keras Wrapper',
      version='2.2',
      description='Wrapper for Keras with support to easy multimodal data and models loading and handling.',
      author='Marc Bolaños - Alvaro Peris',
      author_email='marc.bolanos@ub.edu',
      url='https://github.com/MarcBS/multimodal_keras_wrapper',
      download_url='https://github.com/MarcBS/multimodal_keras_wrapper/archive/master.zip',
      install_requires=['keras',
                        'numpy',
                        'six',
                        'toolz',
                        'cloudpickle',
                        'matplotlib',
                        'sacremoses',
                        'scipy',
                        'future',
                        'cython',
                        'keras_applications',
                        'keras_preprocessing',
                        'sklearn',
                        'tables'
                        ],
      extras_require={
          'cython ': ['cython'],
      },
      packages=find_packages())
