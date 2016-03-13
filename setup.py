from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

files = ['pymaxflow.pyx', 'graph.cpp', 'maxflow.cpp']

extensions = [Extension("pymaxflow", files, 
   language="c++", include_dirs=[np.get_include()])]

setup( ext_modules = cythonize(extensions))
