# These are the imports we're going to use in all scripts.
import sys, os

# Use relative paths so this works on any computer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utility
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/python')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import rfta
import gpytoolbox
import numpy
import scipy
import unittest
import polyscope