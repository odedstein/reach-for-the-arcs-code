# Here I import only the functions I need for these functions
import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ext/gpytoolbox/src')))
import gpytoolbox as gpy

def read_mesh(path):
    try:
        V,F = gpy.read_mesh(path)
    except:
        data = np.load(path, allow_pickle=True)
        V = data[()]['V']
        F = data[()]['F']
    return V,F
