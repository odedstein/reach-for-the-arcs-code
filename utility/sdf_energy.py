# Here I import only the functions I need for these functions
import numpy as np
import gpytoolbox as gpy

# Generates a callback that can be used as a mesh exporter in sdf_flow
def sdf_energy(v,f,u,s):
    # we will find the L2 difference between the signed distance function of the mesh at U and S
    S2 = gpy.signed_distance(u,v,f)[0]
    error = np.sum((S2 - s)**2) / s.shape[0]
    return error