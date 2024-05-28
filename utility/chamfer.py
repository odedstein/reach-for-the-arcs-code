# Here I import only the functions I need for these functions
import numpy as np
import gpytoolbox as gpy

# Generates a callback that can be used as a mesh exporter in sdf_flow
def chamfer(v1,f1,v2,f2,n=1000000):
    P1 = gpy.random_points_on_mesh(v1,f1,n)
    P2 = gpy.random_points_on_mesh(v2,f2,n)
    d1 = gpy.squared_distance(P1,P2,use_aabb=True,use_cpp=True)[0]
    d2 = gpy.squared_distance(P2,P1,use_aabb=True,use_cpp=True)[0]
    return np.sqrt(np.mean(d1)) + np.sqrt(np.mean(d2))