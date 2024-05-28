from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

rng_seed = 34523

# Load ground truth mesh
filename = "data/fertility.obj"
V_gt, F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

# Set up a grid
n = 25
gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
S = sdf(U)

utility.boolean_hulls(U,S)