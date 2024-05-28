from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

seed = 462452

# Load ground truth mesh
filename = "data/armadillo.obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = 0.5*gpy.normalize_points(V_gt) + np.array([[0.5, 0.5, 0.5]])

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

# Set up a grid
n = 20
gx, gy, gz = np.meshgrid(np.linspace(0., 1.0, n+1), np.linspace(0., 1.0, n+1), np.linspace(0., 1.0, n+1))
U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
S = sdf(U)

# SDF to point cloud
neg = S<0
pos = np.logical_not(neg)
res = 32
P_neg,N_neg,f_neg = rfta.sdf_to_point_cloud(U[neg,:], S[neg], rng_seed=seed,
    rasterization_resolution=res,
    parallel=True,
    n_local_searches=10,
    verbose=True)
P_pos,N_pos,f_pos = rfta.sdf_to_point_cloud(U[pos,:], S[pos], rng_seed=seed,
    rasterization_resolution=res,
    parallel=True,
    n_local_searches=10,
    verbose=True)

ps.init()
ps.register_surface_mesh("ground truth", V_gt, F_gt)
ps.register_point_cloud("P_neg", P_neg)
ps.register_point_cloud("P_pos", P_pos)
ps.show()

