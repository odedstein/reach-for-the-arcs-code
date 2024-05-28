from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

seed = 4623452

# Load ground truth mesh
filename = "data/penguin.obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = 0.5*gpy.normalize_points(V_gt) + np.array([[0.5, 0.5, 0.5]])

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

# Set up a grid
n = 30
gx, gy, gz = np.meshgrid(np.linspace(0., 1.0, n+1), np.linspace(0., 1.0, n+1), np.linspace(0., 1.0, n+1))
U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
S = sdf(U)

# Rasterize
neg = S<0
pos = np.logical_not(neg)
res = 360
print("Negative spheres...")
P_neg = rfta.outside_points_from_rasterization(U[neg,:], S[neg], rng_seed=seed,
    narrow_band=True,
    res=res, verbose=True)
print("Positive spheres...")
P_pos = rfta.outside_points_from_rasterization(U[pos,:], S[pos], rng_seed=seed,
    narrow_band=True,
    res=res, verbose=True)

ps.init()
ps.register_surface_mesh("ground truth", V_gt, F_gt)
ps.register_point_cloud("P_neg", P_neg)
ps.register_point_cloud("P_pos", P_pos)
ps.show()
