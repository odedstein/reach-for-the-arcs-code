from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

# rng
rng_seed = 34523

# Load ground truth mesh
filename = "data/penguin.obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

# Set up a grid
n = 20
gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
S = sdf(U)

# Marching squares
V_mc, F_mc = gpy.marching_cubes(S, U, n+1, n+1, n+1)

# Reach for the Arcs
V, F, P, N = rfta.reach_for_the_arcs(U, S,
    fine_tune_iters=3,
    parallel=False, rng_seed=rng_seed,
    return_point_cloud=True, verbose=True)

# Ideal point cloud and its reconstruction
_, I, b = gpy.squared_distance(U, V_gt, F_gt, use_cpp=True,use_aabb=True)
P_best = np.sum(V_gt[F_gt[I,:],:]*b[...,None], axis=1)
N_best = gpy.per_face_normals(V_gt, F_gt)[I,:]
V_best, F_best = rfta.point_cloud_to_mesh(P_best, N_best,
    screening_weight=10., verbose=False)

# Plot
ps.init()
ps.register_surface_mesh("ground truth", V_gt, F_gt)
ps.register_surface_mesh("marching cubes", V_mc, F_mc)
ps.register_surface_mesh("Reach for the Arcs", V, F)
ps.register_point_cloud("Reach for the Arcs - points", P)
ps.register_surface_mesh("PSR with best point cloud", V_best, F_best)
ps.register_point_cloud("PSR with best point cloud - points", P_best)
ps.show()
