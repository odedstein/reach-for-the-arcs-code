from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

# Load ground truth mesh
# filename = "data/penguin.obj"
filename = "data/armadillo.obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)

# Need normals for PSR later
N = gpy.per_face_normals(V_gt, F_gt)

# Run various levels or point_cloud_to_mesh
ns = [50, 100, 1000]
Vs = []
Fs = []
Ps = []
rng = np.random.default_rng(196)
for n in ns:
    P,I,_ = gpy.random_points_on_mesh(V_gt, F_gt, n, rng=rng, return_indices=True)
    Ps.append(P)
    screening_weight = 100
    outer_boundary_type = "dirichlet"
    depth=8
    V,F = rfta.point_cloud_to_mesh(P, N[I,:], screening_weight=screening_weight,
        outer_boundary_type=outer_boundary_type,
        depth=depth,
        verbose = True)
    Vs.append(V)
    Fs.append(F)

ps.init()
ps.register_surface_mesh("ground truth", V_gt, F_gt)
for i in range(len(ns)):
    ps.register_surface_mesh(f"mesh, n={ns[i]}", Vs[i], Fs[i])
    ps.register_point_cloud(f"point cloud, n={ns[i]}", Ps[i])
ps.show()

