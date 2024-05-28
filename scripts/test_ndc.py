from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

rng_seed = 34523

# Load ground truth mesh
filename = "data/fertility.obj"
V_gt, F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)

print(np.max(V_gt,axis=0))
print(np.min(V_gt,axis=0))

V, F = utility.ndc(V_gt, F_gt, 50)

print(np.max(V_gt,axis=0))
print(np.min(V_gt,axis=0))


print("Chamfer error: " + str(utility.chamfer(V, F, V_gt, F_gt)))

# plot all using polyscope
ps.init()
ps_mesh = ps.register_surface_mesh("mesh", V, F)
ps_mesh_gt = ps.register_surface_mesh("mesh_gt", V_gt, F_gt)
ps.show()
