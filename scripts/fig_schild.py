from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

rng_seed = 362088
rng = np.random.default_rng(seed=rng_seed)
seed = lambda : rng.integers(0,np.iinfo(np.int32).max)

results_path = "results/schild/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

filename = "data/schild.png"
poly_list = gpy.png2poly(filename)
V_gt = None
F_gt = None
for poly in poly_list:
    nv = 0 if V_gt is None else V_gt.shape[0]
    pV = poly[::5,:]
    V_gt = pV if V_gt is None else np.concatenate((V_gt, pV), axis=0)
    F_gt = gpy.edge_indices(pV.shape[0],closed=True) if F_gt is None else \
        np.concatenate((F_gt, nv+gpy.edge_indices(pV.shape[0],closed=True)),
            axis=0)
V_gt = 1.5*gpy.normalize_points(V_gt)
#Smooth gt a bit
rso = 0.5
V_gt[F_gt[:,0]] = (1.-rso)*V_gt[F_gt[:,0]] + rso*V_gt[F_gt[:,1]]
V_gt[F_gt[:,1]] = (1.-rso)*V_gt[F_gt[:,1]] + rso*V_gt[F_gt[:,0]]

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

# Set up a grid
n = 6
gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
U = np.vstack((gx.flatten(), gy.flatten())).T
S = sdf(U)

# We now want to resize the SDF and the GT like reach_for_the_arcs does.
trans = np.min(U, axis=0)
U = U - trans[None,:]
scale = np.max(U)
U /= scale
S = S/scale
V_gt = V_gt - trans[None,:]
V_gt /= scale

# Save the gt
utility.write_mesh(f"{results_path}/ground_truth.npy", V_gt, F_gt)

# Save the spheres
np.save(f"{results_path}/U.npy", U)
np.save(f"{results_path}/S.npy", S)

V_rfta, F_rfta, P_rfta, N_rfta = rfta.reach_for_the_arcs(U, S,
    verbose=False, parallel=True,
        rng_seed=rng_seed,
        fine_tune_iters=10,
        max_points_per_sphere=1,
        return_point_cloud=True)


np.save(f"{results_path}/P.npy", P_rfta)
np.save(f"{results_path}/N.npy", N_rfta)
utility.write_mesh(f"{results_path}/rfta.npy", V_rfta, F_rfta)

    