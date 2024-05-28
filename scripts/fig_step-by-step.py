from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

rng_seed = 3563
rng = np.random.default_rng(seed=rng_seed)
seed = lambda : rng.integers(0,np.iinfo(np.int32).max)

results_path = "results/step-by-step/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

filename = "data/chatz.png"
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
V_gt = gpy.normalize_points(V_gt)

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

# Set up a grid
n = 12
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

# Prepare for rasterization and locally_make_feasible
neg = S<0
pos = np.logical_not(neg)
pos,neg = np.nonzero(pos)[0], np.nonzero(neg)[0]
U_pos, U_neg = U[pos,:], U[neg,:]
S_pos, S_neg = S[pos], S[neg]
P_pos_outside = rfta.outside_points_from_rasterization(U_pos, S_pos,
            rng_seed=seed(),
            parallel=False, verbose=False)
P_neg_outside = rfta.outside_points_from_rasterization(U_neg, S_neg,
            rng_seed=seed(),
            parallel=False, verbose=False)
np.save(f"{results_path}/P_pos_outside.npy", P_pos_outside)
np.save(f"{results_path}/P_neg_outside.npy", P_neg_outside)

P_pos_locally_feasible, N_pos, f_pos = \
rfta.locally_make_feasible(U_pos, S_pos, P_pos_outside,
        rng_seed=seed(),
        parallel=False, verbose=False)
P_neg_locally_feasible, N_neg, f_neg = \
rfta.locally_make_feasible(U_neg, S_neg, P_neg_outside,
        rng_seed=seed(),
        parallel=False, verbose=False)
np.save(f"{results_path}/P_pos_locally_feasible.npy", P_pos_locally_feasible)
np.save(f"{results_path}/P_neg_locally_feasible.npy", P_neg_locally_feasible)

P_locally_feasible = np.concatenate((P_pos_locally_feasible,
    P_neg_locally_feasible), axis=0)
np.save(f"{results_path}/P_locally_feasible.npy", P_locally_feasible)
N = np.concatenate((N_pos, N_neg), axis=0)
f = np.concatenate((pos[f_pos], neg[f_neg]), axis=0)

V_pre_fine_tune,F_pre_fine_tune = rfta.point_cloud_to_mesh(P_locally_feasible, N,
    parallel=False, verbose=False)
utility.write_mesh(f"{results_path}/pre_fine_tune.npy", V_pre_fine_tune, F_pre_fine_tune)

P_fine_tuned,N,f = rfta.fine_tune_point_cloud(U, S, P_locally_feasible, N, f,
        rng_seed=seed(),
        fine_tune_iters=50,
        parallel=False, verbose=False)
np.save(f"{results_path}/P_fine_tuned.npy", P_fine_tuned)

V_post_fine_tune,F_post_fine_tune = rfta.point_cloud_to_mesh(P_fine_tuned, N,
    parallel=False, verbose=False)
utility.write_mesh(f"{results_path}/post_fine_tune.npy", V_post_fine_tune, F_post_fine_tune)


    