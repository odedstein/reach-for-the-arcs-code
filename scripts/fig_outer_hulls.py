from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

rng_seed = 3523

results_path = "results/outer_hulls/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

filename = "data/arc-du-carrousel.png"
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

angle = 0.
R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
V_gt = V_gt @ R

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

ns = [5, 10, 30, 50, 120]
for n in ns:
    print(f"doing n={n}")
    # Set up gt

    # Set up a grid
    gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
    U = np.vstack((gx.flatten(), gy.flatten())).T
    S = sdf(U)

    # Reach for the Arcs
    print("  RFTA")
    V_rfta, F_rfta = rfta.reach_for_the_arcs(U, S, verbose=False, parallel=True,
        fine_tune_iters=50, max_points_per_sphere=50,
        rng_seed=rng_seed)


    write_path = f"{results_path}/{n}"
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # Write output
    utility.write_mesh( write_path + f"/rfta.npy", V_rfta@R.T, F_rfta)
    utility.write_mesh( write_path + f"/ground_truth.npy", V_gt@R.T, F_gt)
    np.save(write_path + f"/U.npy", U@R.T)
    np.save(write_path + f"/S.npy", S)


    