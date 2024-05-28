from context import *

rng_seed = 34523

results_path = "results/screening-weight-ablation/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Load ground truth mesh
filename = "data/koala.obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)

gpy.write_mesh(results_path + "ground-truth.obj", V_gt, F_gt)

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

# Set up a grid
n = 30
gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
S = sdf(U)

# Marching squares
V_mc, F_mc = gpy.marching_cubes(S, U, n+1, n+1, n+1)
gpy.write_mesh(results_path + "marching-cubes.obj", V_mc, F_mc)
# chamfer error
chf_mc = rfta.chamfer(V_gt, F_gt, V_mc, F_mc)
print("chamfer error (marching cubes) = {}".format(chf_mc))
# hausdorff distance
hd_mc = gpy.approximate_hausdorff_distance(V_gt, F_gt, V_mc, F_mc)

# Reach for the Arcs
# screening weights powers of 2 from 2^{-4} to 2^{8}
screening_weights = np.power(2.0, np.arange(-4, 14, 1))
chfs = np.zeros(screening_weights.shape)
hds = np.zeros(screening_weights.shape)
for i, screening_weight in enumerate(screening_weights):
    print("screening_weight = {}".format(screening_weight))
    V, F, P, N = rfta.reach_for_the_arcs(U, S, verbose = True, parallel = True, return_point_cloud=True, debug_Vgt=V_gt, debug_Fgt=F_gt, fine_tune_iters=20, max_points_per_sphere=30, screening_weight=screening_weight)
    # write output
    gpy.write_mesh(results_path + "ours-{}.obj".format(screening_weight), V, F)
    # chamfer error
    chfs[i] = rfta.chamfer(V_gt, F_gt, V, F)
    hds[i] = gpy.approximate_hausdorff_distance(V_gt, F_gt, V, F)
    # write distances to npy file
    np.save(results_path + "chfs.npy", chfs)
    np.save(results_path + "hds.npy", hds)

chfs = np.load(results_path + "chfs.npy")
hds = np.load(results_path + "hds.npy")
# plot errors

plt.plot(screening_weights, chfs, label="chamfer error")
# log
plt.yscale("log")
plt.xscale("log")
plt.ylim(0.01, 0.1)
# add horizontal bar at MC value
plt.axhline(y=chf_mc, color='r', linestyle='-', label="marching cubes")
plt.savefig(results_path + "chamfer-error.eps")
plt.clf()
plt.plot(screening_weights, hds, label="hausdorff distance")
# log
plt.yscale("log")
plt.xscale("log")
# set y limits to 0.1 and 0.01
plt.ylim(0.01, 0.1)
# add horizontal bar at MC value
plt.axhline(y=hd_mc, color='r', linestyle='-', label="marching cubes")
plt.savefig(results_path + "hausdorff-distance.eps")
plt.clf()

