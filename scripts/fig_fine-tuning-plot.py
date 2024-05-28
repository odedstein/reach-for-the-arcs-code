from context import *

rng_seed = 34523

results_path = "results/fine-tuning-plot/"
# make directory if it doesnt exist
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Load ground truth mesh
filename = "data/37358_sf.obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)

gpy.write_mesh( results_path + "ground_truth.obj", V_gt, F_gt)

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

# Set up a grid
n = 30
gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
S = sdf(U)

# Marching squares
V_mc, F_mc = gpy.marching_cubes(S, U, n+1, n+1, n+1)
gpy.write_mesh( results_path + "marching_cubes.obj", V_mc, F_mc)
mc_chamfer = rfta.chamfer(V_gt, F_gt, V_mc, F_mc)
mc_hausdorff = gpy.approximate_hausdorff_distance(V_gt, F_gt, V_mc, F_mc)
np.save(results_path + "marching_cubes_chamfer.npy", mc_chamfer)
np.save(results_path + "marching_cubes_hausdorff.npy", mc_hausdorff)
print("marching cubes chamfer: ", mc_chamfer) # 0.05604792619876185
print("marching cubes hausdorff: ", mc_hausdorff) # 0.13994899951746878

iterations = np.arange(0, 20, 1)
chamfer_distances = np.zeros(iterations.shape)
hausdorff_distances = np.zeros(iterations.shape)
for i in iterations:
    # Reach for the Arcs
    V, F, P, N = rfta.reach_for_the_arcs(U, S, verbose = True, parallel = True, return_point_cloud=True, debug_Vgt=V_gt, debug_Fgt=F_gt, fine_tune_iters=i, max_points_per_sphere=30)

    # Write
    gpy.write_mesh( results_path + f"ours_{i}.obj", V, F)
    # chamfer distance
    chamfer_distances[i] = rfta.chamfer(V_gt, F_gt, V, F)
    hausdorff_distances[i] = gpy.approximate_hausdorff_distance(V_gt, F_gt, V, F)
    # write distances to npy file
    np.save(results_path + "chamfer_distances.npy", chamfer_distances)
    np.save(results_path + "hausdorff_distances.npy", hausdorff_distances)

chamfer_distances = np.load(results_path + "chamfer_distances.npy")
hausdorff_distances = np.load(results_path + "hausdorff_distances.npy")

# pick only the first 10
# iterations = iterations[:10]
# chamfer_distances = chamfer_distances[:10]
# hausdorff_distances = hausdorff_distances[:10]

# plot distances vs iterations
plt.plot(iterations, chamfer_distances, label="chamfer")
plt.savefig(results_path + "chamfer_distances.eps", bbox_inches='tight')
plt.clf()
plt.plot(iterations, hausdorff_distances, label="hausdorff")
plt.savefig(results_path + "hausdorff_distances.eps", bbox_inches='tight')
plt.clf()

# now make plots logarithmic (base 10)
plt.plot(iterations, chamfer_distances, label="chamfer")
plt.yscale('log')
# add a line with the value of the marching cubes chamfer distance
plt.axhline(y=mc_chamfer, color='r', linestyle='-', label="marching cubes")
plt.ylim(0.01, 1)
plt.savefig(results_path + "chamfer_distances_log.eps", bbox_inches='tight')
plt.clf()
plt.plot(iterations, hausdorff_distances, label="hausdorff")
plt.yscale('log')
# add a line with the value of the marching cubes hausdorff distance
plt.axhline(y=mc_hausdorff, color='r', linestyle='-', label="marching cubes")
plt.ylim(0.01, 1)
plt.savefig(results_path + "hausdorff_distances_log.eps", bbox_inches='tight')
plt.clf()

# now plot everything together (in log scale)
# plt.plot(iterations, chamfer_distances, label="chamfer")
# plt.plot(iterations, hausdorff_distances*0.5, label="hausdorff")
# plt.yscale('log')
# # add a line with the value of the marching cubes chamfer distance
# plt.axhline(y=mc_chamfer, color='r', linestyle='-', label="marching cubes")
# # add a line with the value of the marching cubes hausdorff distance
# plt.axhline(y=mc_hausdorff, color='r', linestyle='-', label="marching cubes")
# # set axis from 0.01 to 1
# plt.ylim(0.01, 1)
# plt.savefig(results_path + "distances_log.eps", bbox_inches='tight')
# plt.clf()