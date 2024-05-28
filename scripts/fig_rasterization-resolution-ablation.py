from context import *

rng_seed = 34523

results_path = "results/rasterization-resolution-ablation/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Load ground truth mesh
filename = "data/fertility.obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)

gpy.write_mesh(results_path + "ground-truth.obj", V_gt, F_gt)

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

# Set up a grid
ns = [ 10, 20, 50, 80, 100 ]
ns = [ 20 ]
for n in ns:
    print("n = {}".format(n))
    gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
    U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    S = sdf(U)

    # Marching squares
    V_mc, F_mc = gpy.marching_cubes(S, U, n+1, n+1, n+1)
    gpy.write_mesh(results_path + "marching-cubes-{}.obj".format(n), V_mc, F_mc)
    # Reach for the Arcs
    # screening weights powers of 2 from 2^{-4} to 2^{8}
    resolutions = np.arange(30, 256, 10)
    resolutions = np.array([50, 100, 150, 200]) 
    # 0.05
    # 1.2
    # 1.89
    # 2.7
    chfs = np.zeros(resolutions.shape)
    hds = np.zeros(resolutions.shape)
    for i, resolution in enumerate(resolutions):
        print("Resolution =  {}".format(resolution))
        time_start = time.time()
        V, F, P, N = rfta.reach_for_the_arcs(U, S, verbose = True, parallel = True, return_point_cloud=True, debug_Vgt=V_gt, debug_Fgt=F_gt, fine_tune_iters=20, max_points_per_sphere=3, rasterization_resolution=resolution)
        time_end = time.time()
        print("Time = {}".format(time_end - time_start))
        # write output
        gpy.write_mesh(results_path + "ours-{}-{}.obj".format(n, resolution), V, F)