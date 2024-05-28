from context import *

parser = argparse.ArgumentParser(description='Large quantitative experiment.')
parser.add_argument('--run', action=argparse.BooleanOptionalAction)
parser.set_defaults(run=False)
parser.add_argument('--plot', action=argparse.BooleanOptionalAction)
parser.set_defaults(plot=False)
args = parser.parse_args()



rng_seed = 34523

results_path = "results/runtime/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Load ground truth mesh
filename = "data/bunny_fixed.obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)

gpy.write_mesh(results_path + "ground-truth.obj", V_gt, F_gt)

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

# Set up a grid
ns = [ 5, 10, 15, 20, 30, 40, 50, 70, 100, 150 ]

errors_ours = np.zeros(len(ns))
errors_rfts = np.zeros(len(ns))
errors_mc = np.zeros(len(ns))

num_repeats = 10

if args.run:
    times_mc = np.zeros(len(ns))
    times_rfts = np.zeros(len(ns))
    times_ours = np.zeros(len(ns))
    for (i,n) in enumerate(ns):
        print("n = {}".format(n))
        gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
        U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
        S = sdf(U)

        # Marching squares
        # time
        start = time.time()
        for _ in range(num_repeats):
            V_mc, F_mc = gpy.marching_cubes(S, U, n+1, n+1, n+1)
        end = time.time()
        times_mc[i] = (end - start)/num_repeats
        print("MC Time = {}".format(times_mc[i]))
        gpy.write_mesh(results_path + "marching-cubes-{}.obj".format(n), V_mc, F_mc)
        errors_mc[i] = utility.chamfer(V_mc, F_mc, V_gt, F_gt)
        # Reach for the Arcs
        # screening weights powers of 2 from 2^{-4} to 2^{8}
        # Choose an initial surface for reach_for_the_spheres
        start = time.time()
        for _ in range(num_repeats):
            V0, F0 = gpy.icosphere(2)
            V, F = gpy.reach_for_the_spheres(U, None, V0, F0, S=S, verbose=False)
        end = time.time()
        times_rfts[i] = (end - start) / num_repeats
        errors_rfts[i] = utility.chamfer(V, F, V_gt, F_gt)
        print("RFTS Time = {}".format(times_rfts[i]))
        gpy.write_mesh(results_path + "rfts-{}.obj".format(n), V, F)


        start = time.time()
        for _ in range(num_repeats):
            V, F, P, N = rfta.reach_for_the_arcs(U, S, verbose = False, parallel = True, return_point_cloud=True, debug_Vgt=V_gt, debug_Fgt=F_gt, fine_tune_iters=10, max_points_per_sphere=3)
        end = time.time()
        times_ours[i] = (end - start) / num_repeats
        errors_ours[i] = utility.chamfer(V, F, V_gt, F_gt)
        print("RFTA Time = {}".format(times_ours[i]))
        # write output
        gpy.write_mesh(results_path + "ours-{}.obj".format(n), V, F)

    np.save(results_path + "times-mc.npy", times_mc)
    np.save(results_path + "times-rfts.npy", times_rfts)
    np.save(results_path + "times-ours.npy", times_ours)
    np.save(results_path + "errors-mc.npy", errors_mc)
    np.save(results_path + "errors-rfts.npy", errors_rfts)
    np.save(results_path + "errors-ours.npy", errors_ours)

if args.plot:
    times_mc = np.load(results_path + "times-mc.npy")
    times_rfts = np.load(results_path + "times-rfts.npy")
    times_ours = np.load(results_path + "times-ours.npy")
    # plot logarithmic plots of times against grid cells (n^3):
    num_grid_cells = np.array(ns)**3
    plt.figure()
    plt.plot(num_grid_cells, times_mc, label="Marching Cubes")
    plt.plot(num_grid_cells, times_rfts, label="Reach for the Spheres")
    plt.plot(num_grid_cells, times_ours, label="Reach for the Arcs")
    plt.xlabel("Grid Cells")
    plt.ylabel("Time (s)")
    plt.legend()
    # make log
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(results_path + "runtime-log.pdf", bbox_inches='tight')
    # save eps too
    plt.savefig(results_path + "runtime-log.eps", bbox_inches='tight')
    # and png
    plt.savefig(results_path + "runtime-log.png", bbox_inches='tight')

    plt.close()

    # similar figure but for the error
    errors_mc = np.load(results_path + "errors-mc.npy")
    errors_rfts = np.load(results_path + "errors-rfts.npy")
    errors_ours = np.load(results_path + "errors-ours.npy")
    plt.figure()
    plt.plot(num_grid_cells, errors_mc, label="Marching Cubes")
    plt.plot(num_grid_cells, errors_rfts, label="Reach for the Spheres")
    plt.plot(num_grid_cells, errors_ours, label="Reach for the Arcs")
    plt.xlabel("Grid Cells")
    plt.ylabel("Chamfer Distance")
    plt.legend()
    # make log
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(results_path + "error-log.pdf", bbox_inches='tight')
    # save eps too
    plt.savefig(results_path + "error-log.eps", bbox_inches='tight')
    # and png
    plt.savefig(results_path + "error-log.png", bbox_inches='tight')
    plt.close()


    # runtime of just ours
    plt.figure()
    plt.plot(num_grid_cells, times_ours, label="Reach for the Arcs")
    # also plot ticks for data
    plt.plot(num_grid_cells, times_ours, 'o', color='black')
    plt.xlabel("Grid Cells")
    plt.ylabel("Time (s)")
    plt.legend()
    # make log
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(results_path + "runtime-ours-log.pdf", bbox_inches='tight')
    # save eps too
    plt.savefig(results_path + "runtime-ours-log.eps", bbox_inches='tight')