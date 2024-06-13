# This script replicates the results from Figure 12
from context import *

parser = argparse.ArgumentParser(description='Large quantitative experiment.')
parser.add_argument('--num_shapes', type=int, default=100, help='number of shapes to use')
parser.add_argument('--run', action=argparse.BooleanOptionalAction)
parser.set_defaults(run=False)
parser.add_argument('--plot', action=argparse.BooleanOptionalAction)
parser.set_defaults(plot=False)
args = parser.parse_args()

rng_seed = 34523

results_path = "results/batch-size-ablation/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Load ground truth mesh
# turtle pope: RonJon1234, and is licensed under Creative Commons - Attribution
filename = "data/pope.obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)

gpy.write_mesh(results_path + "ground-truth.obj", V_gt, F_gt)

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

# Set up a grid
# only run all these if your computer can handle it (since it benchmarks the non-batching option of our code, which is very slow at large resolutions)
# ns = [ 10, 15, 20, 30, 40, 55 ]
ns = [ 10, 15, 20, 30]
batch_sizes = [ 0, 1000, 10000 ]

if args.run:
    chamfer_errors = np.zeros((len(ns), len(batch_sizes)))
    runtimes = np.zeros((len(ns), len(batch_sizes)))
    for n_ind,n in enumerate(ns):
        print("n = {}".format(n))
        gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
        U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
        S = sdf(U)

        # Marching squares
        # V_mc, F_mc = gpy.marching_cubes(S, U, n+1, n+1, n+1)
        # gpy.write_mesh(results_path + "marching-cubes-{}.obj".format(n), V_mc, F_mc)
        # Reach for the Arcs
        # screening weights powers of 2 from 2^{-4} to 2^{8}
        
        
        for i, bs in enumerate(batch_sizes):
            print("Batch Size =  {}".format(bs))
            time_start = time.time()
            V, F, P, N = rfta.reach_for_the_arcs(U, S, verbose = False, parallel = True, return_point_cloud=True, debug_Vgt=V_gt, debug_Fgt=F_gt, fine_tune_iters=10, max_points_per_sphere=10, batch_size=bs)
            time_end = time.time()
            runtimes[n_ind, i] = time_end-time_start
            # write output
            gpy.write_mesh(results_path + "ours-{}-{}.obj".format(n, bs), V, F)
            chamfer_errors[n_ind, i] = utility.chamfer(V_gt, F_gt, V, F)
            # save
            np.save(results_path + "chamfer-errors.npy", chamfer_errors)
            np.save(results_path + "runtimes.npy", runtimes)

if args.plot:
    # load
    chamfer_errors = np.load(results_path + "chamfer-errors.npy")
    runtimes = np.load(results_path + "runtimes.npy")
    # grid sizes are the cube of ns
    ns = np.array(ns)
    ns = ns**3
    # plot errors vs grid size, logarithmic
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Chamfer Error")
    ax.set_title("Chamfer Error vs Grid Size")
    ax.plot(ns, chamfer_errors[:,0], label="Batch Size = 0")
    ax.plot(ns, chamfer_errors[:,1], label="Batch Size = 1000")
    ax.plot(ns, chamfer_errors[:,2], label="Batch Size = 10000")
    # set ylim between 0.1 and 0.01
    ax.set_ylim([0.005, 0.2])
    ax.legend()
    fig.savefig(results_path + "chamfer-errors.png")
    # eps
    fig.savefig(results_path + "chamfer-errors.eps")
    # pdf
    fig.savefig(results_path + "chamfer-errors.pdf")

    # plot runtimes vs grid size, logarithmic
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime vs Grid Size")
    ax.plot(ns, runtimes[:,0], label="Batch Size = 0")
    ax.plot(ns, runtimes[:,1], label="Batch Size = 1000")
    ax.plot(ns, runtimes[:,2], label="Batch Size = 10000")
    ax.legend()
    fig.savefig(results_path + "runtimes.png")
    # eps
    fig.savefig(results_path + "runtimes.eps")
    # pdf
    fig.savefig(results_path + "runtimes.pdf")
