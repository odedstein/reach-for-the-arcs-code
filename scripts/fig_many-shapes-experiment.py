from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps
import time
import csv
from multiprocessing import Process, Queue, TimeoutError
from collections import defaultdict
import pandas as pd
import seaborn as sns

global path_to_all_meshes, resolutions, methods, metrics, path_to_results_csv, path_to_zero_genus_results_csv, path_to_above_zero_genus_results_csv, error_sums, shape_counts, timing_sums, error_sums_genus, shape_counts_genus, timing_sums_genus, selected_mesh_files, num_shapes, timeout_duration

parser = argparse.ArgumentParser(description='Large quantitative experiment.')
parser.add_argument('--num_shapes', type=int, default=100, help='number of shapes to use')
parser.add_argument('--run', action=argparse.BooleanOptionalAction)
parser.set_defaults(run=False)
parser.add_argument('--reload', action=argparse.BooleanOptionalAction)
parser.set_defaults(reload=False)
parser.add_argument('--metrics', action=argparse.BooleanOptionalAction)
parser.set_defaults(metrics=False)
parser.add_argument('--table', action=argparse.BooleanOptionalAction)
parser.set_defaults(table=False)
parser.add_argument('--plot', action=argparse.BooleanOptionalAction)
parser.set_defaults(plot=False)
args = parser.parse_args()


# number of shapes
num_shapes = 100

# rng
rng_seed = 1
np.random.seed(rng_seed)
random.seed(rng_seed)

# methods
# methods = [ "mc", "ndc", "rfts", "rfta" ]
methods = [ "mc", "rfts", "rfta" ]
# methods = [ "ndc" ] #separately because this only works on my laptop

# metrics
metrics = [ "chf", "hd", "sdf" ]

# path to meshes
# some usual test meshes
# path_to_all_meshes = "data/many-shapes-experiment/"
# thingi10k
path_to_all_meshes = "../../shapes/thingi10k-tetwilded/"
# Get all mesh files
all_mesh_files = os.listdir(path_to_all_meshes)
# Shuffle the list
random.shuffle(all_mesh_files)


if args.reload:
    selected_mesh_files = all_mesh_files[:num_shapes]
else:
    # otherwise, just get a list of all the folders of the type "results/many-shapes-experiment/{mesh}"
    selected_mesh_files = os.listdir("results/many-shapes-experiment")
    # only the ones that are subdirectories
    selected_mesh_files = [mesh for mesh in selected_mesh_files if os.path.isdir(f"results/many-shapes-experiment/{mesh}")]

# print(selected_mesh_files)
# print list of selected meshes
# print(sorted(selected_mesh_files))
    
path_to_results = "results/many-shapes-experiment/"


# path to results csv
path_to_results_csv = "results/many-shapes-experiment/results.csv"
path_to_avg_results_csv = "results/many-shapes-experiment/avg_results.csv"
path_to_zero_genus_results_csv = "results/many-shapes-experiment/zero_genus_results.csv"
path_to_above_zero_genus_results_csv = "results/many-shapes-experiment/above_zero_genus_results.csv"
path_to_above_threshold_genus_results_csv = "results/many-shapes-experiment/above_threshold_genus_results.csv"
genus_threshold = 10.0

# resolutions
resolutions = [ 6, 10, 20, 30, 40, 50, 60, 80, 100 ]
# resolutions = [ 6, 40, 60 ]

# timeout duration
timeout_duration = 1000

# now loop over every mesh, every resolution, every method, and add row with error metrics to a CSV file
# CSV file will have columns: mesh, resolution, method, metric, error

def is_mesh_broken(V, F):
    return np.any(np.isnan(V)) or np.any(np.isnan(F)) or np.any(np.isinf(V)) or np.any(np.isinf(F)) or np.any(np.isneginf(V)) or np.any(np.isneginf(F)) or F.shape[0] == 0 or V.shape[0] == 0 or F.shape[1] != 3 or V.shape[1] != 3

# auxiliary function to run method
def run_method(U, S, method, savepath, rot_matrix, V_gt, F_gt, queue):
    try:
        if method == "mc":
            n = int(np.round(U.shape[0]**(1/3)))
            V, F = gpy.marching_cubes(S, U, n, n, n)
        elif method == "ndc":
            n = int(np.round(U.shape[0]**(1/3)))
            V, F = utility.ndc(V_gt, F_gt, n)
        elif method == "rfta":
            V, F = rfta.reach_for_the_arcs(U, S, parallel=True, max_points_per_sphere=30, fine_tune_iters=20, rng_seed=rng_seed)
        elif method == "rfts":
            # Choose an initial surface for reach_for_the_spheres
            V0, F0 = gpy.icosphere(2)
            # Reconstruct triangle mesh
            V, F = gpy.reach_for_the_spheres(U, None, V0, F0, S=S, verbose=False)
        else:
            queue.put(f"Method {method} not found")
            raise RuntimeError(f"Unknown method {method}")      
        if is_mesh_broken(V, F):
            queue.put(f"Broken or invalid mesh detected")
            raise RuntimeError(f"Broken or invalid mesh detected")
        if savepath is not None:
            gpy.write_mesh(savepath, V @ np.linalg.inv(rot_matrix), F)
    except:
        queue.put(f"Method {method} failed")
        raise RuntimeError(f"Method {method} failed")
    
def our_energy(U,S,v,f):
    d2, I, b = gpy.squared_distance(U, v, f, use_cpp=True, use_aabb=True)
    d = np.sqrt(d2)
    g = np.abs(S)-d
    return 1000*np.sum(g**2.0)/U.shape[0]

# auxiliary function to compute metrics
def compute_error(V_gt, F_gt, V, F, U, S, R, metric):
    if metric == "chf":
        return utility.chamfer(V_gt, F_gt, V, F)
    elif metric == "hd":
        return gpy.approximate_hausdorff_distance(V_gt, F_gt, V, F)
    elif metric == "sdf":
        return our_energy(U,S,V @ R,F)
    else:
        assert False, "unknown metric"

# auxiliary function to compute SDF data
def get_sdf_data(V_gt, F_gt, resolution):
    # Create and abstract SDF function that is the only connection to the shape
    sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

    # Set up a grid
    n = resolution
    gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
    U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    S = sdf(U)
    return U, S

def calculate_genus(V, F):
    # Count vertices
    num_vertices = V.shape[0]

    # Count faces
    num_faces = F.shape[0]

    # Construct a set to count unique edges
    edges = set()
    for face in F:
        # Add each edge of the triangle to the set
        # Sorting each tuple to avoid duplicating edges in reverse order
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))

    # Count edges
    num_edges = len(edges)

    # Calculate genus
    genus = 1 - (num_vertices - num_edges + num_faces) / 2
    return genus



def rotation_matrix(axis, angle):
    """
    Create a rotation matrix corresponding to the rotation around a general axis by a specified angle.

    R = dd^T + cos(theta)*(I - dd^T) + sin(theta)*skew(d)

    Parameters:
    axis : array
        Axis around which to rotate.
    angle : float
        Angle, in radians, by which to rotate.

    Returns:
    numpy.ndarray
        A rotation matrix.
    """

    # Ensure the axis is a unit vector
    axis = axis / np.linalg.norm(axis)

    # Components of the axis vector
    x, y, z = axis

    # Construct the skew-symmetric matrix
    skew_sym = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    # Identity matrix
    I = np.eye(3)

    # Outer product of the axis vector with itself
    outer = np.outer(axis, axis)

    # Rotation matrix
    R = outer + np.cos(angle) * (I - outer) + np.sin(angle) * skew_sym

    return R

def generate_latex_table(csv_file, methods, resolutions, metrics):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Filter the DataFrame based on user specifications
    df_filtered = df[df['method'].isin(methods) & df['resolution'].isin(resolutions) & df['metric'].isin(metrics)]

    # Pivot the DataFrame to get the desired format
    df_pivot = df_filtered.pivot_table(index='resolution', columns=['metric', 'method'], values='error')

    # Reorder columns based on methods_order
    new_columns = [(metric, method) for metric in metrics for method in methods if (metric, method) in df_pivot.columns]
    df_pivot = df_pivot.reindex(columns=new_columns)

    # Generate the LaTeX table
    latex_table = df_pivot.to_latex()

    return latex_table

def generate_combined_latex_table(csv_files, methods_order, resolutions, metrics, captions, method_names, metric_names):
    # Table format with vertical lines
    # table_format = "l" + "|c" * (len(methods_order) * len(metrics))
    table_format = "c" + "".join("|" + "c" * len(methods_order) for _ in metrics)

    # Initialize the header of the LaTeX table
    header_row = "Grid & " + " & ".join(f"{metric} {method}" for metric in metric_names for method in method_names) + " \\\\\n"
    
    # Initialize the LaTeX table with the header and the table format
    latex_table = "\\begin{tabular}{" + table_format + "}\n"
    latex_table += "\\toprule\n"
    latex_table += "\\rowcolor{tableheader}\n"
    latex_table += header_row
    latex_table += "\\midrule\n"
    
    # Process each CSV file
    for csv_file, caption in zip(csv_files, captions):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Filter by specified resolutions and metrics
        df_filtered = df[df['resolution'].isin(resolutions) & df['metric'].isin(metrics)]
        
        # Pivot the DataFrame to have methods as columns
        df_pivot = df_filtered.pivot(index='resolution', columns=['method', 'metric'], values='error')
        
        # Reorder the columns based on the specified metric order and methods
        method_metric_tuples = [(method, metric) for metric in metrics for method in methods_order]
        df_pivot = df_pivot.reindex(columns=method_metric_tuples)

        # Find the smallest value for each metric and resolution
        min_values = {}
        for metric in metrics:
            for resolution in resolutions:
                min_values[(resolution, metric)] = df_pivot.xs(metric, level='metric', axis=1).loc[resolution].min()

        # Add the section caption
        latex_table += "\\rowcolor{tablesubheader}\n"
        latex_table += "\\multicolumn{" + str(len(methods_order) * len(metrics) + 1) + "}{c}{" + caption + "} \\\\\n"
        latex_table += "\\midrule\n"
        
        # Add the table content with alternating row colors
        for row_index, (index, row) in enumerate(df_pivot.iterrows()):
            row_color = "\\rowcolor[HTML]{EFEFEF}" if row_index % 2 == 1 else ""
            row_values = [f"${index}^3$"]
            for metric in metrics:
                for method in methods_order:
                    value = row[(method, metric)]
                    if pd.notna(value) and value == min_values[(index, metric)]:
                        row_values.append(f"\\textbf{{{value:.4f}}}")
                    else:
                        row_values.append(f"{value:.4f}" if pd.notna(value) else "-")
            latex_table += row_color + " & ".join(map(str, row_values)) + " \\\\\n"
        
        # Add a midrule after each section
        latex_table += "\\midrule\n"

    # Finalize the tabular environment
    # latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}"

    return latex_table

# Function to create scatter plots with logarithmic axes
def create_log_scatter_plots(data, metrics):
    sns.set(style="whitegrid")

    colors = {
        "rfts": (51.0/255.0, 160.0/255.0, 44.0/255.0, 1.0),         # green_2
        "rfta": (31.0/255.0, 120.0/255.0, 180.0/255.0, 1.0),        # blue_2
        "mc": (255.0/255.0, 127.0/255.0, 0.0/255.0, 1.0),           # orange_2
        "ndc": (106.0/255.0, 61.0/255.0, 154.0/255.0, 1.0),         # purple_2
        # Add more methods and their corresponding colors here
    }

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        metric_data = data[data['metric'] == metric]
        metric_data['resolution'] = np.power(metric_data['resolution'], 3)
        
        # Creating scatter plot with logarithmic x and y axes
        ax = sns.scatterplot(x='resolution', y='error', hue='method', data=metric_data, palette=colors, alpha=0.1, size=0.01)
        methods = metric_data['method'].unique()
        for method in methods:
            method_data = metric_data[metric_data['method'] == method]
            average_data = method_data.groupby('resolution')['error'].mean().reset_index()
            # this color
            color = colors[method]
            sns.lineplot(x='resolution', y='error', data=average_data, ax=ax, label=f'{method} Average', marker='o', color=color, markersize=10, zorder=5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.title(f'Logarithmic Scatter Plot of Error vs Resolution for {metric.upper()} Metric')
        plt.xlabel('Resolution (log scale)')
        plt.ylabel('Error (log scale)')
        plt.legend(title='Method')
        plt.savefig(path_to_results + f'{metric}_log.png', bbox_inches='tight')
        plt.savefig(path_to_results + f'{metric}_log.eps', format='eps', bbox_inches='tight')



def create_separate_plots(data):
    sns.set(style="whitegrid")

    # Filtering data for Chamfer and Hausdorff metrics
    chamfer_data = data[data['metric'] == 'chf'][['mesh', 'resolution', 'method', 'error']]
    hausdorff_data = data[data['metric'] == 'hd'][['mesh', 'resolution', 'method', 'error']]

    # Merging data on mesh, resolution, and method to match corresponding values
    merged_data = pd.merge(chamfer_data, hausdorff_data, on=['mesh', 'resolution', 'method'])
    merged_data.columns = ['mesh', 'resolution', 'method', 'chamfer_error', 'hausdorff_error']

    # Getting unique resolutions
    resolutions = merged_data['resolution'].unique()
    colors = {
        "rfts": (51.0/255.0, 160.0/255.0, 44.0/255.0, 1.0),         # green_2
        "rfta": (31.0/255.0, 120.0/255.0, 180.0/255.0, 1.0),        # blue_2
        "mc": (255.0/255.0, 127.0/255.0, 0.0/255.0, 1.0),           # orange_2
        "ndc": (106.0/255.0, 61.0/255.0, 154.0/255.0, 1.0),         # purple_2
        # Add more methods and their corresponding colors here
    }
    # Creating a scatter plot for each resolution
    for resolution in resolutions:
        plt.figure(figsize=(10, 6))
        res_data = merged_data[merged_data['resolution'] == resolution]
        # specify colors for each method

        sns.scatterplot(x='chamfer_error', y='hausdorff_error', hue='method', data=res_data, palette=colors, alpha=0.5)
        method_groups = res_data.groupby('method')
        for method, group in method_groups:
            avg_chamfer = np.mean(group['chamfer_error'])
            avg_hausdorff = np.mean(group['hausdorff_error'])
            plt.scatter(avg_chamfer, avg_hausdorff, color=colors[method], edgecolor='black', 
                        s=100, label=f'{method} Average', zorder=5)
        plt.xscale('log')
        plt.yscale('log')
        plt.axis('equal')
        plt.title(f'Scatter Plot of Chamfer Error vs Hausdorff Error for Resolution {resolution}')
        plt.xlabel('Chamfer Error (log scale)')
        plt.ylabel('Hausdorff Error (log scale)')
        plt.legend(title='Method')
        # axis equal
        
        plt.savefig(path_to_results + f'resolution_{resolution}.png', bbox_inches='tight')
        plt.savefig(path_to_results + f'resolution_{resolution}.eps', format='eps', bbox_inches='tight')




def main():

    if args.run:
        for mesh in sorted(selected_mesh_files):
            mesh_name = mesh.split(".")[0]
            print(f"mesh = {mesh_name}")
            mesh_results_path = f"results/many-shapes-experiment/{mesh_name}"
            
            
            if args.reload:
                
                filename = path_to_all_meshes + mesh
                try:
                    V_gt, F_gt = gpy.read_mesh(filename)
                    assert not is_mesh_broken(V_gt, F_gt)
                except:
                    continue
                # normalize
                V_gt = gpy.normalize_points(V_gt)
                # rotate randomly
                axis = np.random.rand(3)
                axis = axis / np.linalg.norm(axis)
                angle = np.random.rand() * 2 * np.pi
                R = rotation_matrix(axis, angle)
                V_gt = V_gt @ R
                # make it so that the mesh results directory exists
                os.makedirs(mesh_results_path, exist_ok=True)
                # write ground truth mesh
                gpy.write_mesh(f"results/many-shapes-experiment/{mesh_name}/ground_truth.obj", V_gt @ np.linalg.inv(R), F_gt)
                # write rotation matrix
                np.save(f"results/many-shapes-experiment/{mesh_name}/rotation_matrix.npy", R)
            else:
                # read rotation matrix
                R = np.load(f"results/many-shapes-experiment/{mesh_name}/rotation_matrix.npy")
                # read ground truth mesh
                V_gt, F_gt = gpy.read_mesh(f"results/many-shapes-experiment/{mesh_name}/ground_truth.obj")
                # rotate
                V_gt = V_gt @ R

            
            for resolution in resolutions:
                os.makedirs(f"{mesh_results_path}/{resolution}", exist_ok=True)
                U, S = get_sdf_data(V_gt, F_gt, resolution)
                # save U and S as npy files
                np.save(f"{mesh_results_path}/{resolution}/U.npy", U)
                np.save(f"{mesh_results_path}/{resolution}/S.npy", S)
                for method in methods:
                    # run all methods in queue to avoid memory issues
                    queue = Queue()
                    p = Process(target=run_method, args=(U, S, method, f"{mesh_results_path}/{resolution}/{method}.obj", R, V_gt, F_gt, queue))
                    p.start()
                    p.join(timeout=timeout_duration)
                    if p.is_alive():
                        print(f"Method {method} on mesh {mesh_name} with resolution {resolution} timed out.")
                        p.terminate()
                        p.join()
                    else:
                        print(f"Method {method} on mesh {mesh_name} with resolution {resolution} finished successfully.")
    
    if args.metrics:
        # now loop over every mesh, every resolution, every method, and add row with error metrics to a CSV file
        # CSV file will have columns: mesh, resolution, method, metric, error
        with open(path_to_results_csv, mode='w') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(["mesh", "genus", "resolution", "method", "metric", "error"])
            # loop oever every subdirectiry of results/many-shapes-experiment
            for mesh_path in sorted(os.listdir("results/many-shapes-experiment")):
            # for mesh in sorted(selected_mesh_files):
                mesh_name = mesh_path.split("/")[-1]
                mesh = f"{mesh_name}.obj"
                # mesh_name = mesh.split(".")[0]
                print(f"mesh = {mesh_name}")
                ground_truth_path = f"results/many-shapes-experiment/{mesh_name}/ground_truth.obj"
                rotation_path = f"results/many-shapes-experiment/{mesh_name}/rotation_matrix.npy"
                try:
                    V_gt, F_gt = gpy.read_mesh(ground_truth_path)
                    R = np.load(rotation_path)
                except:
                    continue
                genus = calculate_genus(V_gt, F_gt)
                for resolution in resolutions:
                    U = np.load(f"results/many-shapes-experiment/{mesh_name}/{resolution}/U.npy")
                    S = np.load(f"results/many-shapes-experiment/{mesh_name}/{resolution}/S.npy")
                    all_errors = np.zeros((len(methods), len(metrics)))
                    some_method_failed = False
                    for (i, method) in enumerate(methods):
                        try:
                            reconstruction_path = f"results/many-shapes-experiment/{mesh_name}/{resolution}/{method}.obj"
                            V, F = gpy.read_mesh(reconstruction_path)
                            for (j,metric) in enumerate(metrics):
                                error = compute_error(V_gt, F_gt, V, F, U, S, R, metric)
                                all_errors[i,j] = error
                                print(f"mesh = {mesh_name}, resolution = {resolution}, method = {method}, metric = {metric}, error = {error}")
                        except:
                            some_method_failed = True
                            print(f"mesh = {mesh_name}, resolution = {resolution}, method = {method}, metric = {metric}, error = failed")
                            continue
                    if not some_method_failed:
                        for (i, method) in enumerate(methods):
                            for (j,metric) in enumerate(metrics):
                                results_writer.writerow([mesh_name, genus, resolution, method, metric, all_errors[i,j]])
            results_file.close()

        # now to calculate averages per method, resolution and metric (i.e., across all meshes)
        # CSV file will have columns: resolution, method, metric, error
        # not super effcient (we are reading the whole thing many times, but whatever, it's simpler and this is not a bottleneck)
        with open(path_to_avg_results_csv, mode='w') as avg_results_file, open(path_to_zero_genus_results_csv, mode='w') as zero_genus_results_file, open(path_to_above_zero_genus_results_csv, mode='w') as above_zero_genus_results_file, open(path_to_above_threshold_genus_results_csv, mode='w') as above_threshold_genus_results_file:
            avg_results_writer = csv.writer(avg_results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            avg_results_writer.writerow(["resolution", "method", "metric", "error"])
            zero_genus_results_writer = csv.writer(zero_genus_results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            zero_genus_results_writer.writerow(["resolution", "method", "metric", "error"])
            above_zero_genus_results_writer = csv.writer(above_zero_genus_results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            above_zero_genus_results_writer.writerow(["resolution", "method", "metric", "error"])
            above_threshold_genus_results_writer = csv.writer(above_threshold_genus_results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            above_threshold_genus_results_writer.writerow(["resolution", "method", "metric", "error"])
            for resolution in resolutions:
                for (i, method) in enumerate(methods):
                    for (j,metric) in enumerate(metrics):
                        # read all errors for this method, resolution and metric
                        errors = []
                        errors_genus_zero = []
                        errors_genus_above_zero = []
                        error_genus_above_threshold = []
                        with open(path_to_results_csv, mode='r') as results_file:
                            results_reader = csv.reader(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            for row in results_reader:
                                if row[2] == str(resolution) and row[3] == method and row[4] == metric:
                                    errors.append(float(row[5]))
                                    if float(row[1]) == 0.0:
                                        errors_genus_zero.append(float(row[5]))
                                    else:
                                        errors_genus_above_zero.append(float(row[5]))
                                    if float(row[1]) > genus_threshold:
                                        error_genus_above_threshold.append(float(row[5]))
                        avg_error = np.mean(errors)
                        avg_genus_zero_error = np.mean(errors_genus_zero)
                        avg_genus_above_zero_error = np.mean(errors_genus_above_zero)
                        avg_genus_above_threshold_error = np.mean(error_genus_above_threshold)
                        avg_results_writer.writerow([resolution, method, metric, avg_error])
                        zero_genus_results_writer.writerow([resolution, method, metric, avg_genus_zero_error])
                        above_zero_genus_results_writer.writerow([resolution, method, metric, avg_genus_above_zero_error])
                        above_threshold_genus_results_writer.writerow([resolution, method, metric, avg_genus_above_threshold_error])
            avg_results_file.close()
            zero_genus_results_file.close()
            above_zero_genus_results_file.close()
            above_threshold_genus_results_file.close()

    if args.table:
        captions = [                                    # Captions for each CSV 
                    "Average results over all test shapes",
                    "Average results over all test shapes with zero genus",
                    "Average results over all test shapes with genus over zero",
                    "Average results over all test shapes with genus over ten"
                ]
        method_names = [ "MC", "NDC", "RFTS", "RFTA"]
        metric_names = [ "Chf", "Hdf", "$\mathcal{E}_{SDF}$"]
        csv_files = [path_to_avg_results_csv, path_to_zero_genus_results_csv, path_to_above_zero_genus_results_csv, path_to_above_threshold_genus_results_csv]
        latex_table = generate_combined_latex_table(csv_files, methods, resolutions, metrics, captions, method_names, metric_names)
        print(latex_table)
        latex_table_path = "paper/sections/many-shapes-experiment.tex"
        with open(latex_table_path, 'w') as latex_table_file:
            latex_table_file.write(latex_table)
            latex_table_file.close()

    if args.plot:
        create_log_scatter_plots(pd.read_csv(path_to_results_csv), metrics)
        
        create_separate_plots(pd.read_csv(path_to_results_csv))

if __name__ == "__main__":
    main()