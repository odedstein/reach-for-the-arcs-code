import numpy as np
import scipy as sp
import gpytoolbox as gpy
import platform, os, sys
import math
os_name = platform.system()
if os_name == "Darwin":
    # Get the macOS version
    os_version = platform.mac_ver()[0]
    # print("macOS version:", os_version)

    # Check if the macOS version is less than 14
    if os_version and os_version < "14":
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build-studio')))
    else:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build')))
elif os_name == "Windows":
    # For Windows systems
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build/Debug')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build/Release')))
else:
    # For other systems
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build')))
from rfta_bindings import _fine_tune_point_cloud_iter_cpp_impl

from .point_cloud_to_mesh import point_cloud_to_mesh
from .debug_utility import *


def fine_tune_point_cloud(U, S, P, N, f,
    rng_seed=3452,
    fine_tune_iters=10,
    batch_size=10000,
    screening_weight=10.,
    max_points_per_sphere=3,
    n_local_searches=None,
    local_search_iters=20,
    local_search_t=0.01,
    tol=1e-4,
    clamp_value=np.Inf,
    parallel=False,
    verbose=False,
    debug_Vgt=None,
    debug_Fgt=None):
    """Improve the point cloud with respect to the SDF such that the
    reconstructed surface will fulfill all sphere conditions
    
    IMPORTANT: IF YOU CALL THIS, YOUR SDF POINTS MUST BE IN [0,1]^d

    Parameters
    ----------
    U : (n_sdf,d) numpy array
        points where the SDF is evaluated
    S : (n_sdf,) numpy array
        sdf(U)
    P : (n_p,d) numpy array
        reconstructed point cloud points
    N : (n_p,d) numpy array
        reconstructed point cloud normals
    f : (n_p,) numpy array
        feasible indices that map P back to U
    rng_seed : int, optional (default 3452)
        rng seed where random data is needed
    fine_tune_iters : int, optional (default False)
        how may iterations to fine tune for
    batch_size : int, optional (default 1000)
        how many points in one batch. Set to 0 to disable batching.
    screening_weight : float, optional (default 0.1)
        PSR screening weight
    max_points_per_sphere : int, optional (default 3)
        How many points should there be at most per sphere.
        Set it to 1 to not add any additional points to any sphere.
        Has to be at least 1 (this is a theoretical minimum).
        If set to larger than 1, spheres that the reconstructed surface
        intersects will have at most max_points_per_sphere added.
    n_local_searches : int, optional (default 2*n_sdf**(1/d))
        how many local searches to perform for each sphere in the locally make
        feasible step
    local_search_iters : int, optional (default 20)
        how many iterations to try for in the local search for each sphere in
        the locally make feasible step
    local_search_t : double, optional (default 5e-3)
        how far to move each point in the local search for each point in the
        locally make feasible step
    tol : float, optional (default 1e-4)
        tolerance for determining whether a point is inside a sphere
    parallel : bool, optional (default False)
        whether to parallelize the algorithm or not
    verbose : bool, optional (default False)
        whether to output details on the algorithm when it's running

    
    Returns
    -------
    P : (n_p,d) numpy array
        reconstructed point cloud points
    N : (n_p,d) numpy array
        reconstructed point cloud normals
    f : (n_p,) numpy array
        feasible indices that map P back to U
    """

    d = U.shape[1]
    assert d==2 or d==3, "Only dimensions 2 and 3 supported."
    assert max_points_per_sphere>=1, "There has to be at least one point per sphere."

    n_sdf = U.shape[0]

    # Pick default values if not supplied.
    if n_local_searches is None:
        n_local_searches = math.ceil(2. * n_sdf**(1./d))

    # RNG used to compute random numbers and new seeds during the method.
    rng = np.random.default_rng(seed=rng_seed)
    seed = lambda : rng.integers(0,np.iinfo(np.int32).max)

    if verbose:
        print(f"  Fine tune called with {f.size} / {U.shape[0]} feasible points.")

    for it in range(fine_tune_iters):
        #Generate a random batch of size batch size.
        if batch_size > 0 and batch_size < n_sdf:
            batch = rng.choice(n_sdf, batch_size)
        else:
            batch = np.arange(n_sdf)
            rng.shuffle(batch)

        V,F = point_cloud_to_mesh(P, N,
            screening_weight=screening_weight,
            outer_boundary_type="Neumann",
            parallel=False, verbose=False)

        if(V.size == 0):
            if(verbose):
                print(f"    point_cloud_to_mesh did not produce a mesh.")
            return P, N, f

        P, N, f = fine_tune_point_cloud_iteration(U,
            S,
            V,
            F,
            P,
            N,
            f,
            batch,
            max_points_per_sphere,
            seed(),
            n_local_searches,
            local_search_iters,
            local_search_t,
            tol, clamp_value, parallel, verbose)

        if verbose:
            if debug_Vgt is not None and debug_Fgt is not None:
                print(f"  After fine tuning iter {it}, we have {f.size} points. (Current chamfer error = {chamfer(V, F, debug_Vgt, debug_Fgt)})")
            else:
                print(f"  After fine tuning iter {it}, we have {f.size} points.")
    return P, N, f


def fine_tune_point_cloud_iteration(U, S, 
    V, F,
    P, N, f,
    batch,
    max_points_per_sphere,
    rng_seed,
    n_local_searches,
    local_search_iters,
    local_search_t,
    tol, clamp_value,
    parallel,
    verbose):
    """Even if you batch, please pass the entirety of of U, S to this function.
    """

    P, N, f = _fine_tune_point_cloud_iter_cpp_impl(U.astype(np.float64),
            S.astype(np.float64),
            V.astype(np.float64),
            F.astype(np.int32),
            P.astype(np.float64),
            N.astype(np.float64),
            f.astype(np.int32),
            batch.astype(np.int32),
            max_points_per_sphere,
            rng_seed,
            n_local_searches,
            local_search_iters,
            local_search_t,
            tol, clamp_value, parallel, verbose)

    return P, N, f


