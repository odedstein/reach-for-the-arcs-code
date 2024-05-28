import numpy as np
import scipy as sp
import gpytoolbox as gpy
import time
import platform, os, sys
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
else:
    # For non-macOS systems
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build')))
import math
from rfta_bindings import _locally_make_feasible_cpp_impl

def locally_make_feasible(U, S, P,
    rng_seed=3452,
    n_local_searches=None,
    local_search_iters=20,
    batch_size=10000,
    tol=1e-4,
    clamp_value=np.Inf,
    parallel=False,
    verbose=False):
    """Given a number of SDF samples and points, tries to make each point
    feasible, and returns a list of feasible points at the end.

    IMPORTANT: IF YOU CALL THIS, YOUR SDF POINTS MUST BE IN [0,1]^d

    Parameters
    ----------
    U : (n_sdf,d) numpy array
        points where the SDF is evaluated
    S : (n_sdf,) numpy array
        sdf(U)
    P : (n_p,d) numpy array
        sampled points outside the spheres
    rng_seed : int, optional (default 3452)
        rng seed where random data is needed
    n_local_searches : int, optional (default 2*n_sdf**(1/d))
        how many local searches to perform for each sphere in the locally make
        feasible step
    local_search_iters : int, optional (default 20)
        how many iterations to try for in the local search for each sphere
    batch_size : int, optional (default 1000)
        how many points in one batch. Set to 0 to disable batching.
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

    assert np.min(U)>=0. and np.max(U)<=1.
    assert P.size > 0, "There needs to be at least one point outside the spheres."

    d = U.shape[1]
    assert d==2 or d==3, "Only dimensions 2 and 3 supported."

    n_sdf = U.shape[0]

    # Pick default values if not supplied.
    if n_local_searches is None:
        n_local_searches = math.ceil(2. * n_sdf**(1./d))

    # RNG used to compute random numbers and new seeds during the method.
    rng = np.random.default_rng(seed=rng_seed)
    seed = lambda : rng.integers(0,np.iinfo(np.int32).max)

    # Batching
    if batch_size > 0 and batch_size < n_sdf:
        batch = rng.choice(n_sdf, batch_size)
    else:
        batch = np.arange(n_sdf)
        rng.shuffle(batch)

    P, N, f = _locally_make_feasible_cpp_impl(U.astype(np.float64),
        S.astype(np.float64), P.astype(np.float64),
        batch.astype(np.int32),
        seed(), 
        n_local_searches, local_search_iters,
        tol, clamp_value, parallel, verbose)

    if verbose:
        print(f"    {f.size} / {U.shape[0]} points are feasible.")

    return P, N, f


