import numpy as np
import scipy as sp
import gpytoolbox as gpy
import time
import math

from .outside_points_from_rasterization import outside_points_from_rasterization
from .outside_points_from_sampling import outside_points_from_sampling
from .locally_make_feasible import locally_make_feasible

def sdf_to_point_cloud(U, S,
    rng_seed=3452,
    rasterization_resolution=None,
    n_local_searches=None,
    local_search_iters=20,
    batch_size=10000,
    num_rasterization_spheres=0,
    tol=1e-4,
    clamp_value=np.Inf,
    parallel=False,
    verbose=False,
    debug_Vgt=None,
    debug_Fgt=None):
    """Converts an SDF to a point cloud where all points are valid with respect
    to the spheres, and most spheres should have a tangent point.

    IMPORTANT: IF YOU CALL THIS, YOUR SDF POINTS MUST BE IN [0,1]^d

    Parameters
    ----------
    U : (n_sdf,d) numpy array
        points where the SDF is evaluated
    S : (n_sdf,) numpy array
        sdf(U)
    rng_seed : int, optional (default 3452)
        rng seed where random data is needed
    rasterization_resolution : int, optional (default 8*n_sdf**(1/d))
        the resolution of the rasterization grid
    n_local_searches : int, optional (default 2*n_sdf**(1/d))
        how many local searches to perform for each sphere in the locally make
        feasible step
    local_search_iters : int, optional (default 20)
        how many iterations to try for in the local search for each sphere in
        the locally make feasible step
    batch_size : int, optional (default 10000)
        how many points in one batch. Set to 0 to disable batching.
    num_rasterization_spheres : int, optional (default 0)
        how many spheres to use at most in the rasterization step.
        Set to zero to use all spheres.
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

    d = U.shape[1]
    assert d==2 or d==3, "Only dimensions 2 and 3 supported."

    n_sdf = U.shape[0]

    # Pick default values if not supplied.
    if rasterization_resolution is None:
        rasterization_resolution = 64 * math.ceil(n_sdf**(1./d)/16.)
    if n_local_searches is None:
        n_local_searches = math.ceil(2. * n_sdf**(1./d))

    # RNG used to compute random numbers and new seeds during the method.
    rng = np.random.default_rng(seed=rng_seed)
    seed = lambda : rng.integers(0,np.iinfo(np.int32).max)

    if rasterization_resolution>0:
        if verbose:
            print(f"  Rasterization...")
            t0_rasterization = time.time()

        P = outside_points_from_rasterization(U, S,
            rng_seed=seed(),
            res=rasterization_resolution, num_spheres=num_rasterization_spheres,
            tol=tol, parallel=parallel, verbose=verbose)

        if verbose:
            print(f"  Rasterization took {time.time()-t0_rasterization}s")
            print("")
    else:
        if verbose:
            print(f"  Metropolis-Hastings sampling...")
            t0_sampling = time.time()

        P = outside_points_from_sampling(U, S,
            rng_seed=seed(),
            n_samples=100*U.shape[0],
            tol=tol, parallel=parallel, verbose=verbose)

        if verbose:
            print(f"  Metropolis-Hastings took {time.time()-t0_sampling}s")
            print("")

    # If we found no points at all, return empty arrays here.
    if P.size == 0:
        return np.array([], dtype=np.float64), \
            np.array([], dtype=np.float64), \
            np.array([], dtype=np.int32)

    if verbose:
        print(f"  Locally make feasible...")
        t0_locally_make_feasible = time.time()

    P, N, f = locally_make_feasible(U, S, P,
        rng_seed=seed(),
        n_local_searches=n_local_searches,
        local_search_iters=local_search_iters,
        batch_size=batch_size,
        tol=tol, clamp_value=clamp_value,
        parallel=parallel, verbose=verbose)

    if verbose:
        print(f"  Locally make feasible took {time.time()-t0_locally_make_feasible}s")
        print("")

    return P, N, f

