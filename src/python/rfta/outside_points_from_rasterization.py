import numpy as np
import scipy as sp
import gpytoolbox as gpy
import time
import sys, os
import math

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
elif os_name == "Windows":
    # For Windows systems
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build/Debug')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build/Release')))
else:
    # For other systems
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build')))
from rfta_bindings import _outside_points_from_rasterization_cpp_impl

def outside_points_from_rasterization(U, S,
    rng_seed=3452,
    res=None,
    num_spheres=0,
    tol=1e-4,
    narrow_band=True,
    parallel=False,
    force_cpu=False,
    verbose=False):
    """Converts an SDF to a point cloud where all points should be outside our
    spheres, using rasterization.

    IMPORTANT: IF YOU CALL THIS, YOUR SDF POINTS MUST BE IN [0,1]^d

    Parameters
    ----------
    U : (n_sdf,d) numpy array
        points where the SDF is evaluated
    S : (n_sdf,) numpy array
        sdf(U)
    rng_seed : int, optional (default 3452)
        rng seed where random data is needed
    res : int, optional (default 8*n_sdf**(1/d))
        the resolution of the rasterization grid
    num_spheres : int, optional (default 0)
        how many spheres to use at most in the rasterization step.
        Set to zero to use all spheres.
    tol : float, optional (default 1e-4)
        tolerance for determining whether a point is inside a sphere
    narrow_band : bool, optional (default True)
        return points only in a narrow band around the spheres
    parallel : bool, optional (default False)
        whether to parallelize the algorithm or not
    force_cpu : bool, optional (default False)
        By default, the code chooses whether it wants to use the GPU or CPU
        based on heuristics. Set this to true to force CPU.
    verbose : bool, optional (default False)
        whether to output details on the algorithm when it's running

    
    Returns
    -------
    P : (n_p,d) numpy array
        outside points
    """

    assert np.min(U)>=0. and np.max(U)<=1.

    d = U.shape[1]
    assert d==2 or d==3, "Only dimensions 2 and 3 supported."

    n_sdf = U.shape[0]

    # Pick default values if not supplied.
    if res is None:
        res = 64 * math.ceil(n_sdf**(1./d)/16.)
    # Maximum resolution so your GPU does not run out of memory.
    res = min(1024, res)

    assert res>=2, "Grid must have at least resolution 2."

    # Make sure res is divisible by 2.
    if res%2 != 0:
        res += 1

    if num_spheres > 0 and num_spheres < n_sdf:
        # Sample random indices to U and S
        rng = np.random.default_rng(rng_seed)
        # print(f"Sampling {num_spheres} random spheres from {n_sdf} sdf points.")
        indices = rng.choice(n_sdf, num_spheres, replace=False)
        U = U[indices,:]
        S = S[indices]

    P = _outside_points_from_rasterization_cpp_impl(U.astype(np.float64),
        np.abs(S).astype(np.float64),
        rng_seed, res, tol,
        narrow_band,
        parallel,
        force_cpu,
        verbose)

    return P
