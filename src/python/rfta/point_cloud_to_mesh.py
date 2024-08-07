import numpy as np
import scipy as sp
import gpytoolbox as gpy
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
from rfta_bindings import _point_cloud_to_mesh_cpp_impl

import matplotlib.pyplot as plt


def point_cloud_to_mesh(P, N,
    screening_weight=10.,
    known_inside_pts=None, known_outside_pts=None,
    known_weight=None,
    depth=10,
    outer_boundary_type="Neumann",
    parallel=False,
    verbose=False):
    """Convert a point cloud to a polyline or triangle mesh.

    Parameters
    ----------
    P : (n_p,d) numpy array
        reconstructed point cloud points
    N : (n_p,d) numpy array
        reconstructed point cloud normals
    screening_weight : float, optional (default 1.)
        PSR screening weight
    known_inside_pts : (n_ki,d) numpy array, optional (default None)
        if known, help the surface reconstruction by specifyig points known
        to be on the inside.
    known_outside_pts : (n_ki,d) numpy array, optional (default None)
        if known, help the surface reconstruction by specifyig points known
        to be on the outside.
    known_weight : float, optional (default 100*(screening_weight+1))
        how heavily to weigh known inside/outside information
    depth : int, optional (default 8)
        PSR tree depth
    outer_boundary_type : string, optional (default "Neumann")
        The boundary condition to use for the outer boundary in the Poisson
        surface reconstruction
    parallel : bool, optional (default False)
        whether to parallelize the algorithm or not
    verbose : bool, optional (default False)
        whether to output details on the algorithm when it's running

    
    Returns
    -------
    V : (n,d) numpy array
        reconstructed vertices
    F : (m,d) numpy array
        reconstructed polyline / triangle mesh indices
    """

    dim = P.shape[1]
    assert dim==2 or dim==3, "Only dimensions 2 and 3 supported."

    assert depth>0, "Depth must be a positive integer."
    assert screening_weight>=0., "Screening weight must be a nonnegative scalar."

    # Max depths supported by PSR
    depth = min(12, depth)

    # Set defaults not set
    if known_weight is None:
        known_weight = 100. * (screening_weight + 1.)
    assert known_weight>=0., "Known weight must be a nonnegative scalar."

    # Make empty matrices for known points if not provided
    if known_inside_pts is None:
        known_inside_pts = np.array([], dtype=np.float64)
    if known_outside_pts is None:
        known_outside_pts = np.array([], dtype=np.float64)

    # TODO: We have C++ implementations for both double and float, do we want
    # to use this?
    V,F = _point_cloud_to_mesh_cpp_impl(P.astype(np.float64),
        N.astype(np.float64),
        screening_weight,
        known_inside_pts.astype(np.float64),
        known_outside_pts.astype(np.float64),
        known_weight,
        depth,
        outer_boundary_type,
        parallel, verbose)

    return V,F


