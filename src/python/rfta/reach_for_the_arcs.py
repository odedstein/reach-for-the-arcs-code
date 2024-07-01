import numpy as np
import scipy as sp
import gpytoolbox as gpy
import time
import math

from .sdf_to_point_cloud import sdf_to_point_cloud
from .fine_tune_point_cloud import fine_tune_point_cloud
from .point_cloud_to_mesh import point_cloud_to_mesh

def reach_for_the_arcs(U, S,
    rng_seed=3452,
    return_point_cloud=False,
    fine_tune_iters=10,
    batch_size=10000,
    num_rasterization_spheres=0,
    screening_weight=10.,
    rasterization_resolution=None,
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
    """Perform the Reach for the Arcs algorithm to reconstruct a surface from
    an SDF.
    This works for polylines in 2D or triangle meshes in 3D.

    Parameters
    ----------
    U : (n_sdf,d) numpy array
        points where the SDF is evaluated
    S : (n_sdf,) numpy array
        sdf(U)
    rng_seed : int, optional (default 3452)
        rng seed where random data is needed
    return_point_cloud : bool, optional (default False)
        return the reconstructed point cloud normals or not
    fine_tune_iters : int, optional (default 5)
        how may iterations to do in the fine tuning step
    batch_size : int, optional (default 1000)
        how many points in one batch. Set to 0 to disable batching.
    num_rasterization_spheres : int, optional (default 0)
        how many spheres to use at most in the rasterization step.
        Set to zero to use all spheres.
    screening_weight : float, optional (default 0.1)
        PSR screening weight
    rasterization_resolution : int, optional (default 8*n_sdf**(1/d))
        the resolution of the rasterization grid
    max_points_per_sphere : int, optional (default 3)
        How many points should there be at most per sphere.
        Set it to 1 to not add any additional points to any sphere when
        fine-tuning.
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
    V : (n,d) numpy array
        reconstructed vertices
    F : (m,d) numpy array
        reconstructed polyline / triangle mesh indices
    P : (n_p,d) numpy array, if requested
        reconstructed point cloud points
    N : (n_p,d) numpy array, if requested
        reconstructed point cloud normals
    """

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

    # buio = np.arange(U.shape[0])
    # rng.shuffle(buio)
    # U = U[buio,:]
    # S = S[buio]

    # Resize the SDF points in U and the SDF samples in S so it's in [0,1]^d
    trans = np.min(U, axis=0)
    U = U - trans[None,:]
    scale = np.max(U)
    U /= scale
    S = S/scale
    clamp_value = clamp_value/scale
    if debug_Vgt is not None:
        debug_Vgt = debug_Vgt - trans[None,:]
        debug_Vgt /= scale


    if verbose:
        print(f" --- Starting Reach for the Arcs --- ")
        t0_total = time.time()

    if verbose:
        print(f"SDF to point cloud...")
        t0_sdf_to_point_cloud = time.time()

    # Split the SDF into positive and negative spheres?
    separate_inside_outside = True
    if separate_inside_outside:
        neg = S<0
        pos = np.logical_not(neg)
        pos,neg = np.nonzero(pos)[0], np.nonzero(neg)[0]
        U_pos, U_neg = U[pos,:], U[neg,:]
        S_pos, S_neg = S[pos], S[neg]
        if pos.size > 0:
            if verbose:
                print(f"  positive spheres")
            P_pos,N_pos,f_pos = sdf_to_point_cloud(U_pos, S_pos,
                rng_seed=seed(),
                rasterization_resolution=rasterization_resolution,
                n_local_searches=n_local_searches,
                local_search_iters=local_search_iters,
                batch_size=batch_size,
                num_rasterization_spheres=num_rasterization_spheres,
                tol=tol, clamp_value=clamp_value,
                parallel=parallel, verbose=verbose,
                debug_Vgt=debug_Vgt, debug_Fgt=debug_Fgt)
        else:
            P_pos,N_pos,f_pos = None,None,None
        if neg.size>0:
            if verbose:
                print(f"  negative spheres")
            P_neg,N_neg,f_neg = sdf_to_point_cloud(U_neg, S_neg,
                rng_seed=seed(),
                rasterization_resolution=rasterization_resolution,
                n_local_searches=n_local_searches,
                local_search_iters=local_search_iters,
                batch_size=batch_size,
                num_rasterization_spheres=num_rasterization_spheres,
                tol=tol, clamp_value=clamp_value,
                parallel=parallel, verbose=verbose,
                debug_Vgt=debug_Vgt, debug_Fgt=debug_Fgt)
        else:
            P_neg,N_neg,f_neg = None,None,None

        if P_pos is None or P_pos.size==0:
            P,N,f = P_neg,N_neg,neg[f_neg]
        elif P_neg is None or P_neg.size==0:
            P,N,f = P_pos,N_pos,pos[f_pos]
        else:
            P = np.concatenate((P_pos, P_neg), axis=0)
            N = np.concatenate((N_pos, N_neg), axis=0)
            f = np.concatenate((pos[f_pos], neg[f_neg]), axis=0)
    else:
        P,N,f = sdf_to_point_cloud(U, S,
            rng_seed=seed(),
            rasterization_resolution=rasterization_resolution,
            n_local_searches=n_local_searches,
            local_search_iters=local_search_iters,
            batch_size=batch_size,
            tol=tol, clamp_value=clamp_value,
            parallel=parallel, verbose=verbose,
            debug_Vgt=debug_Vgt, debug_Fgt=debug_Fgt)

    if P is None or P.size==0:
        if verbose:
            print(f"Unable to find any point cloud point.")
        if return_point_cloud:
            return V, F, P, N
        else:
            return V, F

    if verbose:
        print(f"SDF to point cloud took {time.time()-t0_sdf_to_point_cloud}s")
        print("")

    if verbose:
        print(f"Fine tuning point cloud...")
        t0_fine_tune = time.time()

    P,N,f = fine_tune_point_cloud(U, S, P, N, f,
        rng_seed=seed(),
        fine_tune_iters=fine_tune_iters,
        batch_size=batch_size,
        screening_weight=screening_weight,
        max_points_per_sphere=max_points_per_sphere,
        n_local_searches=n_local_searches,
        local_search_iters=local_search_iters,
        local_search_t=local_search_t,
        tol=tol,clamp_value=clamp_value,
        parallel=parallel, verbose=verbose,
        debug_Vgt=debug_Vgt, debug_Fgt=debug_Fgt)

    if verbose:
        print(f"Fine tuning point cloud took {time.time()-t0_fine_tune}s")
        print("")

    if verbose:
        print(f"Converting point cloud to mesh...")
        t0_point_cloud_to_mesh = time.time()

    V,F = point_cloud_to_mesh(P, N,
        screening_weight=screening_weight,
        outer_boundary_type="Neumann",
        parallel=False, verbose=False) # disabled parallelization here because it ocassionally crashes (Misha's version)

    if V is None or V.size==0:
        if verbose:
            print(f"Reconstructing point cloud failed.")
        if return_point_cloud: 
            if P is None or P.size==0:
                return V, F, P, N
            else:
                return V, F, scale*P+trans[None,:], N
        else:
            return V, F

    if verbose:
        print(f"Converting point cloud to mesh took {time.time()-t0_point_cloud_to_mesh}s")
        print("")

    if verbose:
        print(f" --- Finished Reach for the Arcs --- ")
        print(f"Total elapsed time: {time.time()-t0_total}s")

    # Undo recentering
    V = scale*V + trans[None,:]
    P = scale*P + trans[None,:]

    if return_point_cloud:
        return V, F, P, N
    else:
        return V, F
