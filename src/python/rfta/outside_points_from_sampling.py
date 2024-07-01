import numpy as np
import scipy as sp
import gpytoolbox as gpy
import time
from .metropolis_hastings import metropolis_hastings

def outside_points_from_sampling(U, S,
    rng_seed=3452,
    n_samples=1000,
    tol=1e-4,
    parallel=False,
    verbose=False):
    """Converts an SDF to a point cloud where all points should be outside our
    spheres.

    IMPORTANT: IF YOU CALL THIS, YOUR SDF POINTS MUST BE IN [0,1]^d

    Parameters
    ----------
    U : (n_sdf,d) numpy array
        points where the SDF is evaluated
    S : (n_sdf,) numpy array
        sdf(U)
    rng_seed : int, optional (default 3452)
        rng seed where random data is needed
    n_samples : int, optional (default 1000)
        how many volumetric samples to use in Metropolis-Hastings
    tol : float, optional (default 1e-4)
        tolerance for determining whether a point is inside a sphere
    parallel : bool, optional (default False)
        whether to parallelize the algorithm or not
    verbose : bool, optional (default False)
        whether to output details on the algorithm when it's running

    
    Returns
    -------
    P : (n_p,d) numpy array
        outside points
    """

    assert np.min(U)>=0. and np.max(U)<=1.
    dim = U.shape[1]

    # This is slow, but I'm not sure why you would call this function to begin
    # with when rasterization exists.
    rng=np.random.default_rng(rng_seed)
    next_sample = lambda x: rng.normal(x, 0.05)
    Sabs = np.abs(S)
    def min_dist(p):
        #Distance from p to every sphere
        d = np.linalg.norm(U-p[None,:], axis=-1) - Sabs
        return np.clip(np.min(d), 0., np.inf)
    def distr(x):
        #If outside the bounding box, the distr is 0
        if np.any(x)<0. or np.any(x)>1.:
            return 0.
        else:
            return np.exp(100. * min_dist(x))
    P = np.array([])
    while P.shape[0]<n_samples:
        x0 = rng.uniform(dim*[0.], dim*[1.])
        s,f = metropolis_hastings(distr, next_sample, x0, n_samples)
        cond = f>np.exp(0.)
        if np.any(cond):
            if P.size==0:
                P = s[cond,:]
            else:
                P = np.concatenate((P, s[cond,:]), axis=0)

    print(P)

    return P

