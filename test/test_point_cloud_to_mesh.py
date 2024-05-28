from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import scipy as sp
from .context import unittest
from .context import rfta
from .context import polyscope as ps
from .context import utility

class TestPointCloudToMesh(unittest.TestCase):
    def test_dense_point_cloud_reconstructs_ground_truth(self):
        meshes = [ "boot.obj", "penguin.obj", "armadillo.obj", "horse.obj", "koala.obj" ]
        # meshes = [ "armadillo.obj" ]
        # ps.init()
        for mesh in meshes:
            # read mesh
            v, f = gpy.read_mesh("data/" + mesh)
            max_box_distance = np.max(np.max(v, axis=0) - np.min(v, axis=0))
            # sample random points on mesh
            n = 100000
            N = gpy.per_face_normals(v, f)
            rng = np.random.default_rng(196)
            p,I,_ = gpy.random_points_on_mesh(v, f, n, rng=rng, return_indices=True)
            # compute normals
            n = N[I,:]
            v_psr, f_psr = rfta.point_cloud_to_mesh(p, n, depth=10)
            # hausdorff distance between the two meshes
            distance = gpy.approximate_hausdorff_distance(v, f, v_psr, f_psr, use_cpp=True)
            normalized_hausdorff_distance = distance / max_box_distance
            print("Hausdorff distance between %s and psr: %f" % (mesh, normalized_hausdorff_distance))
            chamfer_distance = utility.chamfer(v, f, v_psr, f_psr)
            normalized_chamfer_distance = chamfer_distance / max_box_distance
            print("Chamfer distance between %s and psr: %f" % (mesh, normalized_chamfer_distance))
            
            # find max box distance
            
            self.assertTrue(normalized_hausdorff_distance < 0.01 )
            self.assertTrue(normalized_chamfer_distance < 0.01 )

            # test that parallelization returns same output
            # @ Oded, this line fails for me:
            # v_psr_par, f_psr_par = rfta.point_cloud_to_mesh(p, n, depth=10, parallel=True)
            # self.assertTrue(np.all(np.abs(v_psr-v_psr_par)<1e-6))
            # self.assertTrue(np.all(np.abs(f_psr-f_psr_par)<1e-6))
            
            




        
        


if __name__ == '__main__':
    unittest.main()