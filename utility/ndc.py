# Here I import only the functions I need for these functions
import numpy as np
import gpytoolbox as gpy
import sys, os

# this will be a very bad wrapper of a bash script
def ndc(v,f,resolution):
    # first, write the mesh to an obj file in the current directory
    # normalize the mesh
    v_input = np.copy(v)
    # v_input = gpy.normalize_points(v_input)
    # divide it by two because NDC is like that
    v_input = v_input/2
    gpy.write_mesh( "temp.obj", v_input, f )
    # now call the bash script
    # this is a very bad way to do this
    # but it works
    path_to_bash_script = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ndc_script.sh'))
    # run bash path_to_bash_script temp.obj resolution, supressing console output
    os.system('bash ' + path_to_bash_script + ' temp.obj ' + str(resolution) + ' > /dev/null')
    # this generates a file called temp_reconstruction.obj
    # read it in
    u, g = gpy.read_mesh('temp_reconstruction.obj')
    u = u - (resolution/2) # centered
    # print(np.max(u,axis=0))
    # print(np.min(u,axis=0))
    u = (2 * u / resolution)
    # print(np.max(u,axis=0))
    # print(np.min(u,axis=0))
    # delete the temp files
    os.system('rm temp.obj')
    os.system('rm temp_reconstruction.obj')
    # return the reconstruction
    return u, g
