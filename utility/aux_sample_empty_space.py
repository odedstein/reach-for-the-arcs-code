# Here I import only the functions I need for these functions
import numpy as np
import gpytoolbox as gpy
from gpytoolbox.copyleft import mesh_boolean
import sys, os, platform
import polyscope as ps

os_name = platform.system()
if os_name == "Darwin":
    # Get the macOS version
    os_version = platform.mac_ver()[0]
    # print("macOS version:", os_version)

    # Check if the macOS version is less than 14
    if os_version and os_version < "14":
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build-studio')))
    else:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
elif os_name == "Windows":
    # For Windows systems
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/Debug')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/Release')))
else:
    # For non-macOS systems
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import rfta_bindings
import time

# this will be a very bad wrapper of a bash script
def aux_sample_empty_space(sdf_data, sdf_values, num_points, type):
    # default res: cubic root of sdt_data
    res = 100
    if type=="rasterization-cpu":
        points = np.array([[0,0,0]])
        while points.shape[0]<num_points:
            time_start = time.time()
            points = rfta_bindings._outside_points_from_rasterization_cpp_impl(sdf_data.astype(np.float64),
            np.abs(sdf_values).astype(np.float64),
            0, res, 1e-4,
            True,
            False,
            True,
            False)
            time_end = time.time()
            res = res + 10
            print(points.shape)
        return time_end-time_start
    elif type=="rasterization-gpu":
        points = np.array([[0,0,0]])
        while points.shape[0]<num_points:
            time_start = time.time()
            points = rfta_bindings._outside_points_from_rasterization_cpp_impl(sdf_data.astype(np.float64),
            np.abs(sdf_values).astype(np.float64),
            0, res, 1e-4,
            True,
            False,
            False,
            False)
            time_end = time.time()
            res = res + 10
            # print(points.shape)
        return time_end-time_start
    elif type=="rejection":
        time_start = time.time()
        points = rfta_bindings._outside_points_from_rejection_sampling_cpp_impl(sdf_data.astype(np.float64),
        np.abs(sdf_values).astype(np.float64),
        0, num_points)
        time_end = time.time()
        return time_end-time_start
    else:
        print("Invalid type")
        return None