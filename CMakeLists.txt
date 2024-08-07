cmake_minimum_required(VERSION 3.10)

#https://discourse.cmake.org/t/msvc-runtime-library-completely-ignored/10004
cmake_policy(SET CMP0091 NEW)

project(Bindings
	DESCRIPTION
		"Python bindings"
)

# MSVC needs explicit configuration for multithreading
# Select a multi-threaded statically-linked runtime library
# 	with or without debug information depending on the configuration
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext")

#https://stackoverflow.com/q/46724267
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

include(libigl)
igl_include(glfw)
igl_include(embree)

include(nanoflann)

include(webgpu)
include(glfw3webgpu)
if(TARGET webgpu AND TARGET glfw3webgpu)
	add_compile_definitions(GL_AVAILABLE)
	set(GL_LIBS webgpu glfw3webgpu)
else()
	set(GL_LIBS "")
endif()

# Poisson Recon isn't CMake friendly, so let's setup our own header-only
# target here.
# include(PoissonRecon)
add_library(PoissonRecon INTERFACE)
# target_include_directories(PoissonRecon INTERFACE "${CMAKE_BINARY_DIR}/_deps/poissonrecon-src/Src/")
target_include_directories(PoissonRecon INTERFACE "${EXT_DIR}/PoissonRecon/Src/")

# All libraries to link
set(PROJ_LIBS igl::core igl::embree igl::glfw nanoflann PoissonRecon ${GL_LIBS})

add_library(cpp_rfta
	STATIC
	# SHARED
	# Headers
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/outside_points_from_rasterization.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/outside_points_from_rejection_sampling.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/locally_make_feasible.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/fine_tune_point_cloud_iter.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/point_cloud_to_mesh.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/sAABB.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/resolve_collisions_on_sphere.h"
	# Source
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/outside_points_from_rasterization.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/outside_points_from_rejection_sampling.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/locally_make_feasible.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/fine_tune_point_cloud_iter.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/point_cloud_to_mesh.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/sAABB.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/resolve_collisions_on_sphere.cpp"
	)

target_link_libraries(cpp_rfta "${PROJ_LIBS}")
set_property(TARGET cpp_rfta PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

# Otman's python bindings using pybind11
add_subdirectory(./ext/pybind11/)
pybind11_add_module(rfta_bindings
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/rfta_bindings_core.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_outside_points_from_rasterization.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_outside_points_from_rejection_sampling.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_locally_make_feasible.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_fine_tune_point_cloud_iter.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_point_cloud_to_mesh.cpp"
)

target_link_libraries(rfta_bindings PUBLIC cpp_rfta "${PROJ_LIBS}")
set_property(TARGET rfta_bindings PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

# Manually copy wgpu to the right place
target_copy_webgpu_binaries(rfta_bindings)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	# Extra difficult on mac.
	add_custom_command(
		TARGET rfta_bindings POST_BUILD
		COMMAND
			${CMAKE_INSTALL_NAME_TOOL} "-change"
			"@executable_path/libwgpu_native.dylib"
			"$<TARGET_FILE_DIR:rfta_bindings>/libwgpu_native.dylib"
			"$<TARGET_FILE:rfta_bindings>"
		VERBATIM
	)
endif()

