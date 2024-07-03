# Reach for the Arcs

This repository is the official public code release accompanying the SIGGRAPH 2024 paper [_Reach For the Arcs_:
Reconstructing Surfaces from SDFs via Tangent Points](https://odedstein.com/projects/reach-for-the-arcs/), 
by Silvia Sellán, Yinying Ren, Christopher Batty and Oded Stein.



> [!CAUTION]
> **If you are looking for an implementation of _Reach For the Arcs_, DO NOT USE THIS CODE! Use the `reach_for_the_arcs` function in [Gpytoolbox](https://gpytoolbox.org/latest/reach_for_the_arcs/) instead.** This repository is merely the implementation that can be used to reproduce some of the figures in the original article, made public here for replicability purposes. The version of `reach_for_the_arcs` in [Gpytoolbox](https://gpytoolbox.org/latest/reach_for_the_arcs/) will be updated with bugfixes and enhancements, which the code here will be frozen in time forever. **For the vast majority of use cases, the implementation you want is most likely *NOT* this repository, but [the one in Gpytoolbox](https://gpytoolbox.org/latest/reach_for_the_arcs/).**

> [!CAUTION]
> This code was run and tested on a Mac only. It might work on other operating systems, but we have not tested it there.

## Instructions

Please make sure to clone this repository with all its submodules.

The python version we used to run this code is 3.9.
Please create your own conda environment with the correct python version, and install all the packages from requirements.txt:
```
conda create --name reach-for-the-arcs python=3.9 -y
conda activate reach-for-the-arcs
python -m pip install -r requirements.txt
```

If you do not have OpenGL installed, you need to install OpenGL.
On Ubuntu, for example, you can install the package `libgl1-mesa-dev`.

If you are on Linux, your build might fail if you have the wrong GLIBXX version
on your computer.
You need to make sure you have a compatible version of GLIBCXX_3.4.30.

After that, please build the python/C++ bindings as follows:
```
mkdir build
cd build
cmake ..
make
```
Then you can run the code.

If you install any python packages whatsoever, add them (with the correct version) to the requirements.txt first.
Our main code functionality is contained in `src/`, while the scripts used for all our experiments go in `scripts/`. Run the files starting with `fig_` in that folder to reproduce results from the paper (see the comment in the first line of each one for the correspondence between script and figure number).
Please run the scripts while you are in the root directory of the repository!
Each figure scripts populates a folder inside `results/` with the meshes and images used in the paper figures (a zipped version of each directory is already included in `results/` for your convenience).

## Issues

Please [email us](mailto:sgsellan@cs.toronto.edu) if you encounter any issues when running this code.

## Thanks

Thanks to Abhishek Madan and João Teixeira for testing this repository.

