# Reach for the Arcs

This repository is the official code release for [_Reach For the Arcs_:
Reconstructing Surfaces from SDFs via Tangent Points](https://odedstein.com/projects/reach-for-the-arcs/).



> [!CAUTION]

> **If you are looking for an implementation of _Reach For the Arcs_, DO NOT USE THIS CODE! Use the reach\_for\_the\_arcs function in [Gpytoolbox](https://gpytoolbox.org/latest/reach_for_the_arcs/) instead.** This repository is merely the implementation that can be used to reproduce some of the figures in the original article, made public here for replicability purposes. The version of `reach_for_the_arcs` in [Gpytoolbox](https://gpytoolbox.org/latest/reach_for_the_arcs/) will be updated with bugfixes and enhancements, which the code here will be frozen in time forever. **For the vast majority of use cases, the implementation you want is most likely *NOT* this repository, but [the one in Gpytoolbox](https://gpytoolbox.org/latest/reach_for_the_arcs/).**

The python version we used to run this code is 3.9.
Please create your own conda environment with the correct python version, and install all the packages from requirements.txt:
```
conda create --name reach-for-the-arcs python=3.9 -y
conda activate reach-for-the-arcs
python -m pip install -r requirements.txt
```
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

