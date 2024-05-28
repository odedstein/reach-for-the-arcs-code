# Reach for the Arcs

This repository is the official code release for [_Reach For the Arcs_:
Reconstructing Surfaces from SDFs via Tangent Points](https://odedstein.com/projects/reach-for-the-arcs/).

## If you are looking for an implementation of _Reach For the Arcs_, DO NOT USE THIS CODE! Use the reach\_for\_the\_arcs function in [Gpytoolbox](https://gpytoolbox.org) instead.
This repository is merely the implementation that can be used to reproduce some of the figures in the original article.
The version of `reach_for_the_arcs` in [Gpytoolbox](https://gpytoolbox.org) will be updated with bugfixes and enhancements, which the code here will be frozen in time forever.

The python version used to run this code is 3.9.
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

Code goes in `src/python` or `src/cpp`.
Experiments go in `scripts/`. Run the files in that folder to reproduce results from the paper.

