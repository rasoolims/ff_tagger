# Commands to install the required libraries (including Jupyter) in virtual environment.

```txt
virtualenv nlphw
source nlphw/bin/activate

cd nlphw/
pip install numpy
pip install matplotlib
pip install jupyter
pip install RISE
jupyter-nbextension install rise --py --sys-prefix
jupyter-nbextension enable rise --py --sys-prefix

mkdir dynet-base
cd dynet-base
pip install cython
git clone https://github.com/clab/dynet.git
hg clone https://bitbucket.org/eigen/eigen -r 699b659
cd dynet
mkdir build
cd build
export pth="$PWD"
cmake .. -DEIGEN3_INCLUDE_DIR=$pth/../../eigen/ -DPYTHON=$pth/../../../bin/python
make -j 2
cd python
python ../../setup.py build --build-dir=.. --skip-build install
export DYLD_LIBRARY_PATH=$pth/dynet/:$DYLD_LIBRARY_PATH
export LD_LIBRARY_PATH=$pth/dynet/:$LD_LIBRARY_PATH
```