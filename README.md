# speclus4py 

__Spectral clustering implementation for not only distributed memory systems written in Python.__

[![Latest anaconda-cloud version](
https://anaconda.org/mpecha/speclus4py/badges/version.svg)](https://anaconda.org/mpecha/speclus4py)
[![Platforms](
https://anaconda.org/mpecha/speclus4py/badges/platforms.svg)](https://anaconda.org/mpecha/speclus4py)
[![Install](
https://anaconda.org/mpecha/speclus4py/badges/installer/conda.svg)](https://anaconda.org/mpecha/speclus4py)

This package is related to unsupervised learning using [the spectral clustering technique](https://en.wikipedia.org/wiki/Spectral_clustering). It is written in the Python programming language on top of the [SLEPc](https://slepc.upv.es) and [PETSc/TAO](https://www.mcs.anl.gov/petsc/) frameworks, and it includes many others additional packages like [OpenCV](https://opencv.org), [scipy](https://www.scipy.org), [numpy](https://numpy.org), [numba](http://numba.pydata.org). It takes advantages from distributed memory management, which basically inherits from PETSc. Thus, computations can run on parallel computing architectures such as [beowulfs](https://en.wikipedia.org/wiki/Beowulf_cluster)/small clusters and [supercomputers](https://en.wikipedia.org/wiki/Supercomputer) natively. It does not mean that one cannot use it on laptops and desktop computers. On these, users can effectively utilize computational cores using the message passing interface commonly known as [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) as well. This approach provides distributed data parallelism to the conventional machine learning technique and enables it for possible processing [large-scale data](https://en.wikipedia.org/wiki/Big_data). 


This repository contains the alpha pre-release of the speclus4py package. Do not hesitate to pull a request if you find a bug or have an idea how to extend a package functionality.

## Installation

You can simply install spelus4py using the package management systems [Anaconda](https://www.anaconda.com) or [Conda](https://docs.conda.io/en/latest/).

```bash
conda create -n env-speclus4py python=3.7
conda activate env-speclus4py
conda config --add channels conda-forge
conda config --append channels conda-forge/label/gcc7
conda install -c mpecha speclus4py
```

## Getting Started

Take a look at [Usage](#usage) and the examples located in the *demo/* folder to your first meeting with this package, which might accelerate using spelus4py in your research. You can run the example of 2-phase segmentation of *data/vol_imgs/ball.vti* (volumetric image) by simply typing to a system console a following command that runs a computation on two CPU cores:

```bash
export WORK_DIR=$PWD
mpirun -np 2 python demo/ex1.py 
``` 
If this package was succesfully installed, you can see the result as *a green ball* displayed in a visualization window.

## Usage

The best way to see how speclus4py can help to your research is looking at the *demo* folder.

```python
from mpi4py import MPI

from speclus4py import SPECLUS as clustering
from speclus4py.tools import hdist_2phase

comm = MPI.COMM_WORLD
solver = clustering.solver(comm=comm, verbose=True)

filename = 'data/vol_imgs/ball.vti'

solver.filename_input = filename
solver.sigma = 0.05  # setting value of standard deviation related to RBF kernel
# number of nearest neighbours to those similarities are being computed
solver.connectivity = 6  

solver.setFromOptions()

# Determine the Hamming distance between solution and ground truth
if comm.Get_rank() == 0:
    labels = solver.getLabels()
    if labels is not None:
       hdist_2phase.hdist_2phase_vol_img(filename, labels, verbose=True)

```

## Publications

- Pecha, Marek (2021): *General Technique for Estimating Number of Groups for Spectral Clustering*. TechRxiv. Preprint. [10.36227/techrxiv.13553705](http://doi.org/10.36227/techrxiv.13553705) 

## Acknowledgements

This software can be developed thanks to the financial support of  The Ministry of Education, Youth and Sports from the National Programme of Sustainability (NPU II) project IT4Innovations excellence in science no. LQ1602, the programme for supporting for science and research in the Moravia–Silesia Region 2017 no. RRC/10/2017, the institutional development plan project RPP2020/138, and Grants of SGS (VSB-TUO) no. SP2020/84 and SP2020/114. 

Volumetric images included in the distribution of this package were provided by colleagues from the [Institute for Parallel Processing, Bulgarian Academy of Science](http://www.bas.bg/clpp/en/indexen.htm). The main functionality of this package is programmed in cooperation with [VSB - Technical University of Ostrava](https://www.vsb.cz/en) and [Czech Academy of Sciences (Institute of Geonics)](http://www.ugn.cas.cz/?l=en&p=home).