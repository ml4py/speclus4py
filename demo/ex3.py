import os
from mpi4py import MPI

from speclus4py import SPECLUS as clustering
from speclus4py.types import OperatorType
from speclus4py.tools import hdist_2phase, plot

filename = os.path.join(os.environ['WORK_DIR'], 'data/vol_imgs/ball.vti')

comm = MPI.COMM_WORLD
solver = clustering.solver(comm=comm, verbose=True)

solver.filename_input = filename

solver.operator_type = OperatorType.LAPLACIAN_NORMALIZED
solver.sigma = 0.05
solver.connectivity = 26
solver.vq_recover_indicators = True

solver.eps_tol = 1e-4
solver.eps_nev = 15

solver.rbart_conf_level = 0.05
solver.vq_kms_nclus = 2

solver.setFromOptions()
solver.solve()

if comm.Get_rank() == 0:
    # compute Hamming distance
    labels = solver.getLabels()
    if labels is not None:
        hdist_2phase.hdist_2phase_vol_img(filename, labels, verbose=True, visualize_result=False)

    # get dimension of null space related to Laplacian matrix estimated using reversed Bartlett test
    dim_null = solver.getDimNullSpace()

    # plot profile of eigenvalues
    eigvals = solver.getEigvals()
    if eigvals is not None:
        eigvals = eigvals[0:10]
        plot.screePlotEigenvalues(eigvals, mark_zeros=True, nzeros=dim_null)

    # plot profile of transformed eigenvalues used in reversed Bartlett test
    eigvals = solver.getTransformedEigvals()
    if eigvals is not None:
        eigvals = eigvals[0:10]
        plot.screePlotEigenvalues(eigvals, mark_zeros=True, nzeros=dim_null)


del solver
