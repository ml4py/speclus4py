from mpi4py import MPI

from speclus4py import SPECLUS as clustering
from speclus4py.tools import hdist_2phase

filename = "../data/vol_imgs/ball.vti"

comm = MPI.COMM_WORLD
solver = clustering.solver(comm=comm, verbose=True)

solver.filename_input = filename
solver.sigma = 0.05
solver.connectivity = 6

solver.eps_tol = 1e-4

solver.rbart_conf_level = 0.05
solver.vq_kms_nclus = 2

solver.setFromOptions()
solver.solve()

if comm.Get_rank() == 0:
    labels = solver.getLabels()
    if labels is not None:
        hdist_2phase.hdist_2phase_vol_img(filename, labels, verbose=True, visualize_result=True)

del solver
