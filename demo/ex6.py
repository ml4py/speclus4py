import speclus4py.sys as Sys

from mpi4py import MPI
from speclus4py import SPECLUS as clustering
from speclus4py.tools import generators as DatasetGenerator
from speclus4py.tools import hdist, plot, viewer
from speclus4py.types import DataType, EPSProblemType, GraphType, OperatorType

Sys.function_tracing = True  # allow tracing functions
comm = MPI.COMM_SELF

# generate two spiral dataset containing 550 points (225 points related to each one spiral)
dataset = DatasetGenerator.TwoSpirals(comm=comm, verbose=True)
dataset.nsamples = 550  # number of samples for both spirals
dataset.noise_level = 0.08  # add some noise
dataset.generate()
dataset.view()

samples, labels = dataset.getSamples()

# # setting solver
solver = clustering.solver(comm=comm, verbose=True)
solver.setData(X=samples, data_type=DataType.GENERAL)

# solver params
solver.connectivity = 10
solver.fn_similarity_params = 0.05
solver.graph_type = GraphType.UNDIRECTED

solver.eps_tol = 1e-3
solver.eps_nev = 20
solver.eps_problem_type = EPSProblemType.HEP  # undirected graph produces symmetric operator

solver.rbart_conf_level = 0.1
solver.rbart_sort_eigvals = True
solver.vq_kms_nclus = 2

solver.operator_type = OperatorType.LAPLACIAN_UNNORMALIZED

solver.setFromOptions()
solver.solve()

if comm.Get_rank() == 0:
    predicted_labels = solver.getLabels()
    hdist.hdist_syntetic(labels, predicted_labels, verbose=True)
    viewer.result_syntehic(samples, labels)

    # get dimension of null space related to Laplacian matrix estimated using reversed Bartlett test
    dim_null = solver.getDimNullSpace()

    # plot profile of transformed eigenvalues used in reversed Bartlett test
    eigvals = solver.getTransformedEigvals()
    if eigvals is not None:
        eigvals = eigvals[0:10]
        plot.screePlotEigenvalues(eigvals, mark_zeros=True, nzeros=dim_null)
