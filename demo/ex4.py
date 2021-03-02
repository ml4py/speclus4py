import speclus4py.sys as Sys

from mpi4py import MPI
from speclus4py import SPECLUS as clustering
from speclus4py.tools import generators as DatasetGenerator
from speclus4py.tools import hdist, plot, viewer
from speclus4py.types import DataType, GraphType, OperatorType

Sys.function_tracing = True  # allow tracing called functions
comm = MPI.COMM_SELF

# generate two spiral dataset containing 1000 points (500 points related to each one spiral)
dataset = DatasetGenerator.TwoCircles(comm=comm, verbose=True)
dataset.nsamples = 300  # number of all samples
dataset.radius_outer_circle = 3.
dataset.radius_inner_circle = 1.
dataset.noise_level = 0.01  # adding some noise
dataset.generate()
dataset.view()

samples, labels = dataset.getSamples()

# setting solver
solver = clustering.solver(comm=comm, verbose=True)
solver.setData(X=samples, data_type=DataType.GENERAL)

solver.operator_type = OperatorType.LAPLACIAN_NORMALIZED
solver.connectivity = 3
# if similarity function is not defined, similarity based on the RBF function is used and
# similarity parameter corresponds to standard deviation related to RBF
solver.fn_similarity_params = 0.1
solver.graph_type = GraphType.UNDIRECTED

solver.eps_tol = 1e-4
solver.eps_nev = 10

solver.rbart_conf_level = 0.1
solver.rbart_sort_eigvals = True
solver.rbart_error_correction = True
solver.vq_kms_nclus = 2

solver.rbart_conf_level = 0.05
solver.vq_kms_nclus = 2

solver.setFromOptions()
solver.solve()

if comm.Get_rank() == 0:
    predicted_labels = solver.getLabels()
    hdist.hdist_syntetic(labels, predicted_labels, verbose=True)
    viewer.result_syntehic(samples, predicted_labels)

    # get dimension of null space related to Laplacian matrix estimated using reversed Bartlett test
    dim_null = solver.getDimNullSpace()

    # plot profile of transformed eigenvalues used in reversed Bartlett test
    eigvals = solver.getTransformedEigvals()
    if eigvals is not None:
        eigvals = eigvals[0:5]
        plot.screePlotEigenvalues(eigvals, mark_zeros=True, nzeros=dim_null)
