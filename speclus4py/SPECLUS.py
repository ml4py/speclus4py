import slepc4py, sys, os
slepc4py.init(sys.argv)

import numpy as np

from pathvalidate import is_valid_filepath

from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI

from speclus4py.types import DataObject, OperatorContainer, OperatorType
from speclus4py.viewer import Viewer
from speclus4py.assembler import OperatorAssembler
from speclus4py.eig import EigenSystemSolver
from speclus4py.kms import kms

import speclus4py.bartlett as bartlett


class solver(DataObject, OperatorContainer):
    def __init__(self, comm=MPI.COMM_WORLD, verbose=False):
        DataObject.__init__(self, comm=comm, verbose=verbose)
        OperatorContainer.__init__(self)

        self.__comm = comm
        self.verbose = verbose

        self.__filename_input = ''
        self.__output_dir = ''
        self.__filename_eigvals = PETSc.DEFAULT
        self.__filename_eigvecs = PETSc.DEFAULT
        self.__filename_degs = PETSc.DEFAULT
        self.__filename_errs = PETSc.DEFAULT
        self.__filename_model = PETSc.DEFAULT
        self.__filename_labels = PETSc.DEFAULT

        self.__eig_solver = None
        self.__eps_tol = 1e-4
        self.__eps_nev = 10
        self.__eps_prob_type = SLEPc.EPS.ProblemType.HEP

        self.__rbart_conf_level = 0.05
        self.__rbart_sort_eigvals = False
        self.__rbart_err_correction = True

        self.__vq_kms_tol = 0.1
        self.__vq_kms_max_it = 100
        self.__vq_kms_nclus = PETSc.DEFAULT
        self.__vq_kms_restarts = 5
        self.__vq_normalize_features = False
        self.__vq_recover_indicators = False

        self.__dim_null = -1
        self.__labels = None
        self.__centers = None

        self.__eigsolcalled = False
        self.__rbartcalled = False
        self.__vqcalled = False
        self.__setupcalled = False
        self.__solvecalled = False

    def reset(self):
        OperatorContainer.reset(self)
        del self.data
        if self.__labels:
            del self.__labels
        if self.__centers:
            del self.__centers
        del self.__eig_solver
        self.__eig_solver = None

        self.__dim_null = -1

        self.__eigsolcalled = False
        self.__rbartcalled = False
        self.__vqcalled = False
        self.__setupcalled = False
        self.__solvecalled = False

    def setFromOptions(self):
        OptDB = PETSc.Options()

        # set options for operator
        op_type_string = OptDB.getString('spc_op_type', self.operator_type.value)
        if op_type_string is not self.operator_type.value:
            self.operator_type = OperatorType(op_type_string)
        connectivity = OptDB.getInt('spc_op_connectivity', self.connectivity)
        if connectivity is not self.connectivity:
            self.connectivity = connectivity
        sigma = OptDB.getReal('spc_op_rbf_sigma', self.sigma)
        if sigma is not self.sigma:
            self.sigma = sigma

        # set options for vector quantification using k-means
        vq_kms_nclus = OptDB.getInt('spc_vq_kms_nclus', self.vq_kms_nclus)
        if vq_kms_nclus is not self.vq_kms_nclus:
            self.vq_kms_nclus = vq_kms_nclus
        vq_kms_max_it = OptDB.getInt('spc_vq_kms_max_it', self.vq_kms_max_it)
        if vq_kms_max_it is not self.vq_kms_max_it:
            self.vq_kms_max_it = vq_kms_max_it
        vq_kms_restarts = OptDB.getInt('spc_vq_kms_restarts', self.vq_kms_restarts)
        if vq_kms_restarts is not self.vq_kms_restarts:
            self.vq_kms_restarts = vq_kms_restarts
        vq_kms_tol = OptDB.getReal('spc_vq_kms_tol', self.vq_kms_tol)
        if vq_kms_tol is not self.vq_kms_tol:
            self.vq_kms_tol = vq_kms_tol
        vq_recover_indicators = OptDB.getBool('spc_vq_recover_indicators', self.vq_recover_indicators)
        if vq_recover_indicators is not self.vq_recover_indicators:
            self.vq_recover_indicators = vq_recover_indicators
        vq_normalize_features = OptDB.getBool('spc_vq_norm_features', self.vq_normalize_features)
        if vq_normalize_features is not self.vq_normalize_features:
            self.vq_normalize_features = vq_normalize_features

        # set options for reversed bartlett test for determining null space dimension of Laplacian matrix
        rbart_conf_level = OptDB.getReal('spc_rbart_conf_level', self.rbart_conf_level)
        if rbart_conf_level is not self.rbart_conf_level:
            self.rbart_conf_level = rbart_conf_level
        rbart_error_correction = OptDB.getBool('spc_rbart_err_correction', self.rbart_error_correction)
        if rbart_error_correction is not self.rbart_error_correction:
            self.rbart_error_correction = rbart_error_correction
        rbart_sort_eigvals = OptDB.getBool('spc_rbat_sort_eigvals', self.rbart_sort_eigvals)
        if rbart_sort_eigvals is not self.rbart_sort_eigvals:
            self.rbart_sort_eigvals = rbart_sort_eigvals

        # set input filename
        filename_input = OptDB.getString('spc_filename_input', self.filename_input)
        if filename_input is not self.filename_input:
            self.filename_input = filename_input

        # set output filenames
        filename_eigvals = OptDB.getString('spc_filename_eigvals', self.filename_eigvals)
        if filename_eigvals is not self.filename_eigvals:
            self.filename_eigvals = filename_eigvals
        filename_eigvecs = OptDB.getString('spc_filename_eigvecs', self.filename_eigvecs)
        if filename_eigvecs is not self.filename_eigvecs:
            self.filename_eigvecs = filename_eigvecs
        filename_degs = OptDB.getString('spc_filename_degs', self.filename_degs)
        if filename_degs is not self.filename_degs:
            self.filename_degs = filename_degs
        filename_model = OptDB.getString('spc_filename_model', self.filename_model)
        if filename_model is not self.filename_model:
            self.filename_model = filename_model
        filename_labels = OptDB.getString('spc_filename_labels', self.filename_labels)
        if filename_labels is not self.filename_labels:
            self.filename_labels = filename_labels

    # Output filenames getters/setters
    @property
    def filename_input(self) -> str:
        return self.__filename_input

    @filename_input.setter
    def filename_input(self, name: str):
        if name is self.filename_input:
            return

        if not os.path.exists(name):
            PETSc.Sys.Print('Input file does not exist')
            raise PETSc.Error(62)

        self.reset()
        self.__filename_input = name

    @property
    def filename_eigvals(self) -> (str, PETSc.DEFAULT):
        return self.__filename_eigvals

    @filename_eigvals.setter
    def filename_eigvals(self, p: (str, PETSc.DEFAULT)):
        if not (is_valid_filepath(p) or PETSc.DEFAULT):
            PETSc.Sys.Print('Not valid filename')
            raise PETSc.Error(62)
        self.__filename_eigvals = p

    @property
    def filename_eigvecs(self) -> (str, PETSc.DEFAULT):
        return self.__filename_eigvecs

    @filename_eigvecs.setter
    def filename_eigvecs(self, p: (str, PETSc.DEFAULT)):
        if not (is_valid_filepath(p) or PETSc.DEFAULT):
            PETSc.Sys.Print('Not valid filename')
            raise PETSc.Error(62)

        self.__filename_eigvecs = p

    @property
    def filename_degs(self) -> (str, PETSc.DEFAULT):
        return self.__filename_degs

    @filename_degs.setter
    def filename_degs(self, p: (str, PETSc.DEFAULT)):
        if not (is_valid_filepath(p) or PETSc.DEFAULT):
            PETSc.Sys.Print('Not valid filename')
            raise PETSc.Error(62)

        self.__filename_degs = p

    @property
    def filename_errs(self) -> (str, PETSc.DEFAULT):
        return self.__filename_errs

    @filename_errs.setter
    def filename_errs(self, p: (str, PETSc.DEFAULT)):
        if not (is_valid_filepath(p) or PETSc.DEFAULT):
            PETSc.Sys.Print('Not valid filename')
            raise PETSc.Error(62)

        self.__filename_errs = p

    @property
    def filename_model(self) -> (str, PETSc.DEFAULT):
        return self.__filename_model

    @filename_model.setter
    def filename_model(self, p: (str, PETSc.DEFAULT)):
        if not (is_valid_filepath(p) or PETSc.DEFAULT):
            PETSc.Sys.Print('Not valid filename')
            raise PETSc.Error(62)

        self.__filename_model = p

    @property
    def filename_labels(self) -> (str, PETSc.DEFAULT):
        return self.__filename_labels

    @filename_labels.setter
    def filename_labels(self, p):
        if not (is_valid_filepath(p) or PETSc.DEFAULT):
            PETSc.Sys.Print('Not valid filename')
            raise PETSc.Error(62)

        self.__filename_labels = p

    # Output directory getter/setter
    @property
    def output_dir(self):
        return self.__output_dir

    @output_dir.setter
    def output_dir(self, p):
        if not (is_valid_filepath(p) or PETSc.DEFAULT):
            PETSc.Sys.Print('Not valid filename')
            raise PETSc.Error(62)

        self.__output_dir = p

    @property
    def eps_problem_type(self):
        return self.__eps_prob_type

    @eps_problem_type.setter
    def eps_problem_type(self, type: SLEPc.EPS.ProblemType):
        if type is self.eps_prob_type:
            return

        self.__eps_prob_type = type

        self.__eigsolcalled = False
        self.__rbartcalled = False
        self.__vqcalled = False
        self.__setupcalled = False
        self.__solvecalled = False

    @property
    def eps_tol(self) -> float:
        return self.__eps_tol

    @eps_tol.setter
    def eps_tol(self, tol: float):
        if tol is self.eps_tol:
            return

        if tol <= 0 or tol >= 1:
            PETSc.Sys.Print('Tolerance must be positive float less than 1, e.g. 1e-1')
            raise PETSc.Error(62)

        self.__eps_tol = tol

        self.__eigsolcalled = False
        self.__rbartcalled = False
        self.__vqcalled = False
        self.__setupcalled = False
        self.__solvecalled = False

    @property
    def eps_nev(self) -> int:
        return self.__eps_nev

    @eps_nev.setter
    def eps_nev(self, n: int):
        if n is self.eps_nev:
            return

        if n <= 0:
            PETSc.Sys.Print('Argument muset be grater than 0')
            raise PETSc.Error(62)
        self.__eps_nev = n

        self.__eigsolcalled = False
        self.__rbartcalled = False
        self.__vqcalled = False
        self.__setupcalled = False
        self.__solvecalled = False

    @property
    def rbart_conf_level(self) -> float:
        return self.__rbart_conf_level

    @rbart_conf_level.setter
    def rbart_conf_level(self, conf_level: float):
        if conf_level is self.rbart_conf_level:
            return

        if conf_level < 0. and conf_level > 1.:
            PETSc.Sys.Print('Confidence level for revered Bartlett test must be greater than 0 and less than 1. '
                            'Typically, value of confidence level is chosen as 0.1, 0.05, 0.01, 0.005.')
            raise PETSc.Error(62)
        self.__rbart_conf_level = conf_level

        self.__rbartcalled = False
        self.__vqcalled = False
        self.__solvecalled = False

    @property
    def rbart_error_correction(self) -> bool:
        return self.__rbart_err_correction

    @rbart_error_correction.setter
    def rbart_error_correction(self, flag: bool):
        if flag is self.rbart_error_correction:
            return

        self.__rbart_err_correction = flag

        self.__rbartcalled = False
        self.__vqcalled = False
        self.__solvecalled = False

    @property
    def rbart_sort_eigvals(self) -> bool:
        return self.__rbart_sort_eigvals

    @rbart_sort_eigvals.setter
    def rbart_sort_eigvals(self, flag: bool):
        if flag is self.rbart_sort_eigvals:
            return

        self.__rbart_sort_eigvals = flag

        self.__rbartcalled = False
        self.__vqcalled = False
        self.__solvecalled = False

    @property
    def vq_normalize_features(self) -> bool:
        return self.__vq_normalize_features

    @vq_normalize_features.setter
    def vq_normalize_features(self, flag: bool):
        if flag is self.vq_normalize_features:
            return

        self.__vq_normalize_features = flag

        self.__vqcalled = False
        self.__solvecalled = False

    @property
    def vq_recover_indicators(self) -> bool:
        return self.__vq_recover_indicators

    @vq_recover_indicators.setter
    def vq_recover_indicators(self, flag: bool):
        if flag is self.vq_recover_indicators:
            return

        self.__vq_recover_indicators = flag

        self.__vqcalled = False
        self.__solvecalled = False

    @property
    def vq_kms_max_it(self) -> int:
        return self.__vq_kms_max_it

    @vq_kms_max_it.setter
    def vq_kms_max_it(self, max_it: int):
        if max_it is self.vq_kms_max_it:
            return

        if max_it < 0:
            PETSc.Sys.Print('Maximum number of iterations must > 0')
            raise PETSc.Error(62)
        self.__vq_kms_max_it = max_it

        self.__vqcalled = False
        self.__solvecalled = False

    @property
    def vq_kms_tol(self) -> float:
        return self.__vq_kms_tol

    @vq_kms_tol.setter
    def vq_kms_tol(self, tol: float):
        if tol is self.vq_kms_tol:
            return

        if tol <= 0:
            PETSc.Sys.Print('Tolerance must be greater than 0, e.g. 1e-1')
            raise PETSc.Error(62)

        self.__vq_kms_tol = tol

        self.__vqcalled = False
        self.__solvecalled = False

    @property
    def vq_kms_restarts(self) -> int:
        return self.__vq_kms_restarts

    @vq_kms_restarts.setter
    def vq_kms_restarts(self, restarts: int):
        if restarts is self.vq_kms_restarts:
            return

        if restarts <= 0:
            PETSc.Sys.Print('K-means restarts must be greater 0')
            raise PETSc.Error(62)

        self.vq_kms_restarts = restarts

        self.__vqcalled = False
        self.__solvecalled = False

    @property
    def vq_kms_nclus(self) -> int:
        return self.__vq_kms_nclus

    @vq_kms_nclus.setter
    def vq_kms_nclus(self, n: int):
        if n is self.vq_kms_nclus:
            return

        if n < 2:
            PETSc.Sys.Print('Number of clusters must be greater than 1')
            raise PETSc.Error(62)

        self.__vq_kms_nclus = n

        self.__vqcalled = False
        self.__solvecalled = False

    def loadData(self):
        if self.filename_input is '':
            PETSc.Sys.Print('Filename is not specified')
            raise PETSc.Error(66)

        if self.verbose:
            PETSc.Sys.Print("Loading data from %s" % self.filename_input)

        v = Viewer(filename=self.filename_input, verbose=self.verbose)
        [data, data_type] = v.load()
        self.setData(data, data_type)

        del v

    def assemblyOperator(self):
        if self.data is None:
            self.loadData()

        [data, data_type] = self.getData()

        op_assembler = OperatorAssembler(self.comm, self.verbose)

        op_assembler.setData(data, data_type)
        op_assembler.connectivity = self.connectivity
        op_assembler.sigma = self.sigma
        op_assembler.operator_type = self.operator_type
        op_assembler.mat_type = self.mat_type

        op_assembler.assembly()
        [self.mat_op, self.mat_adj, self.vec_diag] = op_assembler.getOperators()

        del op_assembler

    def setUp(self):
        if self.__setupcalled:
            return

        if self.mat_op is None:
            self.assemblyOperator()

        del self.__eig_solver

        self.__eig_solver = EigenSystemSolver(self.verbose)
        self.__eig_solver.problem_type = self.eps_problem_type
        self.__eig_solver.tol = self.eps_tol
        self.__eig_solver.nev = self.eps_nev
        self.__eig_solver.setOperators(self.mat_op, self.vec_diag, self.operator_type)

        self.__eigsolcalled = False
        self.__vqcalled = False
        self.__solvecalled = False

        self.__setupcalled = True

    def solveEigensystem(self):
        if self.__eigsolcalled:
            return

        if not self.__setupcalled:
            self.setUp()

        self.__eig_solver.solve()

        self.__eigsolcalled = True
        self.__vqcalled = False
        self.__solvecalled = False

    def vectorQuantification(self):
        if self.__vqcalled:
            return

        if not self.__eigsolcalled:
            self.solveEigensystem()

        eigvals, mat_bv = self.__eig_solver.getSolution()
        error = self.__eig_solver.getError()

        if not self.__rbartcalled:
            self.__dim_null = bartlett.determine_nullspace_dimension(eigvals, error_corr=self.rbart_error_correction,
                                                                     error=error, sort=self.rbart_sort_eigvals,
                                                                     conf_level=self.rbart_conf_level,
                                                                     verbose=self.verbose)
            self.__rbartcalled = True
        else:
            PETSc.Sys.Print('Using k=%d connected components (factors) of underlying similarity graph' %
                            self.__dim_null)

        if self.__dim_null <= 1:
            del self.__labels, self.__centers
            self.__labels = self.__centers = None
            return

        # gather data to root processor
        rank = self.comm.Get_rank()
        N = mat_bv.getSize()[0]
        if rank == 0:
            irows = PETSc.IS().createStride(N, 0, 1, PETSc.COMM_SELF)
            icols = PETSc.IS().createStride(self.__dim_null, 0, 1, PETSc.COMM_SELF)
        else:
            irows = PETSc.IS().createStride(0, 0, 1, PETSc.COMM_SELF)
            icols = None  # download 0 columns on other MPI processes

        # gather base of null-space from mat_bv to mat_bv0 (root proc)
        mat_bv0 = mat_bv.createSubMatrices(irows, icols)[0]
        # gather diagonal vector to root proc
        vec_diag0 = self.vec_diag.getSubVector(irows)

        if self.__vq_kms_nclus == PETSc.DEFAULT:
            nclus = self.__dim_null
        else:
            nclus = self.__dim_null if self.__vq_kms_nclus > self.__dim_null else self.__vq_kms_nclus
            if nclus < self.__vq_kms_nclus:
                PETSc.Sys.Print('Warning: Dimension of null-space is less than specified number cluster (%d). '
                                'Vector quantification using k-means proceeds for k=%d' %
                                (self.__vq_kms_nclus, nclus))
        if rank == 0:
            X = mat_bv0.getDenseArray()
            X = X.astype(dtype=np.float32)

            if self.__vq_normalize_features:
                for i in range(N):
                    X[i, :] /= np.linalg.norm(X[i])
            elif self.__vq_recover_indicators:
                vec_tmp = vec_diag0.duplicate()
                vec_diag0.copy(vec_tmp)

                vec_tmp.sqrtabs()
                vec_tmp.reciprocal()
                diag_arr = vec_tmp.getArray()

                for i in range(self.__dim_null):
                    X[:, i] = np.multiply(X[:, i], diag_arr)

            if self.verbose:
                PETSc.Sys.Print('Vector quantification (k-means++, k=%d, tol=%.3f, restarts=%d)'
                                % (nclus, self.vq_kms_tol, self.vq_kms_restarts))

            del self.__labels, self.__centers
            self.__labels, self.__centers = kms(data=X, k=nclus, max_iter=self.__vq_kms_max_it,
                                                restarts=self.vq_kms_restarts, tol=self.vq_kms_tol,
                                                verbose=self.verbose)

            del X

        del mat_bv0

        self.__vqcalled = True
        self.__solvecalled = False

    def solve(self):
        if not self.__setupcalled:
            self.setUp()

        self.solveEigensystem()
        self.vectorQuantification()

        self.__solvecalled = True

    def getLabels(self):
        return self.__labels

    def save(self):
        if self.__labels is None:
            return

        # create on root process
        if self.comm.Get_rank() == 0:
            try:
                if not os.path.exists(self.output_dir) and self.output_dir is not '':
                    os.mkdir(self.output_dir)
            except OSError:
                PETSc.Sys.Print("Creating directory %s failed" % self.output_dir)
                raise PETSc.Error(67)
        self.comm.Barrier()

        if self.filename_eigvecs is PETSc.DEFAULT:
            file = 'eigvecs_%.4f' % self.sigma
        else:
            file = self.filename_eigvecs
        file = os.path.join(self.output_dir, file)

        eigvals, mat_bv = self.__eig_solver.getSolution()

        petsc_viewer = PETSc.Viewer().createBinary(name=file, mode=PETSc.Viewer().Mode.WRITE, comm=self.comm)
        mat_bv.view(petsc_viewer)

        del petsc_viewer

        if self.filename_degs is PETSc.DEFAULT:
            file = 'degs_%.4f' % self.sigma
        else:
            file = self.filename_degs
        file = os.path.join(self.output_dir, file)

        petsc_viewer = PETSc.Viewer().createBinary(name=file, mode=PETSc.Viewer().Mode.WRITE, comm=self.comm)
        self.vec_diag.view(petsc_viewer)

        if self.comm.Get_rank() == 0:
            if self.filename_eigvals is PETSc.DEFAULT:
                file = 'eigvals_%.4f' % self.sigma
            else:
                file = self.filename_eigvals
            file = os.path.join(self.output_dir, file)

            np.save(file, eigvals)

            if self.filename_errs is PETSc.DEFAULT:
                file = 'errs_%.4f' % self.sigma
            else:
                file = self.filename_errs
            file = os.path.join(self.output_dir, file)

            np.save(file, self.__eig_solver.getError())

            if self.filename_model is PETSc.DEFAULT:
                file = 'model_%.4f' % self.sigma
            else:
                file = self.filename_model
            file = os.path.join(self.output_dir, file)

            np.save(file, self.__centers)

            if self.filename_labels is PETSc.DEFAULT:
                file = 'labels_%.4f' % self.sigma
            else:
                file = self.filename_labels
            file = os.path.join(self.output_dir, file)

            np.save(file, self.__labels)

    # TODO adding tree structure to output directory
    # TODO visualization
    # TODO load result
    # TODO view
    # TODO set prefix for eps, set object names
    # TODO hamming distance tool
    # TODO plotting graph tool
