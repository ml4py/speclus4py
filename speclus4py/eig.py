from petsc4py import PETSc
from slepc4py import SLEPc

import numpy as np

from speclus4py.types import OperatorType


class EigenSystemSolver:
    def __init__(self, verbose=False):
        self.__eps = None

        self.__eps_tol = 1e-4
        self.__eps_nev = 10
        self.__eps_prob_type = SLEPc.EPS.ProblemType.HEP

        self.__op = None
        self.__vec_diag = None
        self.__op_B = None

        self.__mat_bv = None
        self.__eigen_vals = None
        self.__errors = None

        self.__solved = False
        self.__autopostsolve = False
        self.__setupcalled = False
        self.__postsolvecalled = False

        self.__verbose = verbose

    def reset(self):
        del self.__op_B
        del self.__mat_bv
        del self.__eigen_vals
        del self.__errors

        self.__op_B = self.__errors = self.__mat_bv = self.__eigen_vals = None

        self.__solved = False
        self.__postsolvecalled = False
        self.__setupcalled = False

    def setOperators(self, op: PETSc.Mat, diag: PETSc.Vec, op_type: OperatorType):
        self.reset()

        self.__op = op
        self.__vec_diag = diag
        self.__op_type = op_type

    @property
    def problem_type(self) -> SLEPc.EPS.ProblemType:
        return self.__eps_prob_type

    @problem_type.setter
    def problem_type(self, type: SLEPc.EPS.ProblemType):
        if type == self.problem_type:
            return

        self.reset()

        self.__eps_prob_type = type

    @property
    def tol(self) -> float:
        return self.__eps_tol

    @tol.setter
    def tol(self, tol: float):
        if tol == self.tol:
            return

        if tol <= 0 or tol >= 1:
            PETSc.Sys.Print('Tolerance must be positive float less than 1, e.g. 1e-1')
            raise PETSc.Error(62)

        self.__eps_tol = tol

        self.__setupcalled = False
        self.__solved = False

    @property
    def nev(self) -> int:
        return self.__eps_nev

    @nev.setter
    def nev(self, n: int):
        if n == self.nev:
            return

        if n <= 0:
            PETSc.Sys.Print('Argument muset be grater than 0')
            raise PETSc.Error(62)

        self.__eps_nev = n

        self.__setupcalled = False
        self.__solved = False

    @property
    def autopostsolve(self) -> bool:
        return self.__autopostsolve

    @autopostsolve.setter
    def autopostsolve(self, flag: bool):
        self.__autopostsolve = flag

    def setUp(self):
        if self.__setupcalled:
            return

        if self.__op is None:
            PETSc.Sys.Print('Operator is not set')
            raise PETSc.Error(64)

        comm = self.__op.getComm()

        if self.__eps is None:
            self.__eps = SLEPc.EPS().create(comm=comm)
        self.__eps.setDimensions(self.__eps_nev, PETSc.DECIDE)
        self.__eps.setTolerances(tol=self.__eps_tol)
        self.__eps.setProblemType(self.__eps_prob_type)

        if self.__eps_prob_type == SLEPc.EPS.ProblemType.GHEP:
            if self.__op_type is not OperatorType.LAPLACIAN_UNNORMALIZED:
                PETSc.Sys.Print('Operator type (%s) is not compatible with generalized eigenproblem related to '
                                'spectral clustering' % self.__op_type.name)
                raise PETSc.Error(56)

            if self.__op_B is None:
                N = self.__op.getSize()[0]

                self.__op_B = PETSc.Mat().createAIJ((N, N), comm=comm)
                self.__op_B.setPreallocationNNZ(1)
                self.__op_B.setFromOptions()
                self.__op_B.setUp()

                self.__op_B.setDiagonal(self.__vec_diag)
                self.__op_B.assemble()

            self.__eps.setOperators(self.__op, self.__op_B)
        else:
            self.__eps.setOperators(self.__op, None)

        if self.__op_type != OperatorType.MARKOV_1 or self.__op_type != OperatorType.MARKOV_2:
            self.__eps.setWhichEigenpairs(self.__eps.Which.SMALLEST_REAL)
        else:
            self.__eps.setWhichEigenpairs(self.__eps.Which.LARGEST_REAL)

        self.__eps.setFromOptions()
        self.__eps.setUp()

        self.__setupcalled = True
        self.__postsolvecalled = False

    def solve(self):
        if not self.__setupcalled:
            self.setUp()

        if self.__verbose:
            eig_prob_str = 'HEP' if self.__eps_prob_type == SLEPc.EPS.ProblemType.HEP else 'GHEP'
            eig_which_str = 'largest' if (self.__op_type == OperatorType.MARKOV_1 or
                                          self.__op_type == OperatorType.MARKOV_2) else 'smallest'
            PETSc.Sys.Print('Solving eigensystem (%s, %s real eigenvalues) ' % (eig_prob_str, eig_which_str))

        self.__eps.solve()
        self.__solved = True

        if self.autopostsolve:
            self.postSolve()

        if self.__verbose:
            self.view()

    def postSolve(self):
        if not self.__solved:
            self.solve()

        nconv = self.__eps.getConverged()

        if nconv == 0:
            PETSc.Sys.Print('0 eigenvalues converged')
            raise PETSc.Error(91)

        del self.__eigen_vals, self.__mat_bv, self.__errors

        self.__eigen_vals = np.zeros(shape=(nconv,))
        for i in range(nconv):
            self.__eigen_vals[i] = self.__eps.getEigenvalue(i).real

        self.__mat_bv = self.__eps.getBV().createMat()

        self.__errors = np.zeros(shape=(nconv,))
        for i in range(nconv):
            self.__errors[i] = self.__eps.computeError(i, etype=SLEPc.EPS.ErrorType.ABSOLUTE)

        self.__postsolvecalled = True

    def getSolution(self) -> (np.ndarray, PETSc.Mat):
        if self.__postsolvecalled:
            self.postSolve()

        return self.__eigen_vals, self.__mat_bv

    def getError(self) -> np.ndarray:
        if not self.__postsolvecalled:
            self.postSolve()

        return self.__errors

    def view(self):
        if not self.__postsolvecalled:
            self.postSolve()

        if self.__eps.getType() == SLEPc.EPS.Type.ARPACK:
            PETSc.Sys.Print('number of iterations (ARPACK): %d' % self.__eps.getIterationNumber())

        PETSc.Sys.Print('          k                  ||Ax-kx||      ')
        PETSc.Sys.Print(' -------------------  ----------------------')

        for i in range(self.__eps.getConverged()):
            PETSc.Sys.Print(' %.16f   %.16g' % (self.__eigen_vals[i], self.__errors[i]))
