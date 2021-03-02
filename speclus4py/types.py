from enum import Enum, auto
from typing import Callable, Union

from petsc4py import PETSc
from mpi4py import MPI

img_exts = {'.bmp', '.dib', '.jpe', '.jpg', '.jpeg', '.jp2', '.png', '.pbm', '.pgm', '.ppm', '.sr', '.ras',
            '.tif', '.tiff'}
vol_img_exts = {'.vti', '.vtk'}


class DataType(Enum):
    NONE = auto()
    GENERAL = auto()
    IMG = auto()
    VOL_IMG = auto()


class OperatorType(Enum):
    LAPLACIAN_UNNORMALIZED = 'laplacian_unnorm'
    LAPLACIAN_NORMALIZED = 'laplacian_norm'
    RANDOM_WALK = 'random_walk'
    MARKOV_1 = 'markov_1'  # by Shi-Meila
    MARKOV_2 = 'markov_2'  # by Ng-Weiss


class GraphType(Enum):
    DIRECTED = auto()
    UNDIRECTED = auto()


class EPSProblemType(Enum):
    HEP = 1
    GHEP = 2
    NHEP = 3
    GNHEP = 4


class DataObject:
    def __init__(self, comm=MPI.COMM_WORLD, verbose=False):
        self.__comm = comm

        self.__X = None
        self.__data_type = DataType.NONE

        self.__similarity_measure_fn = None
        self.__similarity_measure_params = PETSc.DEFAULT

        self.__verbose = verbose

    def reset(self):
        del self.data

    @property
    def verbose(self) -> bool:
        return self.__verbose

    @verbose.setter
    def verbose(self, flag: bool):
        self.__verbose = flag

    @property
    def comm(self) -> MPI.Intracomm:
        return self.__comm

    @comm.setter
    def comm(self, comm: MPI.Intracomm):
        self.__comm = comm

    @property
    def data(self) -> object:
        return self.__X

    @data.setter
    def data(self, X: object):
        self.__X = X

    @data.deleter
    def data(self):
        del self.__X
        self.__X = None
        self.data_type = DataType.NONE

    @property
    def data_type(self) -> DataType:
        return self.__data_type

    @data_type.setter
    def data_type(self, data_type: DataType):
        self.__data_type = data_type

    def setData(self, X: object, data_type: DataType):
        del self.data

        self.data_type = data_type
        self.data = X

    def getData(self) -> (object, DataType):
        return self.data, self.data_type

    def setSimilarityFunc(self, fn, params):
        self.__similarity_measure_fn = fn
        self.__similarity_measure_params = params

    def getSimilarityMeasure(self) -> (Callable, Union[float, list]):
        return self.fn_similarity, self.fn_similarity_params

    @property
    def fn_similarity(self) -> Callable:
        return self.__similarity_measure_fn

    @fn_similarity.setter
    def fn_similarity(self, fn: Callable):
        self.__similarity_measure_fn = fn

    @property
    def fn_similarity_params(self) -> Union[float, list]:
        return self.__similarity_measure_params

    @fn_similarity_params.setter
    def fn_similarity_params(self, params: Union[float, list]):
        self.__similarity_measure_params = params


class OperatorContainer:
    def __init__(self):
        self.__A = None
        self.__L = None
        self.__d = None

        self.__mat_type = PETSc.Mat.Type.AIJ
        self.__operator_type = OperatorType.LAPLACIAN_UNNORMALIZED

        self.__sigma = 0.5
        self.__connectivity = PETSc.DEFAULT

    def reset(self):
        del self.mat_adj, self.mat_op, self.vec_diag

    @property
    def mat_adj(self) -> PETSc.Mat:
        return self.__A

    @mat_adj.setter
    def mat_adj(self, A: PETSc.Mat):
        self.__A = A

    @mat_adj.deleter
    def mat_adj(self):
        del self.__A
        self.__A = None

    @property
    def mat_op(self) -> PETSc.Mat:
        return self.__L

    @mat_op.setter
    def mat_op(self, L: PETSc.Mat):
        self.__L = L

    @mat_op.deleter
    def mat_op(self):
        del self.__L
        self.__L = None

    @property
    def vec_diag(self) -> PETSc.Vec:
        return self.__d

    @vec_diag.setter
    def vec_diag(self, diag: PETSc.Vec):
        self.__d = diag

    @vec_diag.deleter
    def vec_diag(self):
        del self.__d
        self.__d = None

    def getOperators(self) -> (PETSc.Mat, PETSc.Mat, PETSc.Vec):
        return self.mat_op, self.mat_adj, self.vec_diag

    # @property
    # def sigma(self) -> float:
    #     return self.__sigma
    #
    # @sigma.setter
    # def sigma(self, sigma: float):
    #     if sigma == self.sigma:
    #         return
    #
    #     if sigma <= 0:
    #         PETSc.Sys.Print('Standard deviation sigma must be positive')
    #         raise PETSc.Error(62)
    #
    #     self.reset()
    #     self.__sigma = sigma

    @property
    def connectivity(self) -> (int, PETSc.DEFAULT):
        return self.__connectivity

    @connectivity.setter
    def connectivity(self, con: (int, PETSc.DEFAULT)):
        if con == self.connectivity:
            return

        if con <= 0 and con != PETSc.DEFAULT:
            PETSc.Sys.Print('Connectivity must be positive integer')
            raise PETSc.Error(62)

        self.reset()

        self.__connectivity = con

    @property
    def operator_type(self) -> OperatorType:
        return self.__operator_type

    @operator_type.setter
    def operator_type(self, type: OperatorType):
        if type == self.operator_type:
            return

        self.reset()

        self.__operator_type = type

    @property
    def mat_type(self):
        return self.__mat_type

    @mat_type.setter
    def mat_type(self, type):
        if type == self.mat_type:
            return

        self.reset()

        self.__mat_type = type
