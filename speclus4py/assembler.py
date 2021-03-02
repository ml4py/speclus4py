import numpy as np
from numba import jit
import pyflann

from petsc4py import PETSc
from mpi4py import MPI

from speclus4py.types import DataObject, DataType, GraphType, OperatorType, OperatorContainer


@jit(nopython=True)
def get_global_index(x, y, ydim):
    return y + x * ydim


@jit(nopython=True)
def get_global_index_volumetric(x, y, z, xdim, ydim):
    return x + xdim * (y + z * ydim)


@jit(nopython=True)
def compute_gradient(v1, v2, sigma: float):
    abs = np.abs(v1 - v2)
    return np.exp(-abs * abs / (2. * sigma * sigma))


@jit(nopython=True)
def compute_gradient_norm(v1, v2, sigma: float):
    norm = np.linalg.norm(v1 - v2)
    return np.exp(-norm * norm / (2. * sigma * sigma))


class OperatorAssembler(DataObject, OperatorContainer):
    def __init__(self, comm=MPI.COMM_WORLD, verbose=False):
        DataObject.__init__(self, comm, verbose)
        OperatorContainer.__init__(self)

        self.__graph_type = GraphType.DIRECTED

    @property
    def graph_type(self) -> GraphType:
        return self.__graph_type

    @graph_type.setter
    def graph_type(self, t: GraphType):
        self.__graph_type = t

    def setSimilarityFunc(self, fn, params):
        self.__similarity_measure_fn = fn
        self.__similarity_measure_params = params

    def reset(self):
        OperatorContainer.reset(self)

    def __construct_adjacency_matrix_general_data(self):
        data = self.getData()[0]
        # determine dimension of a problem
        N = data.shape[0]

        # building index (FLANN - Fast Library for Approximate Nearest Neighbors)
        pyflann.set_distance_type('euclidean')
        flann = pyflann.FLANN()
        flann.build_index(data)

        # create matrix object
        self.mat_adj = PETSc.Mat()
        self.mat_adj.create(self.comm)
        self.mat_adj.setSizes([N, N])
        self.mat_adj.setType(self.mat_type)

        if self.graph_type == GraphType.DIRECTED:
            self.__construct_adjacency_matrix_general_data_directed_graph(flann)
        else:
            self.__construct_adjacency_matrix_general_data_undirected_graph(flann)

        # finalizing assembly of adjacency matrix
        self.mat_adj.assemble()

        del flann

    def __construct_adjacency_matrix_general_data_directed_graph(self, flann):
        self.mat_adj.setPreallocationNNZ(self.connectivity)
        self.mat_adj.setFromOptions()
        self.mat_adj.setUp()

        # Get function for measuring similarity and its parameters
        sim_func, sim_func_params = self.getSimilarityMeasure()
        if sim_func is None:
            sim_func = compute_gradient_norm
        if sim_func_params == PETSc.DEFAULT:
            sim_func_params = 0.5

        data = self.getData()[0]

        # building adjacency matrix of similarity graph
        i_start, i_end = self.mat_adj.getOwnershipRange()
        for I in range(i_start, i_end):
            v1 = data[I]
            # find nearest neighbours to sample v1
            # sometimes self-adjoint vertex is included, thus finding n+1 nearest neighbours
            result, dist = flann.nn_index(v1, self.connectivity + 1)
            used_nn = 0
            for J in range(0, self.connectivity + 1):
                idx = result[0, J]
                if idx != I and used_nn < self.connectivity:
                    v2 = data[result[0, J]]
                    g = sim_func(v1, v2, sim_func_params)
                    if g > 0.:
                        self.mat_adj[I, idx] = g
                    used_nn += 1
                elif used_nn >= self.connectivity:
                    break

    def __construct_adjacency_matrix_general_data_undirected_graph(self, flann):
        self.mat_adj.setFromOptions()
        self.mat_adj.setUp()

        # Get function for measuring similarity and its parameters
        sim_func, sim_func_params = self.getSimilarityMeasure()
        if sim_func is None:
            sim_func = compute_gradient_norm
        if sim_func_params == PETSc.DEFAULT:
            sim_func_params = 0.5

        data = self.getData()[0]

        # building adjacency matrix of similarity graph
        i_start, i_end = self.mat_adj.getOwnershipRange()

        for I in range(i_start, i_end):
            v1 = data[I]
            # find nearest neighbours to sample v1
            # sometimes self-adjoint vertex is included, thus finding n+1 nearest neighbours
            result, dist = flann.nn_index(v1, self.connectivity + 1)
            for J in range(0, self.connectivity + 1):
                idx = result[0, J]
                if idx != I:
                    v2 = data[result[0, J]]

                    g = sim_func(v1, v2, sim_func_params)
                    if g > 0.:
                        self.mat_adj[I, idx] = g
                        self.mat_adj[idx, I] = g

    def __construct_adjacency_matrix_vol_img(self):
        if self.connectivity != 6 and self.connectivity != 18 and self.connectivity != 26:
            raise Exception('Connectivity (con) must be set to 6, 18, or 26')

        # Get function for measuring similarity and its parameters
        sim_func, sim_func_params = self.getSimilarityMeasure()
        if sim_func is None:
            sim_func = compute_gradient
        if sim_func_params == PETSc.DEFAULT:
            sim_func_params = 0.5

        data = self.getData()[0]

        # determine dimension of a problem
        dims = data.GetDimensions()
        dim_x = dims[0] - 1
        dim_y = dims[1] - 1
        dim_z = dims[2] - 1
        N = dim_x * dim_y * dim_z

        # create matrix object
        self.mat_adj = PETSc.Mat()
        self.mat_adj.create(self.comm)
        self.mat_adj.setSizes([N, N])
        self.mat_adj.setType(self.mat_type)
        self.mat_adj.setPreallocationNNZ(self.connectivity)
        self.mat_adj.setFromOptions()
        self.mat_adj.setUp()

        # compute local derivatives on structured non-uniform grid that is determined using sigma and
        # connectivity of derivatives (6, 18, or 26)
        data_scalars = data.GetCellData().GetScalars()

        i_start, i_end = self.mat_adj.getOwnershipRange()
        for I in range(i_start, i_end):
            # determine (x, y, z)-coordinates
            z = I // (dim_x * dim_y)
            i = I - z * dim_x * dim_y
            y = i // dim_x
            x = i - y * dim_x

            p1 = get_global_index_volumetric(x, y, z, dim_x, dim_y)
            v1 = data_scalars.GetTuple1(p1) / 255.

            if z > 0:
                if self.connectivity > 6 and y > 0:
                    if self.connectivity == 26 and x > 0:
                        p2 = get_global_index_volumetric(x - 1, y - 1, z - 1, dim_x, dim_y)
                        v2 = data_scalars.GetTuple1(p2) / 255.

                        g = sim_func(v1, v2, sim_func_params)
                        self.mat_adj[p1, p2] = g

                    p2 = get_global_index_volumetric(x, y - 1, z - 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = sim_func(v1, v2, sim_func_params)
                    self.mat_adj[p1, p2] = g

                    if self.connectivity == 26 and x < dim_x - 1:
                        p2 = get_global_index_volumetric(x + 1, y - 1, z - 1, dim_x, dim_y)
                        v2 = data_scalars.GetTuple1(p2) / 255.

                        g = sim_func(v1, v2, sim_func_params)
                        self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and x > 0:
                    p2 = get_global_index_volumetric(x - 1, y, z - 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = sim_func(v1, v2, sim_func_params)
                    self.mat_adj[p1, p2] = g

                p2 = get_global_index_volumetric(x, y, z - 1, dim_x, dim_y)
                v2 = data_scalars.GetTuple1(p2) / 255.

                g = sim_func(v1, v2, sim_func_params)
                self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and x < dim_x - 1:
                    p2 = get_global_index_volumetric(x + 1, y, z - 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = sim_func(v1, v2, sim_func_params)
                    self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and y < dim_y - 1:
                    if self.connectivity == 26 and x > 0:
                        p2 = get_global_index_volumetric(x - 1, y + 1, z - 1, dim_x, dim_y)
                        v2 = data_scalars.GetTuple1(p2) / 255.

                        g = sim_func(v1, v2, sim_func_params)
                        self.mat_adj[p1, p2] = g

                    p2 = get_global_index_volumetric(x, y + 1, z - 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = sim_func(v1, v2, sim_func_params)
                    self.mat_adj[p1, p2] = g

                    if self.connectivity == 26 and x < dim_x - 1:
                        p2 = get_global_index_volumetric(x + 1, y + 1, z - 1, dim_x, dim_y)
                        v2 = data_scalars.GetTuple1(p2) / 255.

                        g = sim_func(v1, v2, sim_func_params)
                        self.mat_adj[p1, p2] = g
            if y > 0:
                if self.connectivity > 6 and x > 0:
                    p2 = get_global_index_volumetric(x - 1, y - 1, z, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = sim_func(v1, v2, sim_func_params)
                    self.mat_adj[p1, p2] = g

                p2 = get_global_index_volumetric(x, y - 1, z, dim_x, dim_y)
                v2 = data_scalars.GetTuple1(p2) / 255.

                g = sim_func(v1, v2, sim_func_params)
                self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and x < dim_x - 1:
                    p2 = get_global_index_volumetric(x + 1, y - 1, z, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = sim_func(v1, v2, sim_func_params)
                    self.mat_adj[p1, p2] = g

            if x > 0:
                p2 = get_global_index_volumetric(x - 1, y, z, dim_x, dim_y)
                v2 = data_scalars.GetTuple1(p2) / 255.

                g = sim_func(v1, v2, sim_func_params)
                self.mat_adj[p1, p2] = g

            if x < dim_x - 1:
                p2 = get_global_index_volumetric(x + 1, y, z, dim_x, dim_y)
                v2 = data_scalars.GetTuple1(p2) / 255.

                g = sim_func(v1, v2, sim_func_params)
                self.mat_adj[p1, p2] = g

            if y < dim_y - 1:
                if self.connectivity > 6 and x > 0:
                    p2 = get_global_index_volumetric(x - 1, y + 1, z, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = sim_func(v1, v2, sim_func_params)
                    self.mat_adj[p1, p2] = g

                p2 = get_global_index_volumetric(x, y + 1, z, dim_x, dim_y)
                v2 = data_scalars.GetTuple1(p2) / 255.

                g = sim_func(v1, v2, sim_func_params)
                self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and x < dim_x - 1:
                    p2 = get_global_index_volumetric(x + 1, y + 1, z, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = sim_func(v1, v2, sim_func_params)
                    self.mat_adj[p1, p2] = g

            if z < dim_z - 1:
                if self.connectivity > 6 and y > 0:
                    if self.connectivity == 26 and x > 0:
                        p2 = get_global_index_volumetric(x - 1, y - 1, z + 1, dim_x, dim_y)
                        v2 = data_scalars.GetTuple1(p2) / 255.

                        g = sim_func(v1, v2, sim_func_params)
                        self.mat_adj[p1, p2] = g

                    p2 = get_global_index_volumetric(x, y - 1, z + 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = sim_func(v1, v2, sim_func_params)
                    self.mat_adj[p1, p2] = g

                    if self.connectivity == 26 and x < dim_x - 1:
                        p2 = get_global_index_volumetric(x + 1, y - 1, z + 1, dim_x, dim_y)
                        v2 = data_scalars.GetTuple1(p2) / 255.

                        g = sim_func(v1, v2, sim_func_params)
                        self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and x > 0:
                    p2 = get_global_index_volumetric(x - 1, y, z + 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = sim_func(v1, v2, sim_func_params)
                    self.mat_adj[p1, p2] = g

                p2 = get_global_index_volumetric(x, y, z + 1, dim_x, dim_y)
                v2 = data_scalars.GetTuple1(p2) / 255.

                g = sim_func(v1, v2, sim_func_params)
                self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and x < dim_x - 1:
                    p2 = get_global_index_volumetric(x + 1, y, z + 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = sim_func(v1, v2, sim_func_params)
                    self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and y < dim_y - 1:
                    if self.connectivity == 26 and x > 0:
                        p2 = get_global_index_volumetric(x - 1, y + 1, z + 1, dim_x, dim_y)
                        v2 = data_scalars.GetTuple1(p2) / 255.

                        g = sim_func(v1, v2, sim_func_params)
                        self.mat_adj[p1, p2] = g

                    p2 = get_global_index_volumetric(x, y + 1, z + 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = sim_func(v1, v2, sim_func_params)
                    self.mat_adj[p1, p2] = g

                    if self.connectivity == 26 and x < dim_x - 1:
                        p2 = get_global_index_volumetric(x + 1, y + 1, z + 1, dim_x, dim_y)
                        v2 = data_scalars.GetTuple1(p2) / 255.

                        g = sim_func(v1, v2, sim_func_params)
                        self.mat_adj[p1, p2] = g

        # finalizing assembly of adjacency matrix
        self.mat_adj.assemble()

    def __construct_adjacency_matrix_img(self):
        if self.connectivity != 4 and self.connectivity != 8:
            PETSc.Sys.Print('Connectivity (con) must be set to 4 or 8')
            raise PETSc.Error(62)

        rows = self.data.shape[0]
        cols = self.data.shape[1]
        N = rows * cols

        # Get function for measuring similarity and its parameters
        sim_func, sim_func_params = self.getSimilarityMeasure()
        if sim_func is None:
            if len(self.data.shape) == 3:
                sim_func = compute_gradient_norm
            else:
                sim_func = compute_gradient
        if sim_func_params == PETSc.DEFAULT:
            sim_func_params = 0.5

        data = self.getData()[0]

        # create matrix object
        self.mat_adj = PETSc.Mat()
        self.mat_adj.create(self.comm)
        self.mat_adj.setSizes([N, N])
        self.mat_adj.setType(self.mat_type)
        self.mat_adj.setPreallocationNNZ(self.connectivity)
        self.mat_adj.setFromOptions()
        self.mat_adj.setUp()

        i_start, i_end = self.mat_adj.getOwnershipRange()

        for I in range(i_start, i_end):
            # determine (x, y) coordinates
            x = I // cols
            y = I - x * cols

            p1 = I
            v1 = self.data[x, y] / 255.

            if x > 0:
                if y > 0 and self.connectivity == 8:
                    p2 = get_global_index(x - 1, y - 1, cols)
                    v2 = data[x - 1, y - 1] / 255.
                    self.mat_adj[p1, p2] = sim_func(v1, v2, sim_func_params)

                p2 = get_global_index(x - 1, y, cols)
                v2 = data[x - 1, y] / 255.
                self.mat_adj[p1, p2] = sim_func(v1, v2, sim_func_params)

                if y < cols - 1 and self.connectivity == 8:
                    p2 = get_global_index(x - 1, y + 1, cols)
                    v2 = data[x - 1, y + 1] / 255.
                    self.mat_adj[p1, p2] = sim_func(v1, v2, sim_func_params)

            if y > 0:
                p2 = get_global_index(x, y - 1, cols)
                v2 = data[x, y - 1] / 255.
                self.mat_adj[p1, p2] = sim_func(v1, v2, sim_func_params)

            if y < cols - 1:
                p2 = get_global_index(x, y + 1, cols)
                v2 = data[x, y + 1] / 255.
                self.mat_adj[p1, p2] = sim_func(v1, v2, sim_func_params)

            if x < rows - 1:
                if y > 0 and self.connectivity == 8:
                    p2 = get_global_index(x + 1, y - 1, cols)
                    v2 = data[x + 1, y - 1] / 255.
                    self.mat_adj[p1, p2] = sim_func(v1, v2, sim_func_params)

                p2 = get_global_index(x + 1, y, cols)
                v2 = data[x + 1, y] / 255.
                self.mat_adj[p1, p2] = sim_func(v1, v2, sim_func_params)

                if y < cols - 1 and self.connectivity == 8:
                    p2 = get_global_index(x + 1, y + 1, cols)
                    v2 = data[x + 1, y + 1] / 255.
                    self.mat_adj[p1, p2] = sim_func(v1, v2, sim_func_params)

        # finalizing assembly of adjacency matrix
        self.mat_adj.assemble()

    def assembly(self):
        self.reset()

        data_type = self.getData()[1]

        if self.fn_similarity_params is not None and self.verbose:
            if type(self.fn_similarity_params) == float:
                str_params = ', param=%.2f' % self.fn_similarity_params
            else:
                str_params = ', params=['
                str_params += ''.join('{}, '.format(k) for k in self.fn_similarity_params)
                str_params = str_params[:-2] + ']'
        else:
            str_params = ''

        if data_type == DataType.IMG:
            if self.connectivity == PETSc.DEFAULT:
                self.connectivity = 4

            if self.verbose:
                s = 'Construct operator (%s, GRAPH_%s) for image: connectivity=%d'
                v = (self.operator_type.name, GraphType.UNDIRECTED.name, self.connectivity)

                PETSc.Sys.Print(s % v + str_params)

            self.__construct_adjacency_matrix_img()

        elif data_type == DataType.VOL_IMG:
            if self.connectivity == PETSc.DEFAULT:
                self.connectivity = 6

            if self.verbose:
                s = 'Construct operator (%s, GRAPH_%s) for volumetric image: connectivity=%d'
                v = (self.operator_type.name, self.graph_type.name, self.connectivity)

                PETSc.Sys.Print(s % v + str_params)

            self.__construct_adjacency_matrix_vol_img()

        else:
            if self.connectivity == PETSc.DEFAULT:
                self.connectivity = 3

            if self.verbose:
                s = 'Construct operator (%s,  GRAPH_%s) for general data: connectivity=%d'
                v = (self.operator_type.name, self.graph_type.name, self.connectivity)

                PETSc.Sys.Print(s % v + str_params)

            self.__construct_adjacency_matrix_general_data()


        # if data_type == DataType.IMG:
        #     if self.connectivity == PETSc.DEFAULT:
        #         self.connectivity = 4
        #
        #     if self.verbose:
        #         PETSc.Sys.Print(
        #             'Construct operator (%s) for image: connectivity=%d, sigma=%2g'
        #             % (self.operator_type.name, self.connectivity, self.sigma)
        #         )
        #
        #     self.__construct_adjacency_matrix_img()
        # elif data_type == DataType.VOL_IMG:  # volumetric image
        #     if self.connectivity == PETSc.DEFAULT:
        #         self.connectivity = 6
        #
        #     if self.verbose:
        #         if self.fn_similarity_params is not None:
        #             s = 'Construct operator (%s, GRAPH_ %s) for volumetric image: connectivity=%d, '
        #             v = (self.operator_type.name, self.graph_type.name, self.connectivity)
        #             sv = s % v
        #             if type(self.fn_similarity_params) == float:
        #                 sp = 'param=%.2f' % self.fn_similarity_params
        #             else:
        #                 sp = 'params=('
        #                 sp += ''.join('{}, '.format(k) for k in self.fn_similarity_params)
        #                 sp = sp[:-2] + ')'
        #             sv += sp
        #         else:
        #             s = 'Construct operator (%s, GRAPH_%s) for volumetric image: connectivity=%d params=None'
        #             v = (self.operator_type.name, self.graph_type.name, self.connectivity)
        #             sv = s % v
        #         PETSc.Sys.Print(sv)
        #
        #     exit(-1)
        #
        #     self.__construct_adjacency_matrix_vol_img()
        # else:
        #     if self.connectivity == PETSc.DEFAULT:
        #         self.connectivity = 6
        #
        #     if self.verbose:
        #         PETSc.Sys.Print(
        #             'Construct operator (%s) for general data: connectivity=%d, params=%2g'
        #             % (self.operator_type.name, self.connectivity, self.__similarity_measure_params)
        #         )
        #
        #     self.__construct_adjacency_matrix_general_data()

        N = self.mat_adj.getSize()[0]

        # compute degree matrix D_i = deg(v_i)
        self.vec_diag = self.mat_adj.createVecLeft()
        self.mat_adj.getRowSum(self.vec_diag)

        if self.operator_type != OperatorType.MARKOV_1 or self.operator_type != OperatorType.MARKOV_2:
            self.mat_op = PETSc.Mat().createAIJ((N, N), comm=self.comm)
            self.mat_op.setPreallocationNNZ(self.connectivity + 1)
            self.mat_op.setFromOptions()
            self.mat_op.setUp()

            self.mat_op.setDiagonal(self.vec_diag)
            self.mat_op.assemble()

            # L = D - A
            self.mat_op.axpy(-1., self.mat_adj)
        else:  # P = D^-1 A (MARKOV_1) or Ng, Weiss (MARKOV_2)
            self.mat_op = self.mat_adj.duplicate()
            self.mat_op.setFromOptions()
            self.mat_op.setType(self.mat_type)
            self.mat_op.setUp()
            self.mat_op.copy(self.mat_op)

        if self.operator_type != OperatorType.LAPLACIAN_UNNORMALIZED:
            tmp_vec = self.vec_diag.duplicate()
            self.vec_diag.copy(tmp_vec)

            if self.operator_type == OperatorType.LAPLACIAN_NORMALIZED or self.operator_type == OperatorType.MARKOV_2:
                tmp_vec.sqrtabs()
                tmp_vec.reciprocal()
                self.mat_op.diagonalScale(tmp_vec, tmp_vec)
            elif self.operator_type == OperatorType.MARKOV_1:
                tmp_vec.reciprocal()
                self.mat_op.diagonalScale(tmp_vec)
            else:  # L_rw
                tmp_vec.reciprocal()
                self.mat_op.diagonalScale(tmp_vec)  # left diagonal scale

            del tmp_vec

        self.mat_op.assemble()
