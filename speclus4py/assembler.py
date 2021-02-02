import numpy as np
from numba import jit

from petsc4py import PETSc
from mpi4py import MPI

from speclus4py.types import DataObject, DataType, OperatorType, OperatorContainer


@jit(nopython=True)
def get_global_index(x, y, ydim):
    return y + x * ydim


@jit(nopython=True)
def get_global_index_volumetric(x, y, z, xdim, ydim):
    return x + xdim * (y + z * ydim)


@jit(nopython=True)
def compute_gradient(v1, v2, sigma):
    return np.exp(-np.abs(v1 - v2) / (2. * sigma * sigma))


@jit(nopython=True)
def compute_gradient_norm(v1, v2, sigma):
    return np.exp(-np.linalg.norm(v1 - v2) / (2. * sigma * sigma))


class OperatorAssembler(DataObject, OperatorContainer):
    def __init__(self, comm=MPI.COMM_WORLD, verbose=False):
        DataObject.__init__(self, comm, verbose)
        OperatorContainer.__init__(self)

    def reset(self):
        OperatorContainer.reset(self)

    def __construct_adjacency_matrix_vol_img(self):
        if self.connectivity != 6 and self.connectivity != 18 and self.connectivity != 26:
            raise Exception('Connectivity (con) must be set to 6, 18, or 26')

        data = self.getData()[0]

        # determine dimension of a problem
        dims = data.GetDimensions()
        dim_x = dims[0] - 1  # why -1
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

                        g = compute_gradient(v1, v2, self.sigma)
                        self.mat_adj[p1, p2] = g

                    p2 = get_global_index_volumetric(x, y - 1, z - 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = compute_gradient(v1, v2, self.sigma)
                    self.mat_adj[p1, p2] = g

                    if self.connectivity == 26 and x < dim_x - 1:
                        p2 = get_global_index_volumetric(x + 1, y - 1, z - 1, dim_x, dim_y)
                        v2 = data_scalars.GetTuple1(p2) / 255.

                        g = compute_gradient(v1, v2, self.sigma)
                        self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and x > 0:
                    p2 = get_global_index_volumetric(x - 1, y, z - 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = compute_gradient(v1, v2, self.sigma)
                    self.mat_adj[p1, p2] = g

                p2 = get_global_index_volumetric(x, y, z - 1, dim_x, dim_y)
                v2 = data_scalars.GetTuple1(p2) / 255.

                g = compute_gradient(v1, v2, self.sigma)
                self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and x < dim_x - 1:
                    p2 = get_global_index_volumetric(x + 1, y, z - 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = compute_gradient(v1, v2, self.sigma)
                    self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and y < dim_y - 1:
                    if self.connectivity == 26 and x > 0:
                        p2 = get_global_index_volumetric(x - 1, y + 1, z - 1, dim_x, dim_y)
                        v2 = data_scalars.GetTuple1(p2) / 255.

                        g = compute_gradient(v1, v2, self.sigma)
                        self.mat_adj[p1, p2] = g

                    p2 = get_global_index_volumetric(x, y + 1, z - 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = compute_gradient(v1, v2, self.sigma)
                    self.mat_adj[p1, p2] = g

                    if self.connectivity == 26 and x < dim_x - 1:
                        p2 = get_global_index_volumetric(x + 1, y + 1, z - 1, dim_x, dim_y)
                        v2 = data_scalars.GetTuple1(p2) / 255.

                        g = compute_gradient(v1, v2, self.sigma)
                        self.mat_adj[p1, p2] = g
            if y > 0:
                if self.connectivity > 6 and x > 0:
                    p2 = get_global_index_volumetric(x - 1, y - 1, z, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = compute_gradient(v1, v2, self.sigma)
                    self.mat_adj[p1, p2] = g

                p2 = get_global_index_volumetric(x, y - 1, z, dim_x, dim_y)
                v2 = data_scalars.GetTuple1(p2) / 255.

                g = compute_gradient(v1, v2, self.sigma)
                self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and x < dim_x - 1:
                    p2 = get_global_index_volumetric(x + 1, y - 1, z, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = compute_gradient(v1, v2, self.sigma)
                    self.mat_adj[p1, p2] = g

            if x > 0:
                p2 = get_global_index_volumetric(x - 1, y, z, dim_x, dim_y)
                v2 = data_scalars.GetTuple1(p2) / 255.

                g = compute_gradient(v1, v2, self.sigma)
                self.mat_adj[p1, p2] = g

            if x < dim_x - 1:
                p2 = get_global_index_volumetric(x + 1, y, z, dim_x, dim_y)
                v2 = data_scalars.GetTuple1(p2) / 255.

                g = compute_gradient(v1, v2, self.sigma)
                self.mat_adj[p1, p2] = g

            if y < dim_y - 1:
                if self.connectivity > 6 and x > 0:
                    p2 = get_global_index_volumetric(x - 1, y + 1, z, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = compute_gradient(v1, v2, self.sigma)
                    self.mat_adj[p1, p2] = g

                p2 = get_global_index_volumetric(x, y + 1, z, dim_x, dim_y)
                v2 = data_scalars.GetTuple1(p2) / 255.

                g = compute_gradient(v1, v2, self.sigma)
                self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and x < dim_x - 1:
                    p2 = get_global_index_volumetric(x + 1, y + 1, z, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = compute_gradient(v1, v2, self.sigma)
                    self.mat_adj[p1, p2] = g

            if z < dim_z - 1:
                if self.connectivity > 6 and y > 0:
                    if self.connectivity == 26 and x > 0:
                        p2 = get_global_index_volumetric(x - 1, y - 1, z + 1, dim_x, dim_y)
                        v2 = data_scalars.GetTuple1(p2) / 255.

                        g = compute_gradient(v1, v2, self.sigma)
                        self.mat_adj[p1, p2] = g

                    p2 = get_global_index_volumetric(x, y - 1, z + 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = compute_gradient(v1, v2, self.sigma)
                    self.mat_adj[p1, p2] = g

                    if self.connectivity == 26 and x < dim_x - 1:
                        p2 = get_global_index_volumetric(x + 1, y - 1, z + 1, dim_x, dim_y)
                        v2 = data_scalars.GetTuple1(p2) / 255.

                        g = compute_gradient(v1, v2, self.sigma)
                        self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and x > 0:
                    p2 = get_global_index_volumetric(x - 1, y, z + 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = compute_gradient(v1, v2, self.sigma)
                    self.mat_adj[p1, p2] = g

                p2 = get_global_index_volumetric(x, y, z + 1, dim_x, dim_y)
                v2 = data_scalars.GetTuple1(p2) / 255.

                g = compute_gradient(v1, v2, self.sigma)
                self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and x < dim_x - 1:
                    p2 = get_global_index_volumetric(x + 1, y, z + 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.

                    g = compute_gradient(v1, v2, self.sigma)
                    self.mat_adj[p1, p2] = g

                if self.connectivity > 6 and y < dim_y - 1:
                    if self.connectivity == 26 and x > 0:
                        p2 = get_global_index_volumetric(x - 1, y + 1, z + 1, dim_x, dim_y)
                        v2 = data_scalars.GetTuple1(p2) / 255.

                        g = compute_gradient(v1, v2, self.sigma)
                        self.mat_adj[p1, p2] = g

                    p2 = get_global_index_volumetric(x, y + 1, z + 1, dim_x, dim_y)
                    v2 = data_scalars.GetTuple1(p2) / 255.
                    self.mat_adj[p1, p2] = compute_gradient(v1, v2, self.sigma)

                    if self.connectivity == 26 and x < dim_x - 1:
                        p2 = get_global_index_volumetric(x + 1, y + 1, z + 1, dim_x, dim_y)
                        v2 = data_scalars.GetTuple1(p2) / 255.
                        self.mat_adj[p1, p2] = compute_gradient(v1, v2, self.sigma)

        # finalizing assembly of adjacency matrix
        self.mat_adj.assemble()

    def __construct_adjacency_matrix_img(self):
        if self.connectivity != 4 and self.connectivity != 8:
            PETSc.Sys.Print('Connectivity (con) must be set to 4 or 8')
            raise PETSc.Error(62)

        rows = self.data.shape[0]
        cols = self.data.shape[1]
        N = rows * cols

        if len(self.data.shape) == 3:
            func_compute_gradient = compute_gradient_norm
        else:
            func_compute_gradient = compute_gradient

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
                    self.mat_adj[p1, p2] = func_compute_gradient(v1, v2, self.sigma)

                p2 = get_global_index(x - 1, y, cols)
                v2 = data[x - 1, y] / 255.
                self.mat_adj[p1, p2] = func_compute_gradient(v1, v2, self.sigma)

                if y < cols - 1 and self.connectivity == 8:
                    p2 = get_global_index(x - 1, y + 1, cols)
                    v2 = data[x - 1, y + 1] / 255.
                    self.mat_adj[p1, p2] = func_compute_gradient(v1, v2, self.sigma)

            if y > 0:
                p2 = get_global_index(x, y - 1, cols)
                v2 = data[x, y - 1] / 255.
                self.mat_adj[p1, p2] = func_compute_gradient(v1, v2, self.sigma)

            if y < cols - 1:
                p2 = get_global_index(x, y + 1, cols)
                v2 = data[x, y + 1] / 255.
                self.mat_adj[p1, p2] = func_compute_gradient(v1, v2, self.sigma)

            if x < rows - 1:
                if y > 0 and self.connectivity == 8:
                    p2 = get_global_index(x + 1, y - 1, cols)
                    v2 = data[x + 1, y - 1] / 255.
                    self.mat_adj[p1, p2] = func_compute_gradient(v1, v2, self.sigma)

                p2 = get_global_index(x + 1, y, cols)
                v2 = data[x + 1, y] / 255.
                self.mat_adj[p1, p2] = func_compute_gradient(v1, v2, self.sigma)

                if y < cols - 1 and self.connectivity == 8:
                    p2 = get_global_index(x + 1, y + 1, cols)
                    v2 = data[x + 1, y + 1] / 255.
                    self.mat_adj[p1, p2] = func_compute_gradient(v1, v2, self.sigma)

        # finalizing assembly of adjacency matrix
        self.mat_adj.assemble()

    def assembly(self):
        self.reset()

        data_type = self.getData()[1]

        if data_type == DataType.IMG:
            if self.connectivity == PETSc.DEFAULT:
                self.connectivity = 4

            if self.verbose:
                PETSc.Sys.Print(
                    'Construct operator (%s) for image: connectivity=%d, sigma=%2g'
                    % (self.operator_type.name, self.connectivity, self.sigma)
                )

            self.__construct_adjacency_matrix_img()
        else:  # volumetric image
            if self.connectivity == PETSc.DEFAULT:
                self.connectivity = 6

            if self.verbose:
                PETSc.Sys.Print(
                    'Construct operator (%s) for volumetric image: connectivity=%d, sigma=%2g'
                    % (self.operator_type.name, self.connectivity, self.sigma)
                )

            self.__construct_adjacency_matrix_vol_img()
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
