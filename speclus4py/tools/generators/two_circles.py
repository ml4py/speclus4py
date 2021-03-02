import matplotlib.pyplot as plt
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import speclus4py.sys as Sys


class TwoCircles:
    def __init__(self, comm=MPI.COMM_WORLD, verbose=False):
        self.comm = comm
        self.verbose = verbose

        self.__nsamples = 100
        self.__nsamples_in = -1
        self.__nsamples_out = -1

        self.__radius_outer_circle = 1.0
        self.__radius_inner_circle = 0.5
        self.__noise_level = 0.01

        self.__samples = self.__labels = None

    def __del__(self):
        self.reset()

    def reset(self):
        del self.__samples, self.__labels
        self.__samples = self.__labels = None

    @property
    def comm(self) -> MPI.Intracomm:
        return self.__comm

    @comm.setter
    def comm(self, comm: MPI.Intracomm):
        self.__comm = comm

    @property
    def verbose(self) -> bool:
        return self.__verbose

    @verbose.setter
    def verbose(self, f: bool):
        self.__verbose = f

    @property
    def nsamples(self) -> int:
        return self.__nsamples

    @nsamples.setter
    def nsamples(self, n: int):
        self.__nsamples = n
        self.reset()

    @property
    def radius_outer_circle(self) -> float:
        return self.__radius_outer_circle

    @radius_outer_circle.setter
    def radius_outer_circle(self, r: float):
        self.__radius_outer_circle = r
        self.reset()

    @property
    def radius_inner_circle(self) -> float:
        return self.__radius_inner_circle

    @radius_inner_circle.setter
    def radius_inner_circle(self, r: float):
        self.__radius_inner_circle = r
        self.reset()

    @property
    def noise_level(self) -> float:
        return self.__noise_level

    @noise_level.setter
    def noise_level(self, level: float):
        self.__noise_level = level
        self.reset()

    def getSamples(self) -> (np.ndarray, np.ndarray):
        if self.__samples is None:
            self.generate()

        return self.__samples, self.__labels

    def generate(self):
        Sys.FUNCTION_TRACE_BEGIN()
        self.reset()

        np.random.seed(0)

        self.__nsamples_in = self.nsamples // 2
        self.__nsamples_out = self.nsamples - self.__nsamples_in

        if self.verbose:
            vals = (self.nsamples, self.__nsamples_out, self.__nsamples_in, self.noise_level)
            PETSc.Sys.Print('Generating TwoCircle dataset: #samples=%d (#outer %d and #inner %d), noise=%.2f' % vals)

        linspace_outer = np.linspace(0, 2 * np.pi, self.__nsamples_out, endpoint=False)
        linspace_inner = np.linspace(0, 2 * np.pi, self.__nsamples_in, endpoint=False)

        r_outer = self.radius_outer_circle
        circle_outer_x = r_outer * np.cos(linspace_outer) + np.random.rand(self.__nsamples_out) * r_outer * self.noise_level
        circle_outer_y = r_outer * np.sin(linspace_outer) + np.random.rand(self.__nsamples_out) * r_outer * self.noise_level

        r_inner = self.radius_inner_circle
        circle_inner_x = r_inner * np.cos(linspace_inner) + np.random.rand(self.__nsamples_in) * r_inner * self.noise_level
        circle_inner_y = r_inner * np.sin(linspace_inner) + np.random.rand(self.__nsamples_in) * r_inner * self.noise_level

        coordinates_x = np.append(circle_outer_x, circle_inner_x)
        coordinates_y = np.append(circle_outer_y, circle_inner_y)

        self.__samples = np.vstack([coordinates_x, coordinates_y]).T
        self.__labels = np.hstack((np.zeros(self.__nsamples_out), np.ones(self.__nsamples_in)))

        Sys.FUNCTION_TRACE_END()

    def view(self):
        rank = self.comm.Get_rank()

        if rank == 0:
            data = self.__samples

            if self.noise_level > 0:
                vals = (self.__nsamples_in, self.__nsamples_out, self.noise_level)
                plt.title('Two Circles (#outer=%d, #inner=%d, noise=%.2f)' % vals)
            else:
                plt.title('Two Circles (#outer=%d, #inner=%d)' % (self.__nsamples_in, self.__nsamples_out))

            plt.scatter(data[0:self.__nsamples_out, 0], data[0:self.__nsamples_out, 1])
            plt.scatter(data[self.__nsamples_out:, 0], data[self.__nsamples_out:, 1])
            plt.show()

# TODO shuffle samples
