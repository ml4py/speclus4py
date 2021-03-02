import matplotlib.pyplot as plt
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import speclus4py.sys as Sys


class TwoSpirals:
    def __init__(self, comm=MPI.COMM_WORLD, verbose=False):
        self.comm = comm
        self.verbose = verbose

        self.__nsamples = 10
        self.__nsamples_s1 = -1
        self.__nsamples_s2 = -1

        self.__initial_radius_s1 = np.pi / 4
        self.__initial_radius_s2 = np.pi / 4
        self.__gap = 0.5
        self.__noise_level = 0.1

        self.__samples = self.__labels = None
        pass

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
    def initial_radius1(self) -> float:
        return self.__initial_radius_s1

    @initial_radius1.setter
    def initial_radius1(self, r: float):
        self.__initial_radius_s1 = r
        self.reset()

    @property
    def initial_radius2(self) -> float:
        return self.__initial_radius_s2

    @initial_radius2.setter
    def initial_radius2(self, r: float):
        self.__initial_radius_s1 = r
        self.reset()

    @property
    def gap(self) -> float:
        return self.__gap

    @gap.setter
    def gap(self, g: float):
        self.__gap = g
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

        self.__nsamples_s1 = self.nsamples // 2
        self.__nsamples_s2 = self.nsamples - self.__nsamples_s1

        if self.verbose:
            vals = (self.nsamples, self.__nsamples_s1, self.__nsamples_s2, self.__initial_radius_s1,
                    self.__initial_radius_s2, self.gap, self.noise_level)
            PETSc.Sys.Print('Generating TwoSpiral (arithmetic) dataset: #samples=%d (#s1=%d and #s2=%d), '
                            'initial_radius_s1=%.2f, initial_radius_s2=%.2f, gap=%.2f, noise=%.2f' % vals)

        linspace_s1 = np.linspace(0, 2 * np.pi, self.__nsamples_s1)
        linspace_s2 = np.linspace(0, 2 * np.pi, self.__nsamples_s2)

        s1_r = self.gap * linspace_s1 + self.initial_radius1
        s1_x = s1_r * (np.cos(linspace_s1) + np.random.rand(self.__nsamples_s1) * self.noise_level)
        s1_y = s1_r * (np.sin(linspace_s1) + np.random.rand(self.__nsamples_s1) * self.noise_level)

        s2_r = -(self.gap * linspace_s2 + self.initial_radius2)
        s2_x = s2_r * (np.cos(linspace_s2) + np.random.rand(self.__nsamples_s2) * self.noise_level)
        s2_y = s2_r * (np.sin(linspace_s2) + np.random.rand(self.__nsamples_s2) * self.noise_level)

        coordinates_x = np.append(s1_x, s2_x)
        coordinates_y = np.append(s1_y, s2_y)

        self.__samples = np.vstack([coordinates_x, coordinates_y]).T
        self.__labels = np.hstack((np.zeros(self.__nsamples_s1), np.ones(self.__nsamples_s2)))
        Sys.FUNCTION_TRACE_END()

    def view(self):
        rank = self.comm.Get_rank()

        if rank == 0:
            data = self.__samples

            if self.noise_level > 0:
                v = (self.__nsamples_s1, self.__nsamples_s2, self.gap, self.noise_level)
                s = 'TwoSpiral dataset (#s1=%d, #s2=%d, gap=%.2f, noise=%.2f)'
            else:
                v = (self.__nsamples_s1, self.__nsamples_s2, self.gap)
                s = 'TwoSpiral dataset (#s1=%d, #s2=%d, gap=%.2f)'
            plt.title(s % v)

            plt.scatter(data[0:self.__nsamples_s1, 0], data[0:self.__nsamples_s1, 1])
            plt.scatter(data[self.__nsamples_s2:, 0], data[self.__nsamples_s2:, 1])
            plt.show()
