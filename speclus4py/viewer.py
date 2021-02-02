import pathlib, os

import cv2 as opencv
import vtk

from petsc4py import PETSc

from speclus4py.types import DataType, img_exts, vol_img_exts


class Viewer:
    def __init__(self, filename=os.path.join(''), verbose=False):
        self.__filename = filename
        self.__verbose = verbose

        self.__ocv_imread_mode = opencv.IMREAD_GRAYSCALE
        self.__ocv_equalize_img = False

    @property
    def filename_input(self) -> os.path:
        return self.__filename

    @filename_input.setter
    def filename_input(self, filename: os.path):
        if not os.path.exists(filename):
            PETSc.Sys.Print('Input file does not exist')
            raise PETSc.Error(62)

        self.__filename = filename

    @property
    def ocv_imread_mode(self):
        return self.__ocv_imread_mode

    @ocv_imread_mode.setter
    def ocv_imread_mode(self, mode):
        self.__ocv_imread_mode = mode

    @property
    def ocv_equalize_img(self) -> bool:
        return self.__ocv_equalize_img

    @ocv_equalize_img.setter
    def ocv_equalize_img(self, flag: bool):
        self.__ocv_equalize_img = flag

    def load_image(self):
        try:
            image_data = opencv.imread(self.filename_input, self.ocv_imread_mode)
        except opencv.error:
            PETSc.Sys.Print('Loading file \'%s\' failed' % self.filename_input)
            raise PETSc.Error(66)

        # TODO move to preprocessing data class
        if self.ocv_equalize_img:
            image_data = opencv.equalizeHist(image_data)

        if self.__verbose:
            rows = image_data.shape[0]
            cols = image_data.shape[1]

            PETSc.Sys.Print(' #rows %d \n #cols %d' % (rows, cols))
            if len(image_data.shape) == 3:
                PETSc.Sys.Print(' #channels 3 (color image)')

        return [image_data, DataType.IMG]

    def load_volumetric_image(self):
        vtk_reader = vtk.vtkXMLImageDataReader()
        vtk_reader.SetFileName(self.filename_input)
        vtk_reader.Update()

        image_data = vtk_reader.GetOutput()
        del vtk_reader

        if self.__verbose:
            dims = image_data.GetDimensions()
            N = (dims[0] - 1) * (dims[1] - 1) * (dims[2] - 1)
            PETSc.Sys.Print(' #cells %d loaded' % N)

        return [image_data, DataType.VOL_IMG]

    def load(self) -> (object, DataType):
        ext = pathlib.Path(self.filename_input).suffix

        if ext in vol_img_exts:
            return self.load_volumetric_image()
        elif ext in img_exts:
            return self.load_image()
        else:
            PETSc.Sys.Print('File format \'%s\' is not supported.' % ext)
            raise PETSc.Error(56)
