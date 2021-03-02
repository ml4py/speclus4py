import vtk
import numpy as np
import matplotlib.pyplot as plt


def vis_result_2phase(image_data, labels, colors=None, window_size=(1024, 768)):
    # determine dimension of problem
    dims = image_data.GetDimensions()
    N = (dims[0] - 1) * (dims[1] - 1) * (dims[2] - 1)

    #  The color transfer function maps voxel intensities to colors
    color = vtk.vtkColorTransferFunction()
    # The opacity transfer function is used to control the opacity
    opacity = vtk.vtkPiecewiseFunction()

    opacity.AddPoint(0., 0.0)
    opacity.AddPoint(1., 1.)

    if colors is None:
        color.AddRGBPoint(0., 1., 1., 1.)
        color.AddRGBPoint(1., .467, .646, .345)
        color.AddRGBPoint(2., 1., 0., 0.)
        color.AddRGBPoint(3., 0.5, 0., 0.)

    # determine label of solid
    solid_label = 0
    for i in range(N):
        c = image_data.GetCellData().GetScalars().GetTuple1(i)
        if c == 255.:
            solid_label = labels[i]
            break

    for i in range(N):
        c = image_data.GetCellData().GetScalars().GetTuple1(i)
        if labels[i] == solid_label:
            if c == 255.:
                image_data.GetCellData().GetScalars().SetTuple1(i, 1.)
            else:
                image_data.GetCellData().GetScalars().SetTuple1(i, 2.)
        else:
            if c == 0.:
                image_data.GetCellData().GetScalars().SetTuple1(i, 0.)
            else:
                image_data.GetCellData().GetScalars().SetTuple1(i, 3.)

    # Set other volume properties
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color)
    volume_property.SetScalarOpacity(opacity)
    volume_property.ShadeOn()

    # Volume mapper
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(image_data)

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    # create render
    render = vtk.vtkRenderer()
    render.AddViewProp(volume)
    render.SetBackground(1.0, 1.0, 1.0)
    render.ResetCamera()

    # render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(window_size[0], window_size[1])
    render_window.AddRenderer(render)

    render_window.Render()
    render_interactor = vtk.vtkRenderWindowInteractor()
    render_interactor.SetRenderWindow(render_window)
    render_interactor.Initialize()
    render_interactor.Start()

    # TODO determine solid labels as maximum of image array values


def result_syntehic(data: np.ndarray, labels: np.ndarray):
    label_min = labels.min()
    samples1_idx = labels == label_min
    samples2_idx = np.logical_not(samples1_idx)

    plt.title('Synthetic data result')
    plt.scatter(data[samples1_idx, 0], data[samples1_idx, 1])
    plt.scatter(data[samples2_idx, 0], data[samples2_idx, 1])
    plt.show()


