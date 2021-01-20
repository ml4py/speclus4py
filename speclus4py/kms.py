from petsc4py import PETSc

import cv2 as opencv


def kms(data, k=2, max_iter=100, restarts=5, tol=0.1, verbose=False):
    criteria = (opencv.TERM_CRITERIA_EPS + opencv.TERM_CRITERIA_MAX_ITER, max_iter, tol)
    flags = opencv.KMEANS_PP_CENTERS

    # implement cuda k-means
    # allow to define different stopping criteria
    # online cluster actualization
    rss, labels, centers = opencv.kmeans(data, k, None, criteria, restarts, flags)
    if verbose:
        PETSc.Sys.Print('(k-means) best compactness: %.5e' % rss)

    return labels, centers
