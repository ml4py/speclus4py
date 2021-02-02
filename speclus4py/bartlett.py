from petsc4py import PETSc

import math
import numpy as np
from scipy import stats


def estimateNullspaceDimension(eigvals: np.ndarray, error_corr=False, error=None, sort=False, conf_level=0.05,
                               verbose=True) -> (int, np.ndarray):
    if verbose:
        PETSc.Sys.Print('Determining null-space dimension using reversed Bartlett test (conf_level=%.3f, '
                        'sort_eigvals=%r, error_correction=%r)' % (conf_level, sort, error_corr))
    if error_corr and error is None:
        PETSc.Sys.Print('Absolute errors related to eigenpairs are not specified')
        return -1, None

    d = eigvals.shape[0]  # get number of eigenvalues

    _eigvals = np.copy(eigvals)
    if sort or np.where(_eigvals[np.where(_eigvals >= 0)[0][0]:-1] < 0)[0].size > 0:
        if not sort and verbose:
            PETSc.Sys.Print('  Warning: eigenvalues have been sorted!')
        _eigvals = np.sort(_eigvals)

    if error_corr:
        exp_min = 1e-1
        for i in range(d):
            if abs(error[i]) > 1.:  # sometimes eigensolver returns extremely-bad solution, thus we need check this
                return -1

            exp = math.floor(math.log10(error[i]))
            _eigvals[i] = round(_eigvals[i], -1 * exp)

            if exp < exp_min:
                exp_min = exp

            if _eigvals[i] == 0.:
                _eigvals[i] = pow(10, exp) * (-1 if _eigvals[i] < 0. else 1)
    else:
        if error is not None:
            exp_min = 1e-1
        else:
            exp_min = math.floor(math.log10(np.finfo(float).eps))

        for i in range(d):
            # sometimes eigensolver returns exact zero value which could cause problem with evaluating logarithm in test
            # statistic
            if _eigvals[i] == 0.:
                if error is not None:
                    exp = math.floor(math.log10(error[i]))
                    _eigvals[i] = pow(10, exp) * (-1 if _eigvals[i] < 0. else 1)

                    if exp < exp_min:
                        exp_min = exp

                    if verbose:
                        PETSc.Sys.Print('  Warning: %dth eigenvalue changed 0. -> %.16g' % (i, _eigvals[i]))
                else:
                    _eigvals[i] = np.finfo(float).eps * (-1 if _eigvals[i] < 0. else 1)

    # if eigensolver returns negative eigenvalues, move eigenvalues to be all positive
    if np.sum(_eigvals < 0) > 0:
        _eigvals += 2 * abs(_eigvals[0])

    for k in range(2, d):
        mean = np.mean(_eigvals[0:k])
        s = np.sum([e - mean for e in _eigvals[k:d]])
        if s == 0.:
            s = pow(10, exp_min)

        fact = k - 1 - (2 * k * k + 2) / (6 * k) + (mean * mean) / (s * s)
        # Compute Vq = prod([e / mean for e in _eigvals[0:k]]) with additional checking if result is not exact 0 which
        # causes problem with evaluating logarithm
        Vq = 1.0
        for i in range(k):
            Vq *= _eigvals[i] / mean
            Vq = Vq if Vq > 0. else np.finfo(float).tiny

        df = (k - 1) * (k + 2) / 2.
        tstat = -fact * np.log(Vq)

        p = stats.chi2.cdf(tstat, df)
        if p >= conf_level:
            if verbose:
                PETSc.Sys.Print('  k=%d, tstat=%.5f, df=%d, p-value=%3f (H0 rejected)' % (k, tstat, df, p))

            dim_null = k - 1
            break
        if verbose:
             PETSc.Sys.Print('  k=%d, tstat=%.5f, df=%d, p-value = %3f (H0 accepted)' % (k, tstat, df, p))

    if dim_null == 1:
        PETSc.Sys.Print('  Underlying similarity graph not contain any multiple connected components (factors)')
    elif verbose:
        PETSc.Sys.Print('  Determined k=%d connected components (factors) of underlying similarity graph' % dim_null)

    return dim_null, _eigvals

# TODO move functionality related to transforming eigenvalue profile to separate function
# TODO adapt test for processing eigenvalues related to matrix of random walk
