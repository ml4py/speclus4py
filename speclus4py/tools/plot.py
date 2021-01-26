import numpy as np
import matplotlib.pyplot as plt


def screePlotEigenvalues(eigvals: np.ndarray, plt_title=None, mark_zeros=False, nzeros=-1):
    clrblue = 'tab:blue'
    clrred = 'tab:red'

    plt.yscale('log')
    plt.grid(True, axis='y')

    plt.xlabel('#factor')
    plt.ylabel('eigenvalue (log)')
    if plt_title is None:
        plt.title('Eigenvalues profile')

    if mark_zeros:
        pos = [i for i in range(1, len(eigvals) + 1)]
        plt.plot(pos, eigvals, markersize='0', linewidth=2, color=clrblue, linestyle=':')
        plt.plot(pos[0:nzeros], eigvals[0:nzeros], marker='*', markersize=10, markerfacecolor=clrred,
                 markeredgecolor=clrred, linewidth=0)
        plt.plot(pos[nzeros:], eigvals[nzeros:], marker='o', markersize=8, markerfacecolor=clrblue,
                 markeredgecolor=clrblue, linewidth=0)
    else:
        plt.plot(eigvals, marker='o', markersize='8', markerfacecolor=clrblue, markeredgecolor=clrblue, color=clrblue,
                 linewidth=2, linestyle=':')
    plt.show()
