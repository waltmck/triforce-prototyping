import cvxpy as cp
import numpy as np

"""
Robust MVDR weights computation based on
second order cone programming (SOCP)
"""

# Number of sensors
N = 3


def mvdr_weights_socp(cov, sv, eps):
    # Construct preliminary variables
    a = np.concatenate([sv.real, sv.imag])
    abar = np.concatenate([sv.imag, -sv.real])
    a.shape = (2 * N, 1)
    abar.shape = (2 * N, 1)

    U = np.linalg.cholesky(cov, upper=True)
    Uprime = np.block([[U.real, -U.imag], [U.imag, U.real]])

    # Free variable w has the form
    # [Re(w_1), Re(w_2), Re(w_3), Im(w_1), Im(w_2), Im(w_3)]
    # Tau is a dummy variable used to move the quadratic objective
    # to a quadratic constraint
    w = cp.reshape(cp.Variable(2 * N), (2 * N, 1), order="C")
    tau = cp.Variable(1)

    # Linear objective on tau
    objective = cp.Minimize(tau)

    constraints = [
        # ||U'w||_2 <= tau
        cp.SOC(tau, Uprime @ w),
        # eps ||w||_2 <= a^T w - 1
        cp.SOC((a.T @ w)[0, 0] - 1, eps * w),
        # abar^T w = 0
        abar.T @ w == 0,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return w.value[0:N] + 1j * w.value[N : 2 * N + 1]
