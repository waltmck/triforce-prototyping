import cvxpy as cp
import numpy as np
import math

"""
Doubly-robust SOCP for adaptive beamforming
See https://ieeexplore.ieee.org/document/1261950

This one needs to take an entire matrix of samples instead of just
a covariance matrix. In exchange, we get robustness against sample errors
(up to Frobenius norm gamma) in addition toeps-robustness to steering vector
errors.
"""

# Number of sensors
N = 3


def mvdr_weights_joint(X, sv, eps, gamma):
    # Construct preliminary variables
    a = np.concatenate([sv.real, sv.imag])
    abar = np.concatenate([sv.imag, -sv.real])
    a.shape = (2 * N, 1)
    abar.shape = (2 * N, 1)

    # Assuming the sample errors are i.i.d, we would expect
    # the Frobenius error to scale with the square root of the
    # number of samples.
    gammaprime = gamma * math.sqrt(X.shape[1])

    X_H = np.block([[X.conj().T], [gammaprime * np.eye(3)]])

    Xprime = np.block([[X_H.real, -X_H.imag], [X_H.imag, X_H.real]])

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
        cp.SOC(tau, Xprime @ w),
        # eps ||w||_2 <= a^T w - 1
        cp.SOC((a.T @ w)[0, 0] - 1, eps * w),
        # abar^T w = 0
        abar.T @ w == 0,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return w.value[0:N] + 1j * w.value[N : 2 * N + 1]
