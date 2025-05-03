import cvxpy as cp
import numpy as np

"""
Standard non-robust MVDR using a pseudoinverse computation
This is what is currently implemented in Triforce
"""

# Number of sensors
N = 3


def mvdr_weights_pinv(cov, sv):
    Rinv = np.linalg.pinv(cov)
    num = Rinv @ sv
    den = sv.T @ num
    return (num / den).reshape(N, 1)
