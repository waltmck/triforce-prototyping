import cvxpy as cp
import numpy as np
import os, sys

# ugly python hack to import from sibling directory
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]) + "/algos")

from mvdr_weights_socp import *
from mvdr_weights_pinv import *

# Test whether SOCP reference algorithm gives similar weights
# compared to the pseudoinversed-based implementation.

# Number of sensors
N = 3

# Generate a random covariance matrix (i.e. Hermitian and PSD) for testing
M = np.random.normal(size=(N, N)) + 1j * np.random.normal(size=(N, N))
cov = M.conj().T @ M

# Set steering vector
sv = np.array([1, 1, 1])

print("Non-robust implementation: \n", mvdr_weights_pinv(cov, sv))

print("SOCP with eps=0: \n", mvdr_weights_socp(cov, sv, 0))
print("SOCP with eps=0.01: \n", mvdr_weights_socp(cov, sv, 0.01))
print("SOCP with eps=0.1: \n", mvdr_weights_socp(cov, sv, 0.1))
print("SOCP with eps=1: \n", mvdr_weights_socp(cov, sv, 1))
