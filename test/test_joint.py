import cvxpy as cp
import numpy as np
import os, sys

# ugly python hack to import from sibling directory
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]) + "/algos")

from mvdr_weights_socp import *
from mvdr_weights_pinv import *
from mvdr_weights_joint import *

# Test whether SOCP reference algorithm gives similar weights
# compared to the pseudoinversed-based implementation.

# Number of sensors
N = 3

# Number of samples
M = 64

# Generate random samples and a random covariance matrix
X = np.random.normal(size=(N, M)) + 1j * np.random.normal(size=(N, M))
cov = X @ X.conj().T

# Set steering vector
sv = np.array([1, 1, 1])

print("Non-robust implementation: \n", mvdr_weights_pinv(cov, sv))
print("Jointly robust with eps=0, gamma=0: \n", mvdr_weights_joint(X, sv, 0, 0))
print("Singlely robust with eps=0.1: \n", mvdr_weights_socp(cov, sv, 0.1))
print("Jointly robust with eps=0.1, gamma=0: \n", mvdr_weights_joint(X, sv, 0.1, 0))
print("Jointly robust with eps=0, gamma=1: \n", mvdr_weights_joint(X, sv, 0, 1))
print("Jointly robust with eps=0.1, gamma=1: \n", mvdr_weights_joint(X, sv, 0.1, 1))
