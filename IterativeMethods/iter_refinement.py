# Iterative Refinement

from sor import SOR
from gs import GS

import numpy as np

def refine(x_in):
  x = x_in.copy()
  TOL=1e-3
  i = 0
  while (True):
    i += 1
    r = b - A@x
    y = np.linalg.solve(A,r)
    new_x = y + x
    error = np.linalg.norm(x - new_x, 2)
    if (error < TOL):
      print(f"Error below tolerance - stopping refinement at {i} iterations")
      break
    x = new_x # this is fine to not do .copy() because new_x = x + y assigns a new array object to new_x; skipping copy to safe some time and memory
  return new_x, error


max_iter_GS = 5
max_iter_SOR = 5
ohm = 0.5
np.random.seed(123)
x0 = np.random.normal(loc = 5, scale = 0.5, size = (3,1))
A = np.array([[3,-1,1],
              [3,6,2],
              [3,3,7]])
b = np.array([[1],[0],[4]])

x, error = GS(A, b, x0, max_iter_GS)
new_x, new_error = refine(x)
print(f"GS: new error is {new_error} old error is {error}\n")

x, error = SOR(A, b, x0, max_iter_GS, ohm)
new_x, new_error = refine(x)
print(f"SOR: new error is {new_error} old error is {error}\n")