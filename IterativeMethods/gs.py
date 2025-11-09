# Gauss-Seidel

import numpy as np
np.random.seed(123)

A = np.array([[4, 3], [-1, 2]])
b = np.array([[59.2],[47.0]])
x0 = np.random.normal(loc = 5, scale = 0.5, size = (2,1))
max_iter_GS = 5

def GS(A, b, x0, max_iter):
  '''
  input: A, n-by-n matrix as numpy array, coefficient matrix for Ax = b
  input: b, n-by-1 vector as numpy array, RHS to Ax = b
  input: x, n-by-1 vector as numpy array, initialized vector
  input: max_iter, an int, the max number of iterations user desires

  output: x, n-by-1 vector, solution to Ax = b after max_iter iterations
  '''
  x = x0.copy()
  for k in range(max_iter):

    n, m = A.shape
    x_last = x.copy()

    for i in range(m):
      sum_term = 0;
      for j in range(m):
        if j < i:
          sum_term += A[i,j]*x[j];
        elif j > i:
          sum_term += A[i,j]*x_last[j];

      sum_term = (b[i]) - sum_term; #fill in blank
      x[i] = (1/A[i,i])*sum_term; #fill in blank

    error = np.linalg.norm(x - x_last, 2)

  return x, error

x, error = GS(A, b, x0, max_iter_GS)
residual_norm = np.linalg.norm(b - A@x, 2)

print("the residual under 2-norm is ", f"{residual_norm:.30f}")
print("the error between last two iterations under 2-norm is", f"{error:.30f}")