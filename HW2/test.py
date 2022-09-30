import numpy as np

a1 = np.array([1, 2, 4, 3])
a2 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
a3 = a1.T + a2
# print(a1 * a2)
print(np.array([a1]).T + a2)
print(np.array([a1]).T + a2 + np.array([a1]).T)