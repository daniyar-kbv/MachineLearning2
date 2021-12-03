from sklearn.
import numpy as np

print(np.linspace(0, 1, 4))
print(np.linspace(0.1, 1, 4))
print([int(num) for num in np.linspace(50, 1000, 10)])
print([None] + [num for num in np.linspace(5, 200, 4)])
print([int(num) for num in np.linspace(2, 10, 4)])
print([int(num) for num in np.linspace(1, 10, 4)])