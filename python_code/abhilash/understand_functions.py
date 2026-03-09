import sys,os
notebook_dir = os.getcwd()  # Gets current working directory
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
# sys.path.append(parent_dir)
sys.path.append(parent_dir + "/python_code")
from implementation.Cloth import Cloth
from implementation.utils import createRectangularMesh
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import time

X, T = createRectangularMesh(a=2, b=1, na=3, nb=2, h=0.5)
# Quad side length: 2, 1
# rect x in [-2/2, 2/2] and y in [-1/2, 1/2]
# number of nodes = 3 * 2 = 6
# number of quads = (3-1) * (2-1) = 2

print("X =")
print(X)
print("\nT =")
print(T)