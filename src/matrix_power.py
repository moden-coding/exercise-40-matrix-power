#!/usr/bin/env python3
import numpy as np
from functools import reduce

def matrix_power(a, n):
    pass

def main():
    a = np.array([[1, 2], [3, 4]])
    print("Matrix a:")
    print(a)
    print("a to the power of 3:")
    print(matrix_power(a, 3))
    print("a to the power of -1 (inverse):")
    print(matrix_power(a, -1))

if __name__ == "__main__":
    main()
