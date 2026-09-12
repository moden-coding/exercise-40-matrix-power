#!/usr/bin/env python3

import unittest
from unittest.mock import patch
from functools import reduce

import numpy as np

from src.matrix_power import matrix_power


class TestMatrixPower(unittest.TestCase):

    def test_power_one_returns_the_same_matrix(self):
        a = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(
            a, matrix_power(a, 1),
            err_msg="matrix_power(a, 1) should return a unchanged: raising a "
            "matrix to the power of 1 should not change it.",
        )

    def test_power_zero_returns_the_identity_matrix(self):
        a = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(
            np.eye(2), matrix_power(a, 0),
            err_msg="matrix_power(a, 0) should return the identity matrix: "
            "any square matrix raised to the power of 0 is the identity.",
        )

    def test_positive_and_negative_powers_are_inverses(self):
        a = np.array([[1, 2], [3, 4]])
        for i in range(1, 4):
            a1 = matrix_power(a, i)
            a2 = matrix_power(a, -i)
            np.testing.assert_array_almost_equal(
                np.eye(2), a1 @ a2,
                err_msg="matrix_power(a, %d) @ matrix_power(a, -%d) should be "
                "the identity matrix for a=\n%s: a matrix raised to a negative "
                "power should be the inverse of that matrix raised to the "
                "positive power." % (i, i, a),
            )

    def test_positive_exponents_match_repeated_multiplication(self):
        a = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_almost_equal(
            matrix_power(a, 2), a @ a,
            err_msg="matrix_power(a, 2) should equal a @ a for a=\n%s." % a,
        )
        np.testing.assert_array_almost_equal(
            matrix_power(a, 3), a @ a @ a,
            err_msg="matrix_power(a, 3) should equal a @ a @ a for a=\n%s." % a,
        )
        np.testing.assert_array_almost_equal(
            matrix_power(a, 4), a @ a @ a @ a,
            err_msg="matrix_power(a, 4) should equal a @ a @ a @ a for a=\n%s." % a,
        )

    def test_uses_reduce_rather_than_a_python_loop(self):
        a = np.array([[1, 2], [3, 4]])
        with patch("src.matrix_power.reduce", wraps=reduce) as preduce:
            matrix_power(a, -2)
            preduce.assert_called()


if __name__ == "__main__":
    unittest.main()
