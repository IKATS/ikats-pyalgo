"""
Copyright 2018 CS Systèmes d'Information

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
from unittest.case import TestCase
from ikats.algo.core.correlation.data import is_triangular_matrix


class TestCorrelationDataset(TestCase):
    """
    TU testing the data module: additional tests required by test_loop unittests.
    Note: may be completed later for exhaustive unittest, at low level -white boxes-.
    exchanged between front/back
    """

    def test_is_triangular_matrix(self):
        """
        Test if the checked matrix is triangular
        """
        self.assertTrue(is_triangular_matrix([[1, 0, 3, 5],
                                              [0, 1, 3],
                                              [1, 1],
                                              [-2]], 4),
                        "Tests is_triangular_matrix() function from correlation.data")

        self.assertFalse(is_triangular_matrix([[1, 0, 3, 5],
                                               [0, 1, 3],
                                               [1, 1]], 4),
                         "Tests is_triangular_matrix(): not triangular, missing last row")

        self.assertFalse(is_triangular_matrix([[1, 0, 3],
                                               [0, 1, 3],
                                               [1, 1],
                                               [0]], 4),
                         "Tests is_triangular_matrix(): not triangular: bad length on a row")

        self.assertFalse(is_triangular_matrix([[1, 0, 3, 5],
                                               [0, 1, 3],
                                               [1, 1],
                                               [-2]], 3),
                         "Tests is_triangular_matrix(): not triangular: not the expected size")

        self.assertTrue(is_triangular_matrix([], 0),
                        "Tests is_triangular_matrix(): empty triangular: size==0")
