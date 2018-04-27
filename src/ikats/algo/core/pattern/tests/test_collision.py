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
import unittest

import numpy as np
from scipy.special import binom

from ikats.algo.core.pattern.collision import final_collision_matrix, SparseMatrix
from ikats.algo.core.sax.sliding_sax import SaxResult
from ikats.core.library.spark import ScManager


class TestCollision(unittest.TestCase):
    """
    Check the collision matrix with different sax result
    """

    def test_sparse_matrix_init(self):
        """
        Tests the SparseMatrix init
        """
        # original collision matrix not used:
        #                            [[0, 1, 2, 3, 4],
        #                             [1, 0, 5, 6, 7],
        #                             [2, 5, 0, 8, 9],
        #                             [3, 6, 8, 0, 10],
        #                             [4, 7, 9, 10, 0]]
        #
        # equivalent content using triangular matrix, actually used:
        # => this vis the version is implemented in random_proj
        original_mat = np.array([[0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0],
                                 [2, 5, 0, 0, 0],
                                 [3, 6, 8, 0, 0],
                                 [4, 7, 9, 10, 0]])

        sparse_from_triangular = SparseMatrix(original_mat)

        self.assertEqual(len(sparse_from_triangular.data), 4 + 3 + 2 + 1)

        for coll, (ind_a, ind_b) in sparse_from_triangular.data:
            self.assertEquals(coll, original_mat[ind_a, ind_b])

    def test_sparse_matrix_get_column(self):
        """
        Tests the SparseMatrix get_column
        """
        # original collision matrix not directly used by code
        #  => useful to understand the test
        #                            [[0, 1, 0, 0, 4],
        #                             [1, 0, 5, 6, 7],
        #                             [0, 5, 0, 8, 9],
        #                             [0, 6, 8, 0, 0],
        #                             [4, 7, 9, 0, 0]]
        #
        # equivalent content using triangular matrix, actually used:
        # => this vis the version is implemented in random_proj
        original_mat = np.array([[0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0],
                                 [0, 5, 0, 0, 0],
                                 [0, 6, 8, 0, 0],
                                 [4, 7, 9, 0, 0]])

        sparse_from_triangular = SparseMatrix(original_mat)

        # should find row indexes from original matrix
        #  column [0, 1, 0, 0, 4] in original matrix => rows 1 and 4
        self.assertListEqual(sparse_from_triangular.get_column(0), [1, 4])
        #  column [1, 0, 5, 6, 7] in original matrix => rows 0, 2, 3, 4
        self.assertListEqual(sparse_from_triangular.get_column(1), [0, 2, 3, 4])
        #  column [0, 5, 0, 8, 9] in original matrix => rows 1, 3, 4
        self.assertListEqual(sparse_from_triangular.get_column(2), [1, 3, 4])
        #  column [0, 6, 8, 0, 0] in original matrix => rows 1, 2
        self.assertListEqual(sparse_from_triangular.get_column(3), [1, 2])
        #  column [4, 7, 9, 0, 0 in original matrix => rows 0, 1, 2
        self.assertListEqual(sparse_from_triangular.get_column(4), [0, 1, 2])

    def test_collision_same_words(self):
        """
        The words are all the same
        """

        sc = ScManager.get()

        sax_result = SaxResult(paa=sc.parallelize([]), breakpoints=[], sax_word='abcdabcdabcdabcd')
        sax, _, _ = sax_result.start_sax(4, spark_ctx=sc)
        sequences_size = np.array(sax.collect()).shape[1]
        result, _ = final_collision_matrix(sax=sax, number_of_iterations=6, index_selected=2,
                                           word_len=sequences_size, spark_ctx=sc)

        result = result.data

        # exactly the same words => six cells of maximum of combinations
        nb_cell = 0
        for i in result:
            if i[0] == 6:
                nb_cell += 1
        self.assertEqual(nb_cell, 6)

    def test_collision_different_words(self):
        """
        The words are all different
        """
        nb_paa = 5
        nb_index = 2
        sc = ScManager.get()
        sax_result = SaxResult(paa=sc.parallelize([]), breakpoints=[],
                               sax_word=''.join(['abcde', 'fghij', 'klmno', 'pqrst', 'uvwxy']))

        sax, _, _ = sax_result.start_sax(nb_paa, spark_ctx=sc)
        sequences_size = np.array(sax.collect()).shape[1]
        result, _ = final_collision_matrix(sax=sax,
                                           number_of_iterations=int(binom(nb_paa, nb_index)),
                                           index_selected=nb_index,
                                           word_len=sequences_size,
                                           spark_ctx=sc)
        result = result.data

        # different words => only zero cells in the matrix
        self.assertTrue(len(result) is 0)

    def test_coll_various_words(self):
        """
        Test the collision matrix for same and different words
        The words 0 and 3 are the same, the words 1 and 2 too
        """

        nb_paa = 5
        nb_index = 2
        sc = ScManager.get()
        sax_result = SaxResult(paa=sc.parallelize([]), breakpoints=[],
                               sax_word=''.join(['ababa', 'cdcdc', 'cdcdc', 'ababa']))

        sax, _, _ = sax_result.start_sax(nb_paa, spark_ctx=sc)
        sequences_size = np.array(sax.collect()).shape[1]
        result, _ = final_collision_matrix(sax=sax,
                                           number_of_iterations=int(binom(nb_paa, nb_index)),
                                           index_selected=nb_index,
                                           word_len=sequences_size,
                                           spark_ctx=sc)
        result = result.data
        result.sort(key=lambda x: "{}-{}-{}".format(int(x[0]), int(x[1][0]), int(x[1][1])))
        print(result)
        # the maximum of possible combinations without repetitions is 10
        # two cells of 10 : one for the occurrences between the words 1 and 2, and another for the words 0 and 3
        for i in range(2):
            self.assertTrue(result[i][0] == 10)
        self.assertTrue(int(result[0][1][0]) == 2 and int(result[0][1][1]) == 1)
        self.assertTrue(int(result[1][1][0]) == 3 and int(result[1][1][1]) == 0)

    def test_coll_near_same_words(self):
        """
        The words have 1, or 2, or 3, or 4 occurrences, but there are not exactly the same because words have five
        letters
        """
        nb_paa = 5
        nb_index = 2
        sc = ScManager.get()
        sax_result = SaxResult(paa=sc.parallelize([]), breakpoints=[],
                               sax_word=''.join(['aaaaa', 'abbbb', 'abccc', 'abcdd', 'abcde']))

        sax, _, _ = sax_result.start_sax(nb_paa, spark_ctx=sc)
        sequences_size = np.array(sax.collect()).shape[1]
        result, _ = final_collision_matrix(sax=sax,
                                           number_of_iterations=int(binom(nb_paa, nb_index)),
                                           index_selected=nb_index,
                                           word_len=sequences_size,
                                           spark_ctx=sc)

        # sorted result list
        result = result.data
        result.sort(key=lambda x: "{}-{}-{}".format(int(x[0]), int(x[1][0]), int(x[1][1])))
        print(result)

        # sorted list expected:
        expected_result = [(1.0, (2, 1)), (1.0, (3, 1)), (3.0, (3, 2)), (1.0, (4, 1)), (3.0, (4, 2)), (6.0, (4, 3))]
        expected_result.sort(key=lambda x: "{}-{}-{}".format(int(x[0]), int(x[1][0]), int(x[1][1])))

        self.assertEqual(len(result), len(expected_result))
        for expected_item, res_item in zip(expected_result, result):
            self.assertEqual(expected_item[0], res_item[0], 'nb collisions')
            self.assertEqual(expected_item[1][0], res_item[1][0], 'seq index left-side')
            self.assertEqual(expected_item[1][1], res_item[1][1], 'seq index right-side')


if __name__ == '__main__':
    unittest.main()
