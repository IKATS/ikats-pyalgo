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
from ikats.algo.core.sax.sliding_sax import SaxResult
from ikats.algo.core.pattern.random_proj import NeighborhoodSearch, ConfigRecognition
from ikats.algo.core.pattern.collision import SparseMatrix
from ikats.algo.core.pattern.recognition import OPT_USING_BRUTE_FORCE, OPT_USING_COLLISIONS, \
    _start_alphabet, _get_mindist
from ikats.core.library.spark import ScManager


class TestRecognition(unittest.TestCase):
    """
    Tests the pattern recognition
    """

    @staticmethod
    def _print_mindist_mat(search_info, activate=False):
        """
        Building new TU : this method diplays mindist matrix from the NeighborhoodSearch

        Simply set activate to False to disable useless printing

        :param search_info: tested object
        :type search_info: NeighborhoodSearch
        :param activate:
        :type activate: boolean
        """
        if activate:
            alphabet = _start_alphabet(search_info.alphabet_size)
            nb_seqs = len(search_info.sax)
            mindist_mat = np.zeros((nb_seqs, nb_seqs))
            for i in range(0, nb_seqs):
                for j in range(0, nb_seqs):
                    mindist_mat[i][j] = _get_mindist(search_info.size_sequence, search_info.sax[i],
                                                     search_info.sax[j],
                                                     search_info.mindist_lookup_table,
                                                     alphabet)

            print("mindist distances:")
            print(mindist_mat)

    @staticmethod
    def _print_matrix(test, data, nb_seq, activate=False):
        """
        Building new TU : this method diplays the matrix corresponding to SparseMatrix

        Simply set activate to False to disable useless printing

        :param test: name of the test
        :type test: str
        :param data: data from SparseMatrix
        :type data: list of tuple
        :param nb_seq: nb sequences
        :type nb_seq: int
        :param activate: False to disable useless printing, once the test is well prepared
        :type activate: boolean
        """
        if activate:
            mat = np.zeros((nb_seq, nb_seq))
            for coll, (row, col) in data:
                mat[row, col] = coll
                # not initialized  mat[col, row] = coll
            print(test)
            print("np.array({})".format(str(mat).replace('.', ',').replace(']\n', '],\n')))

    def test_global_same_words_spark(self):
        """
        Test: see _apply_motif_global_same_words(activate_spark=True)
        """
        self._apply_motif_global_same_words(activate_spark=True)

    def test_global_same_words_no_spark(self):
        """
        Test: see _apply_motif_global_same_words(activate_spark=False)
        """
        self._apply_motif_global_same_words(activate_spark=False)

    def _apply_motif_global_same_words(self, activate_spark):
        """
        Test
        - with the global method to search the neighborhood motif,
        - with/without spark jobs according to activate_spark
        - and where the words are all the same
        """
        spark_context = ScManager.get()
        # Build the SAX result with large breakpoints
        sax_result = SaxResult(paa=spark_context.parallelize([]), breakpoints=[-300, -100, 100, 300],
                               sax_word='abcdeabcdeabcdeabcde')
        sax, _, _ = sax_result.start_sax(5, spark_ctx=spark_context)
        # sax is an rdd -> to np.array
        sax = np.transpose(sax.collect())

        breakpoint = sax_result.build_mindist_lookup_table(alphabet_size=5)

        # Build the collision matrix result
        collision_matrix = SparseMatrix(np.array([[0, 0, 0, 0, ],
                                                  [100, 0, 0, 0, ],
                                                  [100, 100, 0, 0, ],
                                                  [100, 100, 100, 0, ]]))

        # two identical cases here: brute force / with collisions
        for method_opt in [OPT_USING_BRUTE_FORCE, OPT_USING_COLLISIONS]:
            #  mindist distances:
            #
            # [[ 0.  0.  0.  0.]
            #  [ 0.  0.  0.  0.]
            #  [ 0.  0.  0.  0.]
            #  [ 0.  0.  0.  0.]]

            # Build the class for motif search
            search_info = NeighborhoodSearch(size_sequence=20,
                                             mindist_lookup_table=breakpoint,
                                             alphabet_size=5,
                                             sax=np.transpose(sax),
                                             radius=0.01,
                                             collision_matrix=collision_matrix)

            recognition_info = ConfigRecognition(is_stopped_by_eq9=True,
                                                 iterations=0,
                                                 min_value=1,
                                                 is_algo_method_global=True,
                                                 activate_spark=activate_spark,
                                                 radius=0.01,
                                                 neighborhood_method=method_opt)

            # neighborhood_method=OPT_USING_BRUTE_FORCE (compare with all the words)
            result = search_info.motif_neighborhood_global(30, recognition_info)

            self._print_mindist_mat(search_info)

            # The words corresponding to the six largest values cells have a MINDIST < radius
            self.assertEqual(len(result), 1)
            # This results are the same : [0,1,2,3]: the 6 groups have been reduced to one inside
            self.assertEqual(result, [[0, 1, 2, 3]])

    def test_global_zero_coll_spark(self):
        """
        Test: see _apply_motif_global_zero_coll(activate_spark=False)
        """
        self._apply_motif_global_zero_coll(activate_spark=True)

    def test_global_zero_coll_no_spark(self):
        """
        Test: see _apply_motif_global_zero_coll(activate_spark=False)
        """
        self._apply_motif_global_zero_coll(activate_spark=False)

    def _apply_motif_global_zero_coll(self, activate_spark):
        """
        Test
        - with the global method to search the neighborhood motif,
        - with/without spark jobs, according to activate_spark
        - and where the words are all different.
        """
        spark_context = ScManager.get()
        # Build the SAX result with different words, and small breakpoints
        sax_result = SaxResult(paa=spark_context.parallelize([]), breakpoints=[-0.3, -0.1, 0.1, 0.3],
                               sax_word='abcdebcdeacdeabdeabceabcd')
        sax, _, _ = sax_result.start_sax(5, spark_ctx=spark_context)
        # sax is an rdd -> to np.array
        sax = np.transpose(sax.collect())
        breakpoint = sax_result.build_mindist_lookup_table(5)

        # Different words => noly zero cells in the collision matrix
        collision_matrix = SparseMatrix(np.zeros((2, 2)))

        # two identical cases here: brute force / with collisions
        for method_opt in [OPT_USING_BRUTE_FORCE, OPT_USING_COLLISIONS]:
            # Build the class for motif search
            search_info = NeighborhoodSearch(size_sequence=20,
                                             mindist_lookup_table=breakpoint,
                                             alphabet_size=5,
                                             sax=np.transpose(sax),
                                             radius=1000,
                                             collision_matrix=collision_matrix)

            recognition_info = ConfigRecognition(is_stopped_by_eq9=True,
                                                 iterations=0,
                                                 min_value=1,
                                                 is_algo_method_global=True,
                                                 activate_spark=activate_spark,
                                                 radius=1000,
                                                 neighborhood_method=method_opt)

            # neighborhood_method=OPT_USING_BRUTE_FORCE
            result = search_info.motif_neighborhood_global(30, recognition_info)

            # There is no similar sequences
            self.assertEqual(len(result), 0)

    def test_global_brute_spark_ex1(self):
        """
        Test: see _apply_motif_global_brute_ex1(activate_spark=None)
        """
        self._apply_motif_global_brute_ex1(activate_spark=True)

    def test_global_brute_no_spark_ex1(self):
        """
        Test: see _apply_motif_global_brute_ex1(activate_spark=None)
        """
        self._apply_motif_global_brute_ex1(activate_spark=None)

    def _apply_motif_global_brute_ex1(self, activate_spark):
        """
        Test
         - with the global method to search the neighborhood motif,
         - with brute force
         - with/without spark jobs according to activate_spark
         - and where the words have only one different letter.
        """

        # Build the SAX result where the words have only one different letter (words: 5 letters)
        sequences = ["abcde", "abcdd", "abcdc", "abcdb", "abcda"]
        tested_sax_word = ''.join(sequences)
        spark_context = ScManager.get()
        sax_result = SaxResult(paa=spark_context.parallelize([]), breakpoints=[-1.1, -1, 0, 1.501],
                               sax_word=tested_sax_word)
        sax, _, nb_seq = sax_result.start_sax(5, spark_ctx=spark_context)
        # sax is an rdd -> to np.array
        sax = np.transpose(sax.collect())

        breakpoint = sax_result.build_mindist_lookup_table(5)

        # Build a collision matrix (the real collision matrix is different, but we take this one for the test)
        collision_matrix = SparseMatrix(np.array([[0, 0, 0, 0, 0, ],
                                                  [30, 0, 0, 0, 0, ],
                                                  [2, 40, 0, 0, 0, ],
                                                  [4, 8, 50, 0, 0, ],
                                                  [6, 10, 20, 60, 0, ]]))

        self._print_matrix("test_global_brute_force_ex1", collision_matrix.data, nb_seq)

        # mindist distances:
        # [[ 0.     0.     3.002  5.002  5.202]
        #  [ 0.     0.     0.     2.     2.2  ]
        #  [ 3.002  0.     0.     0.     0.2  ]
        #  [ 5.002  2.     0.     0.     0.   ]
        #  [ 5.202  2.2    0.2    0.     0.   ]]

        # Using neighborhood_method=OPT_USING_BRUTE_FORCE
        #
        # brute force:  for collisions (0,1) (1,2) (2,3) (3,4) greater than min_value==25
        #
        # for radius 1.9  => global result is [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3, 4], [2, 3, 4]]
        #
        # for radius 2.5  => global result is [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        #                                      => reduced to [[[0, 1, 2, 3, 4], [1, 2, 3, 4]]
        #
        # for radius 3.5  => global result is [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [1, 2, 3, 4]]
        #                                      => reduced to [[0, 1, 2, 3, 4], [1, 2, 3, 4]]
        #
        # for radius 6    => global result is [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
        #                                      => reduced to [[0, 1, 2, 3, 4]]
        #

        for radius, expected_res in [[2.5, [[0, 1, 2, 3, 4], [1, 2, 3, 4]]],
                                     [1.9, [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3, 4], [2, 3, 4]]],
                                     [3.5, [[0, 1, 2, 3, 4], [1, 2, 3, 4]]],
                                     [6, [[0, 1, 2, 3, 4]]]]:

            # Build the class for motif search where the min_value is 25
            search_info = NeighborhoodSearch(size_sequence=20,
                                             mindist_lookup_table=breakpoint,
                                             alphabet_size=5,
                                             sax=np.transpose(sax),
                                             radius=radius,
                                             collision_matrix=collision_matrix)

            # for info: here is the mindist:
            #  (see _print_mindist_mat doc: in order to activate print)
            self._print_mindist_mat(search_info)

            recognition_info = ConfigRecognition(is_stopped_by_eq9=True,
                                                 iterations=0,
                                                 min_value=25,
                                                 is_algo_method_global=True,
                                                 activate_spark=activate_spark,
                                                 radius=radius,
                                                 neighborhood_method=OPT_USING_BRUTE_FORCE)

            search_info.radius = radius
            recognition_info.radius = radius
            result = search_info.motif_neighborhood_global(recognition_info.min_value, recognition_info)

            self.assertEqual(len(result), len(expected_res))
            for group in result:
                self.assertTrue(group in expected_res)

    def test_global_coll_no_spark_ex1(self):
        """
        Tests without spark: see apply_motif_neighborhood_global__with_collisions_ex1(activate_spark=False)
        """
        self._apply_motif_global_coll_ex1(activate_spark=False)

    def test_global_coll_spark_ex1(self):
        """
        Tests with spark: see apply_motif_neighborhood_global__with_collisions_ex1(activate_spark=True)
        """
        self._apply_motif_global_coll_ex1(activate_spark=True)

    def _apply_motif_global_coll_ex1(self, activate_spark):
        """
        Test
          - with the global method to search the neighborhood motif,
          - with/without spark according to activate_spark
          - exploring similarities with collisions heuristic
          - with input: the words have only one different letter.  And every sequence
            Si has collisions with Sj with that matrix.

         Note: results ought to be equal to test_global_brute_no_spark_ex1
        """

        # Build the SAX result where the words have only one different letter (words: 5 letters)
        sequences = ["abcde", "abcdd", "abcdc", "abcdb", "abcda"]
        tested_sax_word = ''.join(sequences)
        spark_context = ScManager.get()
        sax_result = SaxResult(paa=spark_context.parallelize([]), breakpoints=[-1.1, -1, 0, 1.501],
                               sax_word=tested_sax_word)
        sax, _, nb_seq = sax_result.start_sax(5, spark_ctx=spark_context)
        # sax is an rdd -> to np.array
        sax = np.transpose(sax.collect())

        breakpoint = sax_result.build_mindist_lookup_table(5)

        # Build a collision matrix (the real collision matrix is different, but we take this one for the test)
        collision_matrix = SparseMatrix(np.array([[0, 0, 0, 0, 0, ],
                                                  [30, 0, 0, 0, 0, ],
                                                  [2, 40, 0, 0, 0, ],
                                                  [4, 8, 50, 0, 0, ],
                                                  [6, 10, 20, 60, 0, ]]))

        self._print_matrix("test_global_coll_no_spark_ex1",
                           collision_matrix.data,
                           nb_seq)

        # mindist distances:
        # [[ 0.     0.     3.002  5.002  5.202]
        #  [ 0.     0.     0.     2.     2.2  ]
        #  [ 3.002  0.     0.     0.     0.2  ]
        #  [ 5.002  2.     0.     0.     0.   ]
        #  [ 5.202  2.2    0.2    0.     0.   ]]

        # Using neighborhood_method=OPT_USING_COLLISIONS
        #
        #  for collisions (0,1) (1,2) (2,3) (3,4) greater than min_value==25
        #  and with the collisions heuristic: only sequences having collisions with Si or Sj are examined
        #
        # for radius 1.9  => global result is [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3, 4], [2, 3, 4]]
        #
        # for radius 2.5  => global result is [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        #                                      => reduced to [[[0, 1, 2, 3, 4], [1, 2, 3, 4]]
        #
        # for radius 3.5  => global result is [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [1, 2, 3, 4]]
        #                                      => reduced to [[0, 1, 2, 3, 4], [1, 2, 3, 4]]
        #
        # for radius 6    => global result is [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
        #                                      => reduced to [[0, 1, 2, 3, 4]]
        #
        for radius, expected_res in [[2.5, [[0, 1, 2, 3, 4], [1, 2, 3, 4]]],
                                     [1.9, [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3, 4], [2, 3, 4]]],
                                     [3.5, [[0, 1, 2, 3, 4], [1, 2, 3, 4]]],
                                     [6, [[0, 1, 2, 3, 4]]]]:

            # Build the class for motif search where the min_value is 25
            search_info = NeighborhoodSearch(size_sequence=20,
                                             mindist_lookup_table=breakpoint,
                                             alphabet_size=5,
                                             sax=np.transpose(sax),
                                             radius=radius,
                                             collision_matrix=collision_matrix)

            # for info: here is the mindist:
            #  (see _print_mindist_mat doc: in order to activate print)
            self._print_mindist_mat(search_info)

            recognition_info = ConfigRecognition(is_stopped_by_eq9=True,
                                                 iterations=0,
                                                 min_value=25,
                                                 is_algo_method_global=True,
                                                 activate_spark=activate_spark,
                                                 radius=radius,
                                                 neighborhood_method=OPT_USING_COLLISIONS)

            print("radius {}:expected:                 {}".format(radius, expected_res))
            result = search_info.motif_neighborhood_global(recognition_info.min_value, recognition_info)

            print("radius {}:->global with collisions: {}".format(radius, result))

            self.assertEqual(len(result), len(expected_res))
            for group in result:
                self.assertTrue(group in expected_res)

    def test_iter_same_words_spark(self):
        """
        Test: see _apply_motif_iter_same_words(activate_spark=True)
        """
        self._apply_motif_iter_same_words(activate_spark=True)

    def test_iter_same_words_no_spark(self):
        """
        Test: see _apply_motif_iter_same_words(activate_spark=False)
        """
        self._apply_motif_iter_same_words(activate_spark=False)

    def _apply_motif_iter_same_words(self, activate_spark):
        """
        Test
         - with the iterative method to search the neighborhood motif,
         - with/without spark jobs according to activate_spark
         - and where the words are all the same
        """

        spark_context = ScManager.get()
        # Build the SAX result with large breakpoints
        sax_result = SaxResult(paa=spark_context.parallelize([]), breakpoints=[-300, -100, 100, 300],
                               sax_word='abcdeabcdeabcdeabcde')
        sax, _, _ = sax_result.start_sax(5, spark_ctx=spark_context)
        # sax is an rdd -> to np.array
        sax = np.transpose(sax.collect())

        breakpoint = sax_result.build_mindist_lookup_table(alphabet_size=5)

        # Build the collision matrix result

        collision_matrix = SparseMatrix(np.array([[0, 0, 0, 0, ],
                                                  [100, 0, 0, 0, ],
                                                  [99, 97, 0, 0, ],
                                                  [98, 96, 95, 0, ]]))

        # two identical cases here: brute force / with collisions
        for method_opt in [OPT_USING_BRUTE_FORCE, OPT_USING_COLLISIONS]:
            #  mindist distances:
            #
            # [[ 0.  0.  0.  0.]
            #  [ 0.  0.  0.  0.]
            #  [ 0.  0.  0.  0.]
            #  [ 0.  0.  0.  0.]]

            # Build the class for motif search
            search_info = NeighborhoodSearch(size_sequence=20,
                                             mindist_lookup_table=breakpoint,
                                             alphabet_size=5,
                                             sax=np.transpose(sax),
                                             radius=0.01,
                                             collision_matrix=collision_matrix)

            recognition_info = ConfigRecognition(is_stopped_by_eq9=True,
                                                 iterations=4,
                                                 min_value=1,
                                                 is_algo_method_global=False,
                                                 activate_spark=activate_spark,
                                                 radius=0.01,
                                                 neighborhood_method=method_opt)

            # neighborhood_method=OPT_USING_BRUTE_FORCE (compare with all the words)
            result = search_info.motif_neighborhood_iterative(30, recognition_info)

            # The words corresponding to the six largest values cells have a MINDIST < radius,
            # but the iterative method take 2 group of similar sequences (in recognition_info : iterations = 2)
            self.assertEqual(len(result), 1)

            # This results are the same : [[0,1,2,3]]
            self.assertListEqual(result[0], [0, 1, 2, 3])

    def test_iter_zero_coll_spark(self):
        """
        Test: see _apply_motif_iter_zero_coll(activate_spark=True)
        """
        self._apply_motif_iter_zero_coll(activate_spark=True)

    def test_iter_zero_coll_no_spark(self):
        """
        Test: see _apply_motif_iter_zero_coll(activate_spark=False)
        """
        self._apply_motif_iter_zero_coll(activate_spark=False)

    def _apply_motif_iter_zero_coll(self, activate_spark):
        """
        Test
         - with the iterative method to search the neighborhood motif,
         - with/without spark jobs
         - and where the words are all different => no collisions
        """
        spark_context = ScManager.get()
        # Build the SAX result with different words, and small breakpoints
        sax_result = SaxResult(paa=spark_context.parallelize([]), breakpoints=[-0.3, -0.1, 0.1, 0.3],
                               sax_word='abcdebcdeacdeabdeabceabcd')
        sax, _, nb_seq = sax_result.start_sax(5, spark_ctx=spark_context)
        # sax is an rdd -> to np.array
        sax = np.transpose(sax.collect())

        breakpoint = sax_result.build_mindist_lookup_table(nb_seq)

        # Different words => only zero cells in the collision matrix
        collision_matrix = SparseMatrix(np.zeros((nb_seq, nb_seq)))

        # Build the class for motif search
        search_info = NeighborhoodSearch(size_sequence=20,
                                         mindist_lookup_table=breakpoint,
                                         alphabet_size=5,
                                         sax=np.transpose(sax),
                                         radius=1000,
                                         collision_matrix=collision_matrix)

        recognition_info = ConfigRecognition(is_stopped_by_eq9=True,
                                             iterations=100,
                                             min_value=1,
                                             is_algo_method_global=False,
                                             activate_spark=activate_spark,
                                             radius=1000,
                                             neighborhood_method=OPT_USING_BRUTE_FORCE)

        # neighborhood_method=OPT_USING_BRUTE_FORCE
        result = search_info.motif_neighborhood_iterative(30, recognition_info)

        # There is no similar sequences
        self.assertEqual(len(result), 0)

        # neighborhood_method=OPT_USING_COLLISIONS
        recognition_info.neighborhood_method = OPT_USING_COLLISIONS
        result = search_info.motif_neighborhood_iterative(30, recognition_info)

        # There is no similar sequences
        self.assertEqual(len(result), 0)

    def test_iter_brute_ex1_spark(self):
        """
        Test: see _apply_iter_brute_ex1(activate_spark=True)
        """
        self._apply_iter_brute_ex1(activate_spark=True)

    def test_iter_brute_ex1_no_spark(self):
        """
        Test: see _apply_iter_brute_ex1(activate_spark=False)
        """
        self._apply_iter_brute_ex1(activate_spark=False)

    def _apply_iter_brute_ex1(self, activate_spark):
        """
        Tests motif_neighborhood_iterative()
         - the iterative method
         - using the brute force method
         - to search the neighborhood motif
         - with/without spark jobs according to activate_spark
         Note: test where the words have only one different letter.
        """

        # Build the SAX result where the words have only one different letter (words: 5 letters)
        sequences = ["abcde", "abcdd", "abcdc", "abcdb", "abcda"]
        tested_sax_word = ''.join(sequences)
        spark_context = ScManager.get()
        sax_result = SaxResult(paa=spark_context.parallelize([]), breakpoints=[-1.1, -1, 0, 1.501],
                               sax_word=tested_sax_word)
        sax, _, nb_seq = sax_result.start_sax(5, spark_ctx=spark_context)
        # sax is an rdd -> to np.array
        sax = np.transpose(sax.collect())

        breakpoint = sax_result.build_mindist_lookup_table(5)

        # Build a collision matrix
        collision_matrix = SparseMatrix(np.array([[0, 0, 0, 0, 0, ],
                                                  [30, 0, 0, 0, 0, ],
                                                  [2, 40, 0, 0, 0, ],
                                                  [4, 8, 50, 0, 0, ],
                                                  [6, 10, 20, 50, 0, ]]))

        self._print_matrix("test_iterative__brute_no_spark_ex1",
                           collision_matrix.data,
                           nb_seq)

        # mindist distances:
        # [[ 0.     0.     3.002  5.002  5.202]
        #  [ 0.     0.     0.     2.     2.2  ]
        #  [ 3.002  0.     0.     0.     0.2  ]
        #  [ 5.002  2.     0.     0.     0.   ]
        #  [ 5.202  2.2    0.2    0.     0.   ]]

        # Using neighborhood_method=OPT_USING_BRUTE_FORCE
        #
        # iterative:  examining collisions (i,j) per iteration:
        #             (3,4)+(2,3) then (1,2) then (0,1)
        #
        #             (collisions greater than min_value==25)
        #
        # Test with fixed radius 1.9:
        #    - iter=1    => result is [[1,2,3,4],[2, 3, 4]] considering (S2,S3) and (S3,S4) neighborhoods
        #    - iter=2    => result extended with [0,1,2,3,4] considering (S1,S2)
        #    - iter=3    => result extended with [0,1,2] considering (S0,S1)
        #    - iter=100  => result is the same than for iter=3: no more collision available
        #
        for radius, nb_iter, expected_res in [[1.9, 1, [[1, 2, 3, 4], [2, 3, 4]]],
                                              [1.9, 2, [[1, 2, 3, 4], [2, 3, 4], [0, 1, 2, 3, 4]]],
                                              [1.9, 3, [[1, 2, 3, 4], [2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2]]],
                                              [1.9, 100, [[1, 2, 3, 4], [2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2]]]]:

            # Build the class for motif search where the min_value is 25
            search_info = NeighborhoodSearch(size_sequence=20,
                                             mindist_lookup_table=breakpoint,
                                             alphabet_size=5,
                                             sax=np.transpose(sax),
                                             radius=radius,
                                             collision_matrix=collision_matrix)

            # for info: here is the mindist:
            #  (see _print_mindist_mat doc: in order to activate print)
            self._print_mindist_mat(search_info)

            recognition_info = ConfigRecognition(is_stopped_by_eq9=True,
                                                 iterations=nb_iter,
                                                 min_value=25,
                                                 is_algo_method_global=False,
                                                 activate_spark=activate_spark,
                                                 radius=radius,
                                                 neighborhood_method=OPT_USING_BRUTE_FORCE)

            result = search_info.motif_neighborhood_iterative(recognition_info.min_value, recognition_info)

            self.assertEqual(len(result), len(expected_res))
            for group in result:
                self.assertTrue(group in expected_res)

    def test_iter_coll_ex1_spark(self):
        """
        Test: see _apply_iter_coll_no_spark_ex1(activate_spark=True)
        """
        self._apply_iter_coll_no_spark_ex1(activate_spark=True)

    def test_iter_coll_ex1_no_spark(self):
        """
        Test: see _apply_iter_coll_no_spark_ex1(activate_spark=False)
        """
        self._apply_iter_coll_no_spark_ex1(activate_spark=False)

    def _apply_iter_coll_no_spark_ex1(self, activate_spark):
        """
         Tests motif_neighborhood_iterative()
         - the iterative method
         - using the heuristic based upon collisions
         - to search the neighborhood motif

         Note: test where the words have only one different letter.
        """

        # Build the SAX result where the words have only one different letter (words: 5 letters)
        sequences = ["abcde", "abcdd", "abcdc", "abcdb", "abcda"]
        tested_sax_word = ''.join(sequences)
        spark_context = ScManager.get()
        sax_result = SaxResult(paa=spark_context.parallelize([]), breakpoints=[-1.1, -1, 0, 1.501],
                               sax_word=tested_sax_word)
        sax, _, nb_seq = sax_result.start_sax(5, spark_ctx=spark_context)
        # sax is an rdd -> to np.array
        sax = np.transpose(sax.collect())

        breakpoint = sax_result.build_mindist_lookup_table(5)

        # Build a collision matrix
        # Note: this matrix is different from the one from
        #   test test_iterative__brute_no_spark_ex1:
        #    => see zeros are added: coll(3,2) == coll(4,2) == 0
        collision_matrix = SparseMatrix(np.array([[0, 0, 0, 0, 0, ],
                                                  [40, 0, 0, 0, 0, ],
                                                  [2, 40, 0, 0, 0, ],
                                                  [4, 8, 0, 0, 0, ],
                                                  [6, 10, 0, 50, 0, ]]))

        self._print_matrix("test_iterative__brute_no_spark_ex1",
                           collision_matrix.data,
                           nb_seq)

        # mindist distances:
        # [[ 0.     0.     3.002  5.002  5.202]
        #  [ 0.     0.     0.     2.     2.2  ]
        #  [ 3.002  0.     0.     0.     0.2  ]
        #  [ 5.002  2.     0.     0.     0.   ]
        #  [ 5.202  2.2    0.2    0.     0.   ]]

        # Using neighborhood_method=OPT_USING_BRUTE_FORCE
        #
        # iterative:  examining collisions (i,j) per iteration:
        #             (3,4) then (1,2) +(0,1)
        #
        #             (collisions greater than min_value==25)
        #
        # Test with fixed radius 1.9:
        #    - iter=1    => result is [[3, 4]] considering (S3,S4) neighborhood
        #    - iter=2    => result extended with [0,1,2] considering (S0,S1), unchanged for (S1,S2)
        #    - iter=3    => result is the same than for iter=2: no more collision available
        #    - iter=100  => result is the same than for iter=2: no more collision available
        #
        for radius, nb_iter, expected_res in [[1.9, 1, [[3, 4]]],
                                              [1.9, 2, [[3, 4], [0, 1, 2]]],
                                              [1.9, 3, [[3, 4], [0, 1, 2]]],
                                              [1.9, 100, [[3, 4], [0, 1, 2]]]]:

            # Build the class for motif search where the min_value is 25
            search_info = NeighborhoodSearch(size_sequence=20,
                                             mindist_lookup_table=breakpoint,
                                             alphabet_size=5,
                                             sax=np.transpose(sax),
                                             radius=radius,
                                             collision_matrix=collision_matrix)

            # for info: here is the mindist:
            #  (see _print_mindist_mat doc: in order to activate print)
            self._print_mindist_mat(search_info)

            recognition_info = ConfigRecognition(is_stopped_by_eq9=True,
                                                 iterations=nb_iter,
                                                 min_value=25,
                                                 is_algo_method_global=False,
                                                 activate_spark=activate_spark,
                                                 radius=radius,
                                                 neighborhood_method=OPT_USING_COLLISIONS)

            result = search_info.motif_neighborhood_iterative(recognition_info.min_value, recognition_info)

            self.assertEqual(len(result), len(expected_res))
            for group in result:
                self.assertTrue(group in expected_res)


if __name__ == '__main__':
    unittest.main()
