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

import logging
import random
from collections import defaultdict
from itertools import combinations
from operator import add

import numpy as np
from scipy.special import binom

import pyspark.rdd
from ikats.core.library.exception import IkatsException

LOGGER = logging.getLogger(__name__)

"""
This module is providing services about the Collision matrix for Random-projection algorithm.
"""


class SparseMatrix(object):
    """
    SparseMatrix object gathers information about the non-zero cells in a matrix:
       it is a list of tuples: one tuple  ( value, (index on axis0, index on axis1)) defined for each cell
       whose value is > 0.
    """

    def __init__(self, matrix):
        """
        Constructor from the collision matrix.
        :param matrix: the collision matrix. 2D array of integers.
        :type matrix: numpy.ndarray
        """

        index_tuple = np.where(matrix > 0)
        # Example : (array([0, 1]), array([1, 0]))
        index_list = list(zip(index_tuple[0], index_tuple[1]))
        # Example : [(0, 1), (1, 0)]
        value_cell = list(matrix[index_tuple[0], index_tuple[1]])
        # Example : [2,4]

        self.data = list(zip(value_cell, index_list))
        # Example : [(2, (0, 1)), (4, (1, 0))]

    def values_matrix(self, min_value):
        """
        Search the largest value greater than the min_value, and and give the occurrences of this value

        :param min_value: the minimum value
        :type min_value: int or float

        :return: information that corresponds to the largest value cells, and their occurrences in self.data.
        Information is a dict whose keys are the collision values, and whose values are the lists of indices pairs
        (i,j).
        :rtype: dict

        """
        filtered_data = list(filter(lambda x: x[0] > min_value, self.data))

        dict_of_occurrences = defaultdict(list)
        for item in filtered_data:
            dict_of_occurrences[item[0]].append(item[1])

        return dict_of_occurrences

    def get_column(self, index):
        """
        Give the cell indices of the column at index, from this sparse collision matrix.
        Of course: only non-zero values are stocked in the sparse collision matrix.

        Important hypothesis: the context of use: the original matrix  passed to __init__,
        is triangular and is coding for symmetric content:
          - original content is symmetric
            - => this is why searched matching indexes are as well row indexes or column indexes
          - actual content is triangular
            - => no redundant indexes returned in the result

        :param index: the index of the column selected in the matrix
        :type index: int

        :return: the list of the indexes of the column selected without the zeros cells
        :rtype: list
        """
        result = []
        for item in self.data:
            if item[1][0] == index:
                result.append(item[1][1])
            elif item[1][1] == index:
                result.append(item[1][0])
        return result


def final_collision_matrix(sax, number_of_iterations, index_selected, word_len, spark_ctx):
    """
    Build the collision matrix based on SAX result.

    :param sax: the transpose of the result of SAX.
        Example with SAX words 'bdbc', 'cdab', ..., 'accc':
        sax = [['b', 'd', 'b', 'c'], ['c', 'd', 'a', 'b'], ...,  ['a', 'c', 'c', 'c']]
    :type sax: rdd of numpy.ndarray

    :param number_of_iterations: number of iterations used to build the collision matrix (positive integer)
    :type number_of_iterations: int

    :param index_selected: number of columns selected to build the collision matrix
    :type index_selected: int (in [2,size_word-1])

    :param word_len: The len of a sequence (number of letters).
    (correspond to sax_info.paa, or len(sax) )
    :type word_len: int

    :param spark_ctx: spark context
    :type spark_ctx: pyspark.context.SparkContext


    :return: Tuple containing :
        * the sparse matrix is returned instead of the computed collision matrix: this way we avoid a lot of zeros.
        * the number of iteration used in the algorithm
    :rtype: tuple of (SparseMatrix, int)

    :raise exception: IkatsException when an error occurred while processing the collision matrix
    """

    if type(sax) is not pyspark.rdd.PipelinedRDD:
        msg = "Unexpected type : spark rdd expected for sax={}"
        raise IkatsException(msg.format(sax))

    if (type(number_of_iterations) is not int) or number_of_iterations <= 0:
        msg = "Unexpected arg value (expects positive int): number_of_iterations={}"
        raise IkatsException(msg.format(number_of_iterations))

    if type(word_len) is not int or word_len <= 1:
        msg = "Unexpected arg value (expected positive int) : word_len={}"
        raise IkatsException(msg.format(word_len))

    if type(index_selected) is not int or index_selected not in range(2, word_len):
        msg = "Unexpected arg value : integer within [2, word size] expected for index_selected={}"
        raise IkatsException(msg.format(index_selected))

    try:
        # list of all possible combinations of indexes without repetitions
        comb = list(combinations(range(word_len), index_selected))
        # list of the *number_of_iteration* randomly selected in *comb*

        # If the number of iteration is larger than the possible number of iteration...
        number_of_iterations = min(len(comb), number_of_iterations)
        LOGGER.info("number of iteration specified is too large, new value=%s", number_of_iterations)

        list_column_selected = random.sample(comb, number_of_iterations)

        result = spark_ctx.parallelize([])

        # calculate *index_selected* collision matrix.
        for column_selection in list_column_selected:
            # for each iteration: chose randomly N columns (N = index_selected)

            # 1/ Select words and their occurrences
            #
            # INPUT: sax matrix of words: example: [['b', 'd', 'b', 'c'], ['c', 'd', 'a', 'b'],...,['a', 'c', 'c', 'c']]
            # OUTPUT: list of joined words: Example with N=2 : select_col = ['ab', 'cd', ..., 'ac']
            # PROCESS: select correct columns and concat the letters
            selected_col = sax.map(lambda x: ''.join(x[c] for c in column_selection))

            # INPUT: list of joined words: Example with N=2 : select_col = ['ab', 'cd', ..., 'ac']
            # OUTPUT: Example with N=2: selected_col = [ ('ab',0),('cd',1),('ab',2), ...,('ab',13),('ac',*size_word*)]
            # PROCESS: add a key for each element
            selected_col = selected_col.zipWithIndex()

            # INPUT: selected_col = [ ('ab',0), ('cd',1), ('ab',2), ...,('ab',13), ('ac',*size_word*) ]
            # OUTPUT: Example with N=2: group = [ ('ab',[0,2,13]),..., ('ac',[...,*size_word*]) ]
            # PROCESS: group all the values of *index* by key in lists
            # Note that if a word have just one occurrence, it doesn't modify the collision matrix
            selected_col = selected_col.groupByKey()

            # 2/ Determine the values in the collision matrix (and the positions):
            #
            def _spark_flat_collision_position(word):
                """
                Find the position in the position matrix, only if the word have not one occurrence!
                """
                if len(word[1]) > 1:
                    return [[c, 1] for c in list(combinations(word[1], 2))]
                else:
                    # The word has just one occurrence -> does not matter in the collision matrix
                    return []

            # INPUT: selected_col = [ ('ab',[0,2,13]),..., ('ac',[...,*size_word*]) ]
            # OUTPUT: group = [ [(0,2),1], [(0,13), 1],  [(2,13), 1],... ]
            # PROCESS: Find the position in the position matrix
            mapped_rdd = selected_col.flatMap(lambda word: _spark_flat_collision_position(word))

            # The position of the values in the collision matrix are the combinations for all
            # the values in *selected_col*

            # Example with N=2: group = [ [(0,2),1], [(0,13), 1],  [(2,13), 1],... ]
            # It means that the current collision matrix has a 1 value in the positions (0,2); (0,13); (2,13); ...
            # UNIQUE values

            # 3/ add the collision matrix iteratively
            #

            if not mapped_rdd.isEmpty():
                # all the collision matrix are summed (done iteratively for memory economy)
                result = result.union(mapped_rdd).reduceByKey(add)
                # Example : result = [ [(0,2),2], [(0,13), 3],  [(2,13), 1],... ]

        # INPUT: Example : result = [ [(0,2),2], [(0,13), 3],  [(2,13), 1],... ]
        # OUTPUT: Example : result = [ (2, (2,0)), (3, (13,0)),  (1, (13,2)),... ]
        # PROCESS : represent result as a SparseMatrix (class already created)
        # Note that the collision matrix is symmetric -> we want the upper values (for non regression)
        result = result.map(lambda x: (x[1], (x[0][1], x[0][0])))

        # Create an empty SparseMatrix object...
        sparse_matrix = SparseMatrix(np.array([[0]]))
        # ... and fill it !

        sparse_matrix.data = result.collect()
        LOGGER.info('... ended final_collision_matrix')

        return sparse_matrix, number_of_iterations

    except Exception:
        LOGGER.error("... ended final_collision_matrix with error.")
        raise IkatsException("Failed execution: collision()")


def equation9(number_of_sequences, size_alphabet, size_word, errors, index_selected, iterations):
    """
    Give a value that help to decide when we have to stop the algorithm step
    which builds the collision matrix. This value is the probability that we make errors when we build the collision
    matrix multiply by the number of iterations to built the collision matrix.
    For more details, read the article
    "Probabilistic Discovery of Time Series Motifs" of Bill Chiu, Eamonn Keogh, and Stefano Lonardi

    :param number_of_sequences: the number of sequences
    :type number_of_sequences: int

    :param size_alphabet: the size of the alphabet used by SAX
    :type size_alphabet: int

    :param size_word: number of letters in a word
    :type size_word: int

    :param errors: the number of mistakes accepted by comparing few words
    :type errors: int

    :param index_selected: the number of index selected to build the collision matrix
    :type index_selected: int

    :param iterations: number of iterations to built the collision matrix
    :type iterations: int

    :return: the value that help to decide when we have to stop the algorithm
    :rtype: float or int

    """
    result = 0
    for i in range(0, errors + 1):
        result += ((1 - i / size_word) ** index_selected) * binom(size_word, i) \
                  * (((size_alphabet - 1) / size_alphabet) ** i) * (1 / size_alphabet) ** (size_word - i)
    result = result * binom(number_of_sequences, 2)

    # result > 1 should not happen: nevertheless be robust
    if result >= 1:
        # interpreted as: collision is always expected.
        # => for n iterations : n collisions
        #
        # As it is a strict minimum boundary: decrease by one.
        result = iterations - 1
    else:
        # initial result is the probability per iteration that there is a collision
        # => return the expected nb of collisions:
        result = result * iterations

    return result
