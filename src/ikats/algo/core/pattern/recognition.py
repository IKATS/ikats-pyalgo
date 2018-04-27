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
import string

import numpy as np

from ikats.core.library.exception import IkatsException
from ikats.core.library.spark import ScManager

"""
This module provides services about the Random Projection algorithm last step:
this step searches and completes the neighborhood of each pattern detected from the collision matrix.
The service proposed:
  * recognition_spark()
"""

LOGGER = logging.getLogger(__name__)

OPT_USING_BRUTE_FORCE = 1
OPT_USING_COLLISIONS = 2
OPTIONS_RECOG = [OPT_USING_BRUTE_FORCE, OPT_USING_COLLISIONS]


def _start_alphabet(size_alphabet):
    """
    Create an alphabet of a given size

    :param size_alphabet
    :type size_alphabet: int

    :return: an alphabet of a given size
    :rtype: list
    """
    return list(string.ascii_lowercase)[0:size_alphabet]


def _search_index(items_list, item):
    """
    Give the index of an item in a list or return -1 if this item is not in the list

    :param items_list: list with one item or more
    :type items_list: list

    :param item: an item
    :type item: booleen, float, int, list, numpy.ndarray

    :return: the index of the item in the list, or -1
    :rtype: int
    """
    if (item in items_list) is True:
        return items_list.index(item)
    else:
        return -1


def _get_mindist(size_sequence, word_1, word_2, mindist_lookup_table, alphabet):
    """
    Give the MINDIST between two words

    :param size_sequence: number of points in a sequence
    :type size_sequence: int

    :param word_1: a words
    :type word_1: list

    :param word_2: a word
    :type word_2: list

    :param mindist_lookup_table: a table used by MINDIST, it give the distance between two symbols
    :type mindist_lookup_table: numpy.ndarray

    :param alphabet: the alphabet used by the SAX algorithm
    :type alphabet: list

    :return: the MINDIST between the words given
    :rtype: float
    """
    size_word = len(word_1)
    dist = 0
    for i in range(0, size_word):
        dist += (_get_letters_dist(word_1[i], word_2[i], mindist_lookup_table, alphabet)) ** 2
    dist = np.sqrt(size_sequence / size_word) * np.sqrt(dist)
    return dist


def _get_letters_dist(letter_1, letter_2, mindist_lookup_table, alphabet):
    """
    Give the distance between two letters with the breakpoint table

    :param letter_1: a letter
    :type letter_1: string

    :param letter_2: a letter
    :type letter_2: string

    :param mindist_lookup_table: a table used by MINDIST, it give the distance between two symbols
    :type mindist_lookup_table: numpy.ndarray

    :param alphabet: the alphabet used by the SAX algorithm
    :type alphabet: list

    :return:  a distance
    :rtype: float
    """
    index_letter_1 = _search_index(alphabet, letter_1)
    index_letter_2 = _search_index(alphabet, letter_2)
    return mindist_lookup_table[index_letter_1, index_letter_2]


def _brute_force_neighborhood(dist_info, largest_value, alphabet):
    """
    Search all sequences which are similar to the two SAX sequences (Si, Sj) defined by the largest value.
    To do that, we use the brute force: search all SAX sequences which are either similar to Si or Sj.

    Similarity criteria with SAX is: mindist( Sx, Sy ) < dist_info.radius


    :param dist_info: object grouping information required by the search: the set of SAX words, ...
    :type dist_info: NeighborhoodSearch

    :param largest_value: pair of indexes [i,j] that correspond to one largest value in collision matrix,
      identifying similar sequences Si and Sj
    :type largest_value: list of int

    :param alphabet: a list of letters used by SAX
    :type alphabet: list

    :return: list of numbers that correspond to similar sequences, also including i and j from the largest_value.
    :rtype: list
    """
    largest_1 = largest_value[0]
    largest_2 = largest_value[1]
    similar_sequences = [largest_1, largest_2]

    # search all SAX sequences in the neighborhood: mindist( Sx, Si ) < dist_info.radius
    for j in range(len(dist_info.sax)):
        if j != largest_1 and j != largest_2:
            dist_largest_val_0 = _get_mindist(dist_info.size_sequence, dist_info.sax[largest_1], dist_info.sax[j],
                                              dist_info.mindist_lookup_table, alphabet)
            if dist_largest_val_0 >= dist_info.radius:
                dist_largest_val_1 = _get_mindist(dist_info.size_sequence, dist_info.sax[largest_2], dist_info.sax[j],
                                                  dist_info.mindist_lookup_table, alphabet)
                if dist_largest_val_1 < dist_info.radius:
                    similar_sequences.append(j)
            else:
                similar_sequences.append(j)

    return similar_sequences


def _neighborhood_option(dist_info, largest_value, alphabet):
    """
    Search all sequences which are similar to the two SAX sequences (Si, Sj) defined by the largest value.
    To do that, we use the heuristic based upon collision matrix: it supposes that searched sequences Sx have at least
    one collision with either Si or Sj. Retained Sx are either similar to Si or Sj.

    Similarity criteria with SAX is: mindist( Sx, Sy ) < dist_info.radius

    :param dist_info: information used to calculate the distance between two sequences
    :type dist_info: NeighborhoodSearch

    :param largest_value: pair of indexes [i,j] that correspond to one largest value in collision matrix,
      identifying similar sequences Si and Sj

    :param alphabet: a list of letters used by SAX

    :type alphabet: list

    :return: list of numbers that correspond to similar sequences, also including i and j from the largest_value.
    :rtype: list
    """
    largest_1 = largest_value[0]
    largest_2 = largest_value[1]
    similar_sequences_0 = [largest_1, largest_2]
    similar_sequences_1 = [largest_1, largest_2]

    # Extract the column i from the collision matrix (which is not symmetric)
    column_0 = dist_info.collision_matrix.get_column(largest_1)
    column_1 = dist_info.collision_matrix.get_column(largest_2)

    # Search the neighborhood sequences in neighborhood_0 by using MINDIST
    for i in column_0:
        if _get_mindist(dist_info.size_sequence, dist_info.sax[largest_1], dist_info.sax[i],
                        dist_info.mindist_lookup_table,
                        alphabet) < dist_info.radius:
            similar_sequences_0.append(i)

    # Search the neighborhood sequences in neighborhood_1 by using MINDIST
    for i in column_1:
        if _get_mindist(dist_info.size_sequence, dist_info.sax[largest_2], dist_info.sax[i],
                        dist_info.mindist_lookup_table,
                        alphabet) < dist_info.radius:
            similar_sequences_1.append(i)

    return list(set(similar_sequences_0) | set(similar_sequences_1))


def _recognition(dist_info, list_largest_value, option):
    """
    Search all sequences which are similar to two sequences given by the largest value in the collision matrix.
    To do that, we have two different methods, selected by choice argument:
      1. using the brute force (option == OPT_USING_BRUTE_FORCE)
      2. using heuristic with collision matrix  (option == OPT_OPT_USING_COLLISIONS)

    :param dist_info: information used to calculate the distance between two sequences
    :type dist_info: NeighborhoodSearch

    :param list_largest_value: list of [index_i, column] pairs that correspond to the largest values found in the
    collision matrix
    :type list_largest_value: list of list of int

    :param option: the choice of the recognition method: in OPTIONS_RECOG
    :type option: int

    :return: list of list of numbers that correspond to sequences
    :rtype: list

    :raise if option is not a positive integer within OPTIONS_RECOG
    """

    if type(option) is not int or option not in OPTIONS_RECOG:
        raise ValueError('option must be a positive integer within {}'.format(OPTIONS_RECOG))

    list_similar_sequences = []
    alphabet = _start_alphabet(dist_info.alphabet_size)

    # search all sequences in the neighborhood
    for largest_value in list_largest_value:
        index_i = largest_value[0]
        index_j = largest_value[1]
        if _get_mindist(dist_info.size_sequence, dist_info.sax[index_i], dist_info.sax[index_j],
                        dist_info.mindist_lookup_table, alphabet) < dist_info.radius:
            # by using the brute force
            if option == OPT_USING_BRUTE_FORCE:
                similar_sequences = _brute_force_neighborhood(dist_info, largest_value, alphabet)
            # by using the collision_matrix heuristic
            if option == OPT_USING_COLLISIONS:
                similar_sequences = _neighborhood_option(dist_info, largest_value, alphabet)

            list_similar_sequences.append(similar_sequences)
    return list_similar_sequences


def recognition(dist_info, largest_values, option, activate_spark):
    """
    Sparkified Search all sequences which are similar to two sequences given by the largest value in the collision
    matrix.
    To do that, we have two different methods, selected by choice argument:
      1. using the brute force (option == OPT_USING_BRUTE_FORCE)
      2. using heuristic with collision matrix (option == OPT_USING_COLLISIONS)

    :param dist_info: information used to calculate the distance between two sequences
    :type dist_info: NeighborhoodSearch

    :param largest_values: list of list of rows and columns that correspond to the largest values found in the
    collision matrix
    :type largest_values: list

    :param option: the choice of the method
    :type option: int (in OPTIONS_RECOG)

    :param activate_spark: True to force spark, False to force local, None to let the algorithm decide
    :type activate_spark: bool or none

    :return: list of list of numbers that correspond to sequences
    :rtype: list

    :raise exception: IkatsException when an error occurred while processing the collision matrix
    """

    # spark_ctx is inspected in block finally: this is only initialized when algo run under spark context
    spark_ctx = None
    try:

        if type(largest_values) is not list:
            msg = "Unexpected type : list expected for list_largest_value={}"
            raise IkatsException(msg.format(largest_values))

        if type(dist_info.radius) not in [float, int] or dist_info.radius <= 0:
            msg = "Unexpected arg value : positive float expected for radius={}"
            raise IkatsException(msg.format(dist_info.radius))

        if type(option) is not int or option not in OPTIONS_RECOG:
            msg = "Unexpected arg value : positive integer within {} expected for option={}"
            raise IkatsException(msg.format(OPTIONS_RECOG, option))

        if type(activate_spark) is not bool and activate_spark is not None:
            msg = "Unexpected type : booleen type or None expected for activate_spark={}"
            raise IkatsException(msg.format(activate_spark))

        LOGGER.info("Starting recognition ...")
        if activate_spark is None:
            activate_spark = len(dist_info.collision_matrix.data) > 1000 or len(largest_values) > 100

        if len(largest_values) == 0:
            return []

        else:

            if activate_spark is True:
                LOGGER.info("Running using Spark")

                # Create or get a spark Context
                spark_ctx = ScManager.get()

                # Build the RDD with the index list of the largest values
                rdd = spark_ctx.parallelize(largest_values)

                alphabet = _start_alphabet(dist_info.alphabet_size)

                def brute_force_spark(dist_info, largest_values, alphabet):
                    """
                    For details, see the brute force neighborhood algorithm
                    """
                    largest_1 = largest_values[0]
                    largest_2 = largest_values[1]
                    if _get_mindist(dist_info.size_sequence, dist_info.sax[largest_1], dist_info.sax[largest_2],
                                    dist_info.mindist_lookup_table, alphabet) < dist_info.radius:
                        similar_sequences = [largest_1, largest_2]

                        for j in range(len(dist_info.sax)):
                            if j != largest_1 and j != largest_2:
                                dist_val_0 = _get_mindist(dist_info.size_sequence, dist_info.sax[largest_1],
                                                          dist_info.sax[j],
                                                          dist_info.mindist_lookup_table, alphabet)
                                if dist_val_0 >= dist_info.radius:
                                    dist_val_1 = _get_mindist(dist_info.size_sequence, dist_info.sax[largest_2],
                                                              dist_info.sax[j], dist_info.mindist_lookup_table,
                                                              alphabet)
                                    if dist_val_1 < dist_info.radius:
                                        similar_sequences.append(j)
                                else:
                                    similar_sequences.append(j)

                    else:
                        similar_sequences = None
                    return similar_sequences

                def neighborhood_spark(dist_info, largest_values, alphabet):
                    """
                    For details, see the _neighborhood_option algorithm.
                    """
                    largest_1 = largest_values[0]
                    largest_2 = largest_values[1]
                    if _get_mindist(dist_info.size_sequence, dist_info.sax[largest_1], dist_info.sax[largest_2],
                                    dist_info.mindist_lookup_table, alphabet) < dist_info.radius:
                        similar_sequences_0 = [largest_1, largest_2]
                        similar_sequences_1 = [largest_1, largest_2]

                        column_0 = dist_info.collision_matrix.get_column(largest_1)
                        column_1 = dist_info.collision_matrix.get_column(largest_2)

                        for i in column_0:
                            if _get_mindist(dist_info.size_sequence, dist_info.sax[largest_1], dist_info.sax[i],
                                            dist_info.mindist_lookup_table, alphabet) < dist_info.radius:
                                similar_sequences_0.append(i)

                        for i in column_1:
                            if _get_mindist(dist_info.size_sequence, dist_info.sax[largest_2], dist_info.sax[i],
                                            dist_info.mindist_lookup_table, alphabet) < dist_info.radius:
                                similar_sequences_1.append(i)

                        similar_sequences = list(set(similar_sequences_0) | set(similar_sequences_1))

                    else:
                        similar_sequences = None
                    return similar_sequences

                if option == OPT_USING_BRUTE_FORCE:
                    LOGGER.info("- Running using the brute force")
                    rdd = rdd.map(lambda x: brute_force_spark(dist_info=dist_info,
                                                              largest_values=x,
                                                              alphabet=alphabet))

                if option == OPT_USING_COLLISIONS:
                    LOGGER.info("- Running using the neighborhood the collision matrix")
                    rdd = rdd.map(lambda x: neighborhood_spark(dist_info=dist_info,
                                                               largest_values=x,
                                                               alphabet=alphabet))

                collected_data = rdd.filter(lambda x: x is not None).collect()
                LOGGER.info("... ended recognition.")
                return collected_data

            else:
                LOGGER.info("Running without using Spark")
                result = _recognition(dist_info, largest_values, option)
                LOGGER.info("... ended recognition.")
                return result

    except Exception:
        LOGGER.error("... ended recognition with error.")
        raise IkatsException("Failed execution: _recognition()")

    finally:
        if spark_ctx is not None:
            ScManager.stop()
