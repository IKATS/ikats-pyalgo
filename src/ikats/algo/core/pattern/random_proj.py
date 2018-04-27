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
import itertools
import logging
import string
from collections import defaultdict

import numpy as np
from scipy.special import binom

from ikats.algo.core.pattern.collision import final_collision_matrix, SparseMatrix, equation9
from ikats.algo.core.pattern.recognition import OPT_USING_BRUTE_FORCE, OPT_USING_COLLISIONS, recognition
from ikats.algo.core.sax.sliding_sax import sliding_windows, run_sax_on_sequences
from ikats.core.library.exception import IkatsException
from ikats.core.library.spark import ScManager

LOGGER = logging.getLogger(__name__)

# CONFIG_NORM_METHOD is a dict whose key is coding the selected normalizing method

WITH_MEAN = "WITH_MEAN"
WITH_VARIANCE = "WITH_VARIANCE"
MEAN_VARIANCE = "MEAN_VARIANCE"

# flags are [with_mean, with_std]
CONFIG_NORM_METHOD = {WITH_MEAN: [True, False],
                      WITH_VARIANCE: [False, True],
                      MEAN_VARIANCE: [True, True]}

# CONFIG_NORM_MODE is a dict whose key is coding the selected normalizing mode

NO_NORM = "NO_NORM"
LOCAL_NORM = "LOCAL_NORM"
GLOBAL_NORM = "GLOBAL_NORM"
LOCAL_GLOBAL = "LOCAL_GLOBAL"

# flags are [local_norm, global_norm]
CONFIG_NORM_MODE = {NO_NORM: [False, False],
                    LOCAL_NORM: [True, False],
                    GLOBAL_NORM: [False, True],
                    LOCAL_GLOBAL: [True, True]}

# CONFIG_FILTER is a dict whose key is coding the selected coefficients for the linear filter

NO_FILTER = "NO_FILTER"
LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"

# flags are [linear_filter, coefficients]
CONFIG_FILTER = {NO_FILTER: [False, 0, 1],
                 LOW: [True, 0.01, 0.97],
                 MEDIUM: [True, 0.1, 0.9],
                 HIGH: [True, 0.25, 0.7]}

# CONFIG_NEIGHBORHOOD_METHOD is a dict whose key is coding the selected neighborhood method

ITERATIVE_BRUTE_FORCE = "ITERATIVE_BRUTE_FORCE"
GLOBAL_BRUTE_FORCE = "GLOBAL_BRUTE_FORCE"
ITERATIVE_WITH_COLLISIONS = "ITERATIVE_WITH_COLLISIONS"
GLOBAL_WITH_COLLISIONS = "GLOBAL_WITH_COLLISIONS"

# flags are [is_algo_method_global, neighborhood_method]
CONFIG_NEIGHBORHOOD_METHOD = {ITERATIVE_BRUTE_FORCE: [False, OPT_USING_BRUTE_FORCE],
                              GLOBAL_BRUTE_FORCE: [True, OPT_USING_BRUTE_FORCE],
                              ITERATIVE_WITH_COLLISIONS: [False, OPT_USING_COLLISIONS],
                              GLOBAL_WITH_COLLISIONS: [True, OPT_USING_COLLISIONS]}

EMPTY_REGEX_MESSAGE = 'No Regex'


class ConfigSax(object):
    """
    Constructor of the parameters to prepare the sequences, and apply the SAX algorithm.
    """

    def __init__(self, paa, sequences_size, with_mean, with_std, global_norm, local_norm, linear_filter, recovery,
                 coefficients, alphabet_size):
        """
        :param paa: number of Piecewise Aggregate Approximation in a sliding window
        :type paa: int

        :param sequences_size: number of points in a sequence
        :type sequences_size: int

        :param with_mean: sets the output mean to zero
        :type with_mean: bool

        :param with_std: sets the output std to 1 when possible
        :type with_std: bool

        :param global_norm: global normalization
        :type global_norm: bool

        :param local_norm: local normalization (normalize sequence)
        :type local_norm: bool

        :param linear_filter: filter of the linear or/and constant sequences
        :type linear_filter: bool

        :param recovery: recovery coefficient of the sequences that correspond to the percentage of the recovery of the
               previous sequence : in [0, 1].
               Example: recovery == 0.75 means that the overlap between successive sliding
               windows is 75percent <=> the translation is 25percent of the window size.

               Specific values:
                 - 0: if there is no overlap between sequences,
                 - 1: if the next sequence begin one point to the right.
        :type recovery: float

        :param coefficients: coefficients used to filter sequences:
                                - the first coefficient is for the constant sequences filter: a percentage of the
                                  timeseries variance in [0, 1[, typically in [0,0.1]
                                - the second coefficient is for the linear sequences filter: a value compared with the
                                coefficient of determination in ]0,1], typically in [0.9, 1] (see
                                https://en.wikipedia.org/wiki/Coefficient_of_determination)
        :type coefficients: list

        :param alphabet_size: the size of the alphabet used in the SAX algorithm (in [2,26])
        :type alphabet_size: int
        """

        self.paa = paa
        self.sequences_size = sequences_size
        self.with_mean = with_mean
        self.with_std = with_std
        self.global_norm = global_norm
        self.local_norm = local_norm
        self.linear_filter = linear_filter
        self.recovery = recovery
        self.coefficients = coefficients
        self.alphabet_size = alphabet_size


class ConfigCollision(object):
    """
    The configuration of collision management
    """

    def __init__(self, iterations, index, config_sax):
        """
        Constructor of all the parameters to build the collision matrix and calculate
        the result of the Equation 9 (which can be used to stop the random projections algorithm)
        :param iterations: the percentage of the maximum of the combinations possible.

        :type iterations: int

        :param index: number of index selected to build the collision matrix (in [2, word_size])
        :type index: int

        :param config_sax: sax configuration is required for the ConfigCollision:
          - used by computing self.errors, self.nb_iterations
          - kept as part of configuration of collision management

        :type config_sax: ConfigSax
        """

        self.iterations = iterations

        self.index = index

        # computes effective number of iterations from the percentage selected by the user
        # ...
        max_iter = binom(config_sax.paa, index)

        # nb_iterations in [1, ... [
        nb_iterations = max(1, int(max_iter * iterations))

        # robustness: do not exceed 10000 iterations
        # => wait until
        #
        self.nb_iterations = min(nb_iterations, 10000)

        # keep sax config as part of collision config
        self.config_sax = config_sax

        # computes number of accepted errors from config_sax
        self.errors = int(config_sax.paa / 10) ** 2 + int(config_sax.alphabet_size / 10)


class ConfigRecognition(object):
    """
    Constructor of the parameters to search the motif neighborhood.
    """

    def __init__(self, is_stopped_by_eq9, is_algo_method_global, min_value, iterations, radius, neighborhood_method,
                 activate_spark):
        """
        :param is_stopped_by_eq9: choose to use or not the result of the Equation 9 to stop the algorithm
        :type is_stopped_by_eq9: bool

        :param is_algo_method_global: choose the method to find the motif neighborhood : iterative or global
        :type is_algo_method_global: bool

        :param min_value: give the value that stop the motif neighborhood search in [0, iterations to build
                          collision matrix - 1]
        :type min_value: int

        :param iterations: give the number of iterations to do the motif neighborhood search ( > 1)
        :type iterations: int

        :param radius: give the radius of motif neighborhood ( >=  0)
        :type radius: float

        :param neighborhood_method: give the search neighborhood method: value taken from list recognition.OPTIONS_RECOG
        :type neighborhood_method: int

        :param activate_spark: activate or not spark, if None the algorithm chooses
        :type activate_spark: bool or None
        """

        self.is_stopped_by_eq9 = is_stopped_by_eq9
        self.is_algo_method_global = is_algo_method_global
        self.min_value = min_value
        self.iterations = iterations
        self.radius = radius
        self.neighborhood_method = neighborhood_method
        self.activate_spark = activate_spark


def start_alphabet(alphabet_size):
    """
    Create an alphabet of a given size, for the output of the algorithm

    :param alphabet_size: the size of the alphabet
    :type alphabet_size: int

    :return: an alphabet of a given size
    :rtype: list
    """
    # Note strings and lists can be treated equivalently
    # Strings are lists of characters
    return list(string.ascii_uppercase)[0:alphabet_size]


def sequences_info(sequences_list, normalization_coefficients):
    """
    Give a list with the name, and some information on each sequence.

    :param sequences_list: a dictionary with all sequences extract by the sliding window:
        - keys: name of sequence
        - values: the normalized sequence as numpy array giving TS points:  [ [t1, v1 ], ... , [tN, vN] ]

    :type sequences_list: dict

    :param normalization_coefficients: dictionary of additional information

        - keys: name of sequence as string formatted <ts name><rank>
        - values: [ value_offset, scale_factor ] associated to each sequence,
          in order to recompute the original value from normalized value (vi) for each sequence:

            - value= original(vi) = vi * scale_factor + value_offset

    :type normalization_coefficients: dict

    :return: list of elements [ name, information ] where
      - name: reference of the original TS the sequence is derived from
      - information is dictionary providing values for keys 'timestamp', 'size', 'normalization'
    :rtype: list of list of str and dict

    Example for ONE TS (here "422CD30000030001CB")

    [
      ["422CD30000030001CB", {
            "timestamp": 1449715895000,
            "size": 125,
            "normalization": {
                "average": 26.123,
                "deviation": 0.0458
                  }
            }
      ],
      [...]
    ]

    """
    result = []

    # For each sequences
    for name in sorted(sequences_list.keys()):
        begin = sequences_list[name][0, 0]
        seq_information = {"timestamp": begin,
                           "size": sequences_list[name][len(sequences_list[name]) - 1, 0] - begin,
                           "normalization": {"average": normalization_coefficients[name][0],
                                             "deviation": normalization_coefficients[name][1]}}

        # wrap all this information in a single dict for each TS
        result.append([name[:name.index('_')], seq_information])
    return result


def regex_from_pattern_results(pattern_sax, alphabet_size):
    """
    Compute a regex representative of a given pattern sax
    :param pattern_sax: given pattern
    :type pattern_sax: list
    :param alphabet_size: size of alphabet used for sax
    :type alphabet_size: int

    :return: the computed regex
    :rtype: str
    """

    if len(pattern_sax) == 0:
        return EMPTY_REGEX_MESSAGE

    def regex_part(letters):
        """
        Builds the regexp from letters
        :param letters:
        :return:
        """
        set_of_letters = set(letters)
        length_set = len(set_of_letters)

        if length_set == 1:
            return letters[0]
        elif length_set == alphabet_size:
            return "*"
        else:
            return "[" + "".join(sorted(set_of_letters)) + "]"

    return "".join([regex_part(col) for col in zip(*pattern_sax)])


def result_on_sequences_form(algo_result, sequences_list, sax, alphabet_size, paa_sequences=None):
    """
    Encode intermediate information about similar sequences (without indexes from the collision matrix):
      - encoded result is not yet compliant with the final result.

    .. note:: If specified, encoded results contains the paa values for each sequence
    (very useful for post-process the random projection algorithm).


    :param algo_result: the results of the algorithm with the numbers of the similar sequences in the collision matrix
    :type algo_result: list

    :param sequences_list: name, first timestamp and the time of each sequence.
    :type sequences_list: list

    :param sax: sax sequences: list of words
    :type sax: numpy array of str

    :param alphabet_size: size of alphabet used for sax
    :type alphabet_size: int

    :param paa_sequences: The paa values of each sequences, disposed by columns into a matrix.
    If None, the function don't consider it.
    :type paa_sequences : numpy array or NoneType

    :return: list of dictionaries: each dictionary stands for a **group of similar sequences grouped by TS**

        - key: the TS name from the dataset
        - value: dict of similar sequences, as a second-level dictionary holding details about sequences:

            - key: 'seq'
            - value: list of dictionaries describing each sequence (key)
              'size', 'timestamp', 'normalization' pointing to values

    :rtype: list of dict
    """
    LOGGER.info("Running sequences form...")
    if len(algo_result) < 1:
        result = None

    else:
        result = []
        for pattern in algo_result:
            name = defaultdict(list)
            pattern_sax = []

            for sequence in pattern:
                # if sax_paa_value is specified, we introduce the paa values into the sequence_list
                if paa_sequences is not None:
                    sequences_list[sequence][1]["paa_value"] = list(paa_sequences[sequence])

                # make a list of sequences information (sequences_list[sequence][1])
                # for the current TSuid (sequences_list[sequence][0]).
                name[sequences_list[sequence][0]].append(sequences_list[sequence][1])
                pattern_sax.append(sax[sequence])

            for key in name.keys():
                seq = {"seq": name[key]}
                name[key] = seq

            name = dict(name)

            # resume all the sax words of ONE pattern into a regular expression (loss of information)
            ikats_regex = regex_from_pattern_results(pattern_sax, alphabet_size)
            name['regex'] = ikats_regex

            result.append(name)
    LOGGER.info("Formatting sequences done...")

    return result


def result_on_pattern_form(algo_result):
    """
    Generate the information about each detected pattern:

      - encoded result is compliant with 'patterns' property of functional type pattern_groups.

    :param algo_result: result from function result_on_sequences_form from this module.
    :type algo_result: list of dict

    :return: the resulting structure as described by property patterns of functional type pattern_groups.
    :rtype: dict of dict
    """

    if algo_result is None:
        return {}

    else:
        result = {}
        i = 1
        for pattern in algo_result:
            pattern_information = {}
            pattern_information["regex"] = pattern.pop('regex')
            pattern_size = 0
            for tsuid in pattern.keys():
                pattern_size += len((pattern[tsuid])["seq"])
            pattern_information["length"] = pattern_size
            pattern_information["locations"] = pattern
            result["P" + str(i)] = pattern_information
            i += 1
        return result


class NeighborhoodSearch(object):
    """
    The object NeighborhoodSearch provides
    methods searching the motif neighborhood
    """

    def __init__(self, size_sequence, mindist_lookup_table, alphabet_size, sax, radius, collision_matrix):
        """
        Constructor

        :param size_sequence: number of points in a sequence
        :type size_sequence: int

        :param mindist_lookup_table: a table which give the distance between two letters
        :type mindist_lookup_table: numpy.ndarray

        :param alphabet_size: the size of the alphabet used by the SAX algorithm
        :type alphabet_size: int

        :param sax: the result of the SAX algorithm
        :type sax: numpy.ndarray

        :param radius: give the radius of motif neighborhood ( >=  0)
        :type radius: float

        :param collision_matrix: the sparse collision matrix
        :type collision_matrix: SparseMatrix
        """

        self.size_sequence = size_sequence
        self.mindist_lookup_table = mindist_lookup_table
        self.alphabet_size = alphabet_size
        self.sax = sax
        self.radius = radius
        self.collision_matrix = collision_matrix

    @classmethod
    def __clean_repetitions(cls, result):
        """
        Delete all repetitions in a list of lists
        :param result: the result of Neighborhood methods
          - motif_neighborhood_iterative
          - or motif_neighborhood_global
        :type result: list of lists

        :return: list of lists: each list without redundant element
        :rtype: list of lists
        """

        for group, _ in enumerate(result):
            result[group].sort()

        result.sort()
        return list(result for result, _ in itertools.groupby(result))

    def motif_neighborhood_global(self, eq9_result, recognition_info):
        """
        Search the similar sequences from self.collision_matrix, using global variant of algorithm:
        consider all collisions greater than the criteria eq9_result.

        :param eq9_result: this value helps to decide if a collision probably reveal
           similarity: see collision.equation9().
           If collision(i,j) > eq9_result then algorithm ought to evaluate similarity between corresponding sequences.
        :type eq9_result: float

        :param recognition_info: the recognition configuration required to complete this step.
        :type recognition_info: ConfigRecognition

        :return: a list of groups of indexes pointing to similar sequences.
                 Each index refers to collision matrix (SparseMatrix)
        :rtype: list of list of int
        """

        # take the indices of all largest values cells in the collision matrix greater than eq9_result
        list_largest_value = [x for x in self.collision_matrix.data if x[0] > eq9_result]
        largest_values = [x[1] for x in list_largest_value]

        result_algo = recognition(dist_info=self,
                                  largest_values=largest_values,
                                  option=recognition_info.neighborhood_method,
                                  activate_spark=recognition_info.activate_spark)

        # Delete repetitions in the result
        return self.__clean_repetitions(result_algo)

    def motif_neighborhood_iterative(self, eq9_result, recognition_info):
        """
        Search the similar sequences from self.collision_matrix, using iterative variant of algorithm.
        Iterates X times for the X bigger collision values, as long as they are greater than eq9_result.

        :param eq9_result: this value helps to decide if a collision probably reveal similarity:
                           see collision.equation9().
                           If collision(i,j) > eq9_result then algorithm ought to evaluate similarity
                           between corresponding sequences.
        :type eq9_result: float

        :param recognition_info: the information to made the pattern _recognition
        :type recognition_info: class

        :return: a list of groups of indexes pointing to similar sequences.
          Each index refers to collision matrix (SparseMatrix)
        :rtype: list of list of int
        """

        # largest value cells with the occurrences of this value and the list of the values
        largest_values = self.collision_matrix.values_matrix(min_value=eq9_result)
        keys = list(sorted(largest_values.keys()))

        if len(keys) == 0:
            return []

        else:
            # search sequences in the neighborhood of two motifs for the all the cells that corresponding to one value
            result_algo = recognition(dist_info=self,
                                      largest_values=largest_values[keys[-1]],
                                      option=recognition_info.neighborhood_method,
                                      activate_spark=recognition_info.activate_spark)

            # the number of iterations is the number of largest values that will be taken to search similar sequences
            if len(result_algo) > 0:
                i = 1
            else:
                i = 0
            del keys[-1]
            while i < recognition_info.iterations and len(keys) > 0:
                neighborhood = recognition(dist_info=self,
                                           largest_values=largest_values[keys[-1]],
                                           option=recognition_info.neighborhood_method,
                                           activate_spark=recognition_info.activate_spark)
                result_algo += neighborhood
                if len(neighborhood) > 0:
                    i += 1
                del keys[-1]

            # Delete repetitions in the result
            return self.__clean_repetitions(result_algo)


def random_projections(ts_list, sax_info, collision_info, recognition_info):
    """
    The Random Projections Algorithm
    ================================

    This algorithm does the following (detailed for 1 TS but valid for many TS):
        * Apply the sliding window
        * Normalize the TS (global or/and local)
        * Filter the linear sequences (optional) and trivial matches
        * Apply the SAX algorithm
        * Build the collision matrix
        * Find the largest value cells in the collision matrix
        * Search the motif neighborhood

        ..note::
            The algorithm can produce "paa values" (numeric) for each sequence. The problem is the huge length of the
            results.

    **Catalogue implementation is provided**: main_random_projections() is calling random_projections() once all
    configurations ConfigSAX, ConfigCollision, ConfigRecognition are initialized.

    :param ts_list: list of TSUID
    :type ts_list: list

    :param sax_info: the information to make the sliding window and the sax_algorithm
    :type sax_info: ConfigSax

    :param collision_info: the information to build the collision matrix
    :type collision_info: ConfigCollision

    :param recognition_info: the information to made the pattern _recognition
    :type recognition_info: ConfigRecognition

    :return: the list of similar sequences, the sax result, the equation 9 result, and the sequences list
    :type: list, str, float, list
    """
    LOGGER.info("Configurations deduced from user parameters:")
    LOGGER.info("- sliding sax nb paa=%s", sax_info.paa)
    LOGGER.info("- sliding sax alphabet size=%s", sax_info.alphabet_size)
    LOGGER.info("- sliding sax sequences_size=%s", sax_info.sequences_size)
    LOGGER.info("- collision nb indexes=%s", collision_info.index)
    LOGGER.info("- collision nb iterations=%s", collision_info.nb_iterations)
    LOGGER.info("- collision accepted errors=%s", collision_info.errors)
    LOGGER.info("- recognition min_value=%s", recognition_info.min_value)
    LOGGER.info("- recognition iterations=%s", recognition_info.iterations)
    LOGGER.info("- recognition similarity radius=%s", recognition_info.radius)

    # Create or get a spark Context
    LOGGER.info("Running using Spark")
    spark_ctx = ScManager.get()

    # INPUT : all the TS { "ts_name" : [[time1, value1],...], "ts_name2": ... }
    # OUTPUT :  rdd_sequences_list = [ (key, sequence), ... ]
    # rdd_normalization_coefficients = [ (same_key,(un-normalized seq_mean, un-normalized seq_sd)), ...]
    # PROCESS : *sliding_windows* create sequences for each TS (results are RDDs)
    rdd_sequences_list, rdd_normalization_coefficients = sliding_windows(ts_list=ts_list,
                                                                         sax_info=sax_info,
                                                                         spark_ctx=spark_ctx,
                                                                         trivial_radius=recognition_info.radius / 2)
    # INPUT : rdd_sequences_list = [ (key, sequence), ... ]
    # OUTPUT : rdd_sax_result is a SaxResult object containing
    #  * paa (rdd of flatMap) : rdd of large list of all the paa_values concatenated
    #  * breakpoints (list) : list of the breakpoints (len = sax_info.alphabet_size - 1)
    #  * sax_word (large str): large string of all the SAX words concatenated
    # PROCESS : Give the SAX form of the sequences
    rdd_sax_result = run_sax_on_sequences(rdd_sequences_data=rdd_sequences_list,
                                          paa=sax_info.paa,
                                          alphabet_size=sax_info.alphabet_size)

    # INPUT : rdd_sequences_list = [ (key, sequence), ... ]
    # OUTPUT : sequences_list = { key: sequence, ...} NOT AN RDD!
    # PROCESS : transform rdd_sequences_list elements into dict
    sequences_list = rdd_sequences_list.collectAsMap()

    # INPUT : rdd_normalization_coefficients = [ (same_key,(un-normalized seq_mean, un-normalized seq_sd)), ...]
    # OUTPUT : sequences_list = { key: (un-normalized seq_mean, un-normalized seq_sd), ...} NOT AN RDD!
    # PROCESS : transform rdd_normalization_coefficients elements into dict
    normalization_coefficients = rdd_normalization_coefficients.collectAsMap()

    # Keep only necessary information of each sequence
    sequences_list = sequences_info(sequences_list, normalization_coefficients)

    # *paa_sequence* is a "conversion" of *sax* from letters to numbers (matrix with same shape)
    # (usefull for past-processing the random projection algorithm).
    breakpoints = [str(i) for i in rdd_sax_result.breakpoints]

    # Build the table which give the distance between two letters (need just sax_result.breakpoints)
    mindist_lookup_table = rdd_sax_result.build_mindist_lookup_table(sax_info.alphabet_size)

    # Give the SAX result in a array (need rdd_sax_result.sax_word and sax_result.paa)
    rdd_sax, paa_result, number_of_sequences = rdd_sax_result.start_sax(sax_info.paa, spark_ctx=spark_ctx)

    LOGGER.info("- filtered number of words=%s", number_of_sequences)

    if number_of_sequences == 1:
        LOGGER.info("- sliding window find just one sequence, no collision matrix computed.")
        collision_matrix = SparseMatrix(np.array([[0]]))
    else:

        # Build the collision matrix, the number of iteration can change
        # (if the len of a sequence is too small for example nb_iteration can be < nb_iteration specified)
        collision_matrix, collision_info.nb_iterations = final_collision_matrix(
            sax=rdd_sax,
            number_of_iterations=collision_info.nb_iterations,
            index_selected=collision_info.index,
            word_len=sax_info.paa,
            spark_ctx=spark_ctx)

    # *collision_matrix* is a sparse matrix : light in memory

    # Give the result of the Equation 9
    eq9_result = equation9(number_of_sequences=number_of_sequences,
                           size_alphabet=sax_info.alphabet_size,
                           size_word=sax_info.paa,
                           errors=collision_info.errors,
                           index_selected=collision_info.index,
                           iterations=collision_info.nb_iterations)

    sax = rdd_sax.collect()
    paa_result = np.transpose(paa_result)

    distance_info = NeighborhoodSearch(size_sequence=sax_info.sequences_size,
                                       mindist_lookup_table=mindist_lookup_table,
                                       alphabet_size=sax_info.alphabet_size,
                                       sax=sax,
                                       radius=recognition_info.radius,
                                       collision_matrix=collision_matrix)

    LOGGER.info("- theoretical Eq9 limit: min collisions = %s for accepted errors=%s", eq9_result,
                collision_info.errors)

    # Check the eq9_result with min_value
    if eq9_result < recognition_info.min_value:
        LOGGER.warning("- setting Eq9 limit to min_value=%s: because Eq9 < min_value", recognition_info.min_value)
        eq9_result = recognition_info.min_value
    if eq9_result < 1:
        LOGGER.warning("- setting Eq9 limit to 1: because Eq9 < 1")
        eq9_result = 1

    # find the motif neighborhood by using the largest value cells in the collision matrix
    if recognition_info.is_algo_method_global is True:
        algo_result = distance_info.motif_neighborhood_global(eq9_result, recognition_info)
    else:
        algo_result = distance_info.motif_neighborhood_iterative(eq9_result, recognition_info)

    # Give the results with the names of sequences and not their number in the collision matrix
    algo_result = result_on_sequences_form(algo_result, sequences_list, sax, sax_info.alphabet_size, paa_result)

    algo_result = result_on_pattern_form(algo_result)

    # Give the alphabet used in the SAX algorithm
    alphabet = start_alphabet(sax_info.alphabet_size)

    result = {'patterns': algo_result,
              'break_points': breakpoints,
              'disc_break_points': alphabet}

    if spark_ctx is not None:
        ScManager.stop()
        LOGGER.info("Ended Spark session.")

    return result


def main_random_projections(ts_list,
                            sequences_size,
                            overlap=0.8,
                            paa=10,
                            alphabet_size=8,
                            normalize_mode=LOCAL_NORM,
                            normalize_method=MEAN_VARIANCE,
                            linear_filter=MEDIUM,
                            collision_iterations=1,
                            collision_nb_indexes=2,
                            neighborhood_method=GLOBAL_WITH_COLLISIONS,
                            neighborhood_iterations=10,
                            neighborhood_radius=1.5):
    """
    The random projections algorithm

    :param ts_list: list containing dict {TSUID, FUNCID} for each TS
    :type ts_list: list of dict

    :param sequences_size: The number of points in a sequence. Positive number required.
    :type sequences_size: int

    :param overlap: The overlap of two consecutive sliding windows is expressed in percentage of the window size
    :type overlap: float or int (0.8 by default)

    :param paa: The number of PAAs for each sliding window: also defines the size of words built by SAX
    :type paa: int (10 by default)

    :param alphabet_size: The number of letters used by the SAX algorithm
    :type alphabet_size: int (8 by default)

    :param normalize_mode: Choose to normalize each timeseries (GLOBAL_NORM), or/and each timeseries sequence
    :type normalize_mode: str in CONFIG_NORM_MODE (LOCAL_NORM by default)

    :param normalize_method: Normalize with or without mean, with or without variance
    :type normalize_method: str in CONFIG_NORM_METHOD (MEAN_VARIANCE by default)

    :param linear_filter: Filter linear and constant sequences: such sequences are ignored by the processing. LOW delete
                          sequences which are extremely linear or constant, HIGH delete sequences which look like linear
                          or constant. HIGH delete much more sequences than LOW.
    :type linear_filter: str in CONFIG_FILTER (MEDIUM by default)

    :param collision_iterations: Percentage of the maximum of the combinations possible that correspond to the maximum
                                 of iterations to build the collision matrix, usually greater than 50%
    :type collision_iterations: int (1 by default)

    :param collision_nb_indexes: The number of index selected to build the collision matrix. For this advanced
                                 parameter, default value is usually used: 2
    :type collision_nb_indexes: int (2 by default)

    :param neighborhood_method: Choose the method to find the motif neighborhood. GLOBAL is faster than ITERATIVE. The
                                version BRUTE_FORCE is more precise, but slower than the other version WITH_COLLISIONS
    :type neighborhood_method: str in CONFIG_NEIGHBORHOOD_METHOD (GLOBAL_WITH_COLLISIONS by default)

    :param neighborhood_iterations: Give the number of iterations to do the motif neighborhood search
    :type neighborhood_iterations: int (10 by default)

    :param neighborhood_radius: Give the radius of motif neighborhood
    :type neighborhood_radius: float or int (1.5 by default)

    :return: the group of sets of similar sequences
    :rtype: pattern_groups
    """

    try:
        if type(sequences_size) is not int or sequences_size < 1:
            msg = "Unexpected arg value : positive integer expected for sequences_size={}"
            raise IkatsException(msg.format(sequences_size))

        if type(overlap) not in [float, int] or overlap < 0 or overlap > 1:
            msg = "Unexpected arg value : float within [0,1] expected for overloap={}"
            raise IkatsException(msg.format(overlap))

        if type(paa) is not int or paa < 1:
            msg = "Unexpected arg value : positive integer expected for paa={}"
            raise IkatsException(msg.format(paa))

        if type(alphabet_size) is not int or alphabet_size not in range(2, 27):
            msg = "Unexpected arg value : positive integer within [2,26] expected for alphabet_size={}"
            raise IkatsException(msg.format(alphabet_size))

        if type(collision_iterations) not in [int, float] or collision_iterations <= 0 or collision_iterations > 1:
            msg = "Unexpected arg value : float within ]0,1] expected for collision_iterations={}"
            raise IkatsException(msg.format(collision_iterations))

        if type(collision_nb_indexes) is not int or collision_nb_indexes not in range(2, paa - 1):
            msg = "Unexpected arg value : integer greater than 1 expected for index={}"
            raise IkatsException(msg.format(collision_nb_indexes))

        if type(neighborhood_iterations) is not int or neighborhood_iterations < 1:
            msg = "Unexpected arg value : positive integer expected for iterations={}"
            raise IkatsException(msg.format(neighborhood_iterations))

        if type(neighborhood_radius) not in [int, float] or neighborhood_radius <= 0:
            msg = "Unexpected arg value : positive integer expected for radius={}"
            raise IkatsException(msg.format(neighborhood_radius))

        LOGGER.info("Running Random Projections algorithm")

        # get all the tsuid (necessary for the format of the result)
        tsuid_list = []
        for ts_ref in ts_list:
            tsuid_list.append(ts_ref['tsuid'])

        config_sax = ConfigSax(paa=paa,
                               sequences_size=sequences_size,
                               with_mean=CONFIG_NORM_METHOD[normalize_method][0],
                               with_std=CONFIG_NORM_METHOD[normalize_method][1],
                               global_norm=CONFIG_NORM_MODE[normalize_mode][1],
                               local_norm=CONFIG_NORM_MODE[normalize_mode][0],
                               linear_filter=CONFIG_FILTER[linear_filter][0],
                               recovery=overlap,
                               coefficients=[CONFIG_FILTER[linear_filter][1], CONFIG_FILTER[linear_filter][2]],
                               alphabet_size=alphabet_size)

        config_collision = ConfigCollision(iterations=collision_iterations,
                                           index=collision_nb_indexes,
                                           config_sax=config_sax)

        max_iterations = binom(paa, collision_nb_indexes)

        config_recognition = ConfigRecognition(is_stopped_by_eq9=True,
                                               is_algo_method_global=CONFIG_NEIGHBORHOOD_METHOD[neighborhood_method][0],
                                               min_value=int(0.05 * max_iterations),
                                               iterations=neighborhood_iterations,
                                               radius=neighborhood_radius,
                                               neighborhood_method=CONFIG_NEIGHBORHOOD_METHOD[neighborhood_method][1],
                                               activate_spark=None)

        result = random_projections(ts_list=tsuid_list,
                                    sax_info=config_sax,
                                    collision_info=config_collision,
                                    recognition_info=config_recognition)

        LOGGER.info("Ending Random Projections algorithm")

        return {"context": {"tsuids": tsuid_list},
                "sax": {"break_points": result['break_points'],
                        "disc_break_points": result['disc_break_points']},
                "patterns": result['patterns']}

    except Exception:
        LOGGER.error("Ending with error Random Projections algorithm")
        raise IkatsException("Failed Random Projections")
