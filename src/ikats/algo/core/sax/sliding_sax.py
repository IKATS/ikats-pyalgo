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
import pyspark.rdd
from scipy.stats import pearsonr
import numpy as np
from ikats.algo.core.paa import run_paa
from ikats.algo.core.sax import SAX
from ikats.core.library.exception import IkatsException
from ikats.core.resource.api import IkatsApi

LOGGER = logging.getLogger(__name__)


def extract_seq_from_ts(ts_br, rank, alpha, sequences_size):
    """
    Extract the rank'th sequence of ts_br

    :param ts_br: the current TS broadcasted
    :type ts_br: pyspark.broadcast.Broadcast

    :param rank: the ranks of the chosen sequence
    :type rank: int

    :param alpha: the current value of alpha (number of points between the beginning of two sequences)
    :param alpha: int

    :param sequences_size: the size of the chosen sequence (result of sax_info.sequence_size)
    :type sequences_size: int

    :return: a sequence (list) in ts_br : [[timestamp1, ts_value1] , ...]
    of size sequence_size
    :rtype: list

    extract_seq_from_ts()[:,1] to extract ts_values
    """
    return ts_br.value[(rank * alpha):(rank * alpha) + sequences_size]


class SlidingWindow(object):
    """
    Sliding SAX algorithm
    =====================

    The Sliding SAX algorithm is the SAX algorithm with a sliding window, adapted to the
    random projection algorithm first step.

    This algorithm do the following (detailed for 1 TS but valid for many TS):
      * beginning with sliding_windows() function:
         * Apply the sliding window
         * Normalize the TS (global or/and local)
         * Filter the linear sequences if demanded
         * Ignore the trivial matches
      * ending with run_sax_on_sequences()
         * Apply the SAX algorithm

    **Note: catalogue implementation is not yet provided by this module**:

      - sliding_windows() and run_sax_on_sequences() are used at higher level by implementation from pattern.random_proj

    """

    @staticmethod
    def normalize(timeseries_val, sax_info):
        """
        Normalize time_serie_val and give the mean and the variance.

        :param timeseries_val: (timeseries) values to normalize
        :type timeseries_val: RDD of list

        :param sax sax_info: information about normalization required:
            * with_mean (bool): accept or not to normalize with the mean (centering)
            * with_std (bool): accept to normalize with the variance (scaling)
        :type sax_info: ConfigSax

        :return: the timeseries normalized and the timeseries mean, and standard-deviation
        :rtype: ( RDD of numpy.array, float or int, float or int )
        """

        # Give the mean
        if sax_info.with_mean is True:
            timeseries_mean = timeseries_val.mean()
        else:
            timeseries_mean = 0

        # Give the standard-deviation
        if sax_info.with_std is True:
            timeseries_std = timeseries_val.std()
        else:
            timeseries_std = 1

        # If the standard-deviation is zero, the division is not define, this is a special case
        if timeseries_std in [0, 1]:
            timeseries_val = timeseries_val - timeseries_mean
        else:
            timeseries_val = (timeseries_val - timeseries_mean) / timeseries_std

        return timeseries_val, timeseries_mean, timeseries_std

    @staticmethod
    def prepare(ts_br,
                alpha,
                rdd_list_seq,
                sax_info,
                var_timeseries=1):
        """
        Function to create a sequence from a TS and test its linearity (if required).
        Result is an RDD containing all the (filtered) sequenceS for ONE TS.

        :param ts_br: the data set broadcasted containing ONE TS
        :type ts_br: pyspark.broadcast.Broadcast

        :param alpha: the number of points between the beginning of two sequences S(sliding_rank) and S(sliding_rank+1)
        :type alpha: int

        :param rdd_list_seq: list of "rank" (index of the sequences)
        :type rdd_list_seq: pyspark.rdd.RDD


        :returns: An RDD containing all the sequences for the chosen ts_name.
        Each sequence is a tuple composed by:
            * rank (int) : the "rank" (index) of the sequence
            * dataset (array) containing the timestamps, and timeseries values (local_norm if required)
            * local mean of the sequence (float) (if required)
            * local std of the sequence (if required)
        :type sax_info: ConfigSax

        :param var_timeseries: The global variance of the current TS
        :type var_timeseries: float

        Example of results:
        [(2, array([[ 2.        , -0.11547005],
             [ 3.        ,  0.34641016],
             [ 4.        ,  1.27017059],
             [ 5.        , -1.5011107 ]]), 3.25, 2.1650635094610968),
        (3, array([[ 3.        ,  0.52414242],
             [ 4.        ,  1.36277029],
             [ 5.        , -1.15311332],
             [ 6.        , -0.73379939]]), 2.75, 2.384848003542364)]

        """

        # 1/ Select the sliding sequences with the length required
        # ---------------------------------------------------------
        #
        # get all the sequences of the current TS
        #

        # INPUT : rdd_list_seq = [0, 1, ..., number_of_sequences]
        # OUTPUT : rdd_seq = [(0, seq_of_ts_values1), (1, seq_of_ts_values2),...] where 0,1,... are in rdd_list_seq
        # and *seq_of_ts_values* represent just one sequence of ts_values.
        # PROCESS : get all the sequences of the current TS
        rdd_seq = rdd_list_seq.map(lambda x: (x, extract_seq_from_ts(ts_br=ts_br,
                                                                     rank=x,
                                                                     alpha=alpha,
                                                                     sequences_size=sax_info.sequences_size)))

        # 2/ Local norm (if required)
        # ----------------------------
        #
        if sax_info.local_norm is True:

            def _spark_local_norm(val):
                """
                Local norm, with a management of std = 0 (if the seq values if fully == 0)
                """
                if np.std(val[1][:, 1]) == 0:
                    results = np.array(list(zip(val[1][:, 0], (val[1][:, 1] - np.mean(val[1][:, 1])))))
                else:
                    results = np.array(list(zip(val[1][:, 0],
                                                (val[1][:, 1] - np.mean(val[1][:, 1])) / np.std(val[1][:, 1]))))
                # note that the np.array(list(zip())) return the dataset [ [timestamp1, value_normalized1], [,],...]

                return val[0], results, np.mean(val[1][:, 1]), np.std(val[1][:, 1])

            # INPUT : rdd_seq = [(0, seq_of_ts_values0), (1, seq_of_ts_values1),...]
            # OUTPUT : rdd_seq = [ (rank, seq_of_ts_values, local_mean, local_std), ...]
            # PROCESS : return tuples containing (rank, values, local_mean, local_std) for each seq
            rdd_seq = rdd_seq.map(lambda x: _spark_local_norm(x))

        # Else : NO local norm (local_mean = 0, local_std = 1)
        else:
            # INPUT :rdd_seq = [(0, seq_of_ts_values0), (1, seq_of_ts_values1),...]
            # OUTPUT : rdd_seq = [ (rank, seq_of_ts_values, 0, 1), ...]
            # PROCESS : return tuples containing (rank, values, 0, 1) for each seq
            rdd_seq = rdd_seq.map(lambda x: (x[0],
                                             np.array(list(zip(x[1][:, 0], x[1][:, 1]))),
                                             0,
                                             1))

        # 3/ Filter : keep the sequence in some conditions (if required)
        # -----------------------------------------------------------------
        #
        if sax_info.linear_filter is True:

            # INPUT : rdd_seq =  [ (rank, seq_of_ts_values, local_mean, local_std), ...]
            # OUTPUT : same within the sequences with low variance (quasi-constant)
            # PROCESS : keep the sequence if is not close to a constant (variance very low)
            rdd_seq = rdd_seq.filter(lambda x: np.std(x[1][:, 1]) > sax_info.coefficients[0] * var_timeseries)

            # rdd_seq can contain [] : it means that the seq is filtered.

            def _spark_not_linear(sequence):
                """
                Function to test if a sequence is not linear (return True).
                """
                # if already filtered (rdd_seq.collect() = [])
                if len(sequence) == 0:
                    return False
                else:
                    # Test the linearity of the seq
                    return pearsonr(sequence[:, 0], sequence[:, 1])[0] ** 2 < sax_info.coefficients[1]

            # INPUT : rdd_seq =  [ (rank, seq_of_ts_values, local_mean, local_std), ...]
            # OUTPUT : same but within linear sequences
            # PROCESS : keep rdd_seq if it's not linear
            rdd_seq = rdd_seq.filter(lambda x: _spark_not_linear(x[1]))

        return rdd_seq

    @staticmethod
    def trivial_match(rdd_seq, ts_br, epsilon, alpha, sequences_size):
        """
        Search the trivial match and get the key of all the sequences of
        rdd_seq which are kept (no trivial match)

        Note that trivial matches search is computed UNTIL THE NEXT NON MATCHING SEQUENCE

        :param rdd_seq: ensemble of sequences of ONE TS (linear filter already done)
        :type rdd_seq: pyspark.rdd.PipelinedRDD

        rdd_seq : [ (key, the_sequence, local mean, local std) , ... ]
        note that these sequences have been already filtered (linear filter)

        :param ts_br: the current TS broadcasted
        :type ts_br: pyspark.broadcast.Broadcast

        :param epsilon: a value compared to the distance
        :type epsilon: float

        :return: list of all the keys of the sequences of the current TS
        which are kept after a trivial match filter
        :rtype: list
        """

        # INPUT: [ (key1, the_sequence, local mean, local std) , (key2, ...), ... ]
        # OUTPUT: [('t', key1), ('t', key2),... ]
        # PROCESS : select "ranks", and add a temporary key 't' (see below)
        rdd_seq_keys = rdd_seq.map(lambda x: ("t", x[0]))
        # example: [('t', 0), ('t', 3), ('t', 15), ('t', 21)]
        # if list of keys is [0, 3, 15, 21] (rank of the filtered sequences)

        # INPUT : [("t", key1), ("t", key2), ("t", key3), ...]
        # OUTPUT : [ ("t", (key1, key2)),('t', (key1, key3)), ('t', (key2, key3)), ...]
        # PROCESS : get all the combinations (recursive) without replacement (alternative to 'cartesian')
        # of all the keys.
        rdd_seq_keys = rdd_seq_keys.join(rdd_seq_keys).filter(lambda x: x[1][0] < x[1][1])
        # example : [('t', (0, 3)),
        #            ('t', (0, 15)),
        #            ('t', (0, 21)),
        #            ('t', (3, 15)),
        #            ('t', (3, 21)),
        #            ('t', (15, 21))]

        # INPUT : [ ("t", (key1, key2)),('t', (key1, key3)), ('t', (key2, key3)), ...]
        # OUTPUT : [ (key1, key2), (key1, key3), (key2, key3), ...] all the combinations of the seq without replacement
        # PROCESS : remove the temporal key "t"
        rdd_seq_keys = rdd_seq_keys.map(lambda x: x[1])
        # example: [(0, 3), (0, 15), (0, 21), (3, 15), (3, 21), (15, 21)]

        # INPUT : [ (key1, key2), (key1, key3), (key2, key3), ...]
        # OUTPUT : [ (key1, [key2, key3]), (key2, [key3]),...]
        # PROCESS : get lists of all the combinations by keys
        rdd_seq_keys = rdd_seq_keys.groupByKey().map(lambda x: (x[0], list(x[1])))

        # example : [(0, [3, 15, 21]), (3, [15, 21]), (15, [21])]
        # note that x[1] is initially a ResultIterable object -> changed into list for operations

        def _spark_not_trivial(rdd_content, epsilon=epsilon, ts_br=ts_br, alpha=alpha, sequences_size=sequences_size):
            """
            Test trivial match a sequence (current_rank'th), and all the following sequences,
            UNTIL THE NEXT NON MATCHING SEQUENCE.
            Note that we work on sequences already filtered (linear filter) -> list

            :param rdd_content: content of rdd_seq_keys.
            Example : rdd_content = (0, [1, 3, 4, 8, 9])
            meaning that here, the sequences 2, 5, 6, 7 have been filtered (linear filter) -> not considered

            :return: list composed by tuple :
                * rank (int): the rank of the sequence to consider
                * flag (bool) : False if the sequence is not kept (trivial match),
                True instead (only if rank=current_rank or rank=first_non_match)
            Example: [ (0, True), (1, False), (3, False), (4, True)] if the algo stop at 4 (rank0 and rank4 don't match)

            :rtype : list of tuple
            """
            # current_rank : the rank of the sequence we want to compare (0 in the example)
            # num_seq : the rank of the first sequence which match not with our current sequence
            # Example : rdd_content = (0, [1, 3, 4, 8, 9]) -> current_rank = 0
            # rdd_content[1][num_seq] = 1 if num_seq=0
            current_rank = rdd_content[0]
            num_seq = 0

            # seq of reference (the current_rank'th)
            ref = np.array(extract_seq_from_ts(ts_br=ts_br,
                                               rank=current_rank,
                                               alpha=alpha,
                                               sequences_size=sequences_size))[:, 1]

            seq_to_compare = np.array(extract_seq_from_ts(ts_br=ts_br,
                                                          rank=rdd_content[1][num_seq],
                                                          alpha=alpha,
                                                          sequences_size=sequences_size))[:, 1]

            # If trivial match found between the 2 sequences OR all the sequences have been tested
            while np.linalg.norm(ref - seq_to_compare) < epsilon and num_seq < (len(rdd_content[1]) - 1):
                # The second match with the first -> we continue
                num_seq += 1
                seq_to_compare = np.array(extract_seq_from_ts(ts_br=ts_br,
                                                              rank=rdd_content[1][num_seq],
                                                              alpha=alpha,
                                                              sequences_size=sequences_size))[:, 1]

            # num_seq is the ranks of the first sequence which don't match with our current sequence
            # (the current_rank'th)
            result = [current_rank]
            # if num_seq (remember : the first non matching rank) exist -> add it to results
            if num_seq < (len(rdd_content[1])):
                result.append(rdd_content[1][num_seq])
            else:
                # else: add "None" to results
                result.append(None)

            return result

        # INPUT : [(0, [3, 15, 21]), (3, [15, 21]), (15, [21, None])]
        # OUTPUT :  [[0, 21] -> seq0 has match with seq3 and seq 15
        #           [3, 21] -> seq3 has match with seq15
        #           [15, None]] -> seq15 has match with seq21
        # PROCESS: Test trivial match for each sequence until not matching.
        rdd_trivial_keys = rdd_seq_keys.map(lambda x: _spark_not_trivial(x))

        # Collect as a dict
        # rdd_trivial_keys = [[0,21], [3,21], [15, None]]
        # Example : trivial_key = {0:21, 3:21, 15:None }
        trivial_key = rdd_trivial_keys.collectAsMap()

        # Find the true trivial match
        i = min(trivial_key.keys())  # first sequence key: example: 0

        # List of all the keys we keep
        # Example: expected result: [0, 21]
        keys_kept = [i]

        # Stop criterion
        not_stop = True

        # For list in `trivial_key`
        while not_stop:
            # If the next non matching sequence exist: [0, 21], [3, 21]
            if trivial_key.get(i) is not None:
                i = trivial_key.get(i)  # example: 21
                keys_kept.append(i)
            else:
                not_stop = False

        return keys_kept

    @staticmethod
    def rename_seq(sequence,
                   ts_name, nb_points, alpha,
                   timeseries_mean, ts_scale_factor):
        """
        Function to rename the sequences, with an unique id ('ts_name'0*rank*).
        The "rank" (key of each seq) is replaced by this name

        :param sequence: ONE sequence
        :type sequence: tuple (key, one sequence, local_mean, local_std)

        :param ts_name: the name of the current TS
        :type ts_name: str

        :param nb_points: the number of points of the current TS (useful for encoding a new name for s)
        :type nb_points: int

        :param alpha: the current value of alpha (number of points between the beginning of two sequences)
        :param alpha: int

        :param timeseries_mean: the global mean of the current TS
        :type timeseries_mean: float

        :param ts_scale_factor: the sd of the current TS
        :type ts_scale_factor: float

        :return: a new sequence, with a new name (key), and the tuple (un-normalized seq_mean, un-normalize seq_std)
         or None if *s* is empty.
        :rtype: tuple or NONE

        Example of new keys: 'ts_name0001','ts_name0002',...,'ts_name2394'
        Example of result : (key, sequence, seq_mean, seq_std, (un-normalized seq_mean, un-normalize seq_std))
        """
        try:
            # Encoding format for the <number> part of names <ts name>_<number> of the sequences
            template_num_fmt = '{:0' + str(len(str(int(nb_points / alpha)))) + '}'

            new_name = ts_name + '_' + template_num_fmt.format(sequence[0])

            # Note that *coeff_normalization* is useless if globa_norm=False (timeseries_mean=0, and ts_scale_factor=1)
            coeff_normalization = (timeseries_mean + sequence[2] * ts_scale_factor, sequence[3] * ts_scale_factor)

            # return the same tuple but the name (s[0]) has changed, and add coeff norm
            # note that the result must be a tuple key, value
            return new_name, sequence[1:], coeff_normalization

        # if *s* is empty
        except IndexError:
            return None


def sliding_windows(ts_list, sax_info, spark_ctx, trivial_radius=0.0):
    """
    Apply the sliding window in all the timeseries with or without local normalization,
    and optional exclusion of linear sequences.

    * global normalisation of each TS (permit to compare the TS)
    * generate the sequences :
        * linear filter (don't keep the linear sequences : useless)
        * local normalisation (permit to compare sequences for trivial matching)
        * trivial matching (compare each sequence of a TS and suppress redundant sequences)

    Also, the trivial matching sequences are ignored in order to reduce the result size:
    see complete definition of **trivial match** in the paper 'Probabilistic Discovery of timeseries Motifs
    http://www.cs.ucr.edu/~eamonn/SIGKDD_Motif.pdf'.

    :param ts_list: list of TSUID
    :type ts_list: list

    :param sax_info: the information to make the sliding window and the sax_algorithm
    :type sax_info: ConfigSax object

    :param spark_ctx: spark context : result of ScManager.get()
    :type spark_ctx: pyspark.context.SparkContext

    :param trivial_radius: optional, default 0.0. Defines the radius selected by the user
       to detect trivial matches in order to reduce the number of sequences.
       This value may be replace by automatically estimated radius, if the latter is greater.

       - effective radius == maximum( trivial_radius , estimated radius )

    :type trivial_radius: float

    :return: the tuple being (result, result_normalization) where:
        * result: is an RDD composed by a list of : (seq_key, sequence_list)
        * result_normalization is an RDD composed by a list of:
          (same_seq_key,(un-normalized seq_mean, un-normalized seq_sd))
    Where sequence_list is the list [ [timestamp1, ts_value1], ...]
    :rtype: RDD, RDD

    :raise exception: IkatsException when an error occurred while processing the sliding window
    """

    if type(ts_list) is not list:
        msg = "Unexpected type : list expected for ts_list={}"
        raise IkatsException(msg.format(type(ts_list)))

    if type(sax_info.paa) is not int or sax_info.paa < 2:
        msg = 'Unexpected arg value : integer greater than 1 expected for paa={}'
        raise IkatsException(msg.format(sax_info.paa))

    if type(sax_info.sequences_size) is not int or sax_info.sequences_size < 2:
        msg = 'Unexpected arg value : integer greater than 1 expected for sequences_size={}'
        raise IkatsException(msg.format(sax_info.sequences_size))

    if type(sax_info.global_norm) is not bool:
        msg = 'Unexpected arg value : boolean expected for global_norm={}'
        raise IkatsException(msg.format(sax_info.global_norm))

    if type(sax_info.local_norm) is not bool:
        msg = 'Unexpected arg value : boolean expected for local_norm={}'
        raise IkatsException(msg.format(sax_info.local_norm))

    if type(sax_info.linear_filter) is not bool:
        msg = 'Unexpected arg value : boolean expected for linear_filter={}'
        raise IkatsException(msg.format(sax_info.linear_filter))

    if type(sax_info.recovery) not in [int, float] or sax_info.recovery < 0 or sax_info.recovery > 1:
        msg = 'Unexpected arg value : float within [0,1] expected for recovery={}'
        raise IkatsException(msg.format(sax_info.recovery))

    if type(sax_info.coefficients) is not list or len(sax_info.coefficients) is not 2:
        msg = 'Unexpected type : list of two values expected for coefficients={}'
        raise IkatsException(msg.format(sax_info.coefficients))

    if type(spark_ctx) is not pyspark.context.SparkContext:
        msg = 'Unexpected type : SparkContext expected for spark_ctx={}'
        raise IkatsException(msg.format(type(spark_ctx)))

    try:
        LOGGER.info('Starting sliding_windows ...')

        rdd_result = spark_ctx.parallelize([])
        rdd_result_normalization = spark_ctx.parallelize([])

        # 'For' loop on each TS (avoiding memory dump)
        for ts_name in ts_list:

            # extract the current TS (as numpy.array)
            LOGGER.info("Extracting TS: %s", ts_name)
            current_ts = IkatsApi.ts.read(tsuid_list=ts_name)[0]

            # number of points of the current TS
            nb_points = len(current_ts)

            # if the TS can't produce at least 1 sequence (not enough points)
            if nb_points < sax_info.sequences_size:
                LOGGER.info("Not enough points in timeseries '%s' (%s requested)", nb_points, sax_info.sequences_size)
                continue

            # 1/ global normalization of the current TS (if required):
            # --------------------------------------------------------
            #
            if sax_info.global_norm is True:
                current_ts[:, 1], timeseries_mean, ts_scale_factor = SlidingWindow.normalize(
                    timeseries_val=current_ts[:, 1],
                    sax_info=sax_info)

                # once normalized: actual standard deviation is...
                timeseries_stddev = 1 if sax_info.with_std else ts_scale_factor

            else:
                timeseries_stddev = current_ts[:, 1].std()
                timeseries_mean = 0
                # ts_scale_factor is set to 1: no rescaling needed in coeff_normalization
                ts_scale_factor = 1

            # The entire TS is broadcasted between all the cores (<< 8Gb) -> much more performances
            ts_br = spark_ctx.broadcast(current_ts)
            # Note that broadcast var are immutable -> no global norm on these objects

            # alpha is the number of points between the beginning of two sequences S(sliding_rank) and S(sliding_rank+1)
            alpha = int(round((1 - sax_info.recovery) * sax_info.sequences_size))  # modulo sequence_size

            # sax_info.recovery : percentage of the recovery of the previous sequence (in [0, 1]).
            # Example: recovery == 0.75 means that the overlap between successive sliding
            #   windows is 75percent <=> the translation is 25percent of the window size.
            #
            # Specific values:
            #  - 0: if there is no overlap between sequences (no recovery -> the translation is 100% of the window size)
            #  - 1: if the next sequence begin one point to the right (max recovery).

            # Number of available sequences:
            if alpha == 0:
                # recovery = 1 => no recovery
                alpha = 1

            nb_seq = 1 + int((nb_points - sax_info.sequences_size) / alpha)

            # list of "rank" (index of the sequences)
            rdd_list_seq = spark_ctx.parallelize(range(nb_seq))

            # 2/ Local linear filters + normalisations (if required)
            # -------------------------------------------------------
            # * linear filter (the linear seq are useless)
            # * local normalization of the non linear seq
            #

            # Generate the sequences and apply linear filter (if required)
            rdd_seq = SlidingWindow.prepare(ts_br=ts_br,
                                            alpha=alpha,
                                            rdd_list_seq=rdd_list_seq,
                                            var_timeseries=1,
                                            sax_info=sax_info)

            # 3 / Evaluate next sequences, ignoring trivial matches
            # ---------------------------------------------------------
            # - epsilon used by is_trivial_match(): radius defining similarity between Si and Sj if D(Si, Sj) <= epsilon
            #
            estimated_radius = (0.1 * timeseries_stddev) * np.sqrt(sax_info.sequences_size)
            # choose the most restrictive radius: the greater one ignores more sequences
            epsilon = max(trivial_radius, estimated_radius)

            # Trivial matching filter (don't work of nb_seq = 1...)
            if rdd_seq.count() <= 1:
                LOGGER.info("Not enough sequences to test trivial match in this TS: continue...")
                # But the sequence'll be used below
            else:
                # get all the keys of the sequence kept (no trivial match)
                ranks = SlidingWindow.trivial_match(rdd_seq=rdd_seq,
                                                    ts_br=ts_br,
                                                    epsilon=epsilon,
                                                    alpha=alpha,
                                                    sequences_size=sax_info.sequences_size)

                # INPUT : rdd_seq = [(key, sequence, local_mean, local_std),...]
                # OUTPUT : same but with less elements
                # PROCESS : filter the data: get all the sequences which not match trivially with an other seq
                rdd_seq = rdd_seq.filter(lambda x: x[0] in ranks)

            # 4/ Encoding (unique) seq names
            # -------------------------------
            #

            # INPUT : rdd_seq with keys = [1,2,3,...]
            # OUTPUT : [( key, (sequence, seq_mean, seq_sd), (coeff_normalization) ), ...]
            # where coeff_normalization is : un-normalized seq_mean, un-normalized seq_sd
            # keys are now :['ts_name0001','ts_name0002',...,'ts_name2394']
            # PROCESS : Rename each key : Example: 'ts_name0001','ts_name0002',...,'ts_name2394'
            rdd_seq = rdd_seq.map(lambda x: SlidingWindow.rename_seq(sequence=x,
                                                                     ts_name=ts_name,
                                                                     nb_points=nb_points,
                                                                     alpha=alpha,
                                                                     timeseries_mean=timeseries_mean,
                                                                     ts_scale_factor=ts_scale_factor))

            # 6/ Build the results
            # ---------------------
            #  * rdd_result_normalization : (key,(un-normalized seq_mean, un-normalized seq_sd))
            #  * rdd_result : (key, sequence_list, seq_mean, seq_sd)

            # INPUT : rdd_seq = [( key, (sequence, seq_mean, seq_sd), (coeff_normalization) ), ...]
            # OUTPUT : [ (key,(un-normalized seq_mean, un-normalized seq_sd)), ...]
            # PROCESS : get the local normalization coeff
            rdd_coeff_normalization = rdd_seq.map(lambda x: (x[0], x[2]))

            # INPUT : the rdd_result normalization of the CURRENT TS
            # [ (key,(un-normalized seq_mean, un-normalized seq_sd)), ...]
            # OUTPUT : a concatenation of these result with results calculated before
            # PROCESS : add the current TS result_normalization to the huge RDD *rdd_result_normalization*
            rdd_result_normalization = rdd_result_normalization \
                .union(rdd_coeff_normalization) \
                .reduceByKey(lambda a, b: a + b)

            rdd_result_normalization.persist()

            # INPUT : [ (key, (sequence, seq_mean, seq_sd), (coeff_normalization) ), ...]
            # OUTPUT : [ (key, sequence), ... ]
            # PROCESS : remove coeff_normalization from rdd_seq
            rdd_seq = rdd_seq.map(lambda x: (x[0], x[1][0]))

            # add the current TS results to the huge RDD *result*
            rdd_result = rdd_result.union(rdd_seq).reduceByKey(lambda a, b: a + b)

            # the rdd *result* will be in cache after the first action
            rdd_result.persist()

        LOGGER.info('... ended sliding_windows')

        return rdd_result, rdd_result_normalization

    except Exception:
        LOGGER.error("... ended sliding window with error.")
        raise IkatsException("Failed execution: sliding_window()")


class SaxResult(object):
    """
    SaxResult provides the data support (attributes) and the computing services (methods) of sliding SAX processing.
    """

    def __init__(self, paa, breakpoints, sax_word):
        """
        Constructor of SaxResult
        :param paa: the values of the paa (paa is an RDD if spark is used)
        :type paa: list or pyspark.rdd.pipelineRDD

        :param breakpoints: the list of the breakpoints
        :type breakpoints: list of float

        :param sax_word: the sequences in string form
        :type sax_word: str
        """

        self.paa = paa
        self.breakpoints = breakpoints
        self.sax_word = sax_word

    def add_paa(self, i):
        """
        Append this to PAA results
        :param i:
        :return:
        """
        self.paa += i

    def build_mindist_lookup_table(self, alphabet_size):
        """
        Build the MINDIST look up table used by the MINDIST distance between 2 SAX words.
        :param alphabet_size: the size of the alphabet
        :type alphabet_size: int

        :return: table which give the distance between two symbols
        :rtype: numpy.ndarray
        """

        # See expression (6) given by Bill Chiu, Eamonn Keogh, and Stefano Lonardi in Probabilistic Discovery of Time
        # Series Motifs
        table = np.zeros((alphabet_size, alphabet_size))
        for i in range(0, alphabet_size):
            for j in range(0, alphabet_size):
                if abs(i - j) > 1:
                    table[i, j] = self.breakpoints[max(i, j) - 1] - self.breakpoints[min(i, j)]
        return table

    def start_sax(self, nb_paa, spark_ctx):
        """
        Give the SAX result with the columns that correspond to a word.
        Dispose the sax word in a matrix by columns: each column represents a sub-word
        (of len nb_paa).

        :param nb_paa: number of letters in a word
        :type nb_paa: int

        :param spark_ctx: spark context (result of ScManager.get) or None if spark is not used
        :param : pyspark.context.SparkContext

        :return rdd_sax_result: rdd containing a matrix of letters. Each column represents a word. The total size
                of the matrix is the len of the SAX word input.
        :rtype rdd_sax_result: rdd

        Example of sax_result: [['b', 'd', ...], ['d', 'd', ...],['b', 'a',...], ['c', 'b',...],...]
        for the words 'bdbc', 'cdab', ...

        :return paa_result: list of all the paa_values
        :rtype paa_result : list

        :return number_of_sequences: number of row of the output matrix
        :rtype number_of_sequences: int
        """

        # Broadcast sax_word in all the nodes
        data_sax = self.sax_word
        data_paa = self.paa.collect()

        number_of_sequences = int(len(self.sax_word) / nb_paa)

        # the number of seq is huge
        n_seq = spark_ctx.parallelize(range(number_of_sequences))

        # INPUT : n_seq = [0,1,...,number_of_sequences]
        # OUTPUT : rdd_sax_result = [['b', 'd', 'b', 'c'], ['c', 'd', 'a', 'b'],...]
        # PROCESS : Calculate sax words in a matrix of letters
        rdd_sax_result = n_seq.map(lambda x: list(data_sax[(x * nb_paa):((x + 1) * nb_paa)]))

        # INPUT : rdd_nb_paa = [0,1,...,word_length]
        # OUTPUT : rdd_sax_result = [['b', 'd', ...], ['d', 'd', ...],['b', 'a',...], ['c', 'b',...],...]
        # for the words 'bdbc', 'cdab', ...
        # PROCESS : Calculate sax words in a matrix of letters (words by columns)
        # rdd_sax_result = rdd_nb_paa.map(lambda i: [data_sax[(nb_paa * k) + i] for k in range(number_of_sequences)])

        # INPUT : n_seq = [0,1,...,number_of_sequences]
        # OUTPUT : rdd_sax_result = [['1.45', '2.54', '3.76', '5.67'], ...]
        # PROCESS : Calculate paa_values in a matrix of float
        paa_result = n_seq.map(lambda x: list(data_paa[(x * nb_paa):((x + 1) * nb_paa)]))

        paa_result = np.transpose(paa_result.collect())
        n_seq.unpersist()

        return rdd_sax_result, paa_result, number_of_sequences

    def build_breakpoints(self, alphabet_size):
        """
        Build the breakpoints
        :param alphabet_size: number of letters

        Note that this function IS NOT sparkified (need to collect the PAA rdd).
        """
        LOGGER.info("Building breakpoints...")
        # self.paa is an RDD
        paa = self.paa.collect()
        self.breakpoints = SAX.build_breakpoints(paa, alphabet_size)

        LOGGER.info("... breakpoints built")

    def build_sax_word(self):
        """
        Build word based on the points list and the breakpoints.
        The function is sparkified.
        """
        points_list = self.paa
        breakpoints = self.breakpoints
        LOGGER.info("Begin build SAX words with Spark")

        def _spark_word(paa, breaks=breakpoints):
            """
            Internal function that calculate ONE letter corresponding to a paa
            """
            letter = ord('a')
            # use a loop because breakpoints has a low len (sax_info.alphabet_size - 1)
            for breakpoint in breaks:
                if paa >= breakpoint:
                    letter += 1
                else:
                    break
            return chr(letter)

        # Note that here, a collect is used -> low
        word_list = points_list.map(lambda x: _spark_word(x)).collect()

        # Build the word string from the list of letters
        # sax_word is an RDD
        self.sax_word = ''.join(word_list)


def run_sax_on_sequences(rdd_sequences_data, paa, alphabet_size):
    """
    Perform the Symbolic Aggregate Approximation (SAX) on the data provided in **ts_data**

    :param rdd_sequences_data: rdd containing all sequences: returned by function *sliding_windows()*:
    *sequences_data* contain a list of all seq : tuple composed by: (key, sequence_list, seq_mean, seq_sd)
        - keys: an unique key for each seq
        - sequence_list: the normalized sequence as numpy array giving TS points:  [ [t1, v1 ], ... , [tN, vN] ]

    :type rdd_sequences_data: RDD of list

    :param paa: number of letters in output word
    :type paa: int

    :param alphabet_size: number of characters in result word
    :type alphabet_size: int

    :return: the PAA result, the SAX breakpoints and the SAX string
    :rtype: SaxResult object

    Note that each letter have the same signification (same breakpoints between all the seq).

    :raise exception: IkatsException when an error occurred while processing the sax algorithm
    """

    if type(rdd_sequences_data) is not pyspark.rdd.PipelinedRDD:
        msg = "Unexpected type : PipelinedRDD expected for rdd_sequences_data={}"
        raise IkatsException(msg.format(rdd_sequences_data))

    if type(alphabet_size) is not int or alphabet_size not in range(2, 27):
        msg = "Unexpected arg value : integer within [2,26] expected for alphabet_size={}"
        raise IkatsException(msg.format(alphabet_size))

    try:
        LOGGER.info('Starting run_sax_on_sequences ...')

        # Calculate the PAAs on all the sequences

        def _spark_internal(sequence, local_paa=paa):
            """
            Compute the PAA of each sequence *sequence*.
            """
            local_paa_seq = run_paa(ts_data=np.array(sequence), paa_size=local_paa).means
            if len(local_paa_seq) != local_paa:
                local_paa_seq = local_paa_seq[: len(local_paa_seq) - 1]
            return local_paa_seq

        # INPUT : rdd_sequences_data = [(key, sequence_list, seq_mean, seq_sd),...]
        # OUTPUT : paa_seq = one sequence of all the paa concatenated (flatMap)
        # PROCESS : Run PAA on the TS data sequences
        paa_seq = rdd_sequences_data.sortByKey().flatMap(lambda x: _spark_internal(x[1]))
        # Note that *sortByKey()* is necessary for reproducible results

        # Once PAA calculated, then, find breakpoints and SAX words
        sax_result = SaxResult(paa=paa_seq, breakpoints=[], sax_word='')

        # Build the distribution breakpoints: need a flat list of paa
        # Note that this method is not sparkified => need to collect the paa data
        sax_result.build_breakpoints(alphabet_size)

        # Give the SAX result for all sequences (all timeseries)
        # Note that the concatenated entire sax word is collected.
        sax_result.build_sax_word()

        LOGGER.info("... ended run_sax_on_sequences.")
        return sax_result

    except Exception:
        LOGGER.error("... ended run_sax_on_sequences with error.")
        raise IkatsException("Failed execution: run_sax_on_sequences()")
