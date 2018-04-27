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
import numpy as np

from ikats.core.data.ts import TimestampedMonoVal
from ikats.core.library.spark import ScManager, ListAccumulatorParam
from ikats.algo.core.paa import run_paa

from ikats.core.resource.client import TemporalDataMgr

LOGGER = logging.getLogger(__name__)


class SAX(object):
    """
    SAX Algorithm

    The SAX algorithm allows to create a 'word' representing a pattern in a TS

    This algorithm do the following (detailed for 1 TS but valid for many TS):
        * apply PAA on the TS
        * (optional) Normalize the PAA result
        * Divide in N areas with same probability (where N is defined alphabet_size)
        * Assign a letter to each area
        * Locate the letter of each PAA result and build the word

    :example:

        .. code-block:: python

            # Compute SAX on a dataset
            r = run_sax_from_ds(tdm, ds_name='Portfolio', alphabet_size=5, word_size=3, normalize=True)

            # Compute SAX on a list of TSUID
            r = run_sax_from_ts_list(tdm=tdm, ts_list=tsuid_list, word_size=3, alphabet_size=5, normalize=False)

            # Compute SAX on a single TSUID
            r = run_sax_from_tsuid(tdm, '0123456789', word_size=3, alphabet_size=5, normalize=True)
    """

    @classmethod
    def norm(cls, ts_data):
        """
        Normalize the ts_data to have :
            * mean = 0
            * standard deviation = 1

        :param ts_data: TS points list to compute the norm on
        :type ts_data: np.ndarray or TimestampedMonoVal

        :return: the normalized TS
        """

        if type(ts_data) not in [np.ndarray, TimestampedMonoVal]:
            LOGGER.error("ts_data must be a np.ndarray or TimestampedMonoVal (got %s)", type(ts_data))
            raise TypeError("ts_data must be a np.ndarray or TimestampedMonoVal (got %s)" % type(ts_data))
        if type(ts_data) == TimestampedMonoVal:
            # Get the internal data if ts_data is TimestampedMonoVal
            ts_data = ts_data.data

        # Avoid null values on mean or std calculation
        ts_data[:, 1] = np.nan_to_num(ts_data[:, 1])
        values = ts_data[:, 1]
        ts_data[:, 1] = (values - np.mean(values)) / max(1/len(ts_data),np.std(values))
        return ts_data

    @classmethod
    def build_breakpoints(cls, ts_points_value, alphabet_size):
        """
        Build the breakpoints of the ts_points_value based on the alphabet_size
        and the distribution of the ts_points_value

        :param ts_points_value: points to compute the breakpoints on
        :type ts_points_value: list

        :param alphabet_size: number of parts to break into
        :type alphabet_size: int

        :return: the list of breakpoints
        :rtype: list
        """

        if type(ts_points_value) != list:
            LOGGER.error("ts_points_value must be a list (got %s)", type(ts_points_value))
            raise TypeError("ts_points_value must be a list (got %s)" % type(ts_points_value))

        first = 100. / alphabet_size
        percentile = [first * i for i in range(1, alphabet_size)]

        # keepdims =T : output and input have the same dim
        # axis = 0: compute the percentile between each k element of a
        return np.percentile(a=ts_points_value, q=percentile, axis=0, keepdims=True).T[0]

    @classmethod
    def get_word(cls, points_list, breakpoints):
        """
        Build word based on the points list and the breakpoints

        :param points_list: Result of the PAA
        :type points_list: list

        :param breakpoints: List of areas having same probability
        :type breakpoints: list

        :return: the word
        :rtype: str
        """
        word_list = []
        for point in points_list:
            letter = ord('a')
            for breakpoint in breakpoints:
                if point > breakpoint:
                    letter += 1
                else:
                    break
            word_list.append(letter)

        # Build the word string from the list of letters
        return ''.join([chr(x) for x in word_list])


def run_sax_from_ds(tdm, ds_name, alphabet_size, word_size, normalize=False, activate_spark=None):
    """
    Perform the Symbolic Aggregate Approximation (SAX) on the dataset provided in **ds_name**

    :param tdm: temporal data manager object
    :type tdm: TemporalDataMgr

    :param ds_name: dataset name
    :type ds_name: str

    :param alphabet_size: number of characters in result word
    :type alphabet_size: int

    :param word_size: number of segments
    :type word_size: int

    :param activate_spark: True to force spark, False to force local, None to let the algorithm decide
    :type activate_spark: bool or none

    :param normalize: Apply a normalization on the data
    :type normalize: False (default) skips the the normalization, True to apply it

    :return: A list of dict composed of the PAA result, the SAX breakpoints, the SAX string and the points for all TSUID
    :rtype: list
    """
    # Get the tsuid list from the dataset
    # The check of the ds_name type is performed inside tdm
    tsuid_list = tdm.get_data_set(ds_name)['ts_list']

    # Call the calculation of the paa on the tsuid_list gathered
    # The check on the ts_list is performed in run_paa_from_ts_list
    result = run_sax_from_ts_list(tdm=tdm,
                                  ts_list=tsuid_list,
                                  word_size=word_size,
                                  alphabet_size=alphabet_size,
                                  activate_spark=activate_spark,
                                  normalize=normalize)
    return result


def run_sax_from_ts_list(tdm, ts_list, alphabet_size, word_size, normalize=False, activate_spark=None):
    """
    Perform the Symbolic Aggregate Approximation (SAX) on the TSUID list provided in **ts_list**

    Use spark if necessary

    .. note::
        If spark fails. The local computation will be performed

    :param tdm: temporal data manager object
    :type tdm: TemporalDataMgr

    :param ts_list: tsuid list of the TS to calculate the PAA timeseries
    :type ts_list: list

    :param alphabet_size: number of characters in result word
    :type alphabet_size: int

    :param word_size: number of segments
    :type word_size: int

    :param activate_spark: True to force spark, False to force local, None to let the algorithm decide
    :type activate_spark: bool or none

    :param normalize: Apply the normalization of the TS if True (False:default)
    :type normalize: bool

    :return: A list of dict composed of the PAA result, the SAX breakpoints, the SAX string and the points for all TSUID
    :rtype: list
    """

    results = {}

    # Define if spark is necessary
    if activate_spark is None:

        md = tdm.get_meta_data(ts_list)
        sum_points = 0
        for tsuid in md:
            if 'qual_nb_points' in md[tsuid]:
                sum_points += float(md[tsuid]['qual_nb_points'])
            else:
                # No information on number of points, consider using spark
                sum_points = 0
                break
        spark_nb_points_trigger = 1E5
        if sum_points == 0 or sum_points / len(ts_list) > spark_nb_points_trigger:
            # Spark is active if the average number of points per TS is greater than spark_nb_points_trigger points
            activate_spark = True

    if activate_spark:
        LOGGER.info("Running SAX using Spark")

        # Create or get a spark Context
        spark_context = ScManager.get()

        # Build the RDD with TSUIDS
        rdd = spark_context.parallelize(ts_list)

        # Create a broadcast for spark jobs
        broadcast = spark_context.broadcast({
            "host": tdm.host,
            "port": tdm.port,
            "alphabet_size": alphabet_size,
            "word_size": word_size,
            "normalize": normalize,
        })

        # Create an accumulator to store the results of the spark workers
        accumulator = spark_context.accumulator(dict(), ListAccumulatorParam())

        def run_sax_spark(working_tsuid):
            """
            Method called by spark job

            :param working_tsuid: rdd item
            """

            spark_tdm = TemporalDataMgr(host=broadcast.value['host'], port=broadcast.value['port'])

            results = run_sax_from_tsuid(tdm=spark_tdm,
                                         tsuid=working_tsuid,
                                         alphabet_size=broadcast.value['alphabet_size'],
                                         word_size=broadcast.value['word_size'],
                                         normalize=broadcast.value['normalize'])

            accumulator.add({
                working_tsuid: results
            })

        # Get TS content using spark distribution to increase performance
        # noinspection PyBroadException
        try:
            rdd.foreach(run_sax_spark)
        except Exception:
            LOGGER.warning('Something wrong with spark, Using Local Computation')
            activate_spark = False

        for ts in ts_list:
            if ts in accumulator.value:
                results[ts] = accumulator.value[ts]
            else:
                LOGGER.warning("TS %s has encountered an issue during the spark distribution", ts)

        ScManager.stop()

    if not activate_spark:
        LOGGER.info("Running SAX on single instance")

        for ts in ts_list:
            results[ts] = run_sax_from_tsuid(tdm=tdm,
                                             tsuid=ts,
                                             alphabet_size=alphabet_size,
                                             word_size=word_size,
                                             normalize=normalize)

            # print("TS=%s\nnorm=%s\nr=%s\n\n"%(ts,normalize,results[ts]['sax_breakpoints'][0]))

    return results


def run_sax_from_tsuid(tdm, tsuid, word_size, alphabet_size=None, normalize=False):
    """
    Perform the Symbolic Aggregate Approximation (SAX) on the TSUID provided in **tsuid**

    :param tdm: temporal data manager object
    :type tdm: TemporalDataMgr

    :param tsuid: TSUID of the TS to calculate the SAX
    :type tsuid: str

    :param alphabet_size: number of characters in result word
    :type alphabet_size: int

    :param word_size: number of segments
    :type word_size: int

    :param normalize: Apply the normalization on the TS if set to True (False:default)
    :type normalize: bool

    :return: A dict composed of the PAA result, the SAX breakpoints, the SAX string and the TS points
    :rtype: dict

    :raise TypeError: if TSUID is not a string
    """

    if type(tsuid) is not str:
        LOGGER.error("TSUID must be a string (got %s)", type(tsuid))
        raise TypeError("TSUID must be a string (got %s)" % type(tsuid))

    # Get the TS content
    ts_dps = tdm.get_ts(tsuid_list=[tsuid])[0]
    ts_values = TimestampedMonoVal(ts_dps)

    # Call the calculation of the SAX on the content
    result = run_sax(ts_data=ts_values,
                     alphabet_size=alphabet_size,
                     word_size=word_size,
                     normalize=normalize)

    return result


def run_sax(ts_data, word_size, alphabet_size, normalize=True):
    """
    Perform the Symbolic Aggregate Approximation (SAX) on the data provided in **ts_data**

    :param ts_data: TS points list
    :type ts_data: np.ndarray or TimestampedMonoVal

    :param alphabet_size: number of characters in result word
    :type alphabet_size: int

    :param word_size: number of letters in output word
    :type word_size: int

    :param normalize: Apply the normalization on the TS if set to True(default)
    :type normalize: bool

    :return: A dict composed of the PAA result, the SAX breakpoints and the SAX string
    :rtype: dict

    :raise ValueError: if word_size is not a positive integer
    :raise ValueError: alphabet_size is not a positive integer within [1,26]
    :raise TypeError: TSUID is not a string
    """

    # Check inputs

    if type(alphabet_size) is not int or alphabet_size not in range(1, 27):
        raise ValueError('alphabet_size must be a positive integer within [1,26]')
    if type(word_size) is not int or word_size <= 0:
        raise ValueError('word_size must be a positive integer')

    if type(ts_data) not in [np.ndarray, TimestampedMonoVal]:
        LOGGER.error("ts_data must be a np.ndarray or TimestampedMonoVal (got %s)", type(ts_data))
        raise TypeError("ts_data must be a np.ndarray or TimestampedMonoVal (got %s)" % type(ts_data))
    if type(ts_data) == TimestampedMonoVal:
        # Get the internal data if ts_data is TimestampedMonoVal
        ts_data = ts_data.data

    # Normalize the data
    if normalize:
        ts_data = SAX.norm(ts_data)

    # Apply the SAX on the normalized TS
    paa_result = run_paa(ts_data=ts_data, paa_size=word_size).means

    # Build the distribution breakpoints
    breakpoints = SAX.build_breakpoints(paa_result, alphabet_size)

    sax_word = SAX.get_word(points_list=paa_result, breakpoints=breakpoints)

    result = {
        "paa": list(paa_result),
        "sax_breakpoints": list(breakpoints),
        "sax_string": sax_word
    }

    return result
