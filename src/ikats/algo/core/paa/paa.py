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

import time
import logging
import numpy as np

from ikats.core.data.ts import TimestampedMonoVal
from ikats.core.library.spark import ScManager, ListAccumulatorParam
from ikats.core.resource.client import TemporalDataMgr

LOGGER = logging.getLogger(__name__)


class PAA(object):
    """
    PAA Algorithm
    *************

    This algorithm do the following (detailed for 1 TS but valid for many TS):
        * Divide a TS in **word_size** segments having the same period
        * Calculate the number of points contained in each segment (in case the TS is aperiodic or contains holes)
        * Calculate the mean of each segment (sum the value of points inside a segment and divide by the number
          of points of this segment)

    .. note::
        It is commonly used with SAX

    :example:


        .. code-block:: python

            # Compute PAA on a dataset dividing each dataset in 5 parts
            r = run_paa_from_ds(tdm=tdm, ds_name='Portfolio', paa_size=5, out_ts=False)

            # Compute PAA on a list of TSUID
            r = run_paa_from_ts_list(tdm=tdm, ts_list=['TSUID1', 'TSUID2', 'TSUID3'], paa_size=paa_size, out_ts=True)

            # Compute PAA on a single TSUID
            r = run_paa_from_tsuid(tdm=tdm, tsuid="my_tsuid", paa_size=5)

            # Compute PAA on local TS data
            r = run_paa(tsuid=tsuid, ts_data=ts_data, paa_size=2)

            # Compute PAA on local TS data (detailed version)
            segment_period = PAA.segment_period(ts_data=ts_data, paa_size=paa_size)
            segments_index = PAA.segments_index(ts_data=ts_data, segment_period=segment_period)
            result_ts, result_coeff = PAA.compute(ts_data=ts_data, seg_list=segments_index)
"""

    @classmethod
    def segment_period(cls, ts_data, paa_size):
        """
        Calculate the segments period

        :param ts_data: the TS on which calculate the segments
        :type ts_data: np.ndarray or TimestampedMonoVal

        :param paa_size: number of segments to create
        :type paa_size: int

        :return: the period of every segment
        """

        # Check inputs
        if type(ts_data) not in [np.ndarray, TimestampedMonoVal]:
            LOGGER.error("ts_data must be a np.ndarray or TimestampedMonoVal (got %s)", type(ts_data))
            raise TypeError("ts_data must be a np.ndarray or TimestampedMonoVal (got %s)" % type(ts_data))
        if type(ts_data) == TimestampedMonoVal:
            # Get the internal data if ts_data is TimestampedMonoVal
            ts_data = ts_data.data
        if type(paa_size) is not int:
            LOGGER.error("paa_size must be integer (got %s)", type(paa_size))
            raise TypeError("paa_size must be an integer (got %s)" % type(paa_size))
        if paa_size <= 0:
            LOGGER.error("paa_size must be positive (got %s)", paa_size)
            raise ValueError("paa_size must be positive (got %s)" % paa_size)
        if len(ts_data) < paa_size:
            LOGGER.error("PAA can't be calculated. TSUID too short (%s points) compared to paa_size (%s)",
                         len(ts_data), paa_size)
            raise ValueError(
                "PAA can't be calculated. TSUID too short (%s points) compared to paa_size (%s)" % (
                    len(ts_data), paa_size))

        start_date = ts_data[0][0]
        end_date = ts_data[-1][0]
        return (end_date - start_date) / paa_size

    @classmethod
    def segments_index(cls, ts_data, segment_period):
        """
        Calculate the indexes of the TS which correspond to a distributed period among TS length

        :example:

            consider paa_size is 4
                * TS timestamps are ``[ 0, 1, 2, 3, 4, 5, 10, 11, 20, 29, 30]``
                * segment_period will be in this case : 7.5 to have 4 evenly distributed periods:
                    - from  0,  to  7.5
                    - from  7.5 to 15
                    - from 15   to 22.5
                    - from 22.5 to 30
                * Applying this to the ts_data we have the 4 splits:
                    - ``[0,1,2,3,4,5]``
                    - ``[10,11]``
                    - ``[20]``
                    - ``[29, 30]``
                * Finally, we returns the indexes of the beginning of each part + the last to surround all the ts:
                    - ``[0, 6, 8, 9, 11]``

        :param ts_data: TS to compute the PAA onto
        :type ts_data: np.ndarray or TimestampedMonoVal

        :param segment_period: value of the period of each segment
        :type segment_period: float

        :return:
        """


        # Check inputs
        if type(ts_data) not in [np.ndarray, TimestampedMonoVal]:
            LOGGER.error("ts_data must be a np.ndarray or TimestampedMonoVal (got %s)", type(ts_data))
            raise TypeError("ts_data must be a np.ndarray or TimestampedMonoVal (got %s)" % type(ts_data))

        if type(ts_data) == TimestampedMonoVal:
            # Get the internal data if ts_data is TimestampedMonoVal
            ts_data = ts_data.data

        start_date = ts_data[0][0]
        end_date = ts_data[-1][0]

        # Store the ts point index corresponding to the beginning of the segment
        segments_index = [0]

        # Prepare the limit of the segment
        segment_period_next = segment_period + start_date
        for i, _ in enumerate(ts_data):

            # If we overshoot the end of the segment
            if ts_data[i][0] > round(segment_period_next):
                # Store the index as next segment beginning
                segments_index.append(i)
                # Update the end of the next segment
                segment_period_next += segment_period

        # For the last element, provide an end index or replace existing one
        # (in case of threshold management of floats)
        if round(segment_period_next) >= end_date:
            segments_index.append(len(ts_data))
        return segments_index

    @classmethod
    def compute(cls, ts_data, seg_list):
        """
        Compute the PAA based on the ts and the list of segments

        :param ts_data: TS to compute the PAA onto
        :type ts_data: np.ndarray or TimestampedMonoVal

        :param seg_list: list of the indexes corresponding to every period of the TS
        :type seg_list: list

        :return: the new TS (with values replaced by mean value) and the means list
        :rtype: tuple (TimestampedMonoVal, list)
        """

        # Check inputs
        if type(ts_data) not in [np.ndarray, TimestampedMonoVal]:
            LOGGER.error("ts_data must be a np.ndarray or TimestampedMonoVal (got %s)", type(ts_data))
            raise TypeError("ts_data must be a np.ndarray or TimestampedMonoVal (got %s)" % type(ts_data))

        if type(ts_data) == TimestampedMonoVal:
            # Get the internal data if ts_data is TimestampedMonoVal
            ts_data = ts_data.data

        # Results initialization
        result_ts = ts_data[:]
        result_coeff = []

        # For every segment
        for i in range(1, len(seg_list)):
            # Compute the mean using numpy
            mean = np.mean(ts_data[seg_list[i - 1]:seg_list[i], 1])

            # Store it to one of the results
            result_coeff.append(mean)

            # Overwrite the mean value for every point in the segment
            result_ts[seg_list[i - 1]:seg_list[i], 1] = ts_data[seg_list[i - 1]:seg_list[i], 1] * 0 + [mean]
        return TimestampedMonoVal(result_ts), result_coeff


def run_paa_from_ds(tdm, ds_name, paa_size, out_ts=True, save=False, activate_spark=False):
    """
    Compute the Piecewise Aggregation Approximation (PAA) on the dataset provided in **ds_name**

    :param tdm: temporal data manager object
    :type tdm: TemporalDataMgr

    :param ds_name: dataset name
    :type ds_name: str

    :param paa_size: number of segments
    :type paa_size: int

    :param out_ts: True means the result will be a TS, False will return only the means
    :type out_ts: bool

    :param save: True means the new TS will be saved in addition of the return
    :type save: bool

    :param activate_spark: True if spark must be forced
    :type activate_spark: bool

    :return: the array of the new TS resulting of the PAA approximation or the list of values (with len = paa_size)
    :rtype: list

    """

    # Get the tsuid list from the dataset
    # The check of the ds_name type is performed inside tdm
    tsuid_list = tdm.get_data_set(ds_name)['ts_list']

    # Call the calculation of the paa on the tsuid_list gathered
    # The check on the ts_list is performed in run_paa_from_ts_list
    return run_paa_from_ts_list(tdm=tdm,
                                ts_list=tsuid_list,
                                paa_size=paa_size,
                                out_ts=out_ts,
                                save=save,
                                activate_spark=activate_spark)


def run_paa_from_ts_list(tdm, ts_list, paa_size, out_ts=True, save=False, activate_spark=None):
    """
    Compute the Piecewise Aggregation Approximation (PAA) on the **ts_list** provided
    Use spark if necessary

    :param tdm: temporal data manager object
    :type tdm: TemporalDataMgr

    :param ts_list: tsuid list of the TS to calculate the PAA timeseries
    :type ts_list: list

    :param paa_size: number of segments
    :type paa_size: int

    :param out_ts: True means the result will be a TS, False will return only the means
    :type out_ts: bool

    :param save: True means the new TS will be saved in addition of the return
    :type save: bool

    :param activate_spark: True to force spark, False to force local, None to let the algorithm decide
    :type activate_spark: bool or None

    :return: the array of the new TS resulting of the PAA approximation or the list of values (with len = paa_size)
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
        LOGGER.info("Running PAA using Spark")

        # Create or get a spark Context
        spark_context = ScManager.get()

        # Build the RDD with TSUIDS
        rdd = spark_context.parallelize(ts_list)

        # Create a broadcast for spark jobs
        broadcast = spark_context.broadcast({
            "host": tdm.host,
            "port": tdm.port,
            "paa_size": paa_size,
            "out_ts": out_ts,
            "save": save,
        })

        # Create an accumulator to store the results of the spark workers
        accumulator = spark_context.accumulator(dict(), ListAccumulatorParam())

        def run_paa_spark(working_tsuid):
            """
            Method called by spark job

            :param working_tsuid: rdd item
            """

            spark_tdm = TemporalDataMgr(host=broadcast.value['host'], port=broadcast.value['port'])

            # noinspection PyBroadException
            try:
                results = run_paa_from_tsuid(tdm=spark_tdm,
                                             tsuid=working_tsuid,
                                             paa_size=broadcast.value['paa_size'],
                                             out_ts=broadcast.value['out_ts'],
                                             save=broadcast.value['save'])[:]
            except Exception:
                results = []

            accumulator.add({
                working_tsuid: results
            })

        # Get TS content using spark distribution to increase performance
        # noinspection PyBroadException
        try:
            rdd.foreach(run_paa_spark)
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
        LOGGER.info("Running PAA on single instance")
        for ts in ts_list:
            results[ts] = run_paa_from_tsuid(tdm=tdm,
                                             tsuid=ts,
                                             paa_size=paa_size,
                                             out_ts=out_ts,
                                             save=save)

    return results


def run_paa_from_tsuid(tdm, tsuid, paa_size, out_ts=True, save=False):
    """
    Compute the Piecewise Aggregation Approximation (PAA) on the **tsuid** provided

    :param tdm: temporal data manager object
    :type tdm: TemporalDataMgr

    :param tsuid: TSUID of the TS to calculate the PAA timeseries
    :type tsuid: str

    :param paa_size: number of segments
    :type paa_size: int

    :param out_ts: True means the result will be a TS, False will return only the means
    :type out_ts: bool

    :param save: True means the new TS will be saved in addition of the return
    :type save: bool

    :return: the new TS resulting of the PAA approximation or the list of values (with len = paa_size)
    :rtype: list or TimestampMonoVal
    """

    if type(tsuid) is not str:
        LOGGER.error("tsuid must be a string (got %s)", type(tsuid))
        raise TypeError("tsuid must be a string (got %s)" % type(tsuid))

    # Get the TS content
    ts_data = tdm.get_ts(tsuid_list=[tsuid])[0]

    # Call the calculation of the PAA on the content
    results = run_paa(ts_data=ts_data, paa_size=paa_size)

    if save:
        # Get the FId of the TSUID
        fid = tdm.get_func_id_from_tsuid(tsuid)

        # Give a new name to the new TS
        new_fid = '%s_paa%s_%s' % (fid, paa_size, int(time.time()))

        # Import the new TS
        tdm.import_ts_data("%s" % fid, results.ts, new_fid)

    # Prepare the returned value
    if out_ts:
        result = results.ts
    else:
        result = results.means

    return result


def run_paa(ts_data, paa_size):
    """
    Compute the Piecewise Aggregation Approximation (PAA) on the data provided in **ts_data**

    :param ts_data: TS points list
    :type ts_data: np.ndarray or TimestampedMonoVal

    :param paa_size: number of segments
    :type paa_size: int

    :return: the new TS resulting of the PAA approximation or the list of values (with len = paa_size)
    """

    # Calculate the period of a segment
    segment_period = PAA.segment_period(ts_data=ts_data, paa_size=paa_size)

    # Get the indexes of every segment in the TS (to have the match between 'segment_period' and 'TS list index'
    segments_index = PAA.segments_index(ts_data=ts_data, segment_period=segment_period)

    # Compute the PAA
    result_ts, result_coeff = PAA.compute(ts_data=ts_data, seg_list=segments_index)

    class Result(object):
        """
        Results of a PAA are composed of:
           * period: segment period (as a recall)
           * size: size of the PAA applied (equivalent to len(means))
           * means: list of the means of every segment
           * ts: The TS having same timestamps as ts_data but with values replaced by every segment mean value
        """
        period = segment_period
        size = paa_size
        means = result_coeff
        ts = result_ts

    return Result
