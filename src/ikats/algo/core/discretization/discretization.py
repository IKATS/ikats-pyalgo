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
import re
from _collections import defaultdict
from math import ceil
from enum import IntEnum

import numpy as np

from ikats.core.library.exception import IkatsException
from ikats.core.library.spark import ScManager
from ikats.core.resource.api import IkatsApi

LOGGER = logging.getLogger(__name__)


class Operators(IntEnum):
    """
    Enumerate defining available operators available for discretization
    """

    # operator min
    MIN = 0

    # operator max
    MAX = 1

    # operator average
    AVG = 2

    # operator standard deviation
    STD = 3


def discretize_dataset(ds_name,
                       nb_buckets,
                       table_name,
                       operators_list=None,
                       nb_points_by_chunk=100000):
    """
    This function discretizes each time series provided through dataset name input:
    1. Interval between start date and end date of each time series is divided into nb_buckets interval of same size.
    2. Each operator from input list is processed on each bucket previously defined
    3. result is formatted as a table whose entries are :
                - each time series processed in rows
                - combinations of (each operator) X (each bucket number) in columns
    Result table contains also buckets definitions like (bucket_number, start date, end date)

    :param ds_name: name of the dataset processed
    :type ds_name:  str

    :param nb_buckets: number of buckets wanted for each time series of dataset
    :type nb_buckets: int

    :param table_name: name of the table
    :type table_name:  str

    :param operators_list: list of operators to be calculated on buckets from Operators class (see above)
    :type operators_list: list

    :param nb_points_by_chunk: size of chunks in number of points (assuming time series are periodic and without holes)
    :type nb_points_by_chunk: int


    :returns: a dict containing all data awaited by functional ikats type table
    :rtype: dict

    :raises TypeError: if ds_name is not a string or is None
    :raises TypeError: if nb_buckets is not an integer
    :raises ValueError: if nb_buckets is zero
    :raises ValueError: if operators_list is None
    :raises ValueError: if operators_list items are not in Operators class
    :raises TypeError: if operators_list items are not all string
    :raises ValueError: if number of buckets exceeds number of points for one time series
    """

    # Check inputs validity
    if ds_name is None or type(ds_name) is not str:
        raise TypeError('valid dataset name must be defined (got %s, type: %s)' % (ds_name, type(ds_name)))
    try:
        nb_buckets = int(nb_buckets)
    except:
        raise TypeError('Number of buckets must be an integer (got value %s)' % nb_buckets)
    if nb_buckets == 0:
        raise ValueError("Number of buckets must be not null")
    if operators_list is None:
        raise ValueError("operators list must be not null")
    elif type(operators_list) is not list:
        raise ValueError("operators list must be a list")
    elif not operators_list:
        raise ValueError("operators list must not be empty list")
    if table_name is None or re.match('^[a-zA-Z0-9-_]+$', table_name) is None:
        raise ValueError("Error in table name")

    # Check content of operators list provided
    for operator in operators_list:
        if type(operator) is not str:
            raise TypeError('Operator must be a string (got %s)' % (type(operator)))
        if operator not in [op.name for op in Operators]:
            raise ValueError("Operators (string) must be in the following values list : %s"
                             % [op.name for op in Operators])

    # Extract tsuid list from inputs
    tsuid_list = IkatsApi.ds.read(ds_name)['ts_list']

    # Get list of metadata for all TS
    meta_dict = IkatsApi.md.read(tsuid_list)

    # Initialize result
    result = {}

    try:
        LOGGER.info("Running discretization using Spark")
        # Create or get a spark Context
        sc = ScManager.get()

        # running discretization time series by time series
        for index, tsuid in enumerate(tsuid_list):
            result[tsuid] = {}
            LOGGER.info('Processing Discretization for TS %s (%s/%s)', tsuid, index + 1, len(tsuid_list))

            sd = int(meta_dict[tsuid]['ikats_start_date'])
            ed = int(meta_dict[tsuid]['ikats_end_date'])
            nb_points = int(meta_dict[tsuid]['qual_nb_points'])

            # using qual_ref_period if defined, extrapolating otherwise
            if 'qual_ref_period' in meta_dict[tsuid]:
                period = int(float(meta_dict[tsuid]['qual_ref_period']))
            else:
                period = int(float((ed - sd) / nb_points))

            # checking buckets size regarding time series size
            if nb_buckets > nb_points:
                msg = "Number of buckets exceeds number of points for ts (%s, %s)" % (tsuid, IkatsApi.ts.fid(tsuid))
                LOGGER.error(msg)
                raise ValueError(msg)

            # definition of buckets size in ms
            bucket_size_ms = ceil((ed - sd) / nb_buckets)

            # definition of spark chunks size in ms
            chunks_size_ms = nb_points_by_chunk * period

            # definition of buckets start/end dates
            buckets_timestamps = np.hstack((np.arange(sd, ed, bucket_size_ms, dtype=int), ed))
            buckets = [(buckets_timestamps[i] + 1, buckets_timestamps[i + 1]) for i in
                       range(len(buckets_timestamps) - 1)]

            # start date of first bucket is decreased of 1 ms to catch first time series value
            buckets[0] = (buckets[0][0] - 1, buckets[0][1])

            # add bucket number
            data_to_compute = [(a, b[0], b[1]) for a, b in enumerate(buckets)]

            # store buckets definition in results
            result[tsuid]['buckets'] = data_to_compute

            # starting spark process
            # OUTPUT : [(nb_bucket, sd_bucket, ed_bucket), ...]
            inputs = sc.parallelize(data_to_compute, len(data_to_compute))

            # INPUT :  [(nb_bucket, sd_bucket, ed_bucket), ...]
            # OUTPUT : [(nb_bucket, sd_chunk, ed_chunk), ...]
            # PROCESS : cut buckets into chunks of data if smaller and repartition rdd
            rdd_chunks_timestamps = inputs \
                .flatMap(lambda x: (_spark_chunk(x[0], x[1], x[2], chunks_size_ms)))

            # INPUT : [(nb_bucket, sd_chunk, ed_chunk), ...]
            # OUTPUT : [(nb_bucket, data_array), ...]
            # PROCESS : extract data within buckets
            rdd_chunks_data = rdd_chunks_timestamps \
                .map(lambda x: (x[0], IkatsApi.ts.read(tsuid_list=[tsuid], sd=int(x[1]), ed=int(x[2]))[0])) \
                .filter(lambda x: len(x[1]) > 0)

            # INPUT : [(nb_bucket, data_array), ...]
            # OUTPUT : [(nb_bucket, {info1: , info2:, ..., infon:}),...]
            # PROCESS : calculate operators on data chunks
            rdd_chunks_calc = rdd_chunks_data \
                .map(lambda x: _spark_calc_op_on_chunks(x[0], x[1], operators_list)) \
                .filter(lambda x: x is not None)

            # INPUT : [(nb_bucket, {info1: , info2:, ..., infon:}),...]
            # OUTPUT : [(nb_bucket, {info1: , info2:, ..., infon:}),...] reduced by number of bucket
            # PROCESS : reduce operators results on data buckets
            result_by_bucket = rdd_chunks_calc.reduceByKey(lambda x, y: _spark_reduce_op_chunk(x, y)).collect()

            # extract and calculate final results by bucket
            for bucket in result_by_bucket:
                bucket_nb = bucket[0]
                infos = bucket[1]

                result[tsuid][bucket_nb] = {}
                for operator in operators_list:
                    if operator == 'MIN':
                        result[tsuid][bucket_nb]['MIN'] = float(infos["MIN"])
                    if operator == 'MAX':
                        result[tsuid][bucket_nb]['MAX'] = float(infos["MAX"])
                    if operator == 'AVG':
                        # Computation of the final mean
                        avg_value = float(infos["SUM"]) / float(infos["NB_POINTS"])
                        result[tsuid][bucket_nb]['AVG'] = avg_value
                    if operator == 'STD':
                        # Computation of the final mean and standard deviation
                        avg_value = float(infos["SUM"]) / float(infos["NB_POINTS"])
                        # variance is caped to 0 because it could be negative
                        # (but very near to zero) due to substraction of
                        # very near floating point values
                        variance = max(float(float(infos["SQR_SUM"]) / int(infos["NB_POINTS"]) - avg_value ** 2), 0)
                        std_deviation = variance ** 0.5
                        result[tsuid][bucket_nb]['STD'] = std_deviation

        # format result to fit to table type

        description = "Result of Discretize operator with %s buckets for %s" % (nb_buckets, operators_list)
        table = _fill_table_structure_to_store(json_result=result,
                                               operators_list=operators_list,
                                               nb_buckets=nb_buckets,
                                               tsuid_list=tsuid_list,
                                               table_name=table_name,
                                               table_desc=description)

        # Save the table
        IkatsApi.table.create(data=dict(table))
    except Exception as error:
        msg = "Exception raised while discretizing with Spark"
        LOGGER.error(msg + ": %s " % error)
        raise IkatsException(msg)

    finally:
        # Stop spark Context
        ScManager.stop()

    # Return the name of the table saved
    return table_name


def _spark_reduce_op_chunk(chunk_a, chunk_b):
    """
    Reducer for values computation for chunks

    :param chunk_a: first chunk to reduce
    :type chunk_a: dict

    :param chunk_b: second chunk to reduce
    :type chunk_a: dict

    :return: a dict composed of the aggregated basic information of the previous chunks
    :rtype: dict
    """
    result = dict()
    if 'NB_POINTS' in chunk_a and 'NB_POINTS' in chunk_b:
        result['NB_POINTS'] = chunk_a['NB_POINTS'] + chunk_b['NB_POINTS']
    if 'MAX' in chunk_a and 'MAX' in chunk_b:
        result['MAX'] = max(chunk_a['MAX'], chunk_b['MAX'])
    if 'MIN' in chunk_a and 'MIN' in chunk_b:
        result['MIN'] = min(chunk_a['MIN'], chunk_b['MIN'])
    if 'SUM' in chunk_a and 'SUM' in chunk_b:
        result['SUM'] = chunk_a['SUM'] + chunk_b['SUM']
    if 'SQR_SUM' in chunk_a and 'SQR_SUM' in chunk_b:
        result['SQR_SUM'] = chunk_a['SQR_SUM'] + chunk_b['SQR_SUM']

    return result


def _spark_chunk(bucket, sd, ed, chunk_size_ms):
    """
    Cut buckets in chunks of data if smaller

    :param bucket: number of the processed bucket
    :type bucket: int

    :param sd: start date of the processed bucket
    :type sd: int

    :param ed: end date of the processed bucket
    :type ed: int

    :param chunk_size_ms: size of a data chunk in millisecond
    :type chunk_size_ms: int

    :return: list of computed and separated chunks of data within the provided bucket
    :rtype: list of tuple : chunks (bucket number, start date chunk , end date chunk)
    """
    if (ed - sd) > chunk_size_ms:
        # bucket must be divided into smaller chunks
        intervals = np.hstack((np.arange(sd, ed, chunk_size_ms, dtype=np.int64), ed))
        result = [(bucket, intervals[i] + 1, intervals[i + 1]) for i in
                  range(len(intervals) - 1)]

        # start date of first interval is decreased of 1 ms to catch first bucket value
        result[0] = (bucket, result[0][1] - 1, result[0][2])
    else:
        # bucket is small enough, not divided
        result = [(bucket, sd, ed)]
    return result


def _spark_calc_op_on_chunks(bucket, data, operators_list):
    """
    Calculate operators on chunk of data
    return None if no data provided

    :param bucket: bucket number
    :type bucket: int

    :param data: timeseries data
    :type data: 2-d array

    :param operators_list: list of operators calculated on data
    :type operators_list: list

    :return:tuple of (bucket number, result dict of calculated operators on chunk - Keys are operators.)
    :rtype: tuple (int, dict)
    """
    result = {}

    # number of points processed
    nb_points = len(data)
    result['NB_POINTS'] = nb_points

    # keep only values
    values = data[:, 1]

    if values.size:
        for operator in operators_list:
            if operator == 'MIN':
                result['MIN'] = min(values)
            if operator == 'MAX':
                result['MAX'] = max(values)
            if operator == 'AVG' or operator == 'STD':
                result['SUM'] = sum(values)
            if operator == 'STD':
                result['SQR_SUM'] = sum([x ** 2 for x in values])
    else:
        return None

    return bucket, result


def _fill_table_structure_to_store(json_result, operators_list, nb_buckets, tsuid_list, table_name, table_desc):
    """
    :param json_result: dict of collected result from algorithm
    :type json_result: dict

    :param : operators_list: list of operators to be calculated on buckets from Operators class (see above)
    :type : operators_list: list of Operators

    :param : nb_buckets: number of buckets processed
    :type : nb_buckets: int

    :param tsuid_list: list of tsuid processed
    :type tsuid_list: list

    :param table_name: Name of the table to create
    :type table_name: str

    :param table_desc: Description of the table to create
    :type table_desc: str

    :return: dict formatted as awaited by functional type table
    :rtype: dict
    """

    # Filling empty buckets operators result with None
    for tsuid in tsuid_list:
        for no_bucket in range(nb_buckets):
            if no_bucket not in json_result[tsuid]:
                json_result[tsuid][no_bucket] = {}
                for operator in operators_list:
                    json_result[tsuid][no_bucket][operator] = None

    # initializing table structure
    table = defaultdict(dict)

    # filling title
    table['table_desc']['title'] = 'Discretized dataset'
    table['table_desc']['name'] = table_name
    table['table_desc']['desc'] = table_desc

    # filling headers columns
    table['headers']['col'] = dict()
    table['headers']['col']['data'] = ['functional Id']
    table['headers']['col']['data'].extend(
        [operator + "_B" + str(bucket + 1) for bucket in range(nb_buckets) for operator in operators_list])

    # filling headers rows
    table['headers']['row'] = dict()
    table['headers']['row']['data'] = [None]
    table['headers']['row']['data'].extend([IkatsApi.ts.fid(tsuid) for tsuid in tsuid_list])
    table['headers']['row']['default_links'] = {'type': 'bucket_ts', 'context': 'raw'}
    table['headers']['row']['links'] = [None]
    # filling links data of type bucket_ts
    table['headers']['row']['links'].extend(
        [
            {'val': {
                'data': [{'tsuid': tsuid, 'funcId': IkatsApi.ts.fid(tsuid)}],
                'flags': [
                    {'label': "B" + str(no_bucket + 1),  # renaming buckets with numbers from 1 instead of 0
                     'timestamp': int(json_result[tsuid]['buckets'][no_bucket][1])}
                    for no_bucket in range(nb_buckets)]
            }}
            for tsuid in tsuid_list
        ])

    # filling cells content
    table['content']['cells'] = []
    table['content']['cells'].extend([[json_result[tsuid][bucket][operator]
                                       for bucket in range(nb_buckets)
                                       for operator in operators_list]
                                      for tsuid in tsuid_list])

    return table
