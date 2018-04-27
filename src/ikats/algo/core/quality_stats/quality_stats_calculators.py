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
import json

import logging

from ikats.core.resource.api import IkatsApi
from ikats.core.resource.client.temporal_data_mgr import DTYPE

"""
Calculators definitions for Quality stats
"""

# Logger used for Quality stats
LOGGER = logging.getLogger(__name__)


def _spark_ts_read_values(dps):
    """
    Return the values of a TS

    :param dps: raw points returned by IkatsApi
    :type dps: numpy.array

    :return: numpy.array containing the values
    :rtype: numpy.array
    """
    try:
        return dps[:, 1]
    except IndexError:
        return None


def _spark_ts_read_timestamps(dps):
    """
    Return the values of a TS

    :param dps: raw points returned by IkatsApi
    :type dps: numpy.array

    :return: numpy.array containing the timestamps
    :rtype: numpy.array
    """
    try:
        return dps[:, 0]
    except IndexError:
        return None


def _spark_calc_values_chunk(points):
    """
    Compute some basic information about the chunk points values

    The returned information are :
    * count : the number of points in chunk
    * max : the maximum value in chunk
    * min : the minimum value in chunk
    * sum : the sum of the values in chunk
    * sqr_sum : the sum of the square values in chunk (used for variance calculation)


    :param points: list of data values for each point in the chunk
    :type points: numpy.array

    :return: a dict composed of the basic information computed
    :rtype: dict
    """

    try:
        nb_points = len(points)
    except TypeError:
        return None

    if nb_points > 0:
        sum_chunk_value = sum(points)
        square_sum_chunk_value = sum([x * x for x in points])
        max_chunk_value = max(points)
        min_chunk_value = min(points)
    else:
        # Empty chunk, skip it
        return None

    return {
        "count": nb_points,
        "max": max_chunk_value,
        "min": min_chunk_value,
        "sum": sum_chunk_value,
        "sqr_sum": square_sum_chunk_value,
    }


def _spark_reduce_value_chunk(chunk_a, chunk_b):
    """
    Reducer for values computation for chunks

    :param chunk_a: First chunk to reduce
    :type chunk_a: dict

    :param chunk_b: Second chunk to reduce
    :type chunk_a: dict

    :return: a dict composed of the aggregated basic information of the previous chunks
    :rtype: dict
    """

    if chunk_a is None:
        if chunk_b is None:
            # Both chunks are empty, skip it
            return None
        else:
            # Chunk b is the only one filled, use it
            return chunk_b
    else:
        if chunk_b is None:
            # Chunk a is the only one filled, use it
            return chunk_a

    return {
        "count": chunk_a['count'] + chunk_b['count'],
        "max": max(chunk_a['max'], chunk_b['max']),
        "min": min(chunk_a['min'], chunk_b['min']),
        "sum": chunk_a['sum'] + chunk_b['sum'],
        "sqr_sum": chunk_a['sqr_sum'] + chunk_b['sqr_sum'],
    }


def _spark_calc_timestamps_chunk(chunk_index, timestamps):
    """
    Compute the period intervals of the Timeseries

    If there is no intervals, the "intervals" field will be []

    :param chunk_index: identifier of the processed chunk
    :type chunk_index: int

    :param timestamps: the timestamps list (ms since EPOCH)
    :type timestamps: np.array

    :return: index of chunk + associated intervals
    :rtype: list
    """

    # The chunk size is small enough to use the power of NumPy to compute intervals
    intervals_list = list(timestamps[1:] - timestamps[:-1])

    return chunk_index, {
        "sd": timestamps[0],
        "ed": timestamps[-1],
        "intervals": intervals_list
    }


def _calc_inter_chunks_period(rdd_ts_timestamps, rdd_ts_timestamps_info):
    """
    Calculate time intervals occurring between successive chunks.
    Handle the case where the period is bigger than a whole chunk

    Example:
    * Chunk A last point at T1
    * Chunk B has no point
    * Chunk C has first point at T2
    --> An entry will be written into RDD with : int(T2-T1)

    Format returned [interval_value, ...]
    :param rdd_ts_timestamps: RDD containing timestamps only
        Format : ((chunk_index, [timestamps]),...)
    :type rdd_ts_timestamps: RDD

    :param rdd_ts_timestamps_info: RDD containing chunk information : start_date, end_date and intervals
        Format : ((chunk_index, { sd:start_date, ed:end_date, intervals:[intervals] } ),...)
    :type rdd_ts_timestamps_info: RDD

    :return: new RDD containing all the timeseries intervals (with the ones between chunks)
    :rtype: RDD
    """

    # FROM RDD containing chunk information : start_date, end_date and intervals
    # BUILD new RDD containing the list of period
    # OUTPUT FORMAT (interval_value,...)
    rdd_chunk_ts_intervals = rdd_ts_timestamps_info \
        .map(lambda x: x[1]['intervals']) \
        .flatMap(lambda x: x) \
        .map(lambda x: int(x))

    # FROM RDD containing timestamps only
    # BUILD new RDD containing chunk limits information
    # OUTPUT FORMAT [(chunk_index, first_point_timestamp, last_point_timestamp), ...]
    rdd_inter_chunks_to_find_1 = rdd_ts_timestamps \
        .map(lambda x: (x[0], x[1][0], x[1][-1]))

    # Renumber chunk_index to get contiguous numbers
    chunk_renumber_map_flat = rdd_ts_timestamps.map(lambda x: x[0]).collect()
    chunk_renumber_map = dict(zip(chunk_renumber_map_flat, range(len(chunk_renumber_map_flat))))

    # FROM RDD containing chunk limits information
    # BUILD new RDD containing renumbered chunk index
    # OUTPUT FORMAT [(renumbered_chunk_index, first_point_timestamp, last_point_timestamp), ...]
    rdd_inter_chunks_to_find = rdd_inter_chunks_to_find_1 \
        .map(lambda x: (chunk_renumber_map[x[0]], x[1], x[2]))

    # FROM RDD containing renumbered chunk index
    # BUILD new RDD containing
    # - inter-chunk index (inter-chunk "1" is between chunk1 and chunk2, ..., "n" between "n" and "n+1")
    # - boundary timestamp (no matter if it is first or last point)
    # - occurrence count
    # OUTPUT FORMAT [(renumbered_inter_chunk_index, point_timestamp, count), ...]
    rdd_inter_chunks_3 = rdd_inter_chunks_to_find \
        .flatMap(lambda x: ((x[0], (x[1], 1)), (x[0] + 1, (x[2], 1))))

    # FROM RDD containing
    # - inter-chunk index (inter-chunk "1" is between chunk1 and chunk2, ..., "n" between "n" and "n+1")
    # - boundary timestamp (no matter if it is first or last point)
    # - occurrence count
    # BUILD new RDD containing interval period for identical inter-chunk index and aggregated count
    # OUTPUT FORMAT [(interval_period, count), ...]
    rdd_inter_chunks_4 = rdd_inter_chunks_3 \
        .reduceByKey(lambda x, y: (abs(y[0] - x[0]), y[1] + x[1]))

    # FROM RDD containing interval period for identical inter-chunk index and aggregated count
    # BUILD new RDD containing intervals between chunks by removing first and last chunk information
    # OUTPUT FORMAT [(interval_period), ...]
    rdd_inter_chunks = rdd_inter_chunks_4 \
        .filter(lambda x: x[1][1] > 1) \
        .map(lambda x: x[1][0])

    # Combining chunks and inter-chunks intervals to obtain all intervals
    # [interval_value, ...]
    return rdd_chunk_ts_intervals + rdd_inter_chunks


def _save_metadata(tsuid, md_name, md_value, dtype, force_save):
    if not IkatsApi.md.create(
            tsuid=tsuid,
            name=md_name,
            value=md_value,
            data_type=dtype,
            force_update=force_save):
        LOGGER.error("Metadata '%s' couldn't be saved for TS %s", md_name, tsuid)


def calc_qual_stats_value(timeseries, rdd_ts_dps, force_save=True):
    """
    Compute the quality stats related to "values":
    - qual_min_value
    - qual_max_value
    - qual_average
    - qual_nb_points
    - qual_variance

    The RDD parameter is composed of a tuple of information: ([chunk_index, sd, ed], ...)

    the returned value match the following format:
    { tsuid : { metadata_name : metadata_value, ... }, ... }

    :param timeseries: TS to compute metadata from
    :type timeseries: str

    :param rdd_ts_dps: RDD composed of all chunk points
    :type rdd_ts_dps: RDD

    :param force_save: Save metadata even if already present (default True)
    :type force_save: bool

    :return: the metadata information
    :rtype: dict
    """

    # FROM RDD containing data points
    # BUILD new RDD containing values only
    # OUTPUT FORMAT ([value1, value2, ...],...)
    rdd_ts_data = rdd_ts_dps.map(lambda x: _spark_ts_read_values(x[1]))

    # FROM RDD containing values only
    # BUILD new RDD containing information about every chunk
    # OUTPUT FORMAT ({"count": , "max":, "min":, "sum":, "sqr_sum":},...)
    rdd_info_1 = rdd_ts_data.map(lambda x: _spark_calc_values_chunk(x)) \
        .filter(lambda x: x is not None)

    # Get all aggregated information about chunks
    # OUTPUT FORMAT {"count": , "max":, "min":, "sum":, "sqr_sum":}
    info = rdd_info_1.reduce(lambda x, y: _spark_reduce_value_chunk(x, y))

    # Computation of the final variance
    avg_value = float(info["sum"]) / float(info["count"])
    variance = float(
        float(info["sqr_sum"]) / int(info["count"]) - avg_value ** 2
    )

    # Forcing variance to 0 if min and max values are the same to avoid precision issues
    if float(info["min"]) == float(info["max"]):
        variance = 0

    result = {
        timeseries: {
            "qual_min_value": float(info["min"]),
            "qual_max_value": float(info["max"]),
            "qual_nb_points": int(info["count"]),
            "qual_average": avg_value,
            "qual_variance": variance
        }
    }

    # Saving metadata and log any error
    if not IkatsApi.md.create(tsuid=timeseries, name="qual_min_value", value=float(info["min"]),
                              data_type=DTYPE.number, force_update=force_save):
        LOGGER.error("Metadata 'qual_min_value' couldn't be saved for TS %s", timeseries)
    if not IkatsApi.md.create(tsuid=timeseries, name="qual_max_value", value=float(info["max"]),
                              data_type=DTYPE.number, force_update=force_save):
        LOGGER.error("Metadata 'qual_max_value' couldn't be saved for TS %s", timeseries)
    if not IkatsApi.md.create(tsuid=timeseries, name="qual_nb_points", value=int(info["count"]),
                              data_type=DTYPE.number, force_update=force_save):
        LOGGER.error("Metadata 'qual_nb_points' couldn't be saved for TS %s", timeseries)
    if not IkatsApi.md.create(tsuid=timeseries, name="qual_average", value=avg_value,
                              data_type=DTYPE.number, force_update=force_save):
        LOGGER.error("Metadata 'qual_average' couldn't be saved for TS %s", timeseries)
    if not IkatsApi.md.create(tsuid=timeseries, name="qual_variance", value=variance,
                              data_type=DTYPE.number, force_update=force_save):
        LOGGER.error("Metadata 'qual_variance' couldn't be saved for TS %s", timeseries)

    return result


def calc_qual_stats_time(tsuid, rdd_ts_dps, force_save=True):
    """
    Compute the quality stats related to "time":
    - qual_min_period : Lowest period encountered
    - qual_max_period : Highest period encountered
    - qual_ref_period : Most occurring period
    - qual_average_period : global average period
    - qual_hist_period : Period histogram with occurrence count for each period
    - qual_hist_period_pc : Percentage version of the histogram (between [0,1])

    The RDD parameter is composed of a tuple of information: ([chunk_index, sd, ed], ...)

    the returned value match the following format:
    { tsuid : { metadata_name : metadata_value, ... }, ... }

    :param tsuid: TS to compute metadata from
    :type tsuid: str

    :param rdd_ts_dps: RDD composed of all chunk points
    :type rdd_ts_dps: RDD

    :param force_save: Save metadata even if already present (default True)
    :type force_save: bool


    :return: the metadata information
    :rtype: dict
    """

    # FROM RDD containing data points
    # BUILD new RDD containing timestamps only
    # OUTPUT FORMAT ((chunk_index, [timestamps]),...)
    rdd_ts_timestamps = rdd_ts_dps \
        .map(lambda x: (x[0], _spark_ts_read_timestamps(x[1]))) \
        .filter(lambda x: x[1] is not None)

    # FROM RDD containing timestamps only
    # BUILD new RDD containing chunk information : start_date, end_date and intervals
    # OUTPUT FORMAT ((chunk_index, { sd:start_date, ed:end_date, intervals:[intervals] } ),...)
    rdd_ts_timestamps_info = rdd_ts_timestamps \
        .map(lambda x: _spark_calc_timestamps_chunk(x[0], x[1])) \
        .filter(lambda x: x[1]['intervals'] != [])

    # FROM RDD containing timestamps only and RDD containing chunk information
    # BUILD new RDD containing all the intervals (even the ones between chunks)
    # OUTPUT FORMAT [interval_value, ...]
    working_rdd = _calc_inter_chunks_period(rdd_ts_timestamps, rdd_ts_timestamps_info)

    # FROM RDD containing all the intervals (even the ones between chunks)
    # BUILD new RDD containing the occurrences of each interval
    # OUTPUT FORMAT [(interval_value, occurrences_count),...]
    working_rdd = working_rdd \
        .map(lambda x: (x, 1)) \
        .reduceByKey(lambda x, y: x + y)

    # Getting distribution of periods in tuples list where first item is the period and second item
    # is the number of occurrence
    # Casting to int to homogenize the key format
    histogram = working_rdd.map(lambda x: (int(x[0]), int(x[1]))).collect()

    # Calculating the number of intervals to compute the histogram as percentage
    nb_intervals = sum([x[1] for x in histogram])

    # Sort the histogram to provide a way to get interesting information
    # (Sorted by descending occurrences count)
    sorted_histogram = sorted(histogram, key=lambda x: -x[1])

    # Impossible to store more than 255 characters in metadata
    # Limit the count until it fits the max length.
    max_metadata_size = 255
    # Limit the number of bars in histogram to the top 20 most occurring intervals
    histogram_max_count = 20
    # Initialize metadata content
    hist_for_metadata = None
    hist_pc_for_metadata = None
    for count in range(histogram_max_count, 0, -1):
        if hist_for_metadata is None or len(hist_for_metadata) >= max_metadata_size:
            hist_for_metadata = json.dumps(dict((x, y) for x, y in sorted_histogram[0:count]))
        if hist_pc_for_metadata is None or len(hist_pc_for_metadata) >= max_metadata_size:
            hist_pc_for_metadata = json.dumps(dict((x, y) for x, y in sorted_histogram[0:count]))
        if hist_pc_for_metadata is not None and hist_for_metadata is not None:
            if len(hist_pc_for_metadata) < max_metadata_size and len(hist_for_metadata) < max_metadata_size:
                # Break loop if both metadata are filled properly
                break

    # Building algorithm result
    result = {
        tsuid: {
            'qual_ref_period': int(sorted_histogram[0][0]),
            'qual_max_period': int(sorted(histogram, key=lambda x: -x[0])[0][0]),
            'qual_min_period': int(sorted(histogram, key=lambda x: x[0])[0][0]),
            'qual_average_period': float(sum([x[0] * x[1] for x in histogram]) / nb_intervals),
            'qual_hist_period': json.dumps(dict((x, y) for x, y in histogram)),
            'qual_hist_period_percent': json.dumps(dict((x, float(y / nb_intervals)) for x, y in histogram)),
        }
    }

    # Save metadata into Ikats
    _save_metadata(tsuid, "qual_ref_period", result[tsuid]["qual_ref_period"], DTYPE.number, force_save)
    _save_metadata(tsuid, "qual_max_period", result[tsuid]["qual_max_period"], DTYPE.number, force_save)
    _save_metadata(tsuid, "qual_min_period", result[tsuid]["qual_min_period"], DTYPE.number, force_save)
    _save_metadata(tsuid, "qual_average_period", result[tsuid]["qual_average_period"], DTYPE.number, force_save)
    _save_metadata(tsuid, "qual_hist_period", hist_for_metadata, DTYPE.string, force_save)
    _save_metadata(tsuid, "qual_hist_period_percent", hist_pc_for_metadata, DTYPE.string, force_save)

    return result
