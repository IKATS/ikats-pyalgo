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

from math import ceil
from ikats.algo.core.quality_stats.quality_stats_calculators import \
    calc_qual_stats_value, calc_qual_stats_time, LOGGER
from ikats.core.library.exception import IkatsException
from ikats.core.library.spark import ScManager
from ikats.core.resource.api import IkatsApi


def _ts_read(tsuid, start_date, end_date):
    """
    Return the points of a TS

    :param tsuid: TS to get values from
    :type tsuid: str

    :param start_date: start date (ms since EPOCH)
    :type start_date: int

    :param end_date: end date (ms since EPOCH)
    :type end_date: int

    :return: the values of the Timeseries
    :rtype: numpy.array
    """
    return IkatsApi.ts.read(tsuid_list=[tsuid], sd=start_date, ed=end_date)[0]


def _ts_chunk_count(tsuid, md_list, chunk_size):
    """
    Get the count of chunks for a TSUID split into chunks of <chunk_size> points each

    :param tsuid: tsuid to get points from
    :type tsuid: str

    :param md_list: List of metadata
    :type md_list: dict

    :param chunk_size: the size of the chunk
    :type chunk_size: int

    :return: the number of chunks generated
    :rtype: int
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    try:
        number_of_points = int(md_list[tsuid]["qual_nb_points"])
    except Exception:
        raise ValueError("qual_nb_points metadata not found for TSUID %s" % tsuid)
    return int(ceil(number_of_points / chunk_size))


def _ts_chunk(tsuid, index, md_list, chunk_size):
    """
    Get the chunk <index> information for a TSUID split into chunks of <chunk_size> points each

    :param tsuid: tsuid to get points from
    :type tsuid: str

    :param index: the index of the chunk to get
    :type index: int

    :param md_list: List of metadata
    :type md_list: dict

    :param chunk_size: the size of the chunk
    :type chunk_size: int

    :return: information about the chunk (chunk_index, chunk_start_window, chunk_end_window)
    :rtype: list
    """

    # Number of points
    nb_points = int(md_list[tsuid]["qual_nb_points"])

    # Timeseries start date
    start_date = int(md_list[tsuid]["ikats_start_date"])

    # Timeseries end date
    end_date = int(md_list[tsuid]["ikats_end_date"])

    # Extrapolation of the number of points
    delta = int((end_date - start_date) * chunk_size / nb_points)

    # Chunk start date
    chunk_start = start_date + index * delta

    # Chunk end date
    chunk_end = min(end_date, chunk_start + delta)

    return [index, chunk_start, chunk_end]


def calc_quality_stats(ts_list,
                       compute_value=True, compute_time=True,
                       chunk_size=75000, force_save=True):
    """
    Compute the quality statistics

    Returns a dict as follow
        {
            "TSUIDx" : {
                "MetadataX": ValueX,
                ...
            },
            ...
        }

    Don't override default chunk_size unless you know what you are doing.
    It defines the number of points in a single chunk (assuming th TS is periodic)
    Use it only for performances purposes

    :param ts_list: List of TSUID to work onto
    :type ts_list: list

    :param compute_value: boolean indicating to compute metadata related to value
    :type compute_value: bool

    :param compute_time: boolean indicating to compute metadata related to time
    :type compute_time: bool

    :param chunk_size: (Advanced usage) Override the chunk size
    :type chunk_size: int

    :param force_save: Save metadata even if already present (default True)
    :type force_save: bool

    :return: Tuple composed of the input ts list and a dict
             having TSUID as key and a value being sub-dict
             where key is metadata name
    :rtype: tuple dict
    """

    if not compute_value and not compute_time:
        LOGGER.error("You shall compute at least one set of metadata.")
        raise ValueError("You shall compute at least one set of metadata")

    try:
        # Convert tsuid_list [{tsuid:x, fid:x},...] to tsuid_list [tsuid,...]
        tsuid_list = [x['tsuid'] for x in ts_list]

    except TypeError:
        # Already a tsuid_list. No change
        tsuid_list = ts_list

    LOGGER.info('Computing Quality stats for %s TS', len(tsuid_list))

    # Get all metadata
    md_list = IkatsApi.md.read(ts_list=tsuid_list)

    # Initialize results
    results = {}
    for tsuid in tsuid_list:
        results[tsuid] = {}

    try:
        # Get Spark Context
        # Important !!!! Use only this method in Ikats to use a spark context
        spark_context = ScManager.get()

        results = {}
        for index, tsuid in enumerate(tsuid_list):

            LOGGER.info('Processing Quality stats for TS %s (%s/%s)',
                        tsuid, index, len(tsuid_list))

            # Generating information about TSUID chunks
            # ([chunk_index, sd, ed], ...)
            ts_info = []
            for chunk_index in range(_ts_chunk_count(tsuid=tsuid,
                                                     md_list=md_list,
                                                     chunk_size=chunk_size)):
                ts_info.append(_ts_chunk(tsuid=tsuid,
                                         index=chunk_index,
                                         md_list=md_list,
                                         chunk_size=chunk_size))

            # Parallelizing information to work with spark
            # Each chunk can be computed separately, so divided into len(chunks) partitions
            rdd_ts_info = spark_context.parallelize(ts_info, max(8, len(ts_info)))

            # RDD containing the list of points values for every chunk of a TSUID
            # (without timestamps):
            # ([chunk_index, [[timestamp, value], ...], ...)
            rdd_ts_dps = rdd_ts_info \
                .map(lambda x: (x[0], _ts_read(tsuid=tsuid, start_date=x[1], end_date=x[2])))

            # This RDD is used multiple times, caching it to speed up
            rdd_ts_dps.cache()

            if compute_value:
                # Compute metadata related to "value" information
                result = calc_qual_stats_value(tsuid, rdd_ts_dps, force_save=force_save)
                # Append to final results
                if tsuid in results:
                    results[tsuid].update(result[tsuid])
                else:
                    results.update(result)

            if compute_time:
                # Compute metadata related to "time" information
                result = calc_qual_stats_time(tsuid, rdd_ts_dps, force_save=force_save)
                # Append to final results
                if tsuid in results:
                    results[tsuid].update(result[tsuid])
                else:
                    results.update(result)

            # We don't need the cache anymore
            rdd_ts_dps.unpersist()
    except Exception as cause:
        raise IkatsException("Quality stats failure with ...", cause)
    finally:
        ScManager.stop()
    return ts_list, results
