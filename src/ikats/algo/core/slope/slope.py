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
import time

# Create the logger for this algorithm
from math import ceil

import numpy as np

from ikats.core.library.exception import IkatsException, IkatsConflictError, IkatsNotFoundError
from ikats.core.library.spark import ScManager
from ikats.core.resource.api import IkatsApi, DTYPE

LOGGER = logging.getLogger(__name__)


def _get_chunks_count(tsuid, md_list, chunk_size):
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
    except KeyError:
        raise ValueError("qual_nb_points metadata not found for TSUID %s" % tsuid)
    return int(ceil(number_of_points / chunk_size))


def _get_chunk_info(tsuid, index, md_list, chunk_size):
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


def _build_chunk_index_map(rdd, lambda_index):
    """
    Returns a map to convert old chunk index to new one by making all chunk index contiguous, keeping the same order.

    A small information of the RDD is collected in this function.

    To convert old index to new index, just use:
    hash_map = _build_chunk_index_map(rdd, lambda x: x[1][0])
    new_index = hash_map[old_index]
    where the old index is at first sub-element of the second tuple element (Example: (("tsuid",(chunk_index, data))))

    :param rdd: RDD to work on
    :type rdd: RDD

    :param lambda_index: lambda expression used to extract chunk index position from rdd
    :type lambda_index: function

    :return: the map
    :rtype: dict
    """

    try:
        # DESCRIPTION : Renumber chunk_index to get contiguous numbers
        # INPUT : any rdd having the index located with the lambda expression <lambda_index>
        # OUTPUT : A dict having key=old index and value=new index
        chunk_renumber_map_flat = rdd.map(lambda_index).collect()
        chunk_renumber_map = dict(zip(chunk_renumber_map_flat, range(len(chunk_renumber_map_flat))))

        return chunk_renumber_map
    except Exception as err:
        raise IkatsException("Error during chunk index remapping %s" % err)


def _spark_ts_read(tsuid, start_date, end_date):
    """
    Return the points of a TS

    :param tsuid: TS to get values from
    :type tsuid: str

    :param start_date: start date (ms since EPOCH)
    :type start_date: int

    :param end_date: end date (ms since EPOCH)
    :type end_date: int

    :return: the values of the Timeseries
    :rtype: np.array
    """

    return IkatsApi.ts.read(tsuid_list=[tsuid], sd=start_date, ed=end_date)[0]


def _spark_calc_slope(data):
    """
    Compute a slope within points list and locate new point to first point timestamp

    Considering 2 successive points A and B
    A : timestamp=Ta and value=Xa
    B : timestamp=Tb and value=Xb

    The new point will be located at Ta with value X = (Xb - Xa) / (Tb - Ta)

    :param data: data points to compute slope onto
    :type data: np.array

    :return: the new values and timestamps after computation of slope
    :rtype: np.array
    """

    # Subtract values and timestamps to previous one
    delta = data[1:] - data[:-1]

    try:
        # Divide values by timestamps to obtain the slope value
        values = delta[:, 1] / delta[:, 0]
    except ZeroDivisionError:
        return None

    # Combine values with timestamps
    # The result has one item less than original timeseries
    result = np.column_stack((data[:-1, 0], values))

    return result


def _spark_save(fid, data):
    """
    Saves a data corresponding to timeseries points to database by providing functional identifier.

    :param fid: functional identifier for the new timeseries
    :param data: data to save

    :type fid: str
    :type data: np.array

    :return: the TSUID
    :rtype: str

    :raises IkatsException: if TS couldn't be created
    """

    results = IkatsApi.ts.create(fid=fid, data=data, generate_metadata=False, sparkified=True)
    if results['status']:
        return results['tsuid']
    else:
        raise IkatsException("TS %s couldn't be created" % fid)


def _compute_chunk_slope(rdd, fid, save_new_ts=True):
    """
    Compute the slope function on the current chunk

    :param rdd: RDD containing the chunk data points
    :param fid: functional identifier for the new timeseries
    :param save_new_ts: True (default) if TS must be saved to database

    :type rdd: RDD
    :type fid: str
    :type save_new_ts: bool

    :return: The TSUID corresponding to the new timeseries and the timestamp of the last point
    :rtype: tuple (str, int)
    """

    # DESCRIPTION : Build the new slope for this chunk and suppress empty chunks
    # INPUT  : (chunk_data_points, ...)
    # OUTPUT : (computed_ts_data_points, ...)
    rdd_chunk_slope = rdd \
        .map(lambda x: _spark_calc_slope(data=x)) \
        .filter(lambda x: x is not None) \
        .filter(lambda x: len(x) > 0)

    # Caching this RDD because reused several times
    rdd_chunk_slope.cache()

    computed_tsuid = None
    if save_new_ts:
        try:
            # DESCRIPTION : Save the slope into database and return the TSUID
            # INPUT  : (computed_ts_data_points, ...)
            # OUTPUT : TSUID
            computed_tsuid = rdd_chunk_slope \
                .map(lambda x: _spark_save(fid=fid, data=x)) \
                .collect()[0]
        except Exception as err:
            raise IkatsException("TS %s couldn't be saved: %s" % (fid, err))

    try:
        # Get date of last point of the chunks
        end_date = int(rdd_chunk_slope \
                       .map(lambda x: x[-1][0]) \
                       .reduce(lambda x, y: max(x, y)))
    except Exception as err:
        raise IkatsException("End date couldn't be extracted : %s" % err)

    # RDD not needed anymore, unpersist it
    rdd_chunk_slope.unpersist()

    return computed_tsuid, end_date


def _compute_inter_chunks_slope(rdd, fid, save_new_ts=True):
    """
    Compute the slope function between chunks to cover 100% of the points

    Assuming the timeseries looks like this:

    | Chunk      | CHUNK A        | CHUNK B        | CHUNK C        | CHUNK D        |
    |            start date       :                :                :                end date
    |            |                :                :                :                |
    | Timeline   *------*---*---*---------------------*----------------*-------------*---------> Time
    | Other info |              | :                :  |             :  |             |
    |            |              Last point of chunkA  |                First point of chunk D
    |            First point of chunk A               First point of chunk C         |
    |                                                 and Last point of chunk C      Last point of chunk D

    Empty chunks (like Chunk B) may not exist here because they were previously removed.
    However, they are handled if empty chunks still exist.


    with "simplification" of terms used:
    * First point of chunk A = FPA
    * Last point of chunk A = LPA
    * First point of chunk C = FPC
    * Last point of chunk C = LPC (=FPC)
    suffixed with 't' for timestamps and 'v' for value

    The process will find 2 inter-chunks slope to calculate:
    * (FPCv-LPAv)/(FPCt-LPAt) located at LPAt
    * (FPDv-LPCv)/(FPDt-LPCt) located at LPCt

    :param rdd: RDD containing the chunk data points
    :param fid: functional identifier for the new timeseries
    :param save_new_ts: True (default) if TS must be saved to database

    :type rdd: RDD
    :type fid: str
    :type save_new_ts: bool

    :return: The timestamp of the last point
    :rtype: int
    """

    # DESCRIPTION : Get the first and last points in chunk range
    # INPUT  : ([chunk_index_renumbered, ts_data_points],...)
    # OUTPUT : ([chunk_index, first_point, last_point], ...)
    rdd_chunk_first_last_point = rdd \
        .map(lambda x: (x[0], x[1][0], x[1][-1]))

    # DESCRIPTION : Flat the chunks to have a "last point" connected to "first point" for every chunk
    #               (except for the very first and for the very last one)
    #               occurrence_count will be used to remove the extreme points (where occurrence_count=1)
    #               to keep only intermediate duet (where occurrence_count=2)
    # INPUT  : ([chunk_index, first_point, last_point], ...)
    # OUTPUT : ((chunk_start_index, last_point_chunk, occurrence_count),
    #           (chunk_next_index, first_point_next_chunk, occurrence_count), ...)
    rdd_flat_points = rdd_chunk_first_last_point \
        .flatMap(lambda x: ((x[0], (x[1], 1)), (x[0] + 1, (x[2], 1))))

    # DESCRIPTION : Group identical chunk index to compute the slope between the last and first point
    # INPUT  : ((inter_chunk_index, data_point, occurrence_count), ...)
    # OUTPUT : (np.array(last_point_chunk, first_point_next_chunk), ...)
    rdd_prep_slope_value = rdd_flat_points \
        .reduceByKey(lambda x, y: ((x[0], y[0]), x[1] + y[1])) \
        .filter(lambda x: x[1][1] > 1) \
        .map(lambda x: np.array(x[1][0]))

    # Call the slope computation

    _, end_date = _compute_chunk_slope(rdd=rdd_prep_slope_value, fid=fid,
                                       save_new_ts=save_new_ts)

    return end_date


def compute_slope_for_tsuid(tsuid, fid, fid_suffix, md_list, chunk_size, spark_context, save_new_ts=True):
    """
    Compute the slope for a single TS

    Principle:
    * Define the new TS information (TSUID) to allow a multipart TS creation
    * Divide the TS in chunks having same count of points (assuming the TS is periodic)
    * Remove the empty chunks and reorder the chunk indexes to have better performances
    * For each chunk, compute the slope using numpy performance and save the chunk into new TS
    * Build a set of inter-chunks containing the last point of previous chunk and first point of next chunk
    * Call again the slope computation for each inter-chunk and save the new point into new TS
    * Compute and save the minimum metadata for this new TS only once at the end of computation


    :param tsuid: TSUID to compute slope on
    :param fid: Functional identifier of the original timeseries
    :param fid_suffix: Functional identifier suffix of the final timeseries
    :param md_list: List of metadata used for optimisation
    :param chunk_size: Number of data points in chunks (assuming TS is periodic)
    :param spark_context: spark context to use
    :param save_new_ts: True (default) if TS must be saved to database

    :type tsuid: str
    :type fid: str
    :type fid_suffix: str
    :type md_list: dict
    :type chunk_size: int
    :type spark_context: SparkContext
    :type save_new_ts: bool

    :return: tsuid and functional id of the new timeseries
    :rtype: tuple (str, str)
    """

    # New FID matches the old one with a suffix
    computed_fid = "%s%s" % (fid, fid_suffix)

    # Assign a TS
    try:
        IkatsApi.ts.create_ref(fid=computed_fid)
    except IkatsConflictError:
        # Reference already exist
        LOGGER.warning("TS %s already exist and will be overwritten by Slope computation", computed_fid)

    # Information about a chunk :
    # * index of the chunk
    # * start date (timestamp ms since EPOCH)
    # * end date (timestamp ms since EPOCH)
    ts_chunk_info = []
    for chunk_index in range(_get_chunks_count(tsuid=tsuid,
                                               md_list=md_list,
                                               chunk_size=chunk_size)):
        ts_chunk_info.append(_get_chunk_info(tsuid=tsuid,
                                             index=chunk_index,
                                             md_list=md_list,
                                             chunk_size=chunk_size))

    LOGGER.info("Computation of %s chunks for TS:%s", len(ts_chunk_info), fid)

    rdd_ts_info = spark_context.parallelize(ts_chunk_info, max(8, len(ts_chunk_info)))

    # DESCRIPTION : Get the points within chunk range and suppress empty chunks
    # INPUT  : ([chunk_index, start_date, end_date],...)
    # OUTPUT : ([chunk_index, ts_data_points], ...)
    rdd_chunk_data = rdd_ts_info \
        .map(lambda x: (x[0], _spark_ts_read(tsuid=tsuid, start_date=int(x[1]), end_date=int(x[2])))) \
        .filter(lambda x: len(x[1]) > 0)
    rdd_chunk_data.cache()

    # DESCRIPTION : Renumber the chunk index to have contiguous index
    # INPUT  : ([chunk_index, ts_data_points],...)
    # OUTPUT : ([chunk_index_renumbered, chunk_data_points], ...)
    chunk_renumber_map = _build_chunk_index_map(rdd_chunk_data, lambda x: x[0])
    rdd_chunk_renumbered = rdd_chunk_data \
        .map(lambda x: (chunk_renumber_map[x[0]], x[1])) \
        .filter(lambda x: len(x[1]) > 0)

    # Caching this RDD because reused several times
    rdd_chunk_renumbered.cache()

    # DESCRIPTION : Compute and save slope into database
    # INPUT  : ([chunk_index_renumbered, chunk_data_points],...)
    # OUTPUT : (chunk_data_points,...)
    rdd_ts_datapoints = rdd_chunk_renumbered.map(lambda x: x[1])

    # DESCRIPTION : Compute and save slope into database
    # INPUT  : (chunk_data_points,...)
    # OUTPUT : TSUID, last point timestamp
    new_tsuid, new_end_date = _compute_chunk_slope(rdd=rdd_ts_datapoints, fid=computed_fid,
                                                   save_new_ts=save_new_ts)
    # Handle inter chunks computation if needed
    if len(ts_chunk_info) > 1:
        # DESCRIPTION : Compute and save slope into database
        # INPUT  : ([chunk_index_renumbered, ts_data_points],...)
        # OUTPUT : last point timestamp
        new_end_date_inter = _compute_inter_chunks_slope(rdd=rdd_chunk_renumbered,
                                                         fid=computed_fid,
                                                         save_new_ts=save_new_ts)

        # Update new end date (latest point between chunk and inter-chunks computation)
        new_end_date = max(new_end_date, new_end_date_inter)

    # RDD not needed anymore, unpersist it
    rdd_chunk_renumbered.unpersist()
    rdd_chunk_data.unpersist()

    if save_new_ts:
        # Handle metadata creation
        obtained_nb_points = None
        for _ in range(30):
            try:
                obtained_nb_points = IkatsApi.ts.nb_points(tsuid=new_tsuid, ed=new_end_date)
                break
            except IndexError:
                # TS is not flushed yet into database. Retrying after 1 second
                time.sleep(1)
        if obtained_nb_points is None:
            # After 30 tries, the number of points is still unreachable. Trigger error
            raise IkatsNotFoundError("TS %s (%s) couldn't be found" % (computed_fid, new_tsuid))

        if (int(md_list[tsuid]["qual_nb_points"]) - 1) != obtained_nb_points:
            raise ValueError("Wrong number of points obtained for %s" % new_tsuid)

        # There is one less point in new TS due to slope operation (from 2 successive points, build a single one)
        IkatsApi.md.create(tsuid=new_tsuid, name="qual_nb_points", value=obtained_nb_points,
                           data_type=DTYPE.number,
                           force_update=True)
        # The first point of the new TS is the same as original TS
        IkatsApi.md.create(tsuid=new_tsuid, name="ikats_start_date", value=md_list[tsuid]["ikats_start_date"],
                           data_type=DTYPE.date,
                           force_update=True)
        # The last point of the new TS is extracted during computation
        IkatsApi.md.create(tsuid=new_tsuid, name="ikats_end_date", value=new_end_date,
                           data_type=DTYPE.date,
                           force_update=True)

        # Inherit properties from parent TS
        IkatsApi.ts.inherit(tsuid=new_tsuid, parent=tsuid)

    return new_tsuid, computed_fid


def compute_slope(ts_list, fid_suffix="_slope", chunk_size=75000, save_new_ts=True):
    """
    Compute the slope of a list of timeseries using spark

    This implementation computes slope for one TS at a time in a loop.
    To know the details of the computation, see the corresponding method

    :param ts_list: list of TS. Each item is a dict composed of a TSUID and a functional id
    :param fid_suffix: Functional identifier suffix of the final timeseries
    :param chunk_size: Number of points per chunk (assuming the TS is periodic)
    :param save_new_ts: True (default) if TS must be saved to database

    :type ts_list: list of dict
    :type fid_suffix: str
    :type chunk_size: int
    :type save_new_ts: bool

    :return: the new list of derived TS (same order as input)
    :rtype: list of dict

    :raise TypeError: if ts_list type is incompatible
    """

    # Check inputs
    if not isinstance(ts_list, list):
        raise TypeError("ts_list shall be a list")
    if len(ts_list) == 0:
        raise TypeError("ts_list must have at least one element")

    LOGGER.info('Computing Slope for %s TS', len(ts_list))

    tsuid_list = ts_list
    try:
        # Extract TSUID from ts_list
        tsuid_list = [x['tsuid'] for x in ts_list]
    except Exception:
        # Already a tsuid_list.
        # Getting the functional id for each ts
        ts_list = [{'tsuid': x, 'funcId': IkatsApi.fid.read(x)} for x in ts_list]

    # Gather all metadata for the list of TS to compute slope
    md_list = IkatsApi.md.read(tsuid_list)

    # Results will be stored here
    results = []

    try:
        # Get Spark Context
        spark_context = ScManager.get()

        for index, tsuid in enumerate(tsuid_list):
            fid = [x['funcId'] for x in ts_list if x['tsuid'] == tsuid][0]
            LOGGER.info('Processing Slope for TS %s (%s/%s) (%s)', fid, (index + 1), len(tsuid_list), tsuid)

            computed_tsuid, computed_fid = compute_slope_for_tsuid(
                spark_context=spark_context,
                fid=fid,
                fid_suffix=fid_suffix,
                tsuid=tsuid,
                md_list=md_list,
                chunk_size=chunk_size,
                save_new_ts=save_new_ts)

            # Append results to final results
            results.append({"tsuid": computed_tsuid, "funcId": computed_fid})
    except Exception:
        raise
    finally:
        # Stop spark context in all cases
        ScManager.stop()

    return results
