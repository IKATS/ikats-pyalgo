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

from math import ceil

from ikats.core.library.exception import IkatsException
from ikats.core.library.spark import ScManager
from ikats.core.resource.api import IkatsApi

"""
Algorithm performing a cut on dataset based on the criteria about the value of a metric
"""

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


def _spark_get_cut_ranges(data, lambda_expr):
    # List of detected breakpoints
    breakpoints = []

    # Memorize the start date of the current range found
    start_breakpoint = None
    # Memorize the last date of the range found
    last_breakpoint = None

    # Toggle used to detect the first point where the transition occurs
    in_range = False
    for point in data:
        if lambda_expr(point[1]):

            if not in_range:
                # Previous point was not in range and this one is in matching range
                in_range = True
                start_breakpoint = point[0]
            last_breakpoint = point[0]
        else:

            if in_range:
                # Previous point was in range and this one is not in matching range
                in_range = False
                # Pushing the detected range
                breakpoints.append([start_breakpoint, last_breakpoint])
                start_breakpoint = None
                last_breakpoint = None

    if in_range and len(data) > 1:
        # Pushing the last point
        breakpoints.append([start_breakpoint, data[-1][0]])

    return breakpoints


def _find_cut_ranges(spark_context, tsuid, criteria, md_list, chunk_size=75000):
    """
    Find the start and end dates in reference tsuid to be used later for cutting other TS depending on criteria.

    Return example with 3 ranges found:
    [
       [ start_date_of_1st_range, end_date_of_1st_range],
       [ start_date_of_2nd_range, end_date_of_2nd_range],
       [ start_date_of_3rd_range, end_date_of_3rd_range]
    ]

    :param spark_context: Spark context to use
    :param tsuid: reference TSUID to find breakpoints into
    :param criteria: criteria used to find breakpoints
    :param md_list: list of metadata containing the information about reference tsuid
    :param chunk_size: number of points per chunk to aim at (75000 by default)
    :return: array of start and end dates (each item is a list of start and end date representing a cut range)

    :type spark_context: spark_context
    :type tsuid: str
    :type criteria: str
    :type md_list: dict
    :type chunk_size: int
    :rtype: list
    """

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

    LOGGER.info("Computation of %s chunks for TS:%s", len(ts_chunk_info), tsuid)

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

    # DESCRIPTION : Remove chunk index
    # INPUT  : ([chunk_index_renumbered, chunk_data_points],...)
    # OUTPUT : (chunk_data_points,...)
    rdd_ts_datapoints = rdd_chunk_renumbered.map(lambda x: x[1])

    # Build a lambda function to filter the points
    lambda_filter = eval("lambda M : " + criteria)

    # DESCRIPTION : Defines the date ranges matching the criteria
    # INPUT  : (chunk_data_points,...)
    # OUTPUT : ([range_start, range_end],...)
    rdd_ranges_per_chunk = rdd_ts_datapoints.map(lambda x: _spark_get_cut_ranges(data=x, lambda_expr=lambda_filter))

    # DESCRIPTION : Defines the date ranges matching the criteria
    # INPUT  : (chunk_data_points,...)
    # OUTPUT : ([range_start, range_end],...)
    ranges = rdd_ranges_per_chunk.flatMap(lambda x: x).collect()

    return ranges


def _spark_split_ranges(tsuid, md_list, sd, ed, chunk_size=75000):
    """
    Return a new sub range of start date/end date matching the expected number of points

    :param tsuid: TSUID to compute the sub ranges on
    :param md_list: metadata list used to know the period or the number of points
    :param sd: start date of the initial range
    :param ed: end date of the initial range
    :param chunk_size: ideal number of points to have in each chunk of sub range

    :type tsuid: str
    :type md_list: dict
    :type sd: int
    :type ed: int
    :type chunk_size: int

    :return: the new list of sub ranges
    :rtype: list
    """

    if tsuid not in md_list:
        raise ValueError("No metadata for TS %s" % tsuid)

    # Period is the average period of the whole TS by default
    period = ceil((int(md_list[tsuid]["ikats_end_date"]) - int(md_list[tsuid]["ikats_start_date"])) / int(
        md_list[tsuid]["qual_nb_points"]))

    if "qual_ref_period" in md_list[tsuid]:
        # Period overridden by qual_ref_period which is more accurate (if computed)
        period = int(md_list[tsuid]["qual_ref_period"])

    # Compute number of chunks (1 chunk at least)
    nb_chunks = int(max(ceil((ed - sd) / (period * chunk_size)), 1))

    new_range = []
    chunk_ed = sd
    for i in range(nb_chunks):
        chunk_sd = chunk_ed
        chunk_ed = min(ed, chunk_sd + period * (i + 1) * chunk_size)
        new_range.append([chunk_sd, chunk_ed])

    return new_range


def _cut_ts_list_using_range(spark_context,
                             ts_list,
                             md_list,
                             ranges,
                             metric,
                             fid_pattern="%(fid)s__%(M)s__cut",
                             chunk_size=75000):
    """
    Cut a list of TS based on the ranges (array of start date and end date) provided.

    :param spark_context: Spark context to use
    :param ts_list: List of TSUID to cut
    :param md_list: List of metadata corresponding to TSUID
    :param ranges: List of ranges to use for cut
    :param metric: metric to use for pattern
    :param fid_pattern: Pattern of the new FID ('%(fid)s' and '%(metric)s' will be replaced by original FID and metric)
    :param chunk_size: Size of the ideal chunk (in number of points per chunk)


    :type spark_context: spark_context
    :type ts_list: list
    :type md_list: dict
    :type ranges: list
    :type metric: str
    :type fid_pattern: str
    :type chunk_size: int

    :return: The list of new TSUID created
    :rtype: list
    """

    fid_to_tsuid = {}
    ts_list_with_new_tsuid = []
    for timeseries in ts_list:
        new_fid = fid_pattern % ({
            'fid': IkatsApi.ts.fid(tsuid=timeseries),
            'M': metric
        })
        fid_to_tsuid[new_fid] = IkatsApi.ts.create_ref(fid=new_fid)
        ts_list_with_new_tsuid.append((timeseries, new_fid))

    rdd_ts_list = spark_context.parallelize(ts_list_with_new_tsuid, max(8, len(ts_list_with_new_tsuid)))
    rdd_ranges = spark_context.parallelize(ranges, max(8, len(ranges)))

    # DESCRIPTION : Prepare all the cut to perform
    # INPUT  : ((tsuid, new_fid)...) and ([start_date, end_date], ...)
    # OUTPUT : (((tsuid, new_fid), [range_start, range_end]),...)
    rdd_splits = rdd_ts_list.cartesian(rdd_ranges)

    # DESCRIPTION : Cut big matching range into smaller chunk to not overload worker
    # INPUT  : (((tsuid, new_fid), [range_start, range_end]),...)
    # OUTPUT : (((tsuid, new_fid), [range_start, range_end]),...)
    rdd_sub_splits = rdd_splits \
        .map(lambda x: ((x[0][0], x[0][1]), (x[0][0], x[1]))) \
        .map(lambda x: ((x[0][0], x[0][1]),
                        _spark_split_ranges(tsuid=x[0][0],
                                            md_list=md_list,
                                            sd=x[1][1][0],
                                            ed=x[1][1][1],
                                            chunk_size=chunk_size))) \
        .flatMapValues(lambda x: x) \
        .filter(lambda x: len(x[1]) > 0)

    # DESCRIPTION : Extract points
    # INPUT  : (((tsuid, new_fid), [range_start, range_end]),...)
    # OUTPUT : ((tsuid, new_fid, points),...)
    rdd_split_points = rdd_sub_splits \
        .map(lambda x: (x[0][0], x[0][1], IkatsApi.ts.read(tsuid_list=[x[0][0]], sd=x[1][0], ed=x[1][1])[0])) \
        .filter(lambda x: len(x[2]) > 0)

    rdd_split_points.cache()

    # DESCRIPTION : Save points
    # INPUT  : ((tsuid, new_fid, points),...)
    # OUTPUT : ((tsuid, new_fid, new_tsuid),...)
    rdd_split_results = rdd_split_points \
        .map(lambda x: (x[0], x[1],
                        IkatsApi.ts.create(fid=x[1],
                                           data=x[2],
                                           generate_metadata=False,
                                           sparkified=True)['tsuid'])) \
        .distinct()

    # Caching RDD used 2 times
    rdd_split_results.cache()

    # Inherit properties from parent TS
    rdd_split_results.foreach(lambda x: IkatsApi.ts.inherit(tsuid=x[2], parent=x[0]))

    # Get the list of results as a list.
    # Format is : [(tsuid, new_fid, new_tsuid),...]
    import_results = rdd_split_results.collect()

    # Uncaching used RDD
    rdd_split_results.unpersist()

    # Compute and save qual_nb_points
    rdd_split_points \
        .map(lambda x: (x[1], len(x[2]))) \
        .reduceByKey(lambda x, y: x + y) \
        .foreach(lambda x: IkatsApi.md.create(tsuid=fid_to_tsuid[x[0]], name='qual_nb_points', value=x[1]))

    # Compute and save ikats_start_date
    rdd_split_points \
        .map(lambda x: (x[1], min(x[2][:, 0]))) \
        .reduceByKey(lambda x, y: min(x, y)) \
        .foreach(lambda x: IkatsApi.md.create(tsuid=fid_to_tsuid[x[0]], name='ikats_start_date', value=x[1]))

    # Compute and save ikats_end_date
    rdd_split_points \
        .map(lambda x: (x[1], max(x[2][:, 0]))) \
        .reduceByKey(lambda x, y: max(x, y)) \
        .foreach(lambda x: IkatsApi.md.create(tsuid=fid_to_tsuid[x[0]], name='ikats_end_date', value=x[1]))

    # Uncaching used RDD
    rdd_split_points.unpersist()

    # Return the ts_list format sorted in the same order as input ts_list
    return [{'funcId': item[1], 'tsuid': item[2]} for item in
            sorted(import_results, key=lambda x: ts_list.index(x[0]))]


def cut_ds_from_metric(ds_name, metric, criteria, group_by=None, fid_pattern=None, chunk_size=75000):
    """
    Entry point of the method that cut a dataset based on the criteria applied to the TS matching the metric

    The criteria expression is a python expression that will be converted to a lambda expression with 'M' used as metric
    value.
    Example: "M > 7 and M not in [1,2,6]"

    :param ds_name: name of the dataset to use
    :param metric: metric used as reference to find cut ranges
    :param criteria: criteria expression describing the value thresholds.
    :param group_by: metadata to iterate on each value (Default to None to not use this behaviour)
    :param fid_pattern: name of the generated TS.
                        Variables can be used:
                        - %(fid)s   : Functional identifier
                        - %(M)s     : metric
    :param chunk_size: Size of the ideal chunk (in number of points per chunk)

    :type ds_name: str
    :type metric: str
    :type criteria: str
    :type group_by: str or None
    :type fid_pattern: str
    :type chunk_size: int

    :return: the ts list of the generated TS. [{"funcId": "xx", "tsuid":"xx"}]
    :rtype: list

    :raises ValueError: if dataset is empty
    :raises ValueError: if metric is found several times in dataset
    :raises ValueError: if metric is not found in dataset
    :raises ValueError: if group_by doesn't have a matching reference
    :raises KeyError: if error in fid_pattern
    """

    # List of TS present in dataset
    ts_list = IkatsApi.ds.read(ds_name=ds_name)['ts_list']

    if len(ts_list) == 0:
        LOGGER.error("Dataset %s is empty", ds_name)
        raise ValueError("Dataset %s is empty" % ds_name)

    # Get all the metadata
    md_list = IkatsApi.md.read(ts_list=ts_list)

    # List of all possible values encountered for the group by
    groups_list = None
    if group_by not in [None, ""]:
        # Get all the groups for this group by criterion
        groups_list = _find_all_groups(group_by, md_list)
        LOGGER.info("%s groups found for [%s]", len(groups_list), group_by)
    else:
        # Force to None
        group_by = None

    # Find the reference TS and all TS to cut using this ref
    grouped_ts_list = _find_ts_ref_group(ds_name=ds_name,
                                         md_list=md_list,
                                         metric=metric,
                                         ts_list=ts_list,
                                         group_by=group_by,
                                         group_by_list=groups_list)

    # Get Spark Context
    # Important !!!! Use only this method in Ikats to use a spark context
    spark_context = ScManager.get()

    try:
        result = []

        # For each group (processed in alphabetic order)
        for group in sorted(grouped_ts_list):
            result_iter = _cut_from_metric_for_group(chunk_size=chunk_size,
                                                     criteria=criteria,
                                                     ds_name=ds_name,
                                                     fid_pattern=fid_pattern,
                                                     md_list=md_list,
                                                     metric=metric,
                                                     spark_context=spark_context,
                                                     group=grouped_ts_list[group])

            # Sort functional identifiers alphabetically)
            result.extend(sorted(result_iter, key=lambda x: x['funcId']))

        return result
    finally:
        ScManager.stop()


def _cut_from_metric_for_group(chunk_size, criteria, ds_name, fid_pattern, md_list, metric, spark_context, group):
    ts_ref = group['ref'] or None
    ts_list = group['ts_list'] or None

    if ts_ref is None:
        raise ValueError("TS having metric=%s not found in %s" % (metric, ts_list))

    LOGGER.info("TS used as reference is %s for metric=%s", ts_ref, metric)

    # Get the ranges matching the pattern
    ranges = _find_cut_ranges(spark_context=spark_context,
                              tsuid=ts_ref,
                              criteria=criteria,
                              md_list=md_list,
                              chunk_size=chunk_size)
    LOGGER.info("%s ranges found matching the input pattern", len(ranges))

    # Prepare the other TS to cut (remove the ts reference from the list to cut)
    LOGGER.info("Applying cut for the %s remaining TS of the dataset %s matching the group", len(ts_list), ds_name)
    return _cut_ts_list_using_range(spark_context=spark_context,
                                    ts_list=ts_list,
                                    ranges=ranges,
                                    fid_pattern=fid_pattern,
                                    metric=metric,
                                    md_list=md_list,
                                    chunk_size=chunk_size)


def _find_ts_ref_group(ds_name, md_list, metric, ts_list, group_by=None, group_by_list=None):
    """
    From a list of TS, find the TS that will be used as reference and all associated TS to cut with this reference.
    If group_by parameter is set, then filter the list based on its value

    :param ds_name: Name of the dataset (used for logging)
    :param md_list: List of metadata to select timeseries
    :param metric: Name of the metadata "metric" used to identify the reference TS
    :param ts_list: list of all TS of the dataset
    :param group_by: (optional) group by variable name
    :param group_by_list: (optional) group_by variable value used for this iteration

    :type ds_name: str
    :type md_list: dict
    :type metric: str
    :type ts_list: list
    :type group_by: str or None
    :type group_by_list: list or None

    :return: list where each group is a dict containing the reference TS and its list of TS to cut using this reference
             {ref: xx, ts_list:xx}
    :rtype: list
    """

    # Prepare the default results
    result = dict.fromkeys(["no_group"])
    if group_by_list is not None:
        result = dict.fromkeys(group_by_list, None)

    for timeseries in ts_list:

        # Defines which group this TS belongs to
        if group_by is not None:
            group_by_iter = md_list[timeseries][group_by]
        else:
            # Case where no group defined by user
            group_by_iter = "no_group"

        if result[group_by_iter] is None:
            result[group_by_iter] = {"ref": None, "ts_list": []}

        if md_list[timeseries]['metric'] == metric:
            if result[group_by_iter]["ref"] is not None:
                LOGGER.error("Couple Metric/group by %s/%s=%s found several times in dataset %s: ",
                             metric, group_by, group_by_iter, ds_name)
                raise ValueError("Couple Metric/group by %s/%s=%s found several times in dataset %s: " %
                                 (metric, group_by, group_by_iter, ds_name))
            # Reference TS found
            result[group_by_iter]["ref"] = timeseries
        else:
            # Other Timeseries matching the group are appended
            result[group_by_iter]["ts_list"].append(timeseries)
    return result


def _find_all_groups(group_by, md_list):
    """
    Find all possible values for group_by

    :param group_by: name of the metadata to find values on
    :type group_by: str
    :param md_list: list of metadata to find into
    :type md_list: dict
    :return: the list of all unique possible values for this group
    :rtype: list
    """
    # List of all possible values encountered for the group by
    groups_iterators = []
    # Find groups list
    for ts in md_list:
        if group_by not in md_list[ts]:
            raise ValueError("Group by '%s' is not found at least for ts %s", group_by, ts)
        groups_iterators.append(md_list[ts][group_by])
    # Remove duplicates
    return list(set(groups_iterators))
