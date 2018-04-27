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

from ikats.core.data.SparkUtils import SparkUtils
from ikats.core.library.spark import ScManager
from ikats.core.resource.api import IkatsApi

LOGGER = logging.getLogger(__name__)

def _spark_cut_y_chunk(tsuid, start_date, end_date, match_criterion, result_info):
    """
    Apply the cut among Y axis (values) on a TS chunk.

    TS chunk is defined by tsuid, start date and end date.
    The cut is performed based on match_criterion expression
    The result_info contains useful information about where to store the cut points:
        * matching_fid: the FID of the timeseries that will contain matching points of the original tsuid
        * not_matching_fid: the FID of the timeseries that will contain non-matching points of the original tsuid
        * matching_tsuid: the TSUID of the timeseries that will contain matching points of the original tsuid
        * not_matching_tsuid: the TSUID of the timeseries that will contain non-matching points of the original tsuid

    The result contain statistics about the computation as a dict:
        * start_date: date of the first point matching/not matching the criterion in the current chunk
        * end_date: date of the last point matching/not matching the criterion in the current chunk
        * numberOfSuccess: number of points matching/not matching the criterion in the current chunk
        * tsuid: TSUID of the matching/non-matching part

    :param tsuid: original timeseries to work on
    :param start_date: start date of the chunk to use
    :param end_date: end date of the chunk to use
    :param match_criterion: lambda expression defining the match condition
    :param result_info: information about result timeseries

    :type tsuid: str
    :type start_date: int
    :type end_date: int
    :type match_criterion: lambda
    :type result_info: dict

    :return: statistics about the timeseries written (matching and non-matching) to compute metadata (later)
    :rtype: dict
    """

    # Load data points
    data_points = IkatsApi.ts.read(tsuid_list=[tsuid], sd=start_date, ed=end_date)[0]

    # Points extracted from original timeseries that match the criterion
    matching_points = []
    # Points extracted from original timeseries that don't match the criterion
    not_matching_points = []

    # Separate the points into matching/not_matching points list
    for point in data_points:
        if match_criterion(point[1]):
            # Push point to matching timeseries
            matching_points.append(point)
        else:
            # Push point to not matching timeseries
            not_matching_points.append(point)

    # Write both TS (matching/not matching)
    try:
        # Create the Timeseries
        if len(matching_points) > 0:
            result = IkatsApi.ts.create(fid=result_info["matching_fid"], data=matching_points,
                                        generate_metadata=False,
                                        sparkified=True)
            # Add start and end dates to the results
            match_result = {
                "start_date": matching_points[0][0],
                "end_date": matching_points[-1][0],
                "numberOfSuccess": result['numberOfSuccess'],
                "tsuid": result['tsuid']
            }
        else:
            match_result = {
                "start_date": None,
                "end_date": None,
                "numberOfSuccess": 0,
                "tsuid": result_info["matching_tsuid"]
            }

        if len(not_matching_points) > 0:
            result = IkatsApi.ts.create(fid=result_info["not_matching_fid"], data=not_matching_points,
                                        generate_metadata=False,
                                        sparkified=True)

            # Add start and end dates to the results
            no_match_result = {
                "start_date": not_matching_points[0][0],
                "end_date": not_matching_points[-1][0],
                "numberOfSuccess": result['numberOfSuccess'],
                "tsuid": result['tsuid']
            }
        else:
            no_match_result = {
                "start_date": None,
                "end_date": None,
                "numberOfSuccess": 0,
                "tsuid": result_info["not_matching_tsuid"]
            }

        return match_result, no_match_result
    except:
        raise


def _prepare_spark_data(fid_pattern, md_list, ts_list):
    """
    Prepare the data to be used during/after spark transformation.

    The output is composed of 2 lists:

    * ts_list_with_new_fid that contains (as a list of tuples):
      - original TSUID
      - new FID for matching criterion part
      - new FID for not matching criterion part
    * fid2tsuid : a hash map where the key is a FID and the value is the associated TSUID

    :param fid_pattern: pattern used to name the FID of the output TSUID.
           {fid} will be replaced by the FID of the original TSUID FID
           {M} will be replaced by the original TSUID metric name
           {compl} will be replaced by "matching" or "not_matching" depending on the output type.
    :param md_list: list of all metadata associated to the TS_list
    :param ts_list: tsuid/funcId list to use

    :type fid_pattern: str
    :type md_list: dict
    :type ts_list: list

    :return: the TS list to produce and a hash map
    :rtype: list, dict
    """

    # Initialize the first output to an empty list
    ts_list_with_new_fid = []

    # Initialize the hash map to empty dict
    fid2tsuid = {}

    for item in ts_list:
        original_tsuid = item['tsuid']
        original_fid = item['funcId']

        # Prepare the keywords to be replaced in fid_pattern
        # Keyword is written "{keyword}" in the pattern
        replacement_keys = {
            'fid': original_fid,
            'M': md_list[original_tsuid]["metric"],
            'compl': ""
        }

        # Create new TSUID for "matching" pattern and fill the hash map
        new_fid_matching = fid_pattern.format(**replacement_keys)
        fid2tsuid[new_fid_matching] = IkatsApi.ts.create_ref(fid=new_fid_matching)

        # Create new TSUID for "not matching" pattern and fill the hash map
        replacement_keys['compl'] = "_compl"
        new_fid_not_matching = fid_pattern.format(**replacement_keys)
        fid2tsuid[new_fid_not_matching] = IkatsApi.ts.create_ref(fid=new_fid_not_matching)

        # Create a link between original TSUID and the 2 resulting FID
        ts_list_with_new_fid.append((
            original_tsuid,
            new_fid_matching,
            new_fid_not_matching
        ))

    return ts_list_with_new_fid, fid2tsuid


def _format_output(deleted_tsuid, fid2tsuid, ts_list_with_new_fid, index):
    """
    return a formatted list.
    The operations performed on this list are :
    - removing deleted timeseries from the output
    - sorting in ascending order from the raw output

    :param deleted_tsuid: list of deleted timeseries (no point was added in the initialized timeseries)
    :param fid2tsuid: hash map to get TSUID from FID
    :param ts_list_with_new_fid: associative list of original TSUID with matching and not matching FID
    :param index: index to use in ts_list_with_new_fid to select which of matching or not matching to use.

    :type deleted_tsuid: list
    :type fid2tsuid: dict
    :type ts_list_with_new_fid: list
    :type index: int

    :return: the sorted list of tsuid and funcId [{tsuid:x, funcId:x}, ...]
    :rtype: list
    """
    return [
        {'funcId': x[index], 'tsuid': fid2tsuid[x[index]]}
        for x in sorted(ts_list_with_new_fid, key=lambda x: x[index])
        if fid2tsuid[x[index]] not in deleted_tsuid
    ]


def cut_y(original_ts_list, criterion, fid_pattern="{fid}_cutY{compl}", chunk_size=75000):
    """
    Algorithm Cut-Y

    Cut among Y-axis (values) a list of timeseries matching a criterion defined as a python expression.
    Matching and non-matching values are separated into 2 timeseries

    This algorithm uses spark

    From the TS list provided (used as reference), extract 2 TS list:
    * The first one matching the value condition
    * The second one not matching the value condition

    :param original_ts_list: List of TSUID/funcID to use for filtering: [{tsuid:xxx, funcId:xxx}, ...]
    :param criterion: python expression used to define a matching pattern
    :param fid_pattern: pattern used to name the FID of the output TSUID.
           {fid} will be replaced by the FID of the original TSUID FID
           {M} will be replaced by the original TSUID metric name
           {compl} will be replaced by "" or "_compl" depending on the output type (matching/not matching).
    :param chunk_size: the number of points per chunk

    :type original_ts_list: list
    :type criterion: str
    :type fid_pattern: str
    :type chunk_size: int

    :return: 2 lists representing the "matching" and "not matching" list of TS corresponding to the input
    :rtype: list

    :raises ValueError: if ts_list is badly formatted
    :raises TypeError: if ts_list is not a list
    """

    # Check input validity
    if type(original_ts_list) is not list:
        raise TypeError("ts_list shall be a list")
    if len(original_ts_list) == 0:
        raise ValueError("ts_list shall have at least one element")
    for _, item in enumerate(original_ts_list):
        if "tsuid" not in item or "funcId" not in item:
            raise ValueError("ts_list shall have tsuid and funcId defined")

    # Get all the metadata
    md_list = IkatsApi.md.read(ts_list=[x['tsuid'] for x in original_ts_list])

    # Prepare the spark items to parallelize

    # Create and build the data that will be used in spark transformations
    ts_list_with_new_fid, fid2tsuid = _prepare_spark_data(fid_pattern=fid_pattern,
                                                          md_list=md_list,
                                                          ts_list=original_ts_list)
    # Chunks computation
    ts_info = []
    for ts_data in ts_list_with_new_fid:

        # Get the chunks raw information
        chunks = SparkUtils.get_chunks(tsuid=ts_data[0], md_list=md_list, chunk_size=chunk_size)

        # Build a new list containing only used information
        for chunk in chunks:
            ts_info.append({
                "tsuid": ts_data[0],
                "start_date": chunk[1],
                "end_date": chunk[2],
                "matching_fid": ts_data[1],
                "not_matching_fid": ts_data[2],
                "matching_tsuid": fid2tsuid[ts_data[1]],
                "not_matching_tsuid": fid2tsuid[ts_data[2]]
            })

    # Get Spark Context
    # Important !!!! Use only this method in Ikats to use a spark context
    spark_context = ScManager.get()
    try:

        # Prepare the lambda expression. Value is replaced by "Y" variable name
        lambda_criterion = eval("lambda Y : " + criterion)

        # OUTPUT : [{
        #   tsuid:x,
        #   start_date:x,
        #   end_date:x,
        #   matching_fid:x,
        #   not_matching_fid:x,
        #   matching_tsuid:x,
        #   not_matching_tsuid:x
        # }, ...]
        # PROCESS : Parallelize TS chunks information
        rdd_ts_list = spark_context.parallelize(ts_info, max(8, len(ts_info)))

        # INPUT :  [{
        #   tsuid:x,
        #   start_date:x,
        #   end_date:x,
        #   matching_fid:x,
        #   not_matching_fid:x,
        #   matching_tsuid:x,
        #   not_matching_tsuid:x
        # }, ...]
        # OUTPUT : [({
        #  start_date: "date of the first point matching the criterion in the current chunk"
        #  end_date: "date of the last point matching the criterion in the current chunk"
        #  numberOfSuccess: "number of points matching the criterion in the current chunk"
        #  tsuid: "TSUID of the matching part"
        # },
        # {
        #  start_date: "date of the first point not matching the criterion in the current chunk"
        #  end_date: "date of the last point not matching the criterion in the current chunk"
        #  numberOfSuccess: "number of points not matching the criterion in the current chunk"
        #  tsuid: "TSUID of the non-matching part"
        # }), ...]
        # PROCESS : Separate points matching and not-matching the criterion in every chunk. Fill the corresponding TS
        rdd_imported = rdd_ts_list.map(lambda x: _spark_cut_y_chunk(
            tsuid=x['tsuid'],
            start_date=x['start_date'],
            end_date=x['end_date'],
            match_criterion=lambda_criterion,
            result_info={
                "matching_fid": x['matching_fid'],
                "not_matching_fid": x['not_matching_fid'],
                "matching_tsuid": x['matching_tsuid'],
                "not_matching_tsuid": x['not_matching_tsuid']
            }))

        # INPUT : [({
        #  start_date: "date of the first point matching the criterion in the current chunk"
        #  end_date: "date of the last point matching the criterion in the current chunk"
        #  numberOfSuccess: "number of points matching the criterion in the current chunk"
        #  tsuid: "TSUID of the matching part"
        # },
        # {
        #  start_date: "date of the first point not matching the criterion in the current chunk"
        #  end_date: "date of the last point not matching the criterion in the current chunk"
        #  numberOfSuccess: "number of points not matching the criterion in the current chunk"
        #  tsuid: "TSUID of the non-matching part"
        # }), ...]
        # OUTPUT : [(TSUID, nb_points, start_date, end_date), ...]
        # PROCESS : Flat the results and simplify the format to allow quick actions on every item
        rdd_metadata_prep = rdd_imported \
            .flatMap(lambda x: x) \
            .filter(lambda x: x is not None) \
            .map(lambda x: (x['tsuid'], x['numberOfSuccess'], x['start_date'], x['end_date']))

        # Delete empty TSUID
        deleted_tsuid = rdd_metadata_prep \
            .map(lambda x: (x[0], x[1])) \
            .reduceByKey(lambda x, y: x + y) \
            .filter(lambda x: x[1] == 0) \
            .map(lambda x: (x[0], IkatsApi.ts.delete(tsuid=x[0]))) \
            .map(lambda x: x[0]) \
            .collect()

        # This RDD is reused in several branches. Caching it improves the performances
        rdd_metadata_prep.cache()

        # Create metadata qual_nb_points
        rdd_metadata_prep \
            .map(lambda x: (x[0], x[1])) \
            .reduceByKey(lambda x, y: x + y) \
            .filter(lambda x: x[1] > 0) \
            .foreach(lambda x: IkatsApi.md.create(tsuid=x[0], name="qual_nb_points", value=x[1]))

        # Create metadata ikats_start_date
        rdd_metadata_prep \
            .map(lambda x: (x[0], x[2])) \
            .filter(lambda x: x[1] is not None) \
            .reduceByKey(lambda x, y: min(x, y)) \
            .foreach(lambda x: IkatsApi.md.create(tsuid=x[0], name="ikats_start_date", value=x[1]))

        # Create metadata ikats_end_date
        rdd_metadata_prep \
            .map(lambda x: (x[0], x[3])) \
            .filter(lambda x: x[1] is not None) \
            .reduceByKey(lambda x, y: max(x, y)) \
            .foreach(lambda x: IkatsApi.md.create(tsuid=x[0], name="ikats_end_date", value=x[1]))

        # Unpersist the RDD because not used anymore
        rdd_metadata_prep.unpersist()

    finally:
        ScManager.stop()

    # Inherit properties
    for item in ts_list_with_new_fid:
        if fid2tsuid[item[1]] not in deleted_tsuid:
            IkatsApi.ts.inherit(tsuid=fid2tsuid[item[1]], parent=item[0])
        if fid2tsuid[item[2]] not in deleted_tsuid:
            IkatsApi.ts.inherit(tsuid=fid2tsuid[item[2]], parent=item[0])

    # Format and sort the results
    # First output contains the matched data points TS reference
    # Second output contains the not matched (complement) points TS reference
    return (_format_output(deleted_tsuid=deleted_tsuid,
                           fid2tsuid=fid2tsuid,
                           ts_list_with_new_fid=ts_list_with_new_fid,
                           index=1),
            _format_output(deleted_tsuid=deleted_tsuid,
                           fid2tsuid=fid2tsuid,
                           ts_list_with_new_fid=ts_list_with_new_fid,
                           index=2))
