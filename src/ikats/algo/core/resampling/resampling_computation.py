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
from enum import IntEnum
from statistics import mean, median
import numpy as np

from ikats.core.library.exception import IkatsException
from ikats.core.resource.api import IkatsApi
from ikats.core.library.spark import ScManager
from ikats.core.resource.client.temporal_data_mgr import DTYPE

LOGGER = logging.getLogger(__name__)


class ResamplingWay(IntEnum):
    """
    Enumerate defining the way of resampling (up or down)
    """
    # up sampling case
    UP_SAMPLING = 0

    # down sampling case
    DOWN_SAMPLING = 1


class AddingMethod(IntEnum):
    """
    Enumerate defining the method applied to add points while upsampling
    """
    # apply the value before to new points
    VALUE_BEFORE = 0

    # apply a linear interpolation to new points
    LINEAR_INTERPOLATION = 1

    # apply the value after to new points
    VALUE_AFTER = 2


class TimestampPosition(IntEnum):
    """
    Enumerate defining the timestamp position in the considered interval while downsampling
    """

    # Beginning alignment
    BEG = 0

    # Middle alignment
    MID = 1

    # End alignment
    END = 2


class AggregationMethod(IntEnum):
    """
    Enumerate defining the data aggregation method in the considered interval while downsampling
    """
    # new point value is the minimum of original data points values within aggregation range
    MIN = 0

    # new point value is the maximum of original data points values within aggregation range
    MAX = 1

    # new point value is the median of original data points values within aggregation range
    # median value is the value in an ordered list for which there is as much superior values as inferior values
    # it depends on the list size :
    #   odd case : result is middle value (ex [ 1 2 3 4 5] => result = 3)
    #   even case : result is the average between the 2 middle values (ex [ 1 2 3 4 ] => result = 2.5)
    MED = 2

    # new point value is the average of original data points values within aggregation range
    AVG = 3

    # new point value is the first (older) of original data points values within aggregation range
    FIRST = 4

    # new point value is the last (younger) of original data points values within aggregation range
    LAST = 5


def resampling_ts(ts_list,
                  resampling_period,
                  adding_method=AddingMethod.LINEAR_INTERPOLATION,
                  timestamp_position=TimestampPosition.BEG,
                  aggregation_method=AggregationMethod.AVG,
                  nb_points_by_chunk=50000,
                  generate_metadata=False):
    """
    This function resamples each timeseries provided
    several cases to consider:
    (NB: original period is based on metadata qual_ref_period)
      * resampling period > original period : we suppress points
      * resampling period <= original period : we add points (or decompression for the equality case)

    PREREQUISITES:  input timeseries shall be periodic (holes allowed)
                    metadata 'qual_ref_period' shall exist in db for each input timeseries

    For DOWNSAMPLING, the method is to process slices of original data, beginning at the first timestamp, of a size
     multiple of original period and of resampling period, nearest of nb_points_by_chunk parameter value
    We want to provide a data point within this interval:
      * timestamp can be :     - the beginning of the interval ('BEG') - default value
                               - the middle of the interval ('MID')
                               - the end of the interval ('END')

      * value can be :     - the minimum value of points in the interval ('MIN')
                           - the maximum value of points in the interval ('MAX')
                           - the median value of points in the interval (as many values below than above) ('MED')
                           - the average value of points in the interval ('AVG') - default value
                           - the first value of points in the interval ('FIRST')
                           - the last value of points in the interval ('LAST')

    for UPSAMPLING, the method is to process slices of original data, beginning at the first timestamp, of a size
     multiple of original period and of resampling period, nearest of nb_points_by_chunk parameter value

    Within this interval we will add points at the target resampling period with following value :
                               - the value before ('VALUE_BEFORE')
                            or - the value after ('VALUE_AFTER')
                            or - a linear interpolated value ('LINEAR_INTERPOLATION') - default value

    :param ts_list: list composing the TS information to resample [{'tsuid': xxx, 'funcId': yyy },...]
    :type ts_list: list of dict

    :param resampling_period: target period for resampling (in ms)
    :type resampling_period: int

    :param adding_method: Method to use for interpolation (see type AddingMethod for more information)
    :type adding_method: AddingMethod or str or int

    :param timestamp_position: timestamp position in the interval while downsampling
    :type timestamp_position: str ('BEG','MID','END') - see above description

    :param aggregation_method: aggregation method for downsampling
    :type aggregation_method: str ('MIN','MAX','MED','AVG','FIRST','LAST') - see above description

    :param nb_points_by_chunk: user defined number of points used for a spark chunk of data (after resampling)
    :type nb_points_by_chunk: int

    :param generate_metadata: True to generate metadata on-the-fly (ikats_start_date, ikats_end_date, qual_nb_points)
    :type generate_metadata: boolean (default : False)

    :returns: a list of dict [{'tsuid': xxx, 'funcId': yyy },...]
    :rtype: list of dict

    :raises TypeError: if tsuid_list_or_dataset is not a string nor a list
    :raises TypeError: if meta_name is not a string
    :raises ValueError: if dataset not found in database
    :raises ValueError: if meta data needed not found (resampling period, start date, end date, ...)
    """

    # Input checking about format of ts_list
    if type(ts_list) is not list or len(ts_list) == 0 or type(ts_list[0]) is not dict:
        raise TypeError("TS_list has wrong type")

    # Extract tsuid list from inputs
    tsuid_list = [x['tsuid'] for x in ts_list]

    # Be sure this is int
    resampling_period = int(resampling_period)

    # Invalid resampling period
    if resampling_period == 0:
        LOGGER.error("Resampling period must be not null")
        raise ValueError("Resampling period must be not null")

    # Get list of metadata for all TS to resample only once to speed up process
    metadata_list = IkatsApi.md.read(tsuid_list)

    # list of tsuid and functional identifiers to downsample
    downsampling_list = []

    # list of tsuid and functional identifiers to upsample
    upsampling_list = []

    # Initialize the mapping list to no data (filled by resampling actions below)
    # Key is original TSUID
    # Value will contain (for each element of the dict)
    # - resampled tsuid
    # - associated functional id
    mapped_ts_list = {}

    for tsuid in tsuid_list:
        if tsuid in metadata_list and 'qual_ref_period' in metadata_list[tsuid]:
            qual_ref_period = int(float(metadata_list[tsuid]['qual_ref_period']))
            if qual_ref_period >= resampling_period:
                # in case of compressed timeseries, qual_ref_period has no sense
                upsampling_list.append(tsuid)
            elif qual_ref_period < resampling_period:
                downsampling_list.append(tsuid)
        else:
            # Metadata not found
            LOGGER.error("Metadata 'qual_ref_period' for timeseries %s not found in base", tsuid)
            raise ValueError("No ref period available for resampling %s" % tsuid)

    # Perform downsampling actions if at least one timeseries has to be downsampled
    if len(downsampling_list) > 0:
        list_downsampled = downsampling_ts(ts_list=[ts for ts in ts_list if ts['tsuid'] in downsampling_list],
                                           resampling_period=resampling_period,
                                           timestamp_position=timestamp_position,
                                           aggregation_method=aggregation_method,
                                           nb_points_by_chunk=nb_points_by_chunk,
                                           generate_metadata=generate_metadata)
        mapped_ts_list.update(list_downsampled)

    # Perform upsampling actions if at least one timeseries has to be upsampled
    if len(upsampling_list) > 0:
        list_upsampled = upsampling_ts(ts_list=[ts for ts in ts_list if ts['tsuid'] in upsampling_list],
                                       resampling_period=resampling_period,
                                       adding_method=adding_method,
                                       nb_points_by_chunk=nb_points_by_chunk,
                                       generate_metadata=generate_metadata)
        mapped_ts_list.update(list_upsampled)

    # Generate output ts_list in the same order as input
    returned_ts_list = []
    for original_tsuid in tsuid_list:
        returned_ts_list.append({
            "tsuid": mapped_ts_list[original_tsuid]["tsuid"],
            "funcId": mapped_ts_list[original_tsuid]["funcId"]
        })

    return returned_ts_list


def downsampling_ts(ts_list,
                    resampling_period,
                    timestamp_position=TimestampPosition.BEG,
                    aggregation_method=AggregationMethod.AVG,
                    nb_points_by_chunk=50000,
                    generate_metadata=False):
    """
    Apply a downsampling to a list of timeseries and return a new ts list composed of dict {tsuid: "", funcId: ""}

    :param ts_list: list of TS to use to know corresponding functional identifier
    :param resampling_period: period used for resampling
    :param timestamp_position: position of timestamp generated in resampling period interval
    :param aggregation_method: function used to aggregate data within resampling period interval
    :param nb_points_by_chunk: user defined number of points used for a spark chunk of data (after resampling)
    :param generate_metadata: True to generate metadata on-the-fly (ikats_start_date, ikats_end_date, qual_nb_points)

    :type ts_list: list
    :type resampling_period: int
    :type timestamp_position : TimestampPosition or str or int
    :type aggregation_method: AggregationMethod or str or int
    :type nb_points_by_chunk: int
    :type generate_metadata: boolean (default : False)

    :return: dict of downsampled TS. Each composed of 2 items: resampled tsuid and its functional identifier.
            Key is original TS
    :rtype: dict
    """
    # Be sure this is int
    resampling_period = int(resampling_period)

    # Invalid resampling period
    if resampling_period == 0:
        LOGGER.error("Resampling period must be not null")
        raise ValueError("Resampling period must be not null")

    # check input values are in required enum types
    timestamp_position = _check_enum_input(timestamp_position, TimestampPosition)
    aggregation_method = _check_enum_input(aggregation_method, AggregationMethod)

    return _resample(resampling_way=ResamplingWay.DOWN_SAMPLING,
                     ts_list=ts_list,
                     resampling_period=resampling_period,
                     timestamp_position=timestamp_position,
                     aggregation_method=aggregation_method,
                     nb_points_by_chunk=nb_points_by_chunk,
                     generate_metadata=generate_metadata)


def upsampling_ts(ts_list,
                  resampling_period,
                  adding_method=AddingMethod.LINEAR_INTERPOLATION,
                  nb_points_by_chunk=50000,
                  generate_metadata=False):
    """
    Apply an upsampling to a list of timeseries and return a new ts list composed of dict {tsuid: "", funcId: ""}

    :param ts_list: list of TS to use to know corresponding functional identifier
    :param resampling_period: period used for resampling
    :param adding_method: Method to use for upsampling (see type AddingMethod for more information)
    :param nb_points_by_chunk: user defined number of points used for a spark chunk of data (after resampling)
    :param generate_metadata: True to generate metadata on-the-fly (ikats_start_date, ikats_end_date, qual_nb_points)

    :type ts_list: list
    :type resampling_period: int
    :type adding_method: AddingMethod or str or int
    :type nb_points_by_chunk: int
    :type generate_metadata: boolean

    :return: dict of upsampled TS. Each composed of 2 items: resampled tsuid and its functional identifier.
            Key is original TS
    :rtype: dict

    :raise ValueError: if Adding_method value is unknown
    :raise TypeError: if parameter has wrong type
    :raise ValueError: if no metadata found for a TS
    :raise ValueError: if no start date available for a tsuid
    :raise ValueError: if no end date available for a tsuid
    :raise ValueError: if no reference period available for a tsuid
    :raise IkatsException: if spark error occurred
    """

    # Be sure this is int
    resampling_period = int(resampling_period)

    # Invalid resampling period
    if resampling_period == 0:
        LOGGER.error("Resampling period must be not null")
        raise ValueError("Resampling period must be not null")

    # check input value is in required enum type
    adding_method = _check_enum_input(adding_method, AddingMethod)

    return _resample(resampling_way=ResamplingWay.UP_SAMPLING,
                     ts_list=ts_list,
                     resampling_period=resampling_period,
                     adding_method=adding_method,
                     nb_points_by_chunk=nb_points_by_chunk,
                     generate_metadata=generate_metadata)


def _resample(resampling_way,
              ts_list,
              resampling_period,
              adding_method=AddingMethod.LINEAR_INTERPOLATION,
              timestamp_position=TimestampPosition.BEG,
              aggregation_method=AggregationMethod.AVG,
              nb_points_by_chunk=50000,
              generate_metadata=False):
    """
    Function that effectively resamples (UP or DOWN according to resampling_way value) using Spark

    :param resampling_way: way of resampling (UP or DOWN)
    :type ts_list: ResamplingWay

    :param ts_list: list composing the TS information to resample [{'tsuid': xxx, 'funcId': yyy },...]
    :type ts_list: list of dict

    :param resampling_period: target period for resampling (in ms)
    :type resampling_period: int

    :param adding_method: Method to use for interpolation (see type AddingMethod for more information)
    :type adding_method: AddingMethod or str or int

    :param timestamp_position: timestamp position in the interval while downsampling
    :type timestamp_position: str ('BEG','MID','END')

    :param aggregation_method: aggregation method for downsampling
    :type aggregation_method: str ('MIN','MAX','MED','AVG','FIRST','LAST')

    :param nb_points_by_chunk: user defined number of points used for a spark chunk of data (after resampling)
    :type nb_points_by_chunk: int

    :param generate_metadata: True to generate metadata on-the-fly (ikats_start_date, ikats_end_date, qual_nb_points)
    :type generate_metadata: boolean (default : False)

    :returns: a list of dict [{'tsuid': xxx, 'funcId': yyy },...]
    :rtype: list of dict
    """

    if ts_list == []:
        return []

    fid_dict = dict()
    for ts in ts_list:
        fid_dict[ts['funcId']] = ts['tsuid']

    # List of chunks of data and associated information to parallelize with Spark
    data_to_compute = []

    # Extract tsuid list from inputs

    tsuid_list = [x["tsuid"] for x in ts_list]

    # Checking metadata availability before starting resampling
    meta_list = IkatsApi.md.read(tsuid_list)

    # Collecting information from metadata
    for tsuid in tsuid_list:
        if tsuid not in meta_list:
            LOGGER.error("Timeseries %s : no metadata found in base", tsuid)
            raise ValueError("No ikats metadata available for resampling %s" % tsuid)
        if 'ikats_start_date' not in meta_list[tsuid]:
            # Metadata not found
            LOGGER.error("Metadata 'ikats_start_date' for timeseries %s not found in base", tsuid)
            raise ValueError("No start date available for resampling [%s]" % tsuid)
        if 'ikats_end_date' not in meta_list[tsuid]:
            # Metadata not found
            LOGGER.error("meta data 'ikats_end_date' for timeseries %s not found in base", tsuid)
            raise ValueError("No end date available for resampling [%s]" % tsuid)
        if 'qual_ref_period' not in meta_list[tsuid]:
            # Metadata not found
            LOGGER.error("Metadata qual_ref_period' for timeseries %s not found in base", tsuid)
            raise ValueError("No reference period available for resampling [%s]" % tsuid)

        # Original timeseries information retrieved from metadata
        sd = int(meta_list[tsuid]['ikats_start_date'])
        ed = int(meta_list[tsuid]['ikats_end_date'])
        ref_period = int(float(meta_list[tsuid]['qual_ref_period']))

        # Get the functional identifier of the original timeseries
        fid_origin = [x['funcId'] for x in ts_list if x['tsuid'] == tsuid][0]

        # Generate functional id for resulting timeseries
        if resampling_way == ResamplingWay.UP_SAMPLING:
            func_id = "%s_resampled_to_%sms_%s" % (fid_origin, str(resampling_period), str(adding_method))
        else:
            func_id = "%s_resampled_to_%sms_%s_%s" % (
                fid_origin, str(resampling_period), timestamp_position, aggregation_method)

        # Creating new reference in database for new timeseries
        IkatsApi.ts.create_ref(func_id)

        # Prepare data to compute by defining intervals of final size nb_points_by_chunk
        # Chunk intervals computation :

        # Computing elementary size which is the lowest common multiple between ref period and resampling period
        elementary_size = _lowest_common_multiple(ref_period, resampling_period)

        # Seeking the number of elementary size which contains nb of points nearest to nb_points_by_chunk parameter
        # in order to compute the final data chunk size
        nb_points_for_elementary_size = int(elementary_size / resampling_period)
        data_chunk_size = int(nb_points_by_chunk / nb_points_for_elementary_size) * elementary_size

        # Limit the size of data_chunk_size
        if data_chunk_size < elementary_size:
            data_chunk_size = elementary_size

        # Computing intervals for chunk definition
        interval_limits = np.hstack((np.arange(sd,
                                               ed,
                                               data_chunk_size,
                                               dtype=np.int64), ed))

        # from intervals we define chunk of data to compute
        # ex : intervals = [ 1, 2, 3] => 2 chunks [1, 2] and [2, 3]
        if len(interval_limits) > 2:
            # there is more than 2 limits for interval definition, i.e there is more than one chunk to compute
            data_to_compute.extend([(tsuid,
                                     func_id,
                                     i,
                                     interval_limits[i],
                                     interval_limits[i + 1]) for i in range(len(interval_limits) - 1)])
        elif len(interval_limits) > 1:
            # only one chunk to compute
            data_to_compute.append((tsuid, func_id, 0, interval_limits[0], interval_limits[1]))

        # in case last original point and last downsampled point are aligned => add a supplementary chunk to compute
        # last point
        if (interval_limits[-1] - sd) % resampling_period == 0:
            data_to_compute.append((tsuid, func_id, 1, interval_limits[-1], interval_limits[-1] + resampling_period))

    LOGGER.info("Running resampling using Spark")
    # Create or get a spark Context
    spark_context = ScManager.get()

    if resampling_way == ResamplingWay.UP_SAMPLING:
        spark_function = _spark_upsample
        args = adding_method
    else:
        spark_function = _spark_downsample
        args = (timestamp_position, aggregation_method)

    try:

        # OUTPUT : [(TSUID_origin, func_id, chunk_index, sd_interval, ed_interval), ...]
        inputs = spark_context.parallelize(data_to_compute, len(data_to_compute))

        # INPUT :  [(TSUID_origin, func_id, chunk_index, sd_interval, ed_interval), ...]
        # OUTPUT : [((TSUID_origin, func_id), chunk_index, original_data_array), ...]
        # PROCESS : read original data in database / filter chunk with no data
        rdd_data_with_chunk_index = inputs \
            .map(lambda x: ((x[0], x[1]), x[2], IkatsApi.ts.read(tsuid_list=x[0], sd=int(x[3]), ed=int(x[4]))[0])) \
            .filter(lambda x: len(x[2]) > 0)

        if resampling_way == ResamplingWay.UP_SAMPLING:
            # INPUT :  [((TSUID_origin, func_id), chunk_index, original_data_array), ...]
            # OUTPUT : [((TSUID_origin, func_id), original_data_array_with_inter_chunks), ...]
            # PROCESS : compute inter-chunks intervals / filter empty chunks
            rdd_data = _calc_inter_chunks(rdd=rdd_data_with_chunk_index) \
                .map(lambda x: (x[0], x[2])) \
                .filter(lambda x: not (len(x[1]) == 2 and (int(float(x[1][0][0])) == int(float(x[1][1][0])))))
        else:
            # INPUT :  [((TSUID_origin, func_id), chunk_index, original_data_array), ...]
            # OUTPUT : [((TSUID_origin, func_id), original_data_array), ...]
            # PROCESS : suppress useless chunk indexes
            rdd_data = rdd_data_with_chunk_index.map(lambda x: (x[0], x[2]))

        # INPUT :  [((TSUID_origin, func_id), original_data_array_with_inter_chunks), ...]
        # OUTPUT : [((TSUID_origin, func_id), data_resampled_array), ...]
        # PROCESS : resample chunks of data to resampling_period
        rdd_resampled_data = rdd_data.map(
            lambda x: (x[0], spark_function(data=x[1], period=resampling_period, args=args))) \
            .filter(lambda x: len(x[1]) > 0)

        # INPUT :  [((TSUID_origin, func_id), data_resampled_array), ...]
        # OUTPUT : [(TSUID_origin, func_id, TSUID, sd, ed), ...]
        # PROCESS : create resampled data in database / compute global start and end date
        identifiers = rdd_resampled_data \
            .map(lambda x: (x[0][0], x[0][1], _spark_import(fid=x[0][1],
                                                            data=x[1],
                                                            generate_metadata=generate_metadata))) \
            .map(lambda x: ((x[0], x[1], x[2][0]), (x[2][1], x[2][2]))) \
            .reduceByKey(lambda x, y: (min(x[0], y[0]), max(x[1], y[1]))) \
            .map(lambda x: (x[0][0], x[0][1], x[0][2], x[1][0], x[1][1])) \
            .collect()

    except Exception as err:
        msg = "Exception raised while resampling with Spark: %s " % err
        LOGGER.error(msg)
        raise IkatsException(msg)

    finally:
        # Stop spark Context
        ScManager.stop()  # Post-processing : metadata import and return dict building

    # returns dict containing the results of the resampling
    # where key is the original TSUID and values are resampled TSUID and functional identifiers
    returned_dict = {}
    for timeseries in identifiers:
        tsuid_origin = timeseries[0]
        func_id = timeseries[1]
        tsuid = timeseries[2]
        sd = timeseries[3]
        ed = timeseries[4]

        # Import metadata in non temporal database
        _save_metadata(tsuid=tsuid, md_name='qual_ref_period', md_value=resampling_period, data_type=DTYPE.number,
                       force_update=True)
        _save_metadata(tsuid=tsuid, md_name='ikats_start_date', md_value=sd, data_type=DTYPE.date, force_update=True)
        _save_metadata(tsuid=tsuid, md_name='ikats_end_date', md_value=ed, data_type=DTYPE.date, force_update=True)

        # Retrieve imported number of points from database
        qual_nb_points = IkatsApi.ts.nb_points(tsuid=tsuid)
        IkatsApi.md.create(tsuid=tsuid, name='qual_nb_points', value=qual_nb_points, data_type=DTYPE.number,
                           force_update=True)

        # Inherit from parent
        IkatsApi.ts.inherit(tsuid, tsuid_origin)

        # Fill returned list
        returned_dict[tsuid_origin] = {"tsuid": tsuid, 'funcId': func_id}

    return returned_dict


def _spark_import(fid, data, generate_metadata):
    """
    Create chunks of data in temporal database

    :param fid: functional identifier
    :param data: data to store in db
    :param generate_metadata: True to generate metadata on the fly while creating data in db

    :type fid: str
    :type data: numpy array
    :type generate_metadata: boolean

    :return: identifier of ts created, start date of chunk, end date of chunk
    :rtype: tuple (tsuid, sd, ed)
    """
    tsuid = IkatsApi.ts.create(fid=fid,
                               data=data,
                               generate_metadata=generate_metadata,
                               sparkified=True)['tsuid']
    start_date = data[0][0]
    end_date = data[-1][0]

    return tsuid, start_date, end_date


def _save_metadata(tsuid, md_name, md_value, data_type, force_update):
    """
    Saves metadata to Ikats database and log potential errors

    :param tsuid: TSUID to link metadata with
    :param md_name: name of the metadata to save
    :param md_value: value of the metadata
    :param data_type: type of the metadata
    :param force_update: overwrite metadata value if exists (if True)

    :type tsuid: str
    :type md_name: str
    :type md_value: str or int or float
    :type data_type: DTYPE
    :type force_update: bool
    """
    if not IkatsApi.md.create(
            tsuid=tsuid,
            name=md_name,
            value=md_value,
            data_type=data_type,
            force_update=force_update):
        LOGGER.error("Metadata '%s' couldn't be saved for TS %s", md_name, tsuid)


def _lowest_common_multiple(first_lcm, second_lcm):
    """
    Return lowest common multiple.

    :param first_lcm: first number
    :param second_lcm: second number

    :type first_lcm: int
    :type second_lcm: int

    :return: lowest common multiple.
    :rtype: int
    """

    def greatest_common_divisor(first_gcd, second_gcd):
        """
        Return greatest common divisor using Euclid's Algorithm.

        :param first_gcd: 1st number
        :param second_gcd: 2nd number

        :type first_gcd: int
        :type second_gcd: int

        :return: greatest common divisor
        :rtype: int
        """
        while second_gcd:
            first_gcd, second_gcd = second_gcd, first_gcd % second_gcd
        return first_gcd

    return first_lcm * second_lcm // greatest_common_divisor(first_lcm, second_lcm)


def _spark_upsample(data, period, args):
    """
    Data upsampling within a data range

    Warning : first point of data range must be on the original timeseries AND on the resampled one (common point)

    :param data: data to resample (each item contains timestamp and point value)
    :param period: target period for resampling
    :param args : adding point method (VALUE_BEFORE, LINEAR_INTERPOLATION, VALUE_AFTER)

    :type data: numpy array
    :type period: int
    :type args: str

    :return: the upsampled list of points
    """

    adding_method = args

    # No computation if data variable contains no elements to resample
    if len(data) == 0:
        return []

    # First extract timestamps and values from original data
    ts_timestamps = data[:, 0].astype(int, copy=False)
    ts_val = data[:, 1].astype(np.float64, copy=False)

    # Then compute new timestamps vector from original one
    interpolated_timestamps = np.arange(ts_timestamps[0], ts_timestamps[-1] + 1, period)
    LOGGER.debug("Current size=%s, Interpolated size=%s", len(ts_timestamps), len(interpolated_timestamps))

    if adding_method == "LINEAR_INTERPOLATION":
        # In case of linear interpolation use of integrated numpy function to compute new values at once
        new_values = np.interp(interpolated_timestamps, ts_timestamps, ts_val)
        # Then new values are aggregated to new timestamps in resampled data
        resampled_data = np.array([interpolated_timestamps, new_values], dtype=object).T
    else:
        # adding_method is VALUE_BEFORE or VALUE_AFTER

        # Initializing resampled data result array with new timestamps and empty array for new values
        resampled_data = np.array([interpolated_timestamps, np.empty(shape=(len(interpolated_timestamps)))],
                                  dtype=object).T

        # Iterator on original data
        original_data_it = iter(data)

        # Initialization, fill first point of resampled data (value)
        original_point = next(original_data_it)
        resampled_data[0][1] = original_point[1]

        # Store first point value (used only in case VALUE_BEFORE)
        value_before = original_point[1]

        # Iterate over original data
        original_point = next(original_data_it)

        # Boolean used to identify if currently computed data point match exactly to an original data point
        exactly_reached = False

        # Iterate over pre-filled resampled data (without first point already processed)
        for resampled_point in resampled_data[1:]:
            if int(resampled_point[0]) == int(original_point[0]):
                # Next point exactly reached
                exactly_reached = True
            if int(resampled_point[0]) >= int(original_point[0]):
                # Next point reached or exceeded
                # Update value_before
                value_before = original_point[1]
                try:
                    # Update (next) original data point
                    original_point = next(original_data_it)
                except StopIteration:
                    # Last original data point reached => update last resampled point and exit for loop
                    resampled_point[1] = original_point[1]
                    break

            # Filling data with previous/next original data point value
            if exactly_reached is True:
                # Fill with precedent value
                resampled_point[1] = value_before
                # Reinitialize flag
                exactly_reached = False
            elif adding_method == "VALUE_BEFORE":
                # Fill with precedent value
                resampled_point[1] = value_before
            elif adding_method == "VALUE_AFTER":
                # Fill with following value
                resampled_point[1] = original_point[1]

    return resampled_data


def _spark_downsample(data, period, args):
    """
    Data downsampling within a data range

    :param data: data to resample (each item contains timestamp and point value)
    :param period: target period for resampling
    :param args : (timestamp_position, aggregation_method)
        - timestamp_position : position of timestamp generated in resampling period interval
        - aggregation_method : function used to aggregate data within resampling period interval

    :type data: numpy array
    :type period: int
    :type args: tuple (str, str)

    :return: the downsampled list of points
    """

    # parameter reallocation
    timestamp_position, aggregation_method = args

    # No computation if data variable contains no element or only one element
    if len(data) <= 1:
        return data

    # First extract timestamps and values from original data
    ts_timestamps = data[:, 0].astype(int, copy=False)
    ts_val = data[:, 1].astype(np.float64, copy=False)

    # computing timestamp offset according to timestamp_position required
    offset_timestamp_position = period // 2 * TimestampPosition[timestamp_position].value

    # computing interpolated timestamps within data range
    interpolated_timestamps = np.arange(ts_timestamps[0], ts_timestamps[-1] + 1, period)

    LOGGER.debug("Current size=%s, Interpolated size=%s", len(ts_timestamps), len(interpolated_timestamps))

    # Initializing resampled data result array with new timestamps and empty array for new values
    resampled_data = np.array([interpolated_timestamps + offset_timestamp_position,
                               np.empty(shape=(len(interpolated_timestamps)))],
                              dtype=object).T

    # init resampled data values to None
    resampled_data[:, 1:] = None

    # iterator on original data timestamps (index, value)
    iterator = np.ndenumerate(ts_timestamps)

    # initializing iterating values
    index_begin_interval, value = iterator.next()

    # init structure to store indexes of real data
    list_indexes_data = []

    # boolean to identify a StopIteration exception
    end_iteration = False

    # Iterate over interpolated_timestamps (without first timestamp)
    for index_resampled_data, timestamp in enumerate(interpolated_timestamps):

        # seeking index of end interval
        while value < timestamp + period:
            try:
                index_end_interval, value = iterator.next()
            except StopIteration:
                end_iteration = True
                break

        # extract data within resampled period
        if end_iteration:
            if value != timestamp:
                # extraction of whole end of interval
                data_extracted = ts_val[index_begin_interval[0]:]
            else:
                # last point of chunk processed and
                # same timestamp for original and resampled data => ignored
                data_extracted = []
            end_iteration = False
        else:
            # extraction of data considered for aggregation
            data_extracted = ts_val[index_begin_interval[0]:index_end_interval[0]]
            index_begin_interval = index_end_interval

        if len(data_extracted) != 0:
            # applying aggregation method on extracted data
            if aggregation_method == "MIN":
                resampled_data[index_resampled_data][1] = min(data_extracted)
            elif aggregation_method == "MAX":
                resampled_data[index_resampled_data][1] = max(data_extracted)
            elif aggregation_method == "MED":
                resampled_data[index_resampled_data][1] = median(data_extracted)
            elif aggregation_method == "AVG":
                resampled_data[index_resampled_data][1] = mean(data_extracted)
            elif aggregation_method == "FIRST":
                resampled_data[index_resampled_data][1] = data_extracted[0]
            elif aggregation_method == "LAST":
                resampled_data[index_resampled_data][1] = data_extracted[-1]

            # store indexes with real data
            list_indexes_data.append(index_resampled_data)

        index_resampled_data += 1

    return resampled_data[list_indexes_data]


def _spark_filter_successive_chunks(chunk_1, chunk_2):
    """
    Return the chunk having the highest 'chunk_index' between chunk X and Y

    inputs are a tuple composed of (current chunk index, last point of current chunk, first point of previous chunk)

    :param chunk_1: first chunk to compare
    :type chunk_1: tuple

    :param chunk_2: second chunk to compare
    :type chunk_2: tuple

    :return: the chunk with highest chunk_index
    :rtype: tuple
    """
    if max(chunk_1[0], chunk_2[0]) == chunk_1[0]:
        return chunk_1
    return chunk_2


def _calc_inter_chunks(rdd):
    """
    Calculate limits interval between chunks of shape
    (last_point_of_precedent_chunk_with_data, first_point_of_following_chunk_with_data).
    Handle the case of chunk with no data.
    Result is the fusion between new inter-chunks data and original chunk data.

    Example:
    * Chunk A last point at T1
    * Chunk B has no point
    * Chunk C has first point at T2
    --> An entry will be written into RDD with (Tsinfo, chunk_index_C, np.array([T1, T2]))

    Format returned [(TSInfo, chunk_index, data), ...]

    :return: new RDD containing all the intervals (even the ones between chunks)
    :rtype: RDD
    """

    # FROM RDD containing timestamps only
    # BUILD new RDD containing chunk limits information
    # OUTPUT FORMAT [(TSInfo, (chunk_index, first_point, last_point)), ...]
    rdd_inter_chunks_to_find = rdd \
        .map(lambda x: (x[0], (x[1], x[2][0], x[2][-1])))

    # FROM RDD containing chunk limits information
    # BUILD new RDD containing combination of all chunk limit information duet
    #       assuming there is 2 "chunk limit information",
    #       we will obtain: [
    #           (TSInfo,(chunk_info_A, chunk_info_A)),
    #           (TSInfo,(chunk_info_A, chunk_info_B)),
    #           (TSInfo,(chunk_info_B, chunk_info_A)),
    #           (TSInfo,(chunk_info_B, chunk_info_B)),
    #       ]
    # OUTPUT FORMAT [
    #    (TSInfo, ((chunkA_index, first_pointA, last_pointA),
    #              (chunkB_index, first_pointB, last_pointB))
    #    ),
    #  ...]
    rdd_inter_chunks_1 = rdd_inter_chunks_to_find.join(rdd_inter_chunks_to_find)

    # FROM RDD containing combination of all chunk limit information duet
    # BUILD new RDD containing filtered chunk limits information (when chunkA_index > chunkB_index)
    # OUTPUT FORMAT [
    #    (TSInfo, ((chunkA_index, first_pointA, last_pointA),
    #              (chunkB_index, first_pointB, last_pointB))
    #    ),
    #  ...]
    rdd_inter_chunks_2 = rdd_inter_chunks_1.filter(lambda x: x[1][0][0] > x[1][1][0])

    # FROM RDD containing filtered chunk limits information (when chunkA_index > chunkB_index)
    # BUILD new RDD containing minimum chunk limits information
    # OUTPUT FORMAT [
    #    ((TSInfo, chunkA_index), (chunkB_index, last_pointB, first_pointA)),
    #  ...]
    rdd_inter_chunks_3 = rdd_inter_chunks_2.map(lambda x: ((x[0], x[1][0][0]), (x[1][1][0], x[1][0][1], x[1][1][2])))

    # FROM RDD containing minimum chunk limits information
    # BUILD new RDD containing chunk limits information, where chunkB is the chunk having data just before chunkA
    # OUTPUT FORMAT [
    #    ((TSInfo, chunkA_index), (chunkB_index, last_pointB, first_pointA)),
    #  ...]
    rdd_inter_chunks_4 = rdd_inter_chunks_3.reduceByKey(lambda x, y: _spark_filter_successive_chunks(x, y))

    # FROM RDD containing chunk limits information
    # BUILD new RDD containing chunk limits information, where chunkB is the chunk having data just before chunkA
    # OUTPUT FORMAT [
    #    ((TSInfo, chunkA_index, [last_pointB, first_pointA] )),
    #  ...]
    rdd_inter_chunks = rdd_inter_chunks_4.map(lambda x: (x[0][0], x[0][1], np.array([x[1][2], x[1][1]])))

    # Combining chunks (with data) and inter-chunks intervals to obtain all intervals
    # [(TSInfo, chunk_index, data), ...]
    return rdd + rdd_inter_chunks


def _check_enum_input(x_value, x_type):
    """
    Check that input value matches given enum type
    if not, error is raised

    :param x_value: value to check
    :param x_type: type checked

    :type x_value: unknown
    :type x_type: IntEnum, enumerate

    :return: the value as a string
    :rtype: str

    :raises ValueError: if no match
    """
    if type(x_value) is x_type:
        x_value = x_value.name
    elif type(x_value) is int:
        if x_value not in [e.value for e in x_type]:
            raise ValueError("%s (integer) must be in the following range : %s" % (x_value, [e.value for e in x_type]))
        else:
            x_value = x_type(x_value).name
    elif type(x_value) is str:
        if x_value not in [name for name, _ in x_type.__members__.items()]:
            raise ValueError(
                "%s (string) must be in the following values list : %s"
                % (x_value, [name for name, member in x_type.__members__.items()]))
        else:
            x_value = x_type[x_value].name
    else:
        raise TypeError("ERROR : %s must be a string or an integer or an %s" % (x_value, x_type))

    return x_value
