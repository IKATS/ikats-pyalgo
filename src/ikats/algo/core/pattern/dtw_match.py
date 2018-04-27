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
import collections
from collections import defaultdict
import logging
from math import ceil, floor
import time
import traceback

import itertools
from numpy import array as np_array
from numpy import concatenate as np_concatenate
from numpy import empty as np_empty_array
from scipy.spatial import distance as sci_distance
from ikats.algo.core.pattern.normalize import scale as func_scale
from fastdtw import fastdtw, dtw

from ikats.core.library.exception import IkatsException
from ikats.core.library.spark import ScManager
from ikats.core.resource.interface import ResourceLocator
from ikats.core.resource.api import IkatsApi

LOGGER = logging.getLogger(__name__)

ERROR_IMPORT_MSG = "Failed to import {}: pattern_matching on {} with pattern={}"

# CONFIG_NORM_MODES is a dict whose key is coding the selected normalizing mode
# (the mode selected by the user concerning the normalizing process):
# CONFIG_NORM_MODES[x][0]: MeanPattern is True when user demands to adjust the mean to zero on the selected pattern
# CONFIG_NORM_MODES[x][1]: ScalePattern is True when user demands to scale the variance to unit on the selected pattern
# CONFIG_NORM_MODES[x][2]: MeanTarget is True when user demands to adjust the mean to zero on the search ts
# CONFIG_NORM_MODES[x][3]: ScaleTarget is True when user demands to scale the variance to unit on the search ts
#
N_NO_NORM = "NO_NORM"
N_MEAN_VARIANCE = "MEAN_VARIANCE"
N_MEAN_VARIANCE_ONLY_SEARCH = "MEAN_VARIANCE_ONLY_SEARCH"
N_MEAN = "MEAN"
N_MEAN_ONLY_SEARCH = "MEAN_ONLY_SEARCH"
N_VARIANCE = "VARIANCE"
N_VARIANCE_ONLY_SEARCH = "VARIANCE_ONLY_SEARCH"

# flags are:       [MeanPattern, ScalePattern, MeanTarget, ScaleTarget]
CONFIG_NORM_MODES = {N_NO_NORM: [False, False, False, False],
                     N_MEAN_VARIANCE: [True, True, True, True],
                     N_MEAN_VARIANCE_ONLY_SEARCH: [False, False, True, True],
                     N_MEAN: [True, False, True, False],
                     N_MEAN_ONLY_SEARCH: [False, False, True, False],
                     N_VARIANCE: [False, True, False, True],
                     N_VARIANCE_ONLY_SEARCH: [False, False, False, True]}


def _score_dtw(np_pattern_values, np_values, normalize_mode, fast_dtw):
    """
    Computes the score and matching index interval with the fastdtw/dtw distance from library fastdtw applied
    to np_values vs np_pattern_values.

    .. topic:: Requirement

       len( np_pattern_values ) <=  len(np_values)
       Note: in __init_local_superwindows

       Otherwise: exception is raised

    :param np_pattern_values: the selected pattern
    :type np_pattern_values: numpy.array
    :param np_values: the values compared: window selection from the original search target TS
    :type np_values: numpy.array
    :param normalize_mode: selected normalizing mode
    :type normalize_mode: str
    :param fast_dtw: True selects the distance fastdtw.fastdtw; otherwise: selects fastdtw.dtw
    :type fast_dtw: bool
    :return: ( <score>, <the matching index interval> ):

       * <score> is a float: the distance computed
       * <the matching index interval> is the pair of relevant np_values indexes [ <start index>, <last index> ]:

        the relevant indexes are determined by the path processed by fastdtw:
          * ignore first points until last occurrence of first pattern index
          * ignore (N-1) last points aligned to the last pattern index
    :rtype: tuple
    """
    # Gets the configuration selected by the user: defines how to scale the np_values
    config_norm = CONFIG_NORM_MODES.get(normalize_mode, CONFIG_NORM_MODES[N_NO_NORM])
    len_pattern = len(np_pattern_values)
    len_search_ts = len(np_values)

    if len_pattern > len_search_ts:
        msg = "Error: bad usage: len( np_pattern_values ) > len( np_values ): {} > {}"
        raise Exception(msg.format(len_pattern, len_search_ts))

    # Normalize the np_values if
    # - the mean normalizing is demanded on the ts search <=>  config_norm[2] == True
    # - or if the variance normalizing is demanded on the ts search <=>  config_norm[3] == True
    if config_norm[2] or config_norm[3]:
        norm_ts_search = func_scale(data=np_values,
                                    with_mean=config_norm[2],
                                    with_std=config_norm[3])
    else:
        norm_ts_search = np_values

    if fast_dtw is True:
        # To be confirmed:
        fdtw_radius = ConfigPatternMatching.fdtw_radius
        score, path = fastdtw(norm_ts_search, np_pattern_values, fdtw_radius)
    else:
        score, path = dtw(norm_ts_search, np_pattern_values)

    # matching_index_interval: [ <index start>, <index end> ]
    #
    # cancelled: indexes of the points not relevant <=> "not usefully aligned"
    #  - at the beginning,
    #  - and at the end
    #
    # (note: with this example: the pattern is sized 101, the ts window is sized 111)
    #
    # - ignore (M-1) first points aligned to the first pattern index
    #   - because these are interpreted as  "not usefully aligned"
    #   ex: (0,0), (1,0),... (7,0), (8,0), (9,1), ...
    #       => here: cancel points matched by indexes 0 to 7
    #                keep <index start> == 8
    #
    # - ignore (N-1) last points aligned to the last pattern index
    #   - because these are interpreted as "not usefully aligned"
    #   ex: ..., (105,99), (106,100), (107,100), (108,100),..., (110,100)
    #         here: last pattern index is 100
    #         => cancel points matched by indexes 107 to 110 of norm_ts_search,
    #         => keep <index end> == 106
    #
    len_path = len(path)
    if len_path <= 1:
        raise IkatsException("Unexpected path for fastdtw: len(path)={}".format(len_path))

    pattern_index_start = path[0][1]
    pattern_index_last = path[-1][1]

    # "sum(1 for _ in iter)"  is computing the number of elements in iterator iter
    #
    num_cst_begin = sum(1 for _ in itertools.takewhile(lambda x: x[1] == pattern_index_start, path))

    num_cst_last = sum(1 for _ in itertools.takewhile(lambda x: x[1] == pattern_index_last, path[-1::-1]))

    matching_index_interval = [num_cst_begin - 1, len_search_ts - num_cst_last]

    return score, matching_index_interval


def score_manhattan(np_pattern_values, np_values, normalize_mode):
    """
    Computes the score and matching index interval with the Manhattan distance.
    This scoring is computing one distance, it is useless to have arrays
    with different size.

    .. topic:: Requirement

       len( np_pattern_values ) <=  len(np_values)

       Otherwise: exception is raised

    .. topic:: Assumed

       With this scoring: Manhattan is evaluated once, aligned on the left point: the matching index interval is
       always: [ 0, len( np_pattern_values ) - 1 ]

    :param np_pattern_values: the selected pattern
    :type np_pattern_values: numpy.array
    :param np_values: the search TS
    :type np_values: numpy.array
    :param normalize_mode: selected normalizing mode
    :type normalize_mode: str
    :return: ( <score>, <the matching index interval> )
      * <score> is a float
      * <the matching index interval> is always [ 0, len( np_pattern_values ) ]
    :rtype: tuple
    """

    # Gets the configuration selected by the user: defines how to scale the np_values
    config_norm = CONFIG_NORM_MODES.get(normalize_mode, CONFIG_NORM_MODES[N_NO_NORM])
    len_pattern = len(np_pattern_values)
    len_search_ts = len(np_values)

    # matching_index_interval: [ <index start>, <index end> ]
    # this information is useful in case of scoring with DTW distance
    # => kept for compatibility
    # Assumed with Manhattan and Minkowski: vectors should have same size:
    # if it is not the case: this function behaves as follow:
    #    * when ( len_pattern >  len_search_ts )
    #        => raise error
    #    * otherwise: ( len_pattern < len_search_ts )
    #        =>  limited comparison on the first points from index 0 to len_pattern-1
    #
    if len_pattern > len_search_ts:
        msg = "Error: forbidden: len( np_pattern_values ) > len( np_values ): {} > {}"
        raise Exception(msg.format(len_pattern, len_search_ts))
    elif len_pattern == len_search_ts:
        compared_values = np_values
    else:
        compared_values = np_values[0:len_pattern]

    # Normalize the np_values if
    # - the mean normalizing is demanded on the ts search <=>  config_norm[1] == True
    # - or if the variance normalizing is demanded on the ts search <=>  config_norm[3] == True
    #
    # Note: as said above: limit comparison on the first points from index 0 to len_pattern-1
    if config_norm[2] or config_norm[3]:

        norm_ts_search = func_scale(data=compared_values,
                                    with_mean=config_norm[2],
                                    with_std=config_norm[3])
    else:
        norm_ts_search = compared_values

    # Manhattan distance is minkowski distance with degree 1
    score = sci_distance.minkowski(u=np_pattern_values, v=norm_ts_search, p=1)

    matching_index_interval = [0, len_pattern - 1]
    return score, matching_index_interval


def score_fastdtw(np_pattern_values,
                  np_values,
                  normalize_mode):
    """
    Calls the _score_dtw(np_pattern_values, np_values, normalize_mode, True) coding for fast dtw
    implementation.
    :param np_pattern_values: see np_pattern_values in documentation _score_dtw
    :type np_pattern_values: numpy.array
    :param np_values: see np_values in documentation _score_dtw
    :type np_values: numpy.array
    :param normalize_mode: see normalize_mode in documentation _score_dtw
    :type normalize_mode: str
    :return: _score_dtw(np_pattern_values, np_values, normalize_mode, True)
    :rtype: tuple
    """
    return _score_dtw(np_pattern_values, np_values, normalize_mode, True)


def score_dtw(np_pattern_values, np_values, normalize_mode):
    """
    Calls the _score_dtw(np_pattern_values, np_values, normalize_mode, False) coding for dtw
    implementation.
    :param np_pattern_values: see np_pattern_values in documentation _score_dtw
    :type np_pattern_values: numpy.array
    :param np_values: see np_values in documentation _score_dtw
    :type np_values: numpy.array
    :param normalize_mode: see normalize_mode in documentation _score_dtw
    :type normalize_mode: str
    :return: _score_dtw(np_pattern_values, np_values, normalize_mode, False)
    :rtype: tuple
    """
    return _score_dtw(np_pattern_values, np_values, normalize_mode, False)


# Once score_dtw and score_manhattan are defined:
# => define SCORING_FUNCTIONS configuration
#
FAST_DTW_DISTANCE = "FAST_DTW"
DTW_DISTANCE = "DTW"
MANHATTAN_DISTANCE = "MANHATTAN"

SCORING_FUNCTIONS = {FAST_DTW_DISTANCE: score_fastdtw,
                     DTW_DISTANCE: score_dtw,
                     MANHATTAN_DISTANCE: score_manhattan}


class ConfigPatternMatching(object):
    """
    The default configuration of the algorithm processed by find_pattern() is grouped under the class
    ConfigPatternMatching, so that it is programmatically possible to change some properties.

    Note: the ConfigPatternMatching is read by find_pattern(), on the driver node, before distribution.
    """

    # the maximum size of extracted TS chunk, using the TemporalDataManager extract_ts service
    # Recommended usage with TemporalDataManager: at most 100000 points
    max_size_extract = 100000

    # the maximum size of the working space called 'superwindow', assigned to a spark task
    max_size_superwindow = 1 * max_size_extract

    # the maximum size of TS defining the pattern
    max_pattern_size = 5000

    # the  minimum value for the ratio: (size TS search) / (size TS pattern * rate_size_window )
    min_ratio_search_pattern = 5.0

    # Change this flag in order to call the _collect_for_debug
    #   on the RDD resulting from __mapped_extract
    debug_extract_info = False

    # Change this flag in order to call the _collect_for_debug
    #   on the RDD resulting from __mapped_pattern_matching
    debug_pattern_matching_info = False

    # Change this flag in order to call the _collect_for_debug
    #  on the RDD resulting from __mapped_merge_frontier_windows
    debug_extract_shifted_info = False

    # Change this ratio defined for fastdtw.fastdtw used by score_fastdtw
    #
    # fdtw_radius
    fdtw_radius = 2

    # Maximum count of superwindows
    #
    #  Assuming by default: nb executors = 4 and nb cores is 6
    #  ( this is the case with clusters 2016 )
    #
    # Note: this config may be optimized later => we can have several executors per node
    # ...
    EXECUTORS = 9
    CORES_PER_WORKER = 4
    preferred_count_tasks = EXECUTORS * CORES_PER_WORKER

    @classmethod
    def assert_pattern_fidelity(cls, ref_pattern, normalize_mode):
        """
        Load the points in the pattern and do some sanity checking
        :param ref_pattern:
        :type ref_pattern str
        :param normalize_mode:
        :type normalize_mode int
        :return:
        """

        # Get function id from tsuid
        # if throws exception assume it was the fid that was passed
        try:
            ref_pattern = IkatsApi.fid.tsuid(ref_pattern)
        except Exception:
            pass

        # we need to get the pattern herr
        ts_pattern, _, _, nb_points = ConfigPatternMatching.__read_ts_pattern(ref_pattern, normalize_mode)

        if nb_points > cls.max_pattern_size:
            msg = "Bad usage: pattern-size > ConfigPatternMatching.config_max_pattern_size : {} > {}"
            raise Exception(msg.format(nb_points, cls.max_pattern_size))

        return nb_points, ts_pattern

    @classmethod
    def assert_ts_fidelity(cls, ref_search_location,
                           rate_size_window,
                           nb_points_pattern):
        """
        Similar to assert_pattern_fidelity
        Load the points in a ts and do some sanity checking
        Not fully used here but in a future effort where feedback
        can be given by the HMI could be used to pre-check ts without
        having to run all them
        :param ref_search_location:
        :type ref_search_location: str
        :param rate_size_window:
        :type rate_size_window: float
        :param nb_points_pattern:
        :type nb_points_pattern: int
        :return:
        """

        dict_result = IkatsApi.md.read([ref_search_location])

        start_date_search = float(dict_result[ref_search_location]['ikats_start_date'])
        end_date_search = float(dict_result[ref_search_location]['ikats_end_date'])
        nb_points_search = float(dict_result[ref_search_location]['qual_nb_points'])

        size_sliding_window = float(ceil(rate_size_window * nb_points_pattern))
        actual_ratio = nb_points_search / size_sliding_window

        if actual_ratio < cls.min_ratio_search_pattern:
            msg = "Bad usage: ( nb_points_search / size_sliding_window ) < config_min_ratio_search_pattern: {} < {}"
            raise IkatsException(msg.format(actual_ratio, ConfigPatternMatching.min_ratio_search_pattern))

        # preferred size: depends on cluster configuration
        # - see  ConfigPatternMatching.config_max_count_superwindows
        #
        # Note: preferred size is minimized with floor()
        #       => maximize the superwindow_duration variable below
        #   - selon le critere de limitation de nombre de taches: config_max_count_tasks
        #   - selon le criere de limitation de memoire => config_max_points_in_superwindow
        #
        swin_size_pref_by_nb_tasks = float(
            floor(nb_points_search / ConfigPatternMatching.preferred_count_tasks))

        # Choose the size as a trade-off between task limitation / memory limitation
        # => selects the minimum size ...
        superwin_size_preferred = min(swin_size_pref_by_nb_tasks,
                                      ConfigPatternMatching.max_size_superwindow)

        # actual size of superwindow:
        # - at least : superwindow_size >= size_sliding_window
        #              because a task operating on the superwindow may at least evaluate once the pattern
        #
        superwindow_size = max(superwin_size_preferred, size_sliding_window)

        # Duration : the period of superwindow assuming that the TS point have uniform frequency.
        # we add 1 to the duration to avoid side-effects
        superwindow_duration = 1 + int(
            float(end_date_search - start_date_search) * superwindow_size / nb_points_search)

        number_extractions = ceil(nb_points_search / ConfigPatternMatching.max_size_extract)
        my_extract_period = (end_date_search - start_date_search) / number_extractions

        start_dates = [start_date_search + i * superwindow_duration for i in
                       range(ceil(nb_points_search / superwindow_duration))]
        end_dates = start_dates[1:] + [end_date_search]

        task_intervals = [(ref_search_location, rank, list(interval)) for rank, interval in
                          enumerate(zip(start_dates, end_dates))]

        LOGGER.debug("Superwindow size =%s", superwindow_size)
        LOGGER.debug("Superwindow duration =%s", superwindow_duration)
        LOGGER.debug("Target of search: nb_points=%s start_date=%s end_date=%s", nb_points_search,
                     start_date_search,
                     end_date_search)
        LOGGER.debug("Task intervals on target of search:")

        return my_extract_period, task_intervals

    @classmethod
    def __read_ts_pattern(cls, ref_pattern, normalize_mode):
        """
        Read locally the TS pattern, using TemporalDataManager set on ResourceLocator.
        :param ref_pattern: pattern is a TS selection
        :type ref_pattern: str
        :param normalizing_mode: the choice for the normalizing method selected by the user
        :type normalize_mode: str among CONFIG_NORM_MODES keys:
           |N_NO_NORM: no normalizing
           |N_MEAN_VARIANCE: mean-normalizing and var-normalizing applied on pattern and search_target
           |N_MEAN_VARIANCE_ONLY_SEARCH: mean-normalizing and var-normalizing applied on search_target
           |N_MEAN: mean-normalizing applied on pattern and search_target
           |N_MEAN_ONLY_SEARCH: mean-normalizing applied on search_target
           |N_VARIANCE: var-normalizing applied on pattern and search_target
           |N_VARIANCE_ONLY_SEARCH: var-normalizing applied on search_target
        :return: tuple grouping
          - the TS (vector of points: columns [ timestamp, normalized value ]
          - the start date timestamp
          - the end date timestamp
          - the number of points
        :rtype: ( numpy.array, float, float, int )
        """

        # Extracted TS
        # - Numpy array is a 2D array where:
        #   * Column 1 represents the timestamp as numpy.int64 format
        #   * Column 2 represents the value associated to this timestamp as numpy.float64
        #

        res_extract_pattern = IkatsApi.ts.read([ref_pattern])

        ts_pattern = res_extract_pattern[0]
        # start date
        sd_pattern = float(ts_pattern[0][0])
        # end date
        ed_pattern = float(ts_pattern[-1][0])
        # nb points
        nb_points = len(ts_pattern)

        raw_np_pattern_values = ts_pattern[:, 1]
        config_norm = CONFIG_NORM_MODES.get(normalize_mode, CONFIG_NORM_MODES[N_NO_NORM])

        if config_norm[0] or config_norm[1]:
            norm_pattern = func_scale(data=raw_np_pattern_values,
                                      with_mean=config_norm[0],
                                      with_std=config_norm[1])

            ts_pattern[:, 1] = norm_pattern

        return ts_pattern, sd_pattern, ed_pattern, nb_points


def __append_successive_ts(ts_before, ts_after):
    """
    Append points from ts_after to ts_before.

    Assumed:
      - ts_before is before ts_after
      - at most one timestamp may be common:
        last timestamp from ts_before equals the first timestamp from ts_after.
        In that case: the point from ts_after is replacing the one
        from ts_before: this avoids duplicate points.

    :param ts_before: the TS with points before ts_after
    :type ts_before: numpy.array
    :param ts_after: the TS appended to ts_before
    :type ts_after: numpy.array
    :return: ts_before with new points from ts_after
    :rtype: numpy.array
    """
    if len(ts_before) > 0 and len(ts_after) > 0:
        if ts_before[-1][0] == ts_after[0][0]:
            ts_before = np_concatenate((ts_before[0:-1], ts_after), axis=0)
        else:
            ts_before = np_concatenate((ts_before, ts_after), axis=0)
    else:
        if len(ts_before) > 0:
            return ts_before

        if len(ts_after) > 0:
            return ts_after

    return ts_before


def __import_ts_result(result_type, ref_search_location, ref_pattern, data, fid):
    """
    Imports the specified TS result, detects errors, or return created tsuid.
    Assumed here: locally executed.

    :param result_type: name of the result for the logs
    :type result_type: str
    :param ref_search_location: the reference of the TS search
    :type ref_search_location: str
    :param ref_pattern: the reference of the TS pattern
    :type ref_pattern: str
    :param data: the TS imported: arg passed to the import service
    :type data: numpy.array
    :param fid: the functional id: arg passed to the import service
    :type fid: str
    :return: the imported tsuid in case of success
    :rtype:  str
    :raise exception: Exception is raised in case of import error.
    """
    try:

        import_status = IkatsApi.ts.create(data=data, fid=fid)

        if not import_status.get('status', False):
            # numpy is truncating big arrays, calling the str
            raise IkatsException("{}: {}".format(result_type, data))

        imported_tsuid = IkatsApi.fid.tsuid(fid)

        if imported_tsuid is None:
            raise IkatsException("Unexpected undefined TSUID for the imported result: {}".format(result_type))

        return imported_tsuid

    except Exception:
        msg = ERROR_IMPORT_MSG.format(result_type, ref_search_location, ref_pattern)
        raise IkatsException(msg)


def __local_init_superwindows(ref_search_location,
                              rate_size_window,
                              nb_points_pattern):
    """
    Internal function: local computing of the superwindows intervals
    :param ref_search_location: see argument doc from find_pattern
    :type ref_search_location: str
    :param rate_size_window: see argument doc from find_pattern
    :type rate_size_window: float
    :param nb_points_pattern: the number of points defined by the TS pattern
    :type nb_points_pattern: int
    :return: internal result is a tuple:
             nb_points_search,
             start_date_search,
             end_date_search,
             list of tuples (ref_search_location, superwindow_rank, [ superwindow_start_date, superwindow_end_date ] )
    :rtype: tuple
    """

    dict_result = IkatsApi.md.read([ref_search_location])
    start_date_search = float(dict_result[ref_search_location]['ikats_start_date'])
    end_date_search = float(dict_result[ref_search_location]['ikats_end_date'])
    nb_points_search = float(dict_result[ref_search_location]['qual_nb_points'])
    # Checks that the search area is sufficiently larger than the pattern:
    # => compare actual ratio with config_min_ratio_search_pattern
    size_sliding_window = float(ceil(rate_size_window * nb_points_pattern))
    actual_ratio = nb_points_search / size_sliding_window
    if actual_ratio < ConfigPatternMatching.min_ratio_search_pattern:
        msg = "Bad usage: ( nb_points_search / size_sliding_window ) < config_min_ratio_search_pattern: {} < {}"
        raise IkatsException(msg.format(actual_ratio, ConfigPatternMatching.min_ratio_search_pattern))
    # preferred size: depends on cluster configuration
    # - see  ConfigPatternMatching.config_max_count_superwindows
    #
    # Note: preferred size is minimized with floor()
    #       => maximize the superwindow_duration variable below
    #   - selon le critere de limitation de nombre de taches: config_max_count_tasks
    #   - selon le criere de limitation de memoire => config_max_points_in_superwindow -
    #
    swin_size_pref_by_nb_tasks = float(floor(nb_points_search / ConfigPatternMatching.preferred_count_tasks))

    # Choose the size as a trade-off between task limitation / memory limitation
    # => selects the minimum size ...
    superwin_size_preferred = min(swin_size_pref_by_nb_tasks, ConfigPatternMatching.max_size_superwindow)

    # actual size of superwindow:
    # - at least : superwindow_size >= size_sliding_window
    #              because a task operating on the superwindow may at least evaluate once the pattern
    #
    superwindow_size = max(superwin_size_preferred, size_sliding_window)

    # Duration : the period of superwindow assuming that the TS point have uniform frequency.
    # we add 1 to the duration to avoid side-effects
    superwindow_duration = 1 + int(float(end_date_search - start_date_search) * superwindow_size / nb_points_search)

    # iterate on the creation of each superwindow intervals ... until end_date_search
    # is reached

    start_dates = [start_date_search + i * superwindow_duration for i in range(ceil(actual_ratio))]
    end_dates = start_dates[1:] + [end_date_search]

    task_intervals = [(ref_search_location, rank, list(interval)) for rank, interval in
                      enumerate(zip(start_dates, end_dates))]

    LOGGER.debug("Superwindow size =%s", superwindow_size)
    LOGGER.debug("Superwindow duration =%s", superwindow_duration)
    LOGGER.debug("Target of search: nb_points=%s start_date=%s end_date=%s", nb_points_search,
                 start_date_search,
                 end_date_search)
    LOGGER.debug("Task intervals on target of search:")

    return nb_points_search, start_date_search, end_date_search, task_intervals


def __mapped_extract(tsuid, rank, start_superwindow, end_superwindow,
                     extract_period):
    """
    Gets a portion of the ts (or possibly all if small enough)
    This may be done in several extractions if there are more than
    ConfigPatternMatching.max_size_extract points to collect
    :param rdd_def_extract_x: the input RDD cell is a tuple:
    ( tsuid, rank_superwindow, [ start_superwindow, end_superwindow ] )
    :type rdd_def_extract_x: tuple
    :param tsuid
    :type str
    :param rank so that fragments of ts can be reassembled
    :type int
    :param start_superwindow
    :type float
    :param end_superwindow
    :type float
    :param extract_period: the computed duration of extracting period, applied step by step
    :type extract_period: float
    :param tdm: the shared interface of a TemporalDataManager,
    :return: (rank_superwindow, iterable<points>, <meta_info>, <error_info> )
    where <meta info> is a dict having metadata keys ('tsuid', ... ),
    where <error_info> is a list of errors.
    :rtype: tuple
    """

    # input:
    # ( tsuid, rank_superwindow, [ start_superwindow, end_superwindow ] )
    #

    # output:
    # ( rank_superwindow, iterable<points>, <meta_info>, <error_info> )
    meta_info = {'tsuid': tsuid,
                 'start_superwindow': start_superwindow,
                 'end_superwindow': end_superwindow}
    error_info = []
    points = None
    iter_end_date = start_superwindow

    while iter_end_date < end_superwindow:
        iter_start_date = iter_end_date
        iter_end_date = min(iter_end_date + extract_period, end_superwindow)
        try:
            res_extract = IkatsApi.ts.read([tsuid], int(iter_start_date), int(iter_end_date))
            ts = res_extract[0]
            if (ts is not None) and len(ts) > 0:
                if points is None:
                    points = ts
                else:
                    points = __append_successive_ts(points, ts)
        except Exception as err:
            error_info.append("mapped_extract on superwindow[{}] error: {}".format(rank, err))

    # in case when nothing was extracted ...
    if points is None:
        points = np_array([])

    if len(error_info) > 0:
        log = ResourceLocator().get_logger("dtw_match")
        log.error("Errors at the end of: __mapped_extract():")
        for err in error_info:
            log.error(err)
    return rank, points, meta_info, error_info


def __mapped_pattern_matching(win_index,
                              win_iter,
                              win_meta,
                              win_errors,
                              extracted_pattern,
                              rate_sliding,
                              rate_size_window,
                              distance,
                              scores_limitation,
                              normalize_mode):
    """
    The pattern-matching applied on the superwindow defined by  extracted_superwindow

    :param extracted_superwindow: the RDD cell ( <rank>, iterable<points>, <meta_info>, <error_info> )
    :type extracted_superwindow: tuple
    :param win_index the rank of the window
    :type: int
    :param win_iter the ts returned by OpenTSDB
    :type np.array
    :param win_meta TODO:
    :type dict
    :param win_errors error messages collected during collection
    :type dict
    :param extracted_pattern: the TS array with shape (x, 2): list of points
    :type extracted_pattern: numpy.array
    :param rate_sliding: see param definition in find_pattern
    :type rate_sliding: see param type definition in find_pattern
    :param rate_size_window: see param definition in find_pattern
    :type rate_size_window: see param type definition in find_pattern
    :param distance: see param definition in find_pattern
    :type distance: see param type definition in find_pattern
    :param scores_limitation: see param definition in find_pattern
    :type scores_limitation:see param type definition in find_pattern
    :param normalize_mode: see param definition in find_pattern
    :type normalize_mode: see param type definition in find_pattern
    :return: the cell for the rdd_match_pattern
            is equivalent to ( scores, matched_intervals, unprocessed_interval,
                               rank_superwindow, <meta info>, <error_info> )
    :rtype: tuple structured as
        (list of float, list of 2-sized lists of timestamps, 2-sized list of int, int, dict, list )
    """
    # transforms RDD cells:
    #      ( <rank>, iterable<points>, <meta_info>, <error_info> )
    #   -> ( scores, matched_intervals, unprocessed_interval, rank_superwindow, <meta info>, <error_info> )

    # if win_iter is empty => return the provided meta + error information
    #                         with empty result
    # if win_errors is not empty => idem
    #
    # => errors+ meta will be finally collected by the driver
    if win_iter is None or win_iter.size == 0 or (win_errors is not None and len(win_errors) > 0):
        # return (scores, matched_intervals, the_unprocessed_interval, win_index,  win_meta, win_errors)
        return [], [], [None, None], win_index, win_meta, win_errors

    # It turns out some EDF datasets are not evenly distributed
    # Thus when you call certain sections you get an empty array
    # Previously this caused an Exception here
    try:

        timestamps = win_iter[:, 0]
        np_values = win_iter[:, 1]

    except Exception:
        return [], [], [None, None], win_index, win_meta, win_errors

    np_pattern_values = extracted_pattern.value[:, 1]

    # variables defining the returned tuple:
    # - scores: the good scores filtered by arguments scores_limitation and matching_trigger
    # - matched_intervals (same order than scores )
    # - the_unprocessed_interval: the indexes defining unprocessed interval in current superwindow:
    #   [<start index>, <end index> ]
    scores = []
    matched_intervals = []
    # internal variable required to sort pairs ( score, matched_interval )
    internal_scores = []
    the_unprocessed_interval = [None, None]

    # locally used:
    # - pattern_size,
    # - sliding_window_size: maximum warped size
    # - row_count
    # - sliding_increment: defines the moving-right translation of sliding window,
    #   its unit: the number of points
    pattern_size = len(np_pattern_values)

    sliding_window_size = ceil(pattern_size * rate_size_window)
    sliding_increment = ceil(pattern_size * rate_sliding)
    row_count = len(np_values)

    # last_starting_index is computed:
    # the last possible value for the starting index of the last sliding window:
    # start index is among the search_ts range.
    #
    # ex_1: sliding_window_size == 100 , row_count == 500 , sliding_increment == 50
    #  => last_starting_index is 400 == ((500 - 100) // 40) * 40  : it is not possible to go further
    #
    # ex_2: sliding_window_size == 100 , row_count == 550 , sliding_increment == 40
    #  =>  last_starting_index is 440 == ((550 - 100) // 40) * 40
    #
    last_starting_index = ((row_count - sliding_window_size) // sliding_increment) * sliding_increment
    try:
        if row_count >= sliding_window_size:

            scoring_func = SCORING_FUNCTIONS.get(distance, None)
            if scoring_func is None:
                raise IkatsException("Unknown distance from SCORING_FUNCTIONS: distance={}".format(distance))

            for start_ind in range(0, last_starting_index + 1, sliding_increment):
                # matching_index_interval: [ <index start>, <index end> ]
                # this information is useful in case of scoring with DTW distance
                score, matching_index_interval = scoring_func(
                    np_pattern_values=np_pattern_values,
                    np_values=np_values[start_ind:start_ind + sliding_window_size],
                    normalize_mode=normalize_mode)

                internal_matching_interval = [timestamps[start_ind + matching_index_interval[0]],
                                              timestamps[start_ind + matching_index_interval[1]]]

                internal_scores.append((float(score), internal_matching_interval))

            # About the unprocessed sliding indexes
            # ======================================================================================
            #
            #                  sliding increment
            #                     |<-------->|
            #                     | sliding_window_size  |
            #                     |<-------------------->|
            # -------|------------|----------|-----------|---------|--------> superwindow indices
            #        0     last start_ind    |           |      row-count-1
            #            (processed)         |           |
            #                                |    very_last_evaluated_index
            #                                |         (processed)
            #                                |
            #                       first_unprocessed_index
            #                           (unprocessed)
            # ======================================================================================
            # first_unprocessed_index = last start_ind + sliding increment
            #                         = (very_last_evaluated_index - sliding_window_size + 1) + sliding_increment
            #
            # ex_1:  sliding_window_size == 100 , row_count == 500 , sliding_increment == 50
            #  => last_starting_index is 400: it is not possible to go further
            #     => first_unprocessed_index == 450
            #
            # ex_2: sliding_window_size == 100 , row_count == 550 , sliding_increment == 40
            #  =>  last_starting_index is 440 == ((550 - 100) % 40 * 40)
            #     => first_unprocessed_index == 440 + 40 == 480
            first_unprocessed_index = last_starting_index + sliding_increment
            if first_unprocessed_index < row_count:
                the_unprocessed_interval = [first_unprocessed_index, row_count - 1]

            # Post-processing the scores:
            #  - order internal_scores by score (first field of tuple)
            #  - limit them to scores_limitation: this limit is at least respected on each executor
            internal_scores.sort(key=lambda x: x[0])
            len_max = len(internal_scores)
            for item in internal_scores[0:min(scores_limitation, len_max)]:
                scores.append(item[0])
                matched_intervals.append(item[1])

        else:
            # negative last_starting_index
            # <=> row_count - sliding_window_size < 0
            # do not raise exception ... this may occur in nominal execution
            # => will be processed in next shifted step ... or not processed at the end of TS
            #
            # return (scores, matched_intervals, the_unprocessed_interval, win_index,  win_meta, win_errors)
            return [], [], [0, row_count], win_index, win_meta, win_errors

    except Exception as err:

        win_errors = win_errors or []
        list_traceback = traceback.format_exception(None,  # <- type(e) by docs, but ignored
                                                    err, err.__traceback__)
        win_errors.append(
            "mapped_pattern_matching on superwindow[{}] errors:".format(win_index))
        for error in list_traceback:
            win_errors.append(error)

    # rdd_match_pattern <=> list of ( scores, matched_intervals, unprocessed_interval,
    #                                        rank_superwindow, <meta info>, <error_info> )
    return (scores, matched_intervals, the_unprocessed_interval, win_index,
            win_meta, win_errors)


def __parse_search_results(collected_results, raise_exception_finally):
    """
    Parses locally the results collected from the __mapped_pattern_matching:
      - list of ( scores, matched_intervals, unprocessed_interval,
                        rank_superwindow, <meta info>, <error_info>)
    This parsing
      - evaluates the errors: logs and raise IkatsException if raise_exception_finally is True
      - evaluates, encodes the unprocessed intervals

    :param collected_results: the results collected from the __mapped_pattern_matching
    :type list
    :param raise_exception_finally: flag is True when it is demanded to raise
      IkatsException when errors exist in the collected fields <error_info>
    :type raise_exception_finally: bool
    :return: unprocessed_info: dict stocking unprocessed intervals [sd, ed] by superwindow rank
    :rtype: dict
    """

    # On the driver/manager
    #       - driver: local computing of the re-evaluated intervals:
    #         => set unprocessed_info
    #         => logs errors
    unprocessed_info = {}
    error_exists = False
    for status_cell in collected_results:
        # logs error (when needed)
        error = status_cell[-1]
        cell_has__errors = False
        if isinstance(error, collections.Iterable) and len(error) > 0:
            cell_has__errors = True
            for error_item in error:
                LOGGER.error(error_item)

        error_exists = error_exists or cell_has__errors

        # unprocessed_info[ <rank_superwindow> ] = <unprocessed_interval>
        unprocessed_info[status_cell[3]] = status_cell[2]
        LOGGER.debug("Unprocessed_info[rank=%s]=%s", status_cell[3], status_cell[2])

    if raise_exception_finally and error_exists:
        raise IkatsException("At least one error has been collected: see driver/manager logs")

    return unprocessed_info


def __mapped_split_superwindow(input_iterable_points,
                               rank_superwindow,
                               meta_info, unprocessed_info, size_sliding_window):
    """
    Function used by the flatmap operation, in order to reduce and split in two the RDD from step 1 (rdd_extract)
    :param input_iterable_points
    :type np.array
    :param rank_superwindow
    :type int
    :param meta_info (win_meta in __mapped_pattern_matchin)
    :type dict
    :param unprocessed_info: information defining how to reduce the points to the unprocessed intervals:
        unprocessed intervals[start, end] stored by superwindow ranks. Generated in __parse_search_results
    :type unprocessed_info: dict
    :param size_sliding_window: the maximum size of the sliding window searching the pattern
    :type size_sliding_window: float
    :return: tuple of 2 items which is split in two by the flatmap
            | ((rank_superwindow, <rank_frontier_left>, iterable<left_points>, <meta_info>, <left_error_info> ),
            | (rank_superwindow, <rank_frontier_right>, iterable<right_points>, <meta_info>, <right_error_info> ))
    :rtype: tuple
    """
    # where cell_rdd_extract <=>  (rank_superwindow, iterable<points>, <meta_info>, <error_info> )
    #
    # Transforms cell_rdd_extract into result:
    #
    #        (rank_superwindow, iterable<points>, <meta_info>, <error_info> )
    #
    #     -> (rank_superwindow, <rank_frontier_left>, iterable<left_points>, <meta_info>, <left_error_info> ),
    #        (rank_superwindow, <rank_frontier_right>, iterable<right_points>, <meta_info>, <right_error_info> )
    #
    # where:
    #         <rank_frontier_left> = rank_superwindow
    #         <rank_frontier_right> = rank_superwindow + 1
    # Note: assumed here: <error_info> has already been collected: not preserved in
    # <left_error_info> or <right_error_info>
    #

    window_left = None
    window_right = None

    error_left = None
    error_right = None
    try:
        # window_left:
        # remove the last point because we want to keep only the
        # interval strictly before interval already evaluated:
        #   input_iterable_points[0:size_sliding_window]
        #
        meta_info['left_interv'] = "rank={} [{}:{}]".format(rank_superwindow, 0, size_sliding_window)
        window_left = input_iterable_points[0:size_sliding_window]

        # window right: starts from unprocessed index:
        unprocessed_indexes = unprocessed_info.get(rank_superwindow, None)
        if ((unprocessed_indexes is not None) and (type(unprocessed_indexes) is list) and
                (len(unprocessed_indexes) == 2) and (unprocessed_indexes[0] is not None) and
                (unprocessed_indexes[1] is not None)):
            meta_info['right_interv'] = "rank={} [{}:{}]".format(rank_superwindow,
                                                                 unprocessed_indexes[0], unprocessed_indexes[1] + 1)

            window_right = input_iterable_points[unprocessed_indexes[0]:unprocessed_indexes[1] + 1]
        else:
            meta_info['right'] = "rank={} empty array".format(rank_superwindow)
            window_right = np_empty_array([0, 2])

    except Exception as err:
        err_info = "Failed: __mapped_split_superwindow rank={}: err={}"
        # simply duplicate error in left/right cells
        error_left = [err_info.format(rank_superwindow, err)]
        error_right = [err_info.format(rank_superwindow, err)]

    # <rank_frontier_left> = rank_superwindow
    #         <rank_frontier_right> = rank_superwindow + 1
    return ((rank_superwindow, rank_superwindow, window_left, meta_info, error_left),
            (rank_superwindow, rank_superwindow + 1, window_right, meta_info, error_right))


def __mapped_merge_frontier_windows(corrected_frontier_rank, res_iter):
    """
    Merges the list grouping points around the frontier:
    left points concatenated with right points.

    :param corrected_frontier_rank
    :type int
    :param res_iter
    :type iterable
    :return: the rdd cell returned is
      ( <corrected_frontier_rank>, iterable<points>,  <meta_info>, <error_info> )
    :rtype: tuple
    """

    def __complete_error(errors, evaluated_tuple):

        if evaluated_tuple is not None and type(evaluated_tuple) is tuple and len(evaluated_tuple) > 0:
            field_error = evaluated_tuple[-1]
            if type(field_error) is list:
                errors.extend(field_error)

        return errors

    meta_info = None

    error = []

    points = None
    try:
        corrected_frontier_rank = corrected_frontier_rank - 0.5
        # it is needed to reorder the list by <rank_superwindow>

        len_split_superwindow_items = len(res_iter)
        split_superwindow_items = []
        if res_iter is not None:
            for item in res_iter:
                split_superwindow_items.append(item)

        first_item = None if len_split_superwindow_items == 0 else split_superwindow_items[0]
        second_item = None if len_split_superwindow_items < 2 else split_superwindow_items[1]

        error = __complete_error(error, first_item)
        error = __complete_error(error, second_item)

        if len(error) == 0:

            if second_item is None and first_item is not None:
                points = first_item[2]
                meta_info = first_item[3]

            elif first_item is None and second_item is not None:
                points = second_item[2]
                meta_info = second_item[3]

            elif first_item is None and second_item is None:
                points = None
                meta_info = None
                error = [IkatsException("Unexpected: first_item and second_item are None")]

            else:
                # Both defined: first_item and second_item
                # => concatenate the points ...

                # compare the <rank_superwindow>
                original_rank_first = first_item[0]
                original_rank_second = second_item[0]

                if original_rank_first < original_rank_second:
                    left_points = first_item[2]
                    right_points = second_item[2]
                else:
                    left_points = second_item[2]
                    right_points = first_item[2]

                # returned: nominal case:
                #   ( <corrected_frontier_rank>, iterable<points>,  <meta_info>, [] )
                #
                # Note:  <meta_info> is duplicated on each item => return first
                meta_info = first_item[3]
                points = __append_successive_ts(left_points, right_points)

    except Exception as err:
        # returned: error case:
        #   ( <corrected_frontier_rank>, iterable<points>,  <meta_info>, <error_info> )

        error = error or []
        list_traceback = traceback.format_exception(None,  # <- type(e) by docs, but ignored
                                                    err, err.__traceback__)
        error.append("Failed __mapped_merge_frontier_windows at rank={}:".format(corrected_frontier_rank))
        for stack_error in list_traceback:
            error.append(stack_error)

    return corrected_frontier_rank, points, meta_info, error


def __compute_final_result(collected_scores_by_superwindow,
                           ref_pattern,
                           ref_search_location,
                           scores_limitation):
    """
    Computes the results:
      - list of
    :param collected_scores_by_superwindow: list of (  scores, matched_intervals, unprocessed_interval,
                                                       rank_superwindow, <meta info>, <error_info> )
    :type collected_scores_by_superwindow: list
    :param ref_pattern:
    :type ref_pattern:
    :param ref_search_location:
    :type ref_search_location:
    :param scores_limitation:
    :type scores_limitation:
    :return: multiple result: search_funcId, scores_tsuid, intervals_tsuid. Where
      - search_funcId is the funcId of the search target
      - scores_tsuid is the tsuid of the TS representing the scores of pattern-matching
      - intervals_tsuid is the tsuid of the TS representing the corresponding intervals:
        - start-date is the timestamp
        - end-data is the value (float precision is sufficient to be converted into timestamp)
    :rtype: str, str, str
    """
    try:
        results = []
        for collected_item in collected_scores_by_superwindow:
            LOGGER.debug("----__compute_final_result() reading: ----")

            scores = collected_item[0]
            if scores is not None and len(scores) > 0:

                my_intervals = collected_item[1]
                my_size = len(scores)
                for index in range(my_size):  # result <=> ( score, timestamp_start, timestamp_end )
                    results.append((scores[index], my_intervals[index][0], my_intervals[index][1]))

        # - sort the best matches (score, matched interval)
        results.sort(key=lambda x: x[0])

        #  - if required: keep the best ones: limited to scores_limitation
        results = results[0:min(scores_limitation, len(results))]

        # finally: sort by timestamp before creating the TS ...
        results.sort(key=lambda x: x[1])

        # dtype is object: mixt type timestamp=>int64 / value=>float64
        ts_scores = np_empty_array(shape=(len(results), 2), dtype=object)
        ts_intervals = np_empty_array(shape=(len(results), 2), dtype=object)
        res_index = 0
        for res_item in results:
            score, start, end = res_item
            LOGGER.debug("Top[%s]=> score: %s at [ %s, %s ]", res_index + 1, score, start, end)
            ts_scores[res_index][1] = float(score)
            ts_scores[res_index][0] = int(start)

            # end is int converted to float64 ;: do we loose precision ???
            #  - nowadays: this is ok:
            #    13 digits compatible with float64: up to 16 significant digits
            # - Avoid:  s_intervals[res_index][1] = float64(end)
            #           => cause error !!!
            ts_intervals[res_index][1] = int(end)  # int(end-start)
            ts_intervals[res_index][0] = start
            res_index = res_index + 1

        try:
            search_func_id = IkatsApi.ts.fid(ref_search_location)
        except Exception:
            search_func_id = ref_search_location

        # generates integer suffix from current date and a modulo
        delta_timestamp = int((time.time() - time.mktime(time.strptime("2016", "%Y"))) % 100000)

        scores_func_id = "find_pattern_scores_{}_{}".format(search_func_id, delta_timestamp)
        intervals_func_id = "find_pattern_intervals_{}_{}".format(search_func_id, delta_timestamp)
        scores_tsuid = __import_ts_result(result_type="scores",
                                          ref_search_location=ref_search_location,
                                          ref_pattern=ref_pattern,
                                          data=ts_scores,
                                          fid=scores_func_id)

        intervals_tsuid = __import_ts_result(result_type="intervals",
                                             ref_search_location=ref_search_location,
                                             ref_pattern=ref_pattern,
                                             data=ts_intervals,
                                             fid=intervals_func_id)

        return search_func_id, scores_tsuid, intervals_tsuid
    except Exception:
        raise IkatsException("Failed step: computes final result")


def find_pattern(ref_pattern,
                 broadcast_pattern,
                 my_extract_period,
                 task_intervals,
                 my_sliding_window_size,
                 ref_search_location,
                 rate_sliding,
                 rate_size_window,
                 distance,
                 scores_limitation,
                 normalize_mode, my_spark_context):
    """
    Apply the spark-distributed pattern-matching algorithm: this algorithm searches the TS pattern defined
    by **ref_pattern** argument into the unique TS specified by **ref_search_location**.

    The window-sliding processing evaluates the distance between the TS pattern
    and successive subparts of the **ref_search_location** TS. The window-sliding is configured with:

      - the **rate_sliding** argument,
      - the **rate_size_window** argument.

    The distance between subpart and pattern is specified with **distance** argument.

    Additional normalizing process may be specified: see **normalize_mode** argument.

    The best scores -smallest distances- are saved with two TS:

      - TS of scores defining each point by:

        - timestamp: starting timestamp of comparison,
        - value: score of comparison.
      - TS of intervals -as compared intervals may not have all the same duration -,
        defining each point by:

        - timestamp: starting timestamp of comparison,
        - value: ending timestamp of comparison from the ref_search_location TS.

    The returned result provides access to the created TS matched scores, and matched intervals: detailed below.

    Limitations on the selected best scores are set by:

      - the scores_limitation argument.

    Usage constraints: for normal use, this algorithm expects:

      - Hyp1: ref_pattern and ref_search_location refer to TS with the same "usual" frequencies,
        otherwise a pre-processing is required
      - Hyp2: ref_pattern refer to a TS having number of points <= max_pattern_size,
        otherwise a resampling pre-processing is required on both TS pointed by ref_pattern/ref_search_location
      - Hyp3: following metadata are available on the TS ref_search_location:
        ikats_start_date, ikats_end_date, qual_nb_points


    :param ref_pattern: pattern is a TS selection. Only used for an error message in compute final results -> import ts
    in the case where there is an error writing the scores or the interval ts
    :type ref_pattern: str coding the tsuid
    :param broadcast_pattern
    :type spark broadcast variable
    :param my_extract_period will be 1 unless extractions from OpenTSDB are larger than max extraction allowed
    :type int
    :param task_intervals
    :type nested iterable
    :param my_sliding_window_size pattern size * rate_sizing_window
    :type int
    :param ref_search_location: location where is applied the pattern-matching search. Reference to a TS.
    :type ref_search_location: str coding the tsuid
    :param rate_sliding: the computed translation of search window: expressed as a
      percentage of ref_pattern size (size given by number of points): 1.0 is coding 100%.
      Constraint: 0.0 < rate_sliding.
      Beware: it is recommended that rate_sliding <= 1.0, for a fully-covering search.
    :type rate_sliding: float
    :param rate_size_window: the maximal length of search window, important for DTW distance: expressed as a
      percentage of ref_pattern size (size given by number of points): 1.0 coding for 100%. 1.0 <= rate_size_window
    :type rate_size_window: float
    :param distance selecting the scoring
    :type distance: str among
      |FAST_DTW_DISTANCE
      |DTW_DISTANCE,
      |MANHATTAN_DISTANCE
    :param scores_limitation: defines the maximum number of scores kept: this parameter avoids too big results
    :type scores_limitation: int
    :param normalizing_mode: the choice for the local-normalizing method
    :type normalize_mode: str among CONFIG_NORM_MODES keys:
       |N_NO_NORM: no normalizing
       |N_MEAN_VARIANCE: mean-normalizing and var-normalizing applied on pattern and search_target
       |N_MEAN_VARIANCE_ONLY_SEARCH: mean-normalizing and var-normalizing applied on search_target
       |N_MEAN: mean-normalizing applied on pattern and search_target
       |N_MEAN_ONLY_SEARCH: mean-normalizing applied on search_target
       |N_VARIANCE: var-normalizing applied on pattern and search_target
       |N_VARIANCE_ONLY_SEARCH: var-normalizing applied on search_target
    :param my_spark_context
    :type Spark Context
    :return: standard result of pattern-matching: dictionary defining lists of references instead of unique references,
       as it is also designed to be compatible with searches within multiple TS for the front-end VizTool.

      - key: 'ref_tsuids' with value: list of the original tsuids: here the unique tsuid deduced
            from ref_search_location
      - key: 'ref_funcIds' with value: list of the original functional Ids: here the unique funcId deduced
            from ref_search_location
      - key: 'scores' with value: list of score TS: here the unique tsuid of the TS of scores
      - key: 'intervals' with value: list of interval TS: here the unique tsuid of the TS of intervals
    :rtype dict
    """

    # =================================================================================================
    # Old steps moved assert pattern fidelity and assert ts fidelity
    # =================================================================================================
    #  Step 0: driver: checks + read config
    #  ------
    #
    #  Step 1:- driver: reads the pattern TS (and metadata) and broadcast the variable
    #  ------
    # ===================================================================================================
    # Spark Schema
    # ================================================================================================
    #

    #
    #  Step 1:- driver: computes the extracting zones that will be distributed:
    #  ------
    #        - read the metadata from <ref_search_location>:
    #        - start_date
    #        - end-date
    #        - nb_points (in order to decide how to split the extraction)
    #
    #      - parallelize the extraction intervals: superwindows
    #
    #        rdd_def_extract <=> list of ( rank_superwindow, tsuid, start_superwindow, end_superwindow)
    #
    #      See note below: how to choose superwindow mean size
    #
    #  Step 2:- driver -> workers: map the extraction
    #  -------
    #      Note: assumed that an extract of superwindow: it may be required to call several extractions
    #           from DB, considering max_size_extract (in nb points)
    #
    #      - map the extraction extract_ts: rdd_def_extract => rdd_extract
    #
    #        rdd_extract <=> list of (rank_superwindow, list<points>, <meta_info>, <error_info> )
    #
    #        where <error_info>: list grouping errors
    #
    #  Step 3:- driver -> workers: apply the pattern-matching based on sliding-window within the superwindows,
    #  -------                     and applying a remote filtering regarding parameter scores_limitation
    #
    #      - map with function=... : rdd_match_pattern
    #        rdd_match_pattern <=> list of ( <size scores>, scores, matched_intervals, unprocessed_interval,
    #                                        rank_superwindow, <error_info> )
    #        unprocessed_interval: at the end of superwindow: the interval not evaluated (timestamps + nb points)
    #        <error_info>: in case of error: status, message ...
    #
    #  Step 4:  workers -> driver: collect rdd_match_pattern information:
    #  ------
    #      Assumed: [Hyp 8.A] below
    #
    #      - collect: rdd_match_pattern cells (scores+errors)
    #      - parses collected info:
    #           - prepares unprocessed_info for the next step
    #           - handle errors: log + raise local exception
    #
    #
    #  Step 5:  driver -> workers: reorganize superwindows for the unprocessed_intervals on the frontiers
    #  ------
    #    s-step 6.1: split each superwindow into 2 small datasets
    #
    #       - flatmap: the aim is to
    #              - split each superwindow into 2 subparts "left" and "right",
    #              - and to keep only the re-evaluated points
    #
    #         rdd_extract -> rdd_split_superwindows:
    #
    #        (rank_superwindow, iterable<points>, <meta_info>, <error_info> )
    #     -> (rank_superwindow, <rank_frontier_left>, iterable<left_points>, <meta_info>, <left_error_info> ),
    #        (rank_superwindow,  <rank_frontier_right>, iterable<right_points>, <meta_info>, <right_error_info> )
    #
    #  where:
    #         <rank_frontier_left> = rank_superwindow
    #         <rank_frontier_right> = rank_superwindow + 1
    #
    #    s-step 5.2: group datasets sharing same frontier
    #
    #       - groupBy(rank_frontier): rdd_split_superwindows => rdd_group_frontiers:
    #             list of: ( rank_frontier,
    #                        iterable on ( rank_superwindow, rank_frontier, list<points>, <meta_info>, <error>  ) )
    #
    #        <rank_frontier> is the second field of tuples from rdd_split_superwindows
    #
    #    s-step 5.3: merges grouped datasets
    #
    #       - map: merges the two lists grouped under iterable: gathering points before and after the frontier
    #            rdd_group_frontiers => rdd_group_frontiers_filtered => rdd_extract_shifted
    #
    #           rdd_extract_shifted  <=> (rank_frontier, list<points>, <meta_info>, <error> )
    #
    #            Note: rdd_extract_shifted has same structure than rdd_extract
    #
    # Note: the filter is used in order to ignore extreme ranks:
    #       - ignore rank_frontier=0 : left side of first superwindow: not a shared frontier
    #       - and ignore rank_frontier=len(task_intervals) : right side of last superwindow: not a shared frontier
    #
    #
    #  Step 6: driver-> workers: reapply steps 4 on rdd_extract_shifted
    #  ------
    #          rdd_extract_shifted => rdd_match_pattern_frontier
    #
    #  Step 7: computes the final result
    #  -------
    #  [Hyp 7.A]: Hyp assuming that the limitation on best scores is sufficient to collect them locally
    #
    #   step 7.1: workers -> driver:
    #
    #     collects cells from rdd_match_pattern_frontier
    #       rdd_match_pattern_frontier <=>  list of
    #                          (  scores, matched_intervals, unprocessed_interval,
    #                             rank_superwindow, <meta info>, <error_info> )
    #
    #   workers -> driver: rdd_match_pattern_frontier.collect()
    #
    #    step 7.2: driver:
    #
    #      computes and sort final result: sorting the best matches, and importing the results
    #      (see __compute_final_result() )
    #
    #
    # ====================================================================================================

    try:

        # Step 1

        # rdd_def_extract <=> list of ( tsuid, rank_superwindow, [ start_superwindow, end_superwindow ] )
        rdd_def_extract = my_spark_context.parallelize(task_intervals, len(task_intervals))

        LOGGER.info("Ended:  Step 1.b read the metadata from <ref_search_location>")

        # Step 2:- driver -> workers: map the extraction
        # ------------------------------------------------
        #      Note: assumed that an extract of superwindow: it may be required to call several extractions
        #           from DB, considering max_size_extract (in nb points)
        #
        #      - map the extraction extract_ts: rdd_def_extract => rdd_extract
        #        rdd_extract <=> list of (rank_superwindow, list<points>, <meta_info>, <error_info> )
        #
        #       where <error_info>: list grouping errors
        #

        rdd_extract = rdd_def_extract.map(lambda rdd_def_extract_x:
                                          __mapped_extract(tsuid=rdd_def_extract_x[0],
                                                           rank=rdd_def_extract_x[1],
                                                           start_superwindow=rdd_def_extract_x[2][0],
                                                           end_superwindow=rdd_def_extract_x[2][1],
                                                           extract_period=my_extract_period))

        # the rdd_extract is cached, because reused below
        rdd_extract.cache()

        LOGGER.info("Ended:  Step 3:- driver -> workers: map the extraction")

        # Step 3:- driver -> workers: apply the pattern-matching based on sliding-window within the superwindows,
        #    and applying a remote filtering regarding parameters matching_trigger + scores_limitation
        # -----------------------------------------------------------------------------------------------------
        #      - map with function=... : rdd_match_pattern
        #        rdd_match_pattern <=> list of ( <size scores>, scores, matched_intervals, unprocessed_interval,
        #                                        rank_superwindow, <meta info>, <error_info> )
        #        unprocessed_interval: at the end of superwindow: the interval not evaluated (timestamps + nb points)
        #        <error_info>: in case of error: status, message ...
        #
        LOGGER.info("Norm Mode is %s: [MeanPattern, ScalePattern, MeanTarget, ScaleTarget]=%s", normalize_mode,
                    CONFIG_NORM_MODES[normalize_mode])

        rdd_match_pattern = rdd_extract.map(
            lambda extracted_superwindow: __mapped_pattern_matching(win_index=extracted_superwindow[0],
                                                                    win_iter=extracted_superwindow[1],
                                                                    win_meta=extracted_superwindow[2],
                                                                    win_errors=extracted_superwindow[3],
                                                                    extracted_pattern=broadcast_pattern,
                                                                    rate_sliding=rate_sliding,
                                                                    rate_size_window=rate_size_window,
                                                                    distance=distance,
                                                                    scores_limitation=scores_limitation,
                                                                    normalize_mode=normalize_mode))
        LOGGER.info("Ended:  Step  4:- driver -> workers: apply the pattern-matching")

        # Step 4:  workers -> driver: collect rdd_match_pattern information:
        # ------------------------------------------------------------------
        # Assumed: Hyp [8.A] below
        #
        #      - collect: rdd_match_pattern cells (scores+errors)
        #      - parses collected info:
        #           - prepares unprocessed_info for the next step
        #           - handle errors: log + raise local exception

        all_collected_scores = rdd_match_pattern.collect()
        unprocessed_info = __parse_search_results(all_collected_scores, raise_exception_finally=True)

        LOGGER.info("Ended: Step 5:  workers -> driver: collect status")

        #  Step 5:  driver -> workers: reorganize superwindows for the unprocessed_intervals on the frontiers
        # ----------------------------------------------------------------------------------------------------
        # s-step 5.1:
        #       - flatmap: the aim is to
        #              - split each superwindow into 2 subparts "left" and "right",
        #              - and to keep only the re-evaluated points: reducing the RDD in memory
        #
        #         rdd_extract -> rdd_split_superwindows:
        #
        #        (rank_superwindow, iterable<points>, <meta_info>, <error_info> )
        #
        #     -> (rank_superwindow, <rank_frontier_left>, iterable<left_points>, <meta_info>, <left_error_info> ),
        #        (rank_superwindow, <rank_frontier_right>, iterable<right_points>, <meta_info>, <right_error_info> )
        #
        # where:
        #         <rank_frontier_left> = rank_superwindow
        #         <rank_frontier_right> = rank_superwindow + 1
        #
        #  see details in __mapped_split_superwindow

        rdd_split_superwindows = rdd_extract.flatMap(
            lambda cell_rdd_extract: __mapped_split_superwindow(input_iterable_points=cell_rdd_extract[1],
                                                                rank_superwindow=cell_rdd_extract[0],
                                                                meta_info=cell_rdd_extract[2],
                                                                unprocessed_info=unprocessed_info,
                                                                size_sliding_window=my_sliding_window_size))

        LOGGER.info("Ended: Step 6.1: split+reduce each superwindow into left/right unprocessed parts")

        # s-step 5.2:
        #       - groupBy(rank_frontier): rdd_split_superwindows => rdd_group_frontiers:
        #          with rdd_group_frontiers <=>
        #             list of: ( rank_frontier,
        #                        iterable on ( rank_superwindow, rank_frontier, list<points>, <meta_info>, <error>  ) )
        #
        # Note about rdd_group_frontiers:
        #   - the iterable is the unordered set of windows around the frontier (the set is a pair in most cases,
        #     singleton at boundaries)
        #   - <rank_frontier> is the second field of tuples from rdd_split_superwindows
        #
        # Performance note: groupBy() is quite costy sometimes: but in that case it is applied on a small RDD:
        #   - rdd_split_superwindows is quite small in memory as only points around each frontier are kept
        #     ( less points than 2 * sliding_size * ( number tasks -1 ))
        #   - another solution would have been to collect all the points and finish the work locally;
        #     instead of applying groupby
        rdd_group_frontiers = rdd_split_superwindows.groupBy(lambda x: x[1])

        LOGGER.info("Ended: Step  6.2: group by frontier rank")

        #  5-step 5.3:
        #
        # 5.3.1: filter ignored cases
        #
        #     rdd_group_frontiers => rdd_group_frontiers_filtered
        #
        #     the filter is used in order to ignore extreme ranks:
        #       - ignore rank_frontier=0 : left side of first superwindow: not a shared frontier
        #       - and ignore rank_frontier=len(task_intervals) : right side of last superwindow: not a shared frontier
        #
        nb_superwindows = len(task_intervals)
        rdd_group_frontiers_filtered = rdd_group_frontiers.filter(
            lambda x: (x[0] > 0) and (x[0] < nb_superwindows))

        # 5.3.2: final merge
        #
        #   rdd_group_frontiers_filtered => rdd_extract_shifted
        #
        #      with rdd_extract_shifted  <=> (rank_frontier, list<points>, <meta_info>, <error> )
        #
        #      - map: with __mapped_merge_frontier_windows:
        #        - merges the two lists grouped into iterable: gathering points before and after the frontier
        #        - rank_frontier <- rank_frontier - 0.5
        #           Here the aim is to finally sort all the superwindows:
        #           distinct ranks for initial ones and shifted ones
        #
        rdd_extract_shifted = rdd_group_frontiers_filtered.map(lambda frontier:
                                                               __mapped_merge_frontier_windows(
                                                                   corrected_frontier_rank=frontier[0],
                                                                   res_iter=frontier[1]))

        #         if ConfigPatternMatching.debug_extract_shifted_info is True:
        #             rdd_extract_shifted.cache()
        #             __collect_for_debug(rdd_extract_shifted, "rdd_extract_shifted", only_errors=True)

        LOGGER.info("Ended: Step 6.2: group the parts around unprocessed frontiers")

        #  6: driver-> workers: reapply steps 4 on rdd_extract_shifted
        #       rdd_extract_shifted => rdd_match_pattern_frontier
        # ----------------------------------------------------------------
        rdd_match_pattern_frontier = rdd_extract_shifted.map(
            lambda extracted_superwindow: __mapped_pattern_matching(win_index=extracted_superwindow[0],
                                                                    win_iter=extracted_superwindow[1],
                                                                    win_meta=extracted_superwindow[2],
                                                                    win_errors=extracted_superwindow[3],
                                                                    extracted_pattern=broadcast_pattern,
                                                                    rate_sliding=rate_sliding,
                                                                    rate_size_window=rate_size_window,
                                                                    distance=distance,
                                                                    scores_limitation=scores_limitation,
                                                                    normalize_mode=normalize_mode))

        LOGGER.info("Ended: Step 7: reapply pattern matching around the superwindows frontiers")

        # Step 7: computes the final result
        # -------------------------------------------------------------------------------------------------
        # [Hyp 7.A]: Hyp assuming that the limitation on best scores is sufficient to collect them locally
        #
        # step 7.1:
        #   collects cells from rdd_match_pattern_frontier
        #     rdd_match_pattern_frontier <=>  list of
        #                          (  scores, matched_intervals, unprocessed_interval,
        #                             rank_superwindow, <meta info>, <error_info> )
        #
        #   workers -> driver: rdd_match_pattern_frontier.collect()
        #
        # step 7.2:
        #    computes and sort final result: sorting the best matches, and importing the results
        #    (see __compute_final_result() )

        second_collected_scores = rdd_match_pattern_frontier.collect()
        # at this step the unprocessed info is not used
        # but the parsing also evaluates the errors
        __parse_search_results(second_collected_scores, raise_exception_finally=True)

        if second_collected_scores is not None:
            all_collected_scores.extend(second_collected_scores)
        LOGGER.info("Ended: Step 8.1: gathers all results on the same list")

        search_func_id, scores_tsuid, intervals_tsuid = __compute_final_result(all_collected_scores,
                                                                               ref_pattern,
                                                                               ref_search_location,
                                                                               scores_limitation)

        LOGGER.info("Ended: Step 8.2: computes and sort final result")
        return {'ref_tsuids': ref_search_location,
                'ref_funcIds': search_func_id,
                'scores': scores_tsuid,
                'intervals': intervals_tsuid}

    except Exception as ex:
        LOGGER.error("Error occurred in find_pattern()")
        LOGGER.exception(ex)
        raise IkatsException("Error occurred in find_pattern()")


def find_pattern_by_fid(fid_pattern,
                        search_target,
                        rate_sliding,
                        rate_size_window,
                        distance,
                        scores_limitation,
                        normalize_mode):
    """
    Wrapper above find_pattern():
      - computes internal arguments of find_pattern() from the user parameters
      - runs the find_pattern(...)
    which applies the spark-distributed pattern-matching algorithm: this algorithm searches the TS pattern defined
    by **fid_pattern** argument into the unique TS specified by **search_target**.

    The window-sliding processing evaluates the distance between the TS pattern
    and successive subparts of the **search_target** TS. The window-sliding is configured with:

      - the **rate_sliding** argument,
      - the **rate_size_window** argument.

    The distance between subpart and pattern is specified with **distance** argument.

    Additional normalizing process may be specified: see **normalize_mode** argument.

    The best scores -smallest distances- are saved with two TS:

      - TS of scores defining each point by:

        - timestamp: starting timestamp of comparison,
        - value: score of comparison.
      - TS of intervals -as compared intervals may not have all the same duration -,
        defining each point by:

        - timestamp: starting timestamp of comparison,
        - value: ending timestamp of comparison from the ref_search_location TS.

    The returned result provides access to the created TS matched scores, and matched intervals: detailed below.

    Limitations on the selected best scores are set by:

      - the scores_limitation argument.

    Usage constraints: for normal use, this algorithm expects:

      - Hyp1: fid_pattern and search_target refer to timeseries with the same "usual" frequencies,
        otherwise a pre-processing is required
      - Hyp2: fid_pattern refer to a TS having number of points <= max_pattern_size,
        otherwise a resampling pre-processing is required on both TS pointed by ref_pattern/ref_search_location
      - Hyp3: following metadata are available on the TS defined by search_target and fid_pattern:
        ikats_start_date, ikats_end_date, qual_nb_points

    Note: version V1: only the first TS from search_target is taken into account !

    :param fid_pattern: functional identifier - funcId - of the pattern: pattern is a TS selection
    :type fid_pattern: str coding the funcId
    :param search_target: defines the searching area. Either a dataset name,
      or a ts_list [  { 'tsuid' : ..., 'funcId' : ... }, ...,  { 'tsuid' : ..., 'funcId' : ... }]
      or else a tsuid list [ <tsuid1>, <tsuid2>, ... ].  Note: with version V1: only the first TS from
      search_target is taken into account !
     :type search_target: str or list
     :param rate_sliding: the computed translation of search window: expressed as a
      percentage of ref_pattern size (size given by number of points): 1.0 is coding 100%.
      Constraint: 0.0 < rate_sliding.
      Beware: it is recommended that rate_sliding <= 1.0, for a fully-covering search.
    :type rate_sliding: float
    :param rate_size_window: the maximal length of search window, important for DTW distance: expressed as a
      percentage of fid_pattern size (size given by number of points): 1.0 coding for 100%
    :type rate_size_window: float
    :param distance selecting the scoring
    :type distance: str among
      |FAST_DTW_DISTANCE
      |DTW_DISTANCE,
      |MANHATTAN_DISTANCE
    :param scores_limitation: defines the maximum number of scores kept: this parameter avoids too big results
    :type scores_limitation: int
    :param normalize_mode: the choice for the local-normalizing method
    :type normalize_mode: str among CONFIG_NORM_MODES keys:
       |N_NO_NORM: no normalizing
       |N_MEAN_VARIANCE: mean-normalizing and var-normalizing applied on pattern and search_target
       |N_MEAN_VARIANCE_ONLY_SEARCH: mean-normalizing and var-normalizing applied on search_target
       |N_MEAN: mean-normalizing applied on pattern and search_target
       |N_MEAN_ONLY_SEARCH: mean-normalizing applied on search_target
       |N_VARIANCE: var-normalizing applied on pattern and search_target
       |N_VARIANCE_ONLY_SEARCH: var-normalizing applied on search_target
    :return: standard result of pattern-matching: dictionary defining lists of references instead of unique references,
       as it is also designed to be compatible with searches within multiple TS for the front-end VizTool.

      - key: 'ref_tsuids' with value: list of the original tsuids: here the unique tsuid deduced
            from search_target
      - key: 'ref_funcIds' with value: list of the original functional Ids: here the funcId of the first TS from
            search_target
      - key: 'scores' with value: list of score TS: here the unique tsuid of the TS of scores
      - key: 'intervals' with value: list of interval TS: here the unique tsuid of the TS of intervals
    :rtype dict
    """

    pattern_ref = fid_pattern

    try:
        pattern_ref = IkatsApi.fid.tsuid(fid_pattern)
    except Exception:
        pass

    LOGGER.info("pattern reference")

    if pattern_ref is None:
        raise IkatsException("Undefined tsuid for the pattern")

    ctx = ""
    try:
        if normalize_mode not in CONFIG_NORM_MODES:
            raise IkatsException("Unexpected normalizing-mode: {} not in {}".format(normalize_mode,
                                                                                    CONFIG_NORM_MODES.keys()))
        if distance not in SCORING_FUNCTIONS:
            raise IkatsException("Unexpected distance name: {} not in {}".format(distance,
                                                                                 SCORING_FUNCTIONS.keys()))
        if rate_sliding <= 0.0:
            raise Exception("Bad usage: argument rate_sliding={} should be greater than zero.".format(rate_sliding))
        if rate_size_window < 1.0:
            msg = "Bad usage: argument rate_size_window={} should superior or equal to 1.0"
            raise Exception(msg.format(rate_size_window))

        if distance == MANHATTAN_DISTANCE:
            if rate_size_window != 1.0:
                LOGGER.warning("With Manhattan: forcing rate_size_window=%s passed by the user to 1.0",
                               rate_size_window)
                rate_size_window = 1.0

        LOGGER.info("assert pattern fidelity")

        nb_points, ts_pattern = ConfigPatternMatching.assert_pattern_fidelity(ref_pattern=fid_pattern,
                                                                              normalize_mode=normalize_mode)

        LOGGER.info("pattern fidelity done")
        LOGGER.info(nb_points)
        LOGGER.info(len(ts_pattern))

        my_sliding_window_size = ceil(nb_points * rate_size_window)

        # 1/ computes the argument target_ref from the provided search_target
        #

        ctx = "defining the TS from search_target"
        if type(search_target) is str:
            # case when search_target is a dataset name => expanded into a tsuid list
            ctx = "reading the dataset content for name={}".format(search_target)
            dataset_content = IkatsApi.ds.read(search_target)

            if len(dataset_content['ts_list']) == 0:
                ts_list = [search_target]
            else:
                ts_list = dataset_content['ts_list']

        elif type(search_target) is list:
            # check that the list is not empty
            ts_list = search_target
            if len(ts_list) == 0:
                raise IkatsException("Unexpected empty list for argument search_target")
        else:
            raise IkatsException("Unexpected python type for functional type ts_selection")

    except Exception:
        msg = "Error occurred in find_pattern_by_fid(): {}"
        raise IkatsException(msg.format(ctx))

    workloads = {}

    try:
        for target_ref in ts_list:
            workloads[target_ref] = ConfigPatternMatching.assert_ts_fidelity(ref_search_location=target_ref,
                                                                             rate_size_window=rate_size_window,
                                                                             nb_points_pattern=nb_points)
    except Exception:
        pass

    try:
        all_results = defaultdict(list)

        my_spark_context = ScManager.get()

        try:
            broadcast_pattern = my_spark_context.broadcast(ts_pattern)
        except Exception:
            ScManager.get_tu_spark_context()
            my_spark_context = ScManager.get()
            broadcast_pattern = my_spark_context.broadcast(ts_pattern)

        for target_ref in ts_list:

            try:

                if type(target_ref) is dict:
                    target_ref = target_ref.get('tsuid', None)

                if target_ref is None:
                    raise IkatsException("Undefined tsuid from search_target")
                LOGGER.info("Search target: ref=%s", target_ref)

                # 2/ computes the pattern_ref tsuid from the provided fid
                #

                LOGGER.info("Pattern: ref=%s", pattern_ref)

                my_extract_period, task_intervals = workloads[target_ref]

                result = find_pattern(ref_pattern=pattern_ref,
                                      broadcast_pattern=broadcast_pattern,
                                      my_extract_period=my_extract_period,
                                      task_intervals=task_intervals,
                                      my_sliding_window_size=my_sliding_window_size,
                                      ref_search_location=target_ref,
                                      rate_sliding=rate_sliding,
                                      rate_size_window=rate_size_window,
                                      distance=distance,
                                      scores_limitation=scores_limitation,
                                      normalize_mode=normalize_mode,
                                      my_spark_context=my_spark_context)

                for key, value in result.items():
                    all_results[key].append(str(value))
            except Exception:
                # Some ts might be too small or have other problems
                # The preferred choice for now is to ignore
                # but return the ts that are good enough
                continue

    except Exception as e:
        LOGGER.error("Exception in find_by_fid_pattern :%s", e)
    finally:
        my_spark_context.stop()
        time.sleep(2)

    return dict(all_results)
