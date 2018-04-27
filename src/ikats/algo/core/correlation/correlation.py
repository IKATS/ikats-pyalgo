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

from ikats.core.library.spark import ScManager, ListAccumulatorParam
from ikats.core.resource.client import TemporalDataMgr
import numpy as np


def _ccf(ts1, ts2, lag_max=None):
    """
    Produces Cross-correlation function for 1d time series based on np.correlate.

    :returns: cross-correlation function of x and y for negative and positive lag
    :rtype: np.array

    :param ts1: time series
    :param ts2: time series
    :param lag_max: maximum lag between timeseries
                    lag_max is a number of points :
                    correlation will be calculated on [-lag_max .. lag_max]
                    cross correlation function length will be (2 x lag_max +1)
            example :
            x = [ 1 2 3 4 ]
            y = [ 2 3 4 5 ]
            ccf = [-0.45 -0.30  0.25  1.00  0.25 -0.30 -0.45] by default (no lag_max specified)
            if lag_max = 2 , ccf will be [-0.30  0.25  1.00  0.25 -0.30]
            if lag_max = 1 , ccf will be [0.25  1.00  0.25]

    :type ts1: np.array
    :type ts2: np.array
    :type lag_max: positive int (ignored else)
    """

    cvf = np.correlate((ts1 - ts1.mean()), (ts2 - ts2.mean()), 'full') / len(ts1)
    ccf_value = cvf / (np.std(ts1) * np.std(ts2))

    # if a maximum lag is specified
    if lag_max is not None and lag_max >= 0:
        # evaluating if lag_max is shorter than basic lag
        diff = int(((len(ccf_value) - 1) / 2) - lag_max)
        if diff > 0:
            # slicing result to lag_max
            ccf_value = ccf_value[diff: -diff]

    return ccf_value


def spark_ccf(tdm, tsuid_list_or_dataset, lag_max=None, tsuids_out=False,
              cut_ts=False):
    """
    This function calculates the maximum of the cross correlation function matrix between all ts
    in **tsuid_list_or_dataset** IN A DISTRIBUTED MODE (using spark)

    Cross correlation is a correlation between two timeseries whose one is delayed of successive lag
    values. Result of CCF is a timeseries (correlation function of the lag between timeseries).
    This function keep the maximum value of the CCF function generated and pull it in the matrix for
    corresponding timeseries couple.

    :returns: a string matrix (whose size is equal to the number of tsuids in tsuid_list_or_dataset
              plus one line and one column for headers)
    :rtype: ndarray

    :param tdm: Temporal Data Manager client
    :param tsuid_list_or_dataset: list of identifiers of the time series or dataset name
    :param lag_max: maximum lag between timeseries (cf. _ccf function for more details)
    :param tsuids_out: True to fill headers with tsuids
                       False to fill headers with functional ids
    :param cut_ts: Cut the TS list to the min-length if set to True

    :type tdm: TemporalDataMgr
    :type tsuid_list_or_dataset: list of str or str
    :type lag_max: positive int
    :type tsuids_out: boolean
    :type cut_ts: bool

    :raises TypeError: if tdm is not a TemporalDataMgr
    :raises TypeError: if tsuid_list_or_dataset is not a list nor a string
    :raises TypeError: if tsuids_out is not a boolean
    """
    if type(tdm) is not TemporalDataMgr:
        raise TypeError("tdm must be a TemporalDataMgr")

    if type(tsuid_list_or_dataset) is not list and type(tsuid_list_or_dataset) is not str:
        raise TypeError(
            "tsuid_list_or_dataset must be a list of string OR a string")

    if type(tsuids_out) is not bool:
        raise TypeError("tsuids_out must be a boolean")

    if type(cut_ts) is not bool:
        raise TypeError("cut_ts must be a boolean")

    if type(tsuid_list_or_dataset) is list:
        # input is a list of tsuid
        tsuid_list = tsuid_list_or_dataset
    else:
        # input is a dataset name
        dataset = tdm.get_data_set(tsuid_list_or_dataset)
        tsuid_list = dataset['ts_list']

    if tsuids_out:
        ts_list = tsuid_list
    else:
        ts_list = __retrieve_func_id(tdm, tsuid_list)

    md_list = tdm.get_meta_data(tsuid_list)

    # initialize size of time series
    min_ts_size = md_list[tsuid_list[0]]['qual_nb_points']

    if cut_ts:
        for ts in tsuid_list:
            min_ts_size = min(min_ts_size, md_list[ts]['qual_nb_points'])
    else:
        # check time series have same length
        for ts in tsuid_list:
            size_ts = md_list[ts]['qual_nb_points']
            if size_ts != min_ts_size:
                raise ValueError('time series do not have same length')

    # Create or get a spark Context
    sc = ScManager.get()

    # Build the RDD with TSUIDS
    rdd = sc.parallelize(tsuid_list)

    # Create a broadcast for spark jobs
    broadcast = sc.broadcast(
        {"host": tdm.host, "port": tdm.port, "size_of_ts": min_ts_size, "lag_max": lag_max})

    # Create an accumulator to store the results of the spark workers
    accumulator = sc.accumulator(dict(), ListAccumulatorParam())

    def run_ccf_spark(working_tsuids):
        """
        Method called by spark job
        :param working_tsuids: rdd item
        :type working_tsuids: tuple
        """

        # cross correlation is equal to 1 if timeseries are the same
        if working_tsuids[0] == working_tsuids[1]:
            result = 1
        else:
            spark_tdm = TemporalDataMgr(
                host=broadcast.value['host'], port=broadcast.value['port'])

            result = __run_max_ccf_ts_list(tdm=spark_tdm,
                                           tsuids=list(working_tsuids),
                                           size=int(broadcast.value['size_of_ts']),
                                           lag_max=broadcast.value['lag_max'])

        accumulator.add({";".join(list(working_tsuids)): result})

    # Get TS content and perform ccf calculation using spark distribution to increase performance
    # for each element of rdd which is a couple of timeseries
    # the list of couples is first sorted then duplicates are suppressed to avoid doing same calculation
    # as for (a,b) and (b,a)
    rdd.cartesian(rdd).map(
        lambda x: tuple(sorted(list(x)))).distinct().foreach(run_ccf_spark)

    # Retrieving result from accumulator to fill matrix result
    ts_nb = len(tsuid_list)
    matrix_corr = np.zeros((ts_nb, ts_nb))
    for str_couple in accumulator.value:
        couple = str_couple.split(';')
        matrix_corr[tsuid_list.index(couple[0]), tsuid_list.index(couple[1])] = accumulator.value[str_couple]
        matrix_corr[tsuid_list.index(couple[1]), tsuid_list.index(couple[0])] = accumulator.value[str_couple]

    # fill final matrix with headers
    matrix = __fill_headers_to_final_matrix(matrix_corr, ts_list)

    return matrix


def __run_max_ccf_ts_list(tdm, tsuids, size, lag_max):
    """
    Calculates the max of the cross correlation function for a couple of timeseries

    :returns: list of data, list of tsuids
    :rtype: list, list

    :param tdm: temporal data manager
    :param tsuids: list of two tsuids
    :param size: common size for cutting timeseries
    :param lag_max: maximum lag between timeseries (cf. _ccf function for more details)

    :type tsuids: TemporalDataManager
    :type tsuids: list of str
    :type size: int
    :type lag_max: positive int
    """
    try:
        ts_data_list = tdm.get_ts(tsuid_list=tsuids)
    except Exception:
        raise

    # Keep only values column, and cutting to size
    ts1 = np.asarray(ts_data_list[0][:size, 1])
    ts2 = np.asarray(ts_data_list[1][:size, 1])

    # run ccf for positive and negative lag, keeping the absolute max
    ccf_fcn = _ccf(ts1, ts2, lag_max)

    return __get_max_abs_value(ccf_fcn)


def ccf(tdm, tsuid_list_or_dataset, lag_max=None, tsuids_out=False, cut_ts=False):
    """
    This function calculates the maximum of the cross correlation function matrix between all ts
    in tsuid_list_or_dataset in a serial mode.
    The result is normalized (between -1 and 1)

    Cross correlation is a correlation between two timeseries whose one is delayed of successive lag
    values. Result of CCF is a timeseries (correlation function of the lag between timeseries).
    This function keep the maximum value of the CCF function generated and pull it in the matrix for
    corresponding timeseries couple.

    :returns: a string matrix (whose size is equal to the number of tsuids in tsuid_list_or_dataset
              plus one line and one column for headers)
    :rtype: np.ndarray

    :param tdm: Temporal Data Manager client
    :param tsuid_list_or_dataset: list of identifiers of the time series or dataset name
    :param lag_max: maximum lag between timeseries (cf. _ccf function for more details)
    :param tsuids_out: True to fill headers with tsuids
                       False to fill headers with functional ids
    :param cut_ts: Cut the TS list to the min-length if set to True

    :type tdm: TemporalDataMgr
    :type tsuid_list_or_dataset: list of str or str
    :type lag_max: positive int
    :type tsuids_out: boolean
    :type cut_ts: boolean

    :raises TypeError: if tsuids_out is not a boolean
    """
    if type(tsuids_out) is not bool:
        raise TypeError("tsuids_out must be a boolean")

    # retrieve data from temporal data manager
    ts_data_list, tsuid_list = __retrieve_data(
        tdm, tsuid_list_or_dataset)

    if tsuids_out:
        ts_list = tsuid_list
    else:
        ts_list = __retrieve_func_id(tdm, tsuid_list)

    # number and size of time series
    ts_nb = len(ts_data_list)
    ts_size = len(ts_data_list[0])

    if cut_ts:
        for ts in ts_data_list:
            ts_size = min(len(ts), ts_size)
    else:
        # check time series have same length
        for ts in ts_data_list:
            if len(ts) != ts_size:
                raise ValueError('time series do not have same length')

    # matrix initialization
    matrix_corr = np.zeros([ts_nb, ts_nb])

    for index1, _ in enumerate(ts_data_list):
        matrix_corr[index1, index1] = 1
        # Conversion ts1 data from list (keeping only value column) to an array
        ts1 = np.asarray(ts_data_list[index1][:ts_size, 1])
        for index2 in range(index1 + 1, len(ts_data_list)):
            # Conversion ts2 data from list (keeping only value column) to an
            # array
            ts2 = np.asarray(ts_data_list[index2][:ts_size, 1])
            # cross correlation calculation
            # keeping the maximum absolute value between cross correlation with
            # positive and with negative lag
            ccf_fcn = _ccf(ts1, ts2, lag_max)
            max_ccf = __get_max_abs_value(ccf_fcn)

            # fill matrix with result (max of ccf is commutative)
            matrix_corr[index1, index2] = max_ccf
            matrix_corr[index2, index1] = max_ccf

    # fill final matrix with headers
    matrix = __fill_headers_to_final_matrix(matrix_corr, ts_list)

    return matrix


def __get_max_abs_value(array):
    """
    Retrieve the maximum absolute value of an array

    :returns: the maximum absolute value of the input array
    :rtype: real

    :param array: numpy array
    :type array: np.array
    """
    mini = min(array)
    result = max(array)
    if abs(mini) > result:
        result = mini

    return result


def __retrieve_data(tdm, tsuid_list_or_dataset):
    """
    retrieve timeseries data in database

    :returns: list of data, list of tsuids
    :rtype: list, list

    :param tdm: Temporal Data Manager client
    :param tsuid_list_or_dataset: list of identifiers of the time series or dataset name

    :type tdm: TemporalDataMgr
    :type tsuid_list_or_dataset: list of str or str

    :raises TypeError: if tdm is not a TemporalDataMgr
    :raises TypeError: if tsuid_list_or_dataset is not a list nor a string
    :raises TypeError: if tsuids_out is not a boolean
    :raises ValueError: if every time series from the tsuid_list_or_dataset do
                        not have the same length
    """
    if type(tdm) is not TemporalDataMgr:
        raise TypeError("tdm must be a TemporalDataMgr")

    if type(tsuid_list_or_dataset) is not list and type(tsuid_list_or_dataset) is not str:
        raise TypeError(
            "tsuid_list_or_dataset must be a list of string OR a string")

    if type(tsuid_list_or_dataset) is list:
        # input is a list of tsuid
        tsuid_list = tsuid_list_or_dataset
    else:
        # input is a dataset name
        dataset = tdm.get_data_set(tsuid_list_or_dataset)
        tsuid_list = dataset['ts_list']

    # Retrieval of all time series data in data base
    try:
        ts_data_list = tdm.get_ts(tsuid_list=tsuid_list)
    except Exception:
        raise

    return ts_data_list, tsuid_list


def __retrieve_func_id(tdm, tsuid_list):
    """
    Retrieve func ids if exist for input tsuids in tsuid_list

    :returns: list of functional identifiers
    :rtype: list of string

    :param tdm: Temporal Data Manager client
    :param tsuid_list: list of TS Identifiers of the time series

    :type tdm: TemporalDataMgr
    :type tsuid_list: list
    """
    ts_list = []
    try:
        result = tdm.search_functional_identifiers(
            criterion_type='tsuids', criteria_list=tsuid_list)
        dict_fid = {}
        # by default no functional ids found
        for tsuid in tsuid_list:
            dict_fid[tsuid] = 'NOFID_' + tsuid

        # retrieve all functional ids found in database
        for i in range(result.__len__()):
            dict_fid[result[i]['tsuid']] = result[i]['funcId']

        # fill ts_list with results
        for tsuid in tsuid_list:
            ts_list.append(dict_fid[tsuid])

    except ValueError:
        # No functional ids found in database
        for tsuid in tsuid_list:
            ts_list.append('NOFID_' + tsuid)
    except Exception:
        raise

    return ts_list


def __fill_headers_to_final_matrix(matrix_corr, ts_list):
    """
    Defining a matrix of objects with one more column and one more line
    to store ts identifiers (tsuids or functional ids) + correlation coefficients

    :returns: a string matrix with headers
    :rtype: matrix

    :param matrix_corr: Correlation matrix
    :param ts_list: list of identifiers of the time series or dataset name

    :type matrix_corr: np.ndarray
    :type ts_list: list of str or str
    """
    ts_nb = len(ts_list)
    matrix = np.zeros([ts_nb + 1, ts_nb + 1], dtype=object)
    matrix[0, 0] = ''
    for i in range(ts_nb):
        # first line filled with ts functional ids
        matrix[0, i + 1] = ts_list[i]
        # first column filled with ts functional ids
        matrix[i + 1, 0] = ts_list[i]

    # conversion of correlation matrix values float => string
    matrix_corr_strings = np.array(
        ["%.15f" % x for x in matrix_corr.reshape(matrix_corr.size)])
    matrix_corr_strings = matrix_corr_strings.reshape(matrix_corr.shape)

    # sub-matrix is filled with matrix result of correlation formatted in
    # string
    matrix[1:, 1:] = matrix_corr_strings

    return matrix


def pearson_correlation_matrix(tdm, tsuid_list_or_dataset, tsuids_out=False, cut_ts=False):
    """
    DEPRECATED METHOD: see module loop instead

    This function calculates the pearson correlation matrix between all ts
    in tsuid_list_or_dataset

    :returns: a string matrix (whose size is equal to the number of tsuid in tsuid_list_or_dataset
              plus one line and one column for headers)
    :rtype: np.ndarray

    :param tdm: Temporal Data Manager client
    :param tsuid_list_or_dataset: list of identifiers of the time series or dataset name
    :param tsuids_out: True to fill headers with tsuids
                       False to fill headers with functional ids
    :param cut_ts: Cut the TS list to the min-length if set to True

    :type tdm: TemporalDataMgr
    :type tsuid_list_or_dataset: list of str or str
    :type tsuids_out: boolean
    :type cut_ts: boolean

    :raises TypeError: if tsuids_out is not a boolean
    :raises ValueError: if every time series from the tsuid_list_or_dataset do
                        not have the same length
    """
    if type(tsuids_out) is not bool:
        raise TypeError("ERROR : tsuids_out must be a boolean")

    # retrieve data from temporal data manager
    ts_data_list, tsuid_list = __retrieve_data(
        tdm, tsuid_list_or_dataset)

    if tsuids_out:
        ts_list = tsuid_list
    else:
        ts_list = __retrieve_func_id(tdm, tsuid_list)

    # number and size of time series
    ts_nb = len(ts_data_list)
    ts_size = len(ts_data_list[0])

    if cut_ts:
        for ts in ts_data_list:
            ts_size = min(len(ts), ts_size)
        # Cut TS to shortest one
        for index, val in enumerate(ts_data_list):
            ts_data_list[index] = val[:ts_size]
    else:
        # check time series have same length
        for ts in ts_data_list:
            if len(ts) != ts_size:
                raise ValueError('time series do not have same length')

    # matrix initialization
    matrix_corr = np.zeros([ts_nb, ts_nb])

    # calculation is done if there is at least one non-empty time series
    if ts_nb > 0 and ts_size > 0:

        # Conversion ts data from list (keeping only value column) to an array
        ts_data_array = np.zeros([ts_nb, ts_size])
        for i in range(ts_nb):
            ts_data_array[i] = np.asarray(ts_data_list[i][:, 1])

        # Correlation matrix calculation using numpy corrcoef
        matrix_corr = np.corrcoef(ts_data_array)

    return __fill_headers_to_final_matrix(matrix_corr, ts_list)
