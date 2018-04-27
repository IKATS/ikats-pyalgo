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
from math import ceil, floor

import numpy as np

from ikats.core.data.ts import TimestampedMonoVal
from ikats.core.library.exception import IkatsException, IkatsConflictError
from ikats.core.resource.api import IkatsApi

"""
Rollmean Algorithm (also named Sliding window)
"""

# Define a logger for this algorithm
LOGGER = logging.getLogger(__name__)


class Alignment(enumerate):
    """
    Alignment used for Roll mean
    """
    left = 1
    center = 2
    right = 3


def rollmean(ts_data, window_size, alignment=Alignment.center):
    """
    Compute the rollmean on TS data provided

    This algorithm needs:
        * a TS
        * a window range (in number of points or in time (ms))
        * an alignment method for the output

    .. warning::
        The TS must not contain any hole (an interpolation may be applied before calling this algorithm).
        The result will be altered and may not represent the real behaviour of the rollmean algorithm

    Example:
    ~~~~~~~~

    .. code-block:: python

        # Applying a rollmean on a TS stored in ts1_data with a window having 2 points
        # The result will be left-aligned
        r1 = rollmean(ts1_data, window_size=2, alignment=Alignment.left)

    To understand what is done, given this ts1_data:

        +-----------+-------+
        | Timestamp | Value |
        +===========+=======+
        | 1000      | 1     |
        +-----------+-------+
        | 2000      | 10    |
        +-----------+-------+
        | 3000      | 20    |
        +-----------+-------+
        | 4000      | 5     |
        +-----------+-------+
        | 5000      | 8     |
        +-----------+-------+
        | 6000      | 2     |
        +-----------+-------+

    We want to apply a rollmean with a window equal to 2 points.
        * we take the first window [1000;2000]
        * the mean of points is (sum of points in window divided by the size of the window): (1 + 10) / 2 = 5.5
        * Now the alignment is left so the value 5.5 will be assigned to the timestamp 1000
        * we shift the window by one point [2000;3000]
        * Start again until the TS is fully parsed

    About alignment:
    ~~~~~~~~~~~~~~~~

    Assuming
        * K is the window length (in number of points)
        * N is the number of points of the TS

    Then
        * Left alignment corresponds to the range: [0;N-K+1]
        * Center alignment has the same length but shifted by floor(k/2)
        * Right alignment has the same length but shifted by k-1


    About size of the final TS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Due to the mean, we have less points in the result TS than the original TS
    Assuming
        * K is the window length (in number of points)
        * N is the number of points of the TS
    the length of the new TS will be N-K


    About the computation method
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    The code is highly optimized for single instance run.

    To explain the details, we will use the following values (timestamps are removed for clarity)

    ``TS_values = [1, 2, 10, 3, 4, 5, 6, 7, 8, 9, 10]``

    **Steps**

        #. Compute the cumsum
        #. Shift the cumsum by K and subtract
        #. Divide the results by K

    **Step 1**

    ``step1 = [1, 3, 13, 16, 20, 25, 31, 38, 46, 55, 65]``

    **Step 2**

        +---+---+----+----+----+----+----+----+----+----+----+--------+-----------------------+
        | 1 | 3 | 13 | 16 | 20 | 25 | 31 | 38 | 46 | 55 | 65 |        | Original              |
        +---+---+----+----+----+----+----+----+----+----+----+----+---+-----------------------+
        |       |  1 |  3 | 13 | 16 | 20 | 25 | 31 | 38 | 46 | 55 | 65| Shifted               |
        +---+---+----+----+----+----+----+----+----+----+----+----+---+-----------------------+
        | 1 | 3 | 12 | 13 |  7 |  9 | 11 | 13 | 15 | 17 | 19 |        | Result of subtraction |
        +---+---+----+----+----+----+----+----+----+----+----+--------+-----------------------+
    ``step2 = [1, 3, 12, 13,  7,  9, 11, 13, 15, 17, 19]``

    **Step 3**

        +-----+----+------+------+------+------+------+------+------+------+-----------------+
        | 3   | 12 | 13   |  7   |  9   | 11   | 13   | 15   | 17   | 19   | Remove first    |
        +-----+----+------+------+------+------+------+------+------+------+-----------------+
        | 1.5 |  6 |  6.5 |  3.5 |  4.5 |  5.5 |  6.5 |  7.5 |  8.5 |  9.5 | Divide by K(=2) |
        +-----+----+------+------+------+------+------+------+------+------+-----------------+
    ``step3 =  [1.5,  6, 6.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]``


    :param ts_data: input Timeseries to compute the rollmean on
    :type ts_data: numpy.array or TimestampedMonoVal

    :param window_size: Size of the sliding window (in number of points). Mutually exclusive with window_size
    :type window_size: int

    :param alignment: result alignment (left,right,center), default: center
    :type alignment: int

    :return: The new TS
    :rtype: TimestampedMonoVal

    :raise TypeError: Alignment must be taken within Alignment Enumerate
    :raise TypeError: ts_data must be numpy array or TimestampedMonoVal
    :raise ValueError: window_period and window_size are mutually exclusive
    :raise ValueError: window size must be positive integer
    :raise ValueError: window period must be positive integer
    :raise ValueError: window_period xor window_size must be set
    :raise ValueError: window_period must be lower than TS window
    :raise ValueError: window_size must be lower than TS length
    :raise ValueError: Window size is too big compared to TS length

    """

    LOGGER.debug("RollMean arguments:")
    LOGGER.debug(" * window_size: (%s) %s", type(window_size), window_size)
    LOGGER.debug(" * alignment: (%s) %s", type(alignment), alignment)
    LOGGER.debug(" * ts_data: (%s) len=%s", type(ts_data), len(ts_data))

    # Input check
    if type(alignment) != int or alignment not in [1, 2, 3]:
        raise TypeError("Alignment must be taken within Alignment Enumerate")
    if type(ts_data) != np.ndarray and type(ts_data) != TimestampedMonoVal:
        raise TypeError("ts_data must be numpy array or TimestampedMonoVal (got %s)" % type(ts_data))
    if window_size is not None and type(window_size) != int and window_size > 0:
        raise ValueError("window size must be positive integer")
    if window_size is not None and window_size >= len(ts_data):
        raise ValueError("window_size must be lower than TS length")

    # Convert to numpy array
    if type(ts_data) == TimestampedMonoVal:
        ts_data = ts_data.data

    if window_size == 0:
        LOGGER.error("Window size is too big compared to TS length")
        raise ValueError("Window size is too big compared to TS length")

    if window_size == 1:
        # The result is the original TS if window is equal to 1 point
        LOGGER.warning("Window size contains 1 point. The result of rollmean is the original TS")
        return TimestampedMonoVal(ts_data)

    # Work only with values, timestamps are not needed for calculation
    values = ts_data[:, 1]

    # Computation of the roll mean (highly optimized for arithmetic mean)
    ret = np.cumsum(values, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    ts_result_values = ret[window_size - 1:] / window_size

    # Selection of the correct time range (depending on alignment)
    ts_result_timestamps = []
    if alignment == Alignment.left:
        ts_result_timestamps = ts_data[:len(values) - window_size + 1, 0]
    if alignment == Alignment.center:
        ts_result_timestamps = ts_data[floor(window_size / 2):floor(window_size / 2) + len(values) - window_size + 1, 0]
    if alignment == Alignment.right:
        ts_result_timestamps = ts_data[window_size - 1:len(values), 0]

    # Build result TS
    # From 2 series (timestamps and values), build a formatted TS
    # example: timestamps = [1000, 2000, 3000] and values are [42, 15, 0]
    #          --> [[1000, 42],
    #               [2000, 15],
    #               [3000, 0]]
    #           in TimestampedMonoVal format
    ts_result = TimestampedMonoVal(np.dstack((ts_result_timestamps, ts_result_values))[0])

    return ts_result


def rollmean_tsuid(tdm, tsuid, window_size=None, window_period=None, alignment=Alignment.left, save=True):
    """
    Compute the rollmean on TS data provided

    If the Save Flag is set to True, the method returns the new TSUID that have been saved as str
    Otherwise, the returned value is the TS_result as TimestampedMonoVal

    :param tdm: Temporal Data Manager API
    :type tdm: TemporalDataMgr

    :param tsuid: TSUID of the TS to compute rollmean on
    :type tsuid: str

    :param window_period: Size of the sliding window (in ms). Mutually exclusive with window_period
    :type window_period: int

    :param window_size: Size of the sliding window (in number of points). Mutually exclusive with window_size
    :type window_size: int

    :param alignment: result alignment (left,right,center), default: center
    :type alignment: int

    :param save: Flag indicating the need to save (default: True)
    :type save: bool

    :return: The original TSUID or the new TS depending on save flag
    :rtype: str or TimestampedMonoVal

    :raise TypeError: Save flag must be bool
    """

    # input check
    if type(save) != bool:
        raise TypeError("Save flag must be bool")
    if window_size is None and window_period is None:
        raise ValueError("window_period xor window_size must be set")
    if window_size is not None and window_period is not None:
        raise ValueError("window_period and window_size are mutually exclusive")

    # Load TS content
    start_loading_time = time.time()
    ts_data = IkatsApi.ts.read([tsuid])[0]
    LOGGER.debug("TSUID: %s, Gathering time: %.3f seconds", tsuid, time.time() - start_loading_time)

    # Define the window size
    if window_period:
        window_size = get_window_size(tdm=tdm, tsuid=tsuid, ts_data=ts_data, period=window_period)

    # Compute the rolling mean
    start_computing_time = time.time()
    ts_result = rollmean(ts_data=ts_data, window_size=window_size, alignment=alignment)
    LOGGER.debug("TSUID: %s, Computation time: %.3f ms", tsuid, 1000 * (time.time() - start_computing_time))

    if save:
        # Save the result
        start_saving_time = time.time()
        short_name = "rollmean_%s" % window_size
        new_tsuid, new_fid = save_rollmean(tsuid=tsuid, ts_result=ts_result, short_name=short_name)
        LOGGER.debug("TSUID: %s(%s), Result import time: %.3f seconds", new_fid, new_tsuid,
                     time.time() - start_saving_time)
        return new_tsuid, new_fid
    else:
        # No save planned, return the computed TS
        return ts_result


def rollmean_ts_list(tdm, ts_list, window_size=None, window_period=None, alignment=Alignment.left, save=True):
    """
    Compute the rollmean on a provided TS list

    :param tdm: Temporal Data Manager API
    :type tdm: TemporalDataMgr

    :param ts_list: list of TSUID
    :type ts_list: list

    :param window_period: Size of the sliding window (in ms). Mutually exclusive with window_period
    :type window_period: int

    :param window_size: Size of the sliding window (in number of points). Mutually exclusive with window_size
    :type window_size: int

    :param alignment: result alignment (left,right,center), default: center
    :type alignment: int

    :param save: Flag indicating the need to save (default: True)
    :type save: bool

    :return: A list of dict composed of original TSUID and the information about the new TS
    :rtype: list
    """
    result = []
    for tsuid in ts_list:
        if save:
            new_tsuid, new_fid = rollmean_tsuid(tdm=tdm, tsuid=tsuid,
                                                window_size=window_size, window_period=window_period,
                                                alignment=alignment, save=save)

            result.append({
                "tsuid": new_tsuid,
                "funcId": new_fid,
                "origin": tsuid
            })
        else:
            new_tsuid = rollmean_tsuid(tdm=tdm, tsuid=tsuid,
                                       window_size=window_size, window_period=window_period,
                                       alignment=alignment, save=save)
            result.append({
                "tsuid": new_tsuid,
                "funcId": "TS_Not_Saved_from_%s" % tsuid,
                "origin": tsuid
            })
    return result


def rollmean_ds(tdm, ds_name, window_period=None, window_size=None, alignment=Alignment.left, save=True):
    """
    Compute the rollmean on a provided dataset name

    :param tdm: Temporal Data Manager API
    :type tdm: TemporalDataMgr

    :param ds_name: Name of the dataset to work on
    :type ds_name: str

    :param window_period: Size of the sliding window (in ms). Mutually exclusive with window_size
    :type window_period: int

    :param window_size: Size of the sliding window (in number of points). Mutually exclusive with window_period
    :type window_size: int

    :param alignment: result alignment (left,right,center), default: center
    :type alignment: int

    :param save: Flag indicating the need to save (default: True)
    :type save: bool

    :return: A list of dict composed of original TSUID and the information about the new TS
    :rtype: list
    """
    ts_list = tdm.get_data_set(data_set=ds_name)['ts_list']
    return rollmean_ts_list(tdm=tdm, ts_list=ts_list, window_size=window_size, window_period=window_period,
                            alignment=alignment, save=save)


def get_window_size(tdm, tsuid, ts_data, period):
    """
    Gets the window size (in number of points) corresponding to a specific period for the given tsuid

    :param tdm: Temporal Data Manager API
    :type tdm: TemporalDataMgr

    :param tsuid: original TSUID used for computation
    :type tsuid: str

    :param ts_data: input Timeseries to compute the rollmean on
    :type ts_data: numpy.array

    :param period: Size of the sliding window (in ms).
    :type period: int

    :return: the number of points corresponding to the period
    :rtype: int
    """
    if period is None:
        raise TypeError("Period must be provided")
    if type(period) != int and period <= 0:
        raise ValueError("window period must be positive integer")
    if period >= (ts_data[-1][0] - ts_data[0][0]):
        raise ValueError("window_period must be lower than TS window (%s)" % (ts_data[-1][0] - ts_data[0][0]))

    # noinspection PyBroadException
    try:
        meta_data = tdm.get_meta_data(ts_list=[tsuid])
        if 'qual_ref_period' in meta_data[tsuid]:
            window_size = ceil(period / meta_data[tsuid]['qual_ref_period'])
            LOGGER.debug("Period (%sms) -> Window size = %s", period, window_size)
            return window_size
    except Exception:
        pass

    # If no meta data has been found (or if error occurred)
    # Count the number manually
    LOGGER.debug("qual_ref_period metadata not found for tsuid:%s", tsuid)

    # Calculation of effective window size by parsing each date until it founds the period
    for i, _ in enumerate(ts_data):
        if (ts_data[i][0] - ts_data[0][0]) >= period:
            LOGGER.debug("Period (%sms) -> Window size = %s", period, i)
            return i

    raise ValueError("Window size is too big compared to TS length")


def save_rollmean(tsuid, ts_result, short_name="rollmean"):
    """
    Saves the TS to database
    It copies some attributes from the original TSUID, that is why it needs the tsuid

    :param tsuid: original TSUID used for computation
    :type tsuid: str

    :param ts_result: TS resulting of the operation
    :type ts_result: TimestampedMonoVal

    :param short_name: Name used as short name for Functional identifier
    :type short_name: str

    :return: the created TSUID and its associated FID
    :rtype: str, str

    :raise IOError: if an issue occurs during the import
    """
    try:
        # Retrieve timeseries information (funcId)
        original_fid = IkatsApi.ts.fid(tsuid=tsuid)

        # Generate unique functional id for resulting timeseries
        new_fid = '%s_%s' % (str(original_fid), short_name)
        try:
            IkatsApi.ts.create_ref(new_fid)
        except IkatsConflictError:
            # TS already exist, append timestamp to be unique
            new_fid = '%s_%s_%s' % (str(original_fid), short_name, int(time.time() * 1000))
            IkatsApi.ts.create_ref(new_fid)

        # Import timeseries result in DB
        res_import = IkatsApi.ts.create(fid=new_fid,
                                        data=ts_result.data,
                                        parent=tsuid,
                                        generate_metadata=True)

        # Returns the TSUID if no issue happened
        if res_import['status']:
            return res_import['tsuid'], new_fid
        else:
            raise IOError(" IkatsApi.ts.create(fid={}, data=..., ...) failed".format(original_fid))

    except Exception:
        raise IkatsException("save_rollmean() failed")
