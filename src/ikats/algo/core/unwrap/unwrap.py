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
from enum import Enum

import numpy as np
from numpy import pi
from ikats.core.data.SparkUtils import SparkUtils
from ikats.core.library.exception import IkatsException, IkatsConflictError
from ikats.core.library.spark import ScManager
from ikats.core.resource.api import IkatsApi
from ikats.core.resource.client.temporal_data_mgr import DTYPE

LOGGER = logging.getLogger(__name__)


class TSUnit(Enum):
    """
    Units enumeration usable for angle
    """
    Degrees = "Degrees"
    Radians = "Radians"


class Timings(object):
    """
    Timings measurements class.
    Use for performance measurements
    """

    def __init__(self, read=0, compute=0, create=0, points=0):
        # Read time
        self.read = read
        # Computing time
        self.compute = compute
        # Creation time
        self.create = create
        # Number of points
        self.points = points

    def __iadd__(self, other):
        return self + other

    def __add__(self, other):
        """
        Combine other Timings instance with this one
        :param other: the other instance
        :type other: Timings
        :return: this instance
        :rtype: Timings
        """
        self.read += other.read
        self.compute += other.compute
        self.create += other.create
        self.points += other.points
        return self

    def stats(self):
        """
        Return a string composed of 4 information:
        - R : Time to read the original timeseries (per point) + percentage of total spent time
        - C : Computation time (per point) + percentage of total spent time
        - W : Writing time of the result in database (per point) + percentage of total spent time
        - N : Number of points of the time series -

        :return: the string corresponding to the wanted information
        :rtype: str
        """

        total = self.read + self.compute + self.create
        return "R:%.3fp/s(%.1f%%,%.3fs), C:%.3fp/s(%.1f%%,%.3fs), W:%.3fp/s(%.1f%%,%.3fs), N:%dpoints(%.3fs)" % (
            self.points / self.read, 100 * self.read / total, self.read,
            self.points / self.compute, 100 * self.compute / total, self.compute,
            self.points / self.create, 100 * self.create / total, self.create,
            self.points, total
        )


def unwrap_dataset(dataset_name, unit=TSUnit.Radians, discontinuity=pi, fid_pattern="%(fid)s__unwrap"):
    """
    Unwrap a dataset by changing deltas between values to 2*discontinuity complement.
    Unwrap phase of each TS composing the dataset

    :param dataset_name: name of the dataset to unwrap
    :param unit: TS unit : "Degrees" or "Radians" (default)
    :param discontinuity: Maximum discontinuity between values.
    :param fid_pattern: Pattern of the new FID ('%(fid)s' will be replaced by original FID)


    :type dataset_name: str
    :type unit: str or TSUnit
    :type discontinuity: float or None
    :type fid_pattern: str

    :return: a new ts_list
    :rtype: list
    """

    # Getting the TS list from the dataset
    ts_list = IkatsApi.ds.read(ds_name=dataset_name)['ts_list']

    # Unwraps the TS list gathered
    unwrap_ts_list(ts_list=ts_list, fid_pattern=fid_pattern, discontinuity=discontinuity, unit=unit)


def unwrap_ts_list(ts_list, unit=TSUnit.Radians, discontinuity=None, fid_pattern="%(fid)s__unwrap", use_spark=True):
    """
    Unwrap a list of TS by changing deltas between values to 2*discontinuity complement.
    Unwrap phase of each TS composing the dataset

    :param ts_list: list of TSUID to unwrap
    :param unit: TS unit : "Degrees" or "Radians" (default)
    :param discontinuity: Maximum discontinuity between values.
    :param fid_pattern: Pattern of the new FID ('%(fid)s' will be replaced by original FID)
    :param use_spark: Set to True to use spark. True is default

    :type ts_list: list
    :type unit: str or TSUnit
    :type discontinuity: float or None
    :type fid_pattern: str
    :type use_spark: bool

    :return: a new ts_list
    :rtype: list

    :raises TypeError: if input is not well formatted
    """

    if not isinstance(ts_list, list) or len(ts_list) == 0:
        raise TypeError("ts_list shall be a list having at least one TS")

    if discontinuity is None:
        raise ValueError("Discontinuity is not filled")

    results = []
    if use_spark:
        # Get Spark Context
        spark_context = ScManager.get()

        try:

            # Parallelize 1 TS = 1 partition
            rdd_ts_list = spark_context.parallelize(ts_list, len(ts_list))

            rdd_results = rdd_ts_list.map(
                lambda x: unwrap_tsuid(tsuid=x["tsuid"], fid=x["funcId"], fid_pattern=fid_pattern,
                                       discontinuity=discontinuity,
                                       unit=unit))

            # Persist data to not recompute them again
            # (Functional identifier reservation called multiple times through IkatsApi.ts.create_ref)
            rdd_results.cache()

            timings = rdd_results.map(lambda x: x[1]).reduce(lambda x, y: x + y)

            results = rdd_results.map(lambda x: x[0]).collect()

            rdd_results.unpersist()

            LOGGER.debug("Unwrapping %s TS using Spark: %s", len(ts_list), timings.stats())
        finally:
            # Stop the context
            ScManager.stop()
    else:
        timings = Timings()
        for item in ts_list:
            tsuid = item["tsuid"]
            fid = item["funcId"]
            result, tsuid_timings = unwrap_tsuid(tsuid=tsuid, fid=fid, fid_pattern=fid_pattern,
                                                 discontinuity=discontinuity,
                                                 unit=unit)
            results.append(result)
            timings += tsuid_timings

        LOGGER.debug("Unwrapping %s TS: %s", len(ts_list), timings.stats())
    return results


def unwrap_tsuid(tsuid, fid=None, unit=TSUnit.Radians, discontinuity=pi, fid_pattern="%(fid)s__unwrap",
                 chunk_size=75000):
    """
    Unwrap a tsuid  by changing deltas between values to 2*pi complement.
    Unwrap radian phase of the tsuid by changing absolute jumps greater than <discontinuity> to their 2*pi complement.

    :param tsuid: TSUID to unwrap
    :param unit: TS unit : "Degrees" or "Radians" (default)
    :param fid: Functional Identifier corresponding to tsuid (optional, only if known)
    :param discontinuity: Maximum discontinuity between values.
    :param fid_pattern: Pattern of the new FID ('%(fid)s' will be replaced by original FID)
    :param chunk_size: Number of points per chunk (75000 by default)

    :type tsuid: str
    :type unit: str or TSUnit
    :type discontinuity: float or str
    :type fid: str or None
    :type fid_pattern: str
    :type chunk_size:int

    :return: the generated tsuid with associated fid as a dict {tsuid: x, funcId: x}
    :rtype: dict

    :raises IkatsConflictError: if the requested reference already exist
    :raises ValueError: if the discontinuity has a bad value
    """

    md_list = IkatsApi.md.read(ts_list=[tsuid])
    if fid is None:
        fid = IkatsApi.ts.fid(tsuid=tsuid)
    new_fid = fid_pattern % ({'fid': fid})

    if 'qual_nb_points' not in md_list[tsuid]:
        raise IkatsException("Metadata qual_nb_points doesn't exist for %s", fid)

    if isinstance(discontinuity, str):
        if "pi" in discontinuity:
            # Convert "pi" to numpy pi
            try:
                discontinuity = eval(discontinuity)
            except:
                raise ValueError("Bad value for discontinuity")
        else:
            try:
                discontinuity = float(discontinuity)
            except ValueError:
                raise ValueError("Discontinuity is not a number")
            if discontinuity < 0:
                raise ValueError("Discontinuity shall be positive")

    # Abort if the TS already exist with this Functional Identifier
    try:
        new_tsuid = IkatsApi.ts.create_ref(fid=new_fid)
    except IkatsConflictError:
        raise

    unit_str = unit
    if type(unit) == TSUnit:
        unit_str = unit.value

    try:
        # Split TS into chunks
        ts_chunks = SparkUtils.get_chunks(tsuid=tsuid, md_list=md_list, chunk_size=chunk_size)

        # Work on a single chunk at a time to not overload the memory usage per TS
        offset = None
        timings = Timings()
        for chunk_idx, chunk in enumerate(ts_chunks):
            sd = chunk[1]
            ed = chunk[2]
            offset, chunk_timings = unwrap_tsuid_part(tsuid=tsuid, sd=sd, ed=ed,
                                                      new_fid=new_fid, discontinuity=discontinuity,
                                                      last_point_prev=offset, unit=unit_str)
            LOGGER.debug("Processing chunk %s/%s for tsuid %s", chunk_idx + 1, len(ts_chunks), tsuid)

            timings += chunk_timings

        # Copy metadata
        IkatsApi.ts.inherit(tsuid=new_tsuid, parent=tsuid)
        IkatsApi.md.create(tsuid=new_tsuid, name="ikats_start_date", value=md_list[tsuid]["ikats_start_date"],
                           data_type=DTYPE.date)
        IkatsApi.md.create(tsuid=new_tsuid, name="ikats_end_date", value=md_list[tsuid]["ikats_end_date"],
                           data_type=DTYPE.date)
        IkatsApi.md.create(tsuid=new_tsuid, name="qual_nb_points", value=md_list[tsuid]["qual_nb_points"],
                           data_type=DTYPE.number)

        # qual_ref_period also copied because it is the same and is commonly used to display TS
        if "qual_ref_period" in md_list[tsuid]:
            IkatsApi.md.create(tsuid=new_tsuid, name="qual_ref_period", value=md_list[tsuid]["qual_ref_period"],
                               data_type=DTYPE.number)

        LOGGER.debug("Unwrap timings for %s (%s chunks) : %s", new_fid, len(ts_chunks), timings.stats())

    except Exception:
        # If any error occurs, release the incomplete TSUID
        IkatsApi.ts.delete(tsuid=new_tsuid, no_exception=True)
        raise

    return {"tsuid": new_tsuid, "funcId": new_fid}, timings


def unwrap_tsuid_part(tsuid, sd, ed, new_fid, discontinuity, last_point_prev=None, unit=TSUnit.Radians):
    """
    Unwrap a tsuid part by changing deltas between values to 2*pi complement.
    Unwrap radian phase of the tsuid by changing absolute jumps greater than <discontinuity> to their 2*pi complement.

    To connect parts (chunks) together, the algorithm re-uses the last point of the previous chunk (if it exists).
    By adding this point at the beginning of the current range, the unwrap will handle the following cases:
    * The TS chunk bounds times corresponds to a discontinuity to handle
    * The previous chunk unwrapping implied a shift applied to this chunk at the beginning (to prevent from having new
      discontinuities)

    :param tsuid: TSUID to unwrap
    :param sd: Start date of the TSUID part to work on (EPOCH in ms)
    :param ed: End date of the TSUID part to work on (EPOCH in ms)
    :param new_fid: Functional identifier of the unwrapped TS
    :param unit: TS unit : "Degrees" or "Radians" (default)
    :param discontinuity: Maximum discontinuity between values.
    :param last_point_prev: Offset to apply when piping unwraps

    :type tsuid: str
    :type sd: int
    :type ed: int
    :type new_fid: str
    :type unit: str
    :type discontinuity: float
    :type last_point_prev: np.array or None

    :return: the time and value of the last unwrapped point (to compute offset for next chunk) +
             time performance measurements
    :rtype: float, Timings
    """

    # Get Data points
    time_start = time.time()
    try:
        dps = IkatsApi.ts.read(tsuid_list=[tsuid], sd=sd, ed=ed)[0]
    except:
        # Raise again any encountered error regarding TS points reading
        raise
    # Define which point is the first to save (to skip the last point of previous chunk)
    first_point_to_save = 0
    if last_point_prev is not None:
        # if previous chunk has been computed, last_point_prev contains the last point of the previous chunk.
        # This point is used to link this new chunk to the previous one but it is not necessary to save it (already done
        # during previous chunk processing)
        dps = np.vstack((last_point_prev, dps))
        first_point_to_save = 1
    values = dps[:, 1]
    time_read_ts_duration = time.time() - time_start

    time_start = time.time()
    # Compute the unwrap for this chunk
    if unit == TSUnit.Degrees.value:
        # Use Degrees
        new_values = np.rad2deg(list(np.unwrap(p=np.deg2rad(list(values)), discont=np.deg2rad(discontinuity))))
    elif unit == TSUnit.Radians.value:
        # Use Radians
        new_values = np.unwrap(values, discont=discontinuity)
    else:
        raise ValueError("Unhandled TS Unit: %s" % unit)
    new_points = np.column_stack((dps[first_point_to_save:, 0], new_values[first_point_to_save:]))
    time_compute_duration = time.time() - time_start

    # Save it
    time_start = time.time()
    IkatsApi.ts.create(fid=new_fid, data=new_points, generate_metadata=False)
    time_create_ts_duration = time.time() - time_start

    return new_points[-1], Timings(time_read_ts_duration, time_compute_duration, time_create_ts_duration, len(values))
