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
from unittest import TestCase
from unittest import skip

import numpy as np

from ikats.algo.core.slope import LOGGER, compute_slope
from ikats.core.resource.api import IkatsApi, DTYPE


def log_to_stdout(logger_to_use):
    """
    Allow to print some loggers to stdout
    :param logger_to_use: the LOGGER object to redirect to stdout
    """

    logger_to_use.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
    # Create another handler that will redirect log entries to STDOUT
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger_to_use.addHandler(stream_handler)


# Prints the Slope logger to display
log_to_stdout(LOGGER)


def gen_ts(ts_id):
    """
    Generate a TS in database used for test bench where id is defined

    :param ts_id: Identifier of the TS to generate (see content below for the structure)
    :type ts_id: int

    :return: the TSUID and funcId
    :rtype: dict
    """

    # Build TS identifier
    fid = "UNIT_TEST_Slope_%s" % ts_id

    if ts_id == 1:
        ts_content = [[1e12, 5.0],
                      [1e12 + 1000, 6.2],
                      [1e12 + 2000, 6.0],
                      [1e12 + 3600, 8.0],
                      [1e12 + 4000, -15.0],
                      [1e12 + 5000, 2.0],
                      [1e12 + 6000, 6.0],
                      [1e12 + 7000, 3.0],
                      [1e12 + 8000, 2.0],
                      [1e12 + 9000, 42.0],
                      [1e12 + 10000, 8.0],
                      [1e12 + 11000, 8.0],
                      [1e12 + 12000, 8.0],
                      [1e12 + 13000, 8.0]]
    elif ts_id == 2:
        ts_content = [[1e12, 5.0],
                      [1e12 + 1000, 6.2],
                      [1e12 + 2000, 6.0],
                      [1e12 + 3000, 8.0],
                      [1e12 + 4000, -15.0],
                      [1e12 + 5000, 2.0],
                      # Hole
                      [1e12 + 100000, 6.0],
                      # Hole
                      [1e12 + 200000, 3.0],
                      [1e12 + 201000, 2.0],
                      [1e12 + 202000, 42.0],
                      [1e12 + 203000, 8.0],
                      [1e12 + 204000, 8.0],
                      [1e12 + 205000, 8.0],
                      [1e12 + 206000, 8.0]]
    elif ts_id == 3:
        ts_content = [[1e12, 5.0],
                      [1e12 + 1000, 6.2],
                      [1e12 + 2000, 6.0],
                      [1e12 + 3000, 8.0],
                      [1e12 + 4000, -15.0],
                      [1e12 + 5000, 2.0],
                      # Hole
                      [1e12 + 100000, 6.0],
                      # Hole
                      [1e12 + 200000, 3.0],
                      [1e12 + 201000, 2.0],
                      [1e12 + 202000, 42.0],
                      [1e12 + 203000, 8.0],
                      [1e12 + 204000, 8.0],
                      [1e12 + 205000, 8.0],
                      [1e12 + 206000, 8.0],
                      # Hole
                      [1e12 + 400000, 100000.0]]
    else:
        raise NotImplementedError

    try:
        tsuid = IkatsApi.fid.tsuid(fid=fid)
        IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
    except ValueError:
        # No TS to delete
        pass

    # Create the timeseries
    result = IkatsApi.ts.create(fid=fid, data=np.array(ts_content))
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_ref_period", value=1000, data_type=DTYPE.number)
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_nb_points", value=len(ts_content), data_type=DTYPE.number)
    if not result['status']:
        raise SystemError("Error while creating TS %s" % ts_id)

    return {"tsuid": result['tsuid'], "funcId": fid}


def expected_ts(ts_id):
    """
    Returns a ts composed of expected slope results (having same id as gen_ts method)

    :param ts_id: Identifier of the TS to get expected results (see content below for the structure)
    :type ts_id: int

    :return: the TS data points
    :rtype: np.array
    """

    if ts_id == 1:
        ts_content = np.array([
            [1e12, 0.0012],
            [1e12 + 1000, -0.0002],
            [1e12 + 2000, 0.00125],
            [1e12 + 3600, -0.0575],
            [1e12 + 4000, 0.017],
            [1e12 + 5000, 0.004],
            [1e12 + 6000, -0.003],
            [1e12 + 7000, -0.001],
            [1e12 + 8000, 0.04],
            [1e12 + 9000, -0.034],
            [1e12 + 10000, 0.0],
            [1e12 + 11000, 0.0],
            [1e12 + 12000, 0.0]
        ])
    elif ts_id == 2:
        ts_content = np.array([
            [1e12, 0.0012],
            [1e12 + 1000, -0.0002],
            [1e12 + 2000, 0.002],
            [1e12 + 3000, -0.023],
            [1e12 + 4000, 0.017],
            [1e12 + 5000, 4.21053e-5],
            # Hole
            [1e12 + 100000, -3e-5],
            # Hole
            [1e12 + 200000, -0.01],
            [1e12 + 201000, 0.04],
            [1e12 + 202000, -0.034],
            [1e12 + 203000, 0],
            [1e12 + 204000, 0],
            [1e12 + 205000, 0]
        ])
    elif ts_id == 3:
        ts_content = np.array([
            [1e12, 0.0012],
            [1e12 + 1000, -0.0002],
            [1e12 + 2000, 0.002],
            [1e12 + 3000, -0.023],
            [1e12 + 4000, 0.017],
            [1e12 + 5000, 4.21053e-5],
            # Hole
            [1e12 + 100000, -3e-5],
            # Hole
            [1e12 + 200000, -0.01],
            [1e12 + 201000, 0.04],
            [1e12 + 202000, -0.034],
            [1e12 + 203000, 0],
            [1e12 + 204000, 0],
            [1e12 + 205000, 0],
            [1e12 + 206000, 0.51542268]
        ])
    else:
        raise NotImplementedError

    return ts_content


class TestSlope(TestCase):
    """
    Test of Slope computation
    """

    def test_slope_simple(self):
        """
        Compute the slope of a simple TS
        * Going up
        * Going down
        * Constant
        """

        # Defining default results (for cleanup purposes)
        results = None

        # Prepare list of TS
        ts_list = [gen_ts(1)]
        try:
            # Compute
            results = compute_slope(ts_list=ts_list)

            # Get computed data
            obtained_timeseries_data = IkatsApi.ts.read(tsuid_list=[results[0]['tsuid']])[0]

            # Same number of TS generated
            self.assertEqual(len(ts_list), len(results))

            # Compare length
            self.assertEqual(len(expected_ts(1)), len(obtained_timeseries_data))

            # Compare values
            self.assertTrue(np.allclose(
                np.array(expected_ts(1), dtype=np.float64),
                np.array(obtained_timeseries_data, dtype=np.float64),
                atol=1e-2))
        finally:
            for timeseries in ts_list:
                IkatsApi.ts.delete(timeseries['tsuid'], no_exception=True)
            if results is not None:
                for timeseries in results:
                    IkatsApi.ts.delete(timeseries['tsuid'], no_exception=True)

    def test_slope_many_ts(self):
        """
        Compute the slope of a 2 TS
        """

        # Defining default results (for cleanup purposes)
        results = None

        # Prepare list of TS
        ts_list = [gen_ts(1), gen_ts(2)]

        try:
            # Compute
            results = compute_slope(ts_list=ts_list)

            # Get computed data
            obtained_timeseries_data_1 = IkatsApi.ts.read(tsuid_list=[results[0]['tsuid']])[0]
            obtained_timeseries_data_2 = IkatsApi.ts.read(tsuid_list=[results[1]['tsuid']])[0]

            # Same number of TS generated
            self.assertEqual(len(ts_list), len(results))

            # Compare values
            self.assertTrue(np.allclose(
                np.array(expected_ts(1), dtype=np.float64),
                np.array(obtained_timeseries_data_1, dtype=np.float64),
                atol=1e-2))

            self.assertTrue(np.allclose(
                np.array(expected_ts(2), dtype=np.float64),
                np.array(obtained_timeseries_data_2, dtype=np.float64),
                atol=1e-2))

        finally:
            for timeseries in ts_list:
                IkatsApi.ts.delete(timeseries['tsuid'], no_exception=True)
            if results is not None:
                for timeseries in results:
                    IkatsApi.ts.delete(timeseries['tsuid'], no_exception=True)

    def test_slope_same_ts(self):
        """
        Compute the slope of a 2 TS
        """

        # Defining default results (for cleanup purposes)
        results = None

        # Prepare list of identical TS
        ts_item = gen_ts(1)
        ts_list = [ts_item, ts_item]

        try:
            # Compute
            results = compute_slope(ts_list=ts_list)

            # Get computed data
            obtained_timeseries_data_1 = IkatsApi.ts.read(tsuid_list=[results[0]['tsuid']])[0]
            obtained_timeseries_data_2 = IkatsApi.ts.read(tsuid_list=[results[1]['tsuid']])[0]

            # Same number of TS generated
            self.assertEqual(len(ts_list), len(results))

            # Computed TSUID are identical (because they have the same functional identifier)
            self.assertEqual(results[0]['tsuid'], results[1]['tsuid'])

            # Compare values
            self.assertTrue(np.allclose(
                np.array(expected_ts(1), dtype=np.float64),
                np.array(obtained_timeseries_data_1, dtype=np.float64),
                atol=1e-2))

            self.assertTrue(np.allclose(
                np.array(expected_ts(1), dtype=np.float64),
                np.array(obtained_timeseries_data_2, dtype=np.float64),
                atol=1e-2))

        finally:
            for timeseries in ts_list:
                IkatsApi.ts.delete(timeseries['tsuid'], no_exception=True)
            if results is not None:
                for timeseries in results:
                    IkatsApi.ts.delete(timeseries['tsuid'], no_exception=True)

    def test_slope_many_chunks_simple(self):
        """
        Compute the slope of a simple TS implying several chunks
        """

        # Defining default results (for cleanup purposes)
        results = None

        # Prepare list of TS
        ts_list = [gen_ts(1)]
        try:
            # Compute and force chunk_size to be "4" to have several chunks
            results = compute_slope(
                ts_list=ts_list,
                chunk_size=4)

            # Get computed data
            obtained_timeseries_data = IkatsApi.ts.read(tsuid_list=[results[0]['tsuid']])[0]

            # Same number of TS generated
            self.assertEqual(len(ts_list), len(results))

            # Compare length
            self.assertEqual(len(expected_ts(1)), len(obtained_timeseries_data))

            # Compare values
            self.assertTrue(np.allclose(
                np.array(expected_ts(1), dtype=np.float64),
                np.array(obtained_timeseries_data, dtype=np.float64),
                atol=1e-2))

        finally:
            for timeseries in ts_list:
                IkatsApi.ts.delete(timeseries['tsuid'], no_exception=True)
            if results is not None:
                for timeseries in results:
                    IkatsApi.ts.delete(timeseries['tsuid'], no_exception=True)

    def test_slope_special_chunk(self):
        """
        Compute the slope of a TS having chunks containing:
        * no points
        * 1 single point
        * several points
        * last chunk has one point
        """

        # Defining default results (for cleanup purposes)
        results = None

        # Prepare list of TS
        ts_list = [gen_ts(3)]
        try:
            # Compute
            results = compute_slope(ts_list=ts_list, chunk_size=4)

            # Get computed data
            obtained_timeseries_data = IkatsApi.ts.read(tsuid_list=[results[0]['tsuid']])[0]

            # Same number of TS generated
            self.assertEqual(len(ts_list), len(results))

            # Compare values
            self.assertTrue(np.allclose(
                np.array(expected_ts(3), dtype=np.float64),
                np.array(obtained_timeseries_data, dtype=np.float64),
                atol=1e-2))

        finally:
            for timeseries in ts_list:
                IkatsApi.ts.delete(timeseries['tsuid'], no_exception=True)
            if results is not None:
                for timeseries in results:
                    IkatsApi.ts.delete(timeseries['tsuid'], no_exception=True)

    @staticmethod
    @skip("Performance purposes only")
    def test_perf():
        """
        Performances measurement on full EDF
        """

        # Prepare list of TS
        ts_list_origin = IkatsApi.ds.read(ds_name="EDF")["ts_list"]
        md_list = IkatsApi.md.read(ts_list=ts_list_origin)

        results = None
        ts_start = 0
        ts_end = len(ts_list_origin) + 1
        for i in range(ts_start + 1, ts_end):
            for save in [False, True]:
                ts_list = ts_list_origin[ts_start:i]

                time_before = time.time()

                # Compute
                results = compute_slope(ts_list=ts_list, save_new_ts=save)

                time_after = time.time()

                nb_points = sum([int(md_list[x]['qual_nb_points']) for x in md_list if x in ts_list])
                print("PERF %s TS, %s pts, save=%s in %ss (speed: %.2f pts/s)" % (
                    len(ts_list), nb_points, save, (time_after - time_before), nb_points / (time_after - time_before)))

        print(results)
