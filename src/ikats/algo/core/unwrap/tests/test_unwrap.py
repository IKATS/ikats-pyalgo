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

from unittest import TestCase

import logging
import time
import numpy as np

from numpy import pi as pi

from ikats.algo.core.unwrap import unwrap_tsuid
from ikats.algo.core.unwrap.unwrap import unwrap_ts_list, TSUnit
from ikats.core.resource.api import IkatsApi

# Logger definition
LOGGER = logging.getLogger('ikats.algo.core.unwrap.unwrap')
# Log format
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')

# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)


def generate_ts(fid, pattern, splits=0, period=1000):
    """
    Generate a calibrated timeseries used for unwrap tests purposes.

    It consists in repeating a pattern (<splits>+1) times.
    The pattern shall have first and last points producing a discontinuity greater than
    the one provided to unwrap algorithm in order to work properly

    The size of the resulting TS is (<splits>+1) * len(pattern) points

    :param fid: functional identifier of the generated timeseries
    :param pattern: numpy array to duplicate splits+1 times
    :param splits: number of discontinuity to add in timeseries
    :param period: number of ms between each point

    :type fid: str
    :type pattern: np.array
    :type splits: int
    :type period: int

    :return: the tsuid of the generated TS and the number of points
    :rtype: str, int
    """
    LOGGER.info("Generating Test TS ...")

    # Performance measurement
    start_time = time.time()

    # Duplicating pattern <splits> + 1 times
    values = []
    for _ in range(splits + 1):
        values = np.hstack((values, pattern))

    # Add timestamps in front of each point
    ts_content = np.column_stack((
        np.arange(10000000000, 10000000000 + (len(values) * period), step=period),
        values
    ))

    # Create timeseries + metadata
    test_ts = IkatsApi.ts.create(fid=fid, data=ts_content, generate_metadata=True)
    tsuid = test_ts["tsuid"]
    IkatsApi.md.create(tsuid=tsuid, name="qual_ref_period", value=period)

    LOGGER.info("Test TS %s generated in %.3fs", fid, time.time() - start_time)
    return tsuid, len(ts_content)


def calc_expected_ts(splits, pattern, phase, period=1000):
    """
    Following the TS generation used for this test, this method allow to create the corresponding expected TS
    without computing unwrap.

    The rules are simple:
    * The resulting period is the same as original timeseries (no time shift/stretch)
    * The number of points is exactly the same as original timeseries
    * the <pattern> is present <splits>+1 times
    * the <pattern> is shifted up by 2*<phase> each time it begins again (when a split is encountered)

    .. example:

        calc_expected_ts(splits=2, pattern=np.array([-180, -90, 0, 90, 180]), phase=180, period=1000)

        will generate:

        expected_points = np.array([
            [10000000000.0, -180.0],
            [10000001000.0, -90.0],
            [10000002000.0, 0.0],
            [10000003000.0, 90.0],
            [10000004000.0, 180.0],

            [10000005000.0, 360 + -180.0],
            [10000006000.0, 360 + -90.0],
            [10000007000.0, 360 + 0.0],
            [10000008000.0, 360 + 90.0],
            [10000009000.0, 360 + 180.0],

            [10000010000.0, 2 * 360 + -180.0],
            [10000011000.0, 2 * 360 + -90.0],
            [10000012000.0, 2 * 360 + 0.0],
            [10000013000.0, 2 * 360 + 90.0],
            [10000014000.0, 2 * 360 + 180.0],
        ])


    :param splits: number of discontinuity used for generation
    :param pattern: initial pattern used for original timeseries
    :param phase: phase used for detection
    :param period: period between each point (1000 is default)

    :type splits: int
    :type pattern: np.array
    :type phase: float
    :type period: int

    :return: the expected timeseries data points (not saved in database)
    :rtype: np.array
    """

    expected_values = np.array(pattern)

    # Offset (in values) to apply for each iteration
    offset = 0

    # Generate the expected points
    for _ in range(splits):
        # Increase the vertical shift
        offset += 2 * phase

        # Combine the new values to generated ones
        expected_values = np.hstack((expected_values, pattern + offset))

    # Add timestamps (evenly spaced by <period> ms until covering all points)
    return np.column_stack((
        np.arange(10000000000, 10000000000 + (len(expected_values) * period), step=period),
        expected_values
    ))


class TestUnwrap(TestCase):
    """
    Tests the unwrap algorithm
    """

    def test_unwrap_tsuid_degrees(self):
        """
        Nominal test using degrees values
        Check the degrees series are well handled
        """

        # Create test timeseries
        fid = "UT_Unwrap_degrees"
        discontinuity = 180
        pattern_to_use = np.array([-180, -90, 0, 90, 180])
        splits = 2
        # Generated TS has (2+1)*5 points
        tsuid, generated_nb_points = generate_ts(fid=fid, splits=splits, pattern=pattern_to_use)

        # Create expected result
        expected_points = calc_expected_ts(splits=splits, pattern=pattern_to_use, phase=discontinuity)

        fid_pattern = "%(fid)s__unwrap"
        try:

            # Call algorithm
            results, _ = unwrap_tsuid(tsuid=tsuid,
                                      unit=TSUnit.Degrees,
                                      discontinuity=discontinuity,
                                      fid_pattern=fid_pattern)

            # Get results
            obtained_points = IkatsApi.ts.read(tsuid_list=[results["tsuid"]])[0]

            # Check results
            self.assertEqual(generated_nb_points, len(obtained_points))
            self.assertTrue(np.allclose(
                expected_points.astype(float),
                obtained_points.astype(float)
            ))

        except AssertionError:
            raise
        finally:
            # Clear Data

            # Delete the TS
            IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
            # Delete the result TS
            IkatsApi.ts.delete(tsuid=IkatsApi.fid.tsuid(fid=fid_pattern % ({'fid': fid})), no_exception=True)

    def test_unwrap_tsuid_radian(self):
        """
        Nominal test using radian values
        Check the radian series are well handled
        """

        # Create test timeseries
        fid = "UT_Unwrap_radian"
        discontinuity = pi
        pattern_to_use = np.array([-pi, -pi / 2, 0, pi / 2, pi])
        splits = 2
        # Generated TS has (2+1)*5 points
        tsuid, generated_nb_points = generate_ts(fid=fid, splits=splits, pattern=pattern_to_use)
        # Create expected result
        expected_points = calc_expected_ts(splits=splits, pattern=pattern_to_use, phase=discontinuity, period=1000)

        fid_pattern = "%(fid)s__unwrap"
        try:
            # Call algorithm
            results, _ = unwrap_tsuid(tsuid=tsuid,
                                      unit=TSUnit.Radians,
                                      discontinuity=discontinuity,
                                      fid_pattern=fid_pattern)

            # Get results
            obtained_points = IkatsApi.ts.read(tsuid_list=[results["tsuid"]])[0]

            # Check results
            self.assertEqual(generated_nb_points, len(obtained_points))
            self.assertTrue(np.allclose(
                expected_points.astype(float),
                obtained_points.astype(float)
            ))

        except AssertionError:
            raise
        finally:
            # Clear Data

            # Delete the TS
            IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
            # Delete the result TS
            IkatsApi.ts.delete(tsuid=IkatsApi.fid.tsuid(fid=fid_pattern % ({'fid': fid})), no_exception=True)

    def test_unwrap_ts_no_phase_jumps(self):
        """
        Nominal test using radian values without splits
        Check a TS having no phase jumps content is the same as input
        """

        # Create test timeseries
        fid = "UT_Unwrap_no_phase_jumps"
        discontinuity = pi
        pattern_to_use = np.array([-pi, -pi / 2, 0, pi / 2, pi])
        splits = 0
        # Generated TS has (0+1)*5 points
        tsuid, generated_nb_points = generate_ts(fid=fid, splits=splits, pattern=pattern_to_use)
        # Create expected result
        expected_points = calc_expected_ts(splits=splits, pattern=pattern_to_use, phase=discontinuity, period=1000)

        fid_pattern = "%(fid)s__unwrap"
        try:
            # Call algorithm
            results, _ = unwrap_tsuid(tsuid=tsuid,
                                      unit=TSUnit.Radians,
                                      discontinuity=discontinuity,
                                      fid_pattern=fid_pattern)

            # Get results
            obtained_points = IkatsApi.ts.read(tsuid_list=[results["tsuid"]])[0]

            # Check results
            self.assertEqual(generated_nb_points, len(obtained_points))
            self.assertTrue(np.allclose(
                expected_points.astype(float),
                obtained_points.astype(float)
            ))

        except AssertionError:
            raise
        finally:
            # Clear Data

            # Delete the TS
            IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
            # Delete the result TS
            IkatsApi.ts.delete(tsuid=IkatsApi.fid.tsuid(fid=fid_pattern % ({'fid': fid})), no_exception=True)

    def test_unwrap_ts_single_point(self):
        """
        Nominal test using radian values with only one point
        Check a TS having 1 point doesn't produce error
        """

        # Create test timeseries
        fid = "UT_Unwrap_single"
        discontinuity = pi
        pattern_to_use = np.array([42])
        splits = 0
        # Generated TS has 1 point
        tsuid, generated_nb_points = generate_ts(fid=fid, splits=splits, pattern=pattern_to_use)
        # Create expected result
        expected_points = calc_expected_ts(splits=splits, pattern=pattern_to_use, phase=discontinuity, period=1000)

        fid_pattern = "%(fid)s__unwrap"
        try:
            # Call algorithm
            results, _ = unwrap_tsuid(tsuid=tsuid,
                                      unit=TSUnit.Radians,
                                      discontinuity=discontinuity,
                                      fid_pattern=fid_pattern)

            # Get results
            obtained_points = IkatsApi.ts.read(tsuid_list=[results["tsuid"]])[0]

            # Check results
            self.assertEqual(generated_nb_points, len(obtained_points))
            self.assertTrue(np.allclose(
                expected_points.astype(float),
                obtained_points.astype(float)
            ))

        except AssertionError:
            raise
        finally:
            # Clear Data

            # Delete the TS
            IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
            # Delete the result TS
            IkatsApi.ts.delete(tsuid=IkatsApi.fid.tsuid(fid=fid_pattern % ({'fid': fid})), no_exception=True)

    def test_unwrap_ts_many_splits(self):
        """
        Nominal test using a timeseries composed of many discontinuities (5000)
        Check the successive discontinuities doesn't alter the result
        """

        # Create test timeseries : (splits+1) * num items
        fid = "UT_Unwrap_many_splits"
        discontinuity = 180
        pattern_to_use = np.linspace(-180, 180, 5)
        splits = 5000
        # Generated TS has (5000+1)*5 points
        tsuid, generated_nb_points = generate_ts(fid=fid, splits=splits, pattern=pattern_to_use)

        # Create expected result
        expected_points = calc_expected_ts(splits=splits, pattern=pattern_to_use, phase=discontinuity, period=1000)

        fid_pattern = "%(fid)s__unwrap"
        try:
            # Call algorithm
            results, timings = unwrap_tsuid(tsuid=tsuid,
                                            unit=TSUnit.Degrees,
                                            discontinuity=discontinuity,
                                            fid_pattern=fid_pattern)

            # Get results
            obtained_points = IkatsApi.ts.read(tsuid_list=[results["tsuid"]])[0]

            # Check results
            self.assertEqual(generated_nb_points, len(obtained_points))
            self.assertTrue(np.allclose(
                expected_points.astype(float),
                obtained_points.astype(float)
            ))

            LOGGER.info('Stats :%s', timings.stats())

        except AssertionError:
            raise
        finally:
            # Clear Data

            # Delete the TS
            IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
            # Delete the result TS
            IkatsApi.ts.delete(tsuid=IkatsApi.fid.tsuid(fid=fid_pattern % ({'fid': fid})), no_exception=True)

    def test_unwrap_ts_many_chunks(self):
        """
        Nominal test forcing several chunks
        Check the chunks of TS are combined properly into the final TS
        """

        # Create test timeseries
        fid = "UT_Unwrap_many_chunks"
        discontinuity = 180
        pattern_to_use = np.array([-180, -90, 0, 90, 180])
        splits = 3
        # Generated TS has (2+1)*5 points
        tsuid, generated_nb_points = generate_ts(fid=fid, splits=splits, pattern=pattern_to_use)

        # Create expected result
        expected_points = calc_expected_ts(splits=splits, pattern=pattern_to_use, phase=discontinuity)

        fid_pattern = "%(fid)s__unwrap"
        try:
            # Call algorithm
            # Force a small chunk_size to have several chunks (chunk_size=3, instead of 75000 by default)
            results, _ = unwrap_tsuid(tsuid=tsuid,
                                      unit=TSUnit.Degrees,
                                      discontinuity=discontinuity,
                                      fid_pattern=fid_pattern,
                                      chunk_size=3)

            # Get results
            obtained_points = IkatsApi.ts.read(tsuid_list=[results["tsuid"]])[0]

            # Check results
            self.assertEqual(generated_nb_points, len(obtained_points))
            self.assertTrue(np.allclose(
                expected_points.astype(float),
                obtained_points.astype(float)
            ))

        except AssertionError:
            raise
        finally:
            # Clear Data

            # Delete the TS
            IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
            # Delete the result TS
            IkatsApi.ts.delete(tsuid=IkatsApi.fid.tsuid(fid=fid_pattern % ({'fid': fid})), no_exception=True)

    def test_unwrap_ts_list_no_spark(self):
        """
        Nominal test using many TS without spark
        Check that all timeseries are processed
        """

        # Create test timeseries
        discontinuity = 180

        fid_ts1 = "UT_Unwrap_1"
        splits_ts1 = 2
        pattern_to_use1 = np.linspace(-180, 180, 10)
        # Generated TS has (2+1)*10 points
        tsuid_1, generated_nb_points_1 = generate_ts(fid=fid_ts1, splits=splits_ts1, pattern=pattern_to_use1)

        fid_ts2 = "UT_Unwrap_2"
        splits_ts2 = 5
        pattern_to_use2 = np.linspace(-180, 180, 6)
        # Generated TS has (5+1)*6 points
        tsuid_2, generated_nb_points_2 = generate_ts(fid=fid_ts2, splits=splits_ts2, pattern=pattern_to_use2)

        fid_ts3 = "UT_Unwrap_3"
        splits_ts3 = 15
        pattern_to_use3 = np.linspace(-180, 180, 3)
        # Generated TS has (15+1)*3 points
        tsuid_3, generated_nb_points_3 = generate_ts(fid=fid_ts3, splits=splits_ts3, pattern=pattern_to_use3)

        # Create expected result
        expected_points_ts1 = calc_expected_ts(splits=splits_ts1, pattern=pattern_to_use1, phase=discontinuity)
        expected_points_ts2 = calc_expected_ts(splits=splits_ts2, pattern=pattern_to_use2, phase=discontinuity)
        expected_points_ts3 = calc_expected_ts(splits=splits_ts3, pattern=pattern_to_use3, phase=discontinuity)

        ts_list = [
            {"tsuid": tsuid_1, "funcId": fid_ts1},
            {"tsuid": tsuid_2, "funcId": fid_ts2},
            {"tsuid": tsuid_3, "funcId": fid_ts3}
        ]

        fid_pattern = "%(fid)s__unwrap"
        try:
            # Call algorithm
            results = unwrap_ts_list(ts_list=ts_list,
                                     unit="Degrees",
                                     discontinuity=discontinuity,
                                     fid_pattern=fid_pattern,
                                     use_spark=False)

            # Get results
            obtained_points_ts1 = IkatsApi.ts.read(tsuid_list=[results[0]["tsuid"]])[0]
            obtained_points_ts2 = IkatsApi.ts.read(tsuid_list=[results[1]["tsuid"]])[0]
            obtained_points_ts3 = IkatsApi.ts.read(tsuid_list=[results[2]["tsuid"]])[0]

            # Check results
            self.assertEqual(len(results), len(ts_list))
            self.assertEqual(generated_nb_points_1, len(obtained_points_ts1))
            self.assertTrue(np.allclose(
                expected_points_ts1.astype(float),
                obtained_points_ts1.astype(float)
            ))

            self.assertEqual(generated_nb_points_2, len(obtained_points_ts2))
            self.assertTrue(np.allclose(
                expected_points_ts2.astype(float),
                obtained_points_ts2.astype(float)
            ))

            self.assertEqual(generated_nb_points_3, len(obtained_points_ts3))
            self.assertTrue(np.allclose(
                expected_points_ts3.astype(float),
                obtained_points_ts3.astype(float)
            ))

        except AssertionError:
            raise
        finally:
            # Clear Data

            # Delete the TS
            IkatsApi.ts.delete(tsuid=tsuid_1, no_exception=True)
            IkatsApi.ts.delete(tsuid=tsuid_2, no_exception=True)
            IkatsApi.ts.delete(tsuid=tsuid_3, no_exception=True)
            # Delete the result TS
            IkatsApi.ts.delete(tsuid=IkatsApi.fid.tsuid(fid=fid_pattern % ({'fid': fid_ts1})), no_exception=True)
            IkatsApi.ts.delete(tsuid=IkatsApi.fid.tsuid(fid=fid_pattern % ({'fid': fid_ts2})), no_exception=True)
            IkatsApi.ts.delete(tsuid=IkatsApi.fid.tsuid(fid=fid_pattern % ({'fid': fid_ts3})), no_exception=True)

    def test_unwrap_ts_list_spark(self):
        """
        Nominal test using many TS with spark
        Check that all timeseries are processed with spark
        """

        # Create test timeseries
        discontinuity = 180

        fid_ts1 = "UT_Unwrap_1"
        splits_ts1 = 2
        pattern_to_use1 = np.linspace(-180, 180, 10)
        # Generated TS has (2+1)*10 points
        tsuid_1, generated_nb_points_1 = generate_ts(fid=fid_ts1, splits=splits_ts1, pattern=pattern_to_use1)

        fid_ts2 = "UT_Unwrap_2"
        splits_ts2 = 5
        pattern_to_use2 = np.linspace(-180, 180, 6)
        # Generated TS has (5+1)*6 points
        tsuid_2, generated_nb_points_2 = generate_ts(fid=fid_ts2, splits=splits_ts2, pattern=pattern_to_use2)

        fid_ts3 = "UT_Unwrap_3"
        splits_ts3 = 15
        pattern_to_use3 = np.linspace(-180, 180, 3)
        # Generated TS has (15+1)*3 points
        tsuid_3, generated_nb_points_3 = generate_ts(fid=fid_ts3, splits=splits_ts3, pattern=pattern_to_use3)

        # Create expected result
        expected_points_ts1 = calc_expected_ts(splits=splits_ts1, pattern=pattern_to_use1, phase=discontinuity)
        expected_points_ts2 = calc_expected_ts(splits=splits_ts2, pattern=pattern_to_use2, phase=discontinuity)
        expected_points_ts3 = calc_expected_ts(splits=splits_ts3, pattern=pattern_to_use3, phase=discontinuity)

        ts_list = [
            {"tsuid": tsuid_1, "funcId": fid_ts1},
            {"tsuid": tsuid_2, "funcId": fid_ts2},
            {"tsuid": tsuid_3, "funcId": fid_ts3}
        ]

        fid_pattern = "%(fid)s__unwrap"
        try:
            # Call algorithm
            results = unwrap_ts_list(ts_list=ts_list,
                                     unit="Degrees",
                                     discontinuity=discontinuity,
                                     fid_pattern=fid_pattern,
                                     use_spark=True)

            # Get results
            obtained_points_ts1 = IkatsApi.ts.read(tsuid_list=[results[0]["tsuid"]])[0]
            obtained_points_ts2 = IkatsApi.ts.read(tsuid_list=[results[1]["tsuid"]])[0]
            obtained_points_ts3 = IkatsApi.ts.read(tsuid_list=[results[2]["tsuid"]])[0]

            # Check results
            self.assertEqual(len(results), len(ts_list))

            # Check results
            self.assertEqual(len(results), len(ts_list))
            self.assertEqual(generated_nb_points_1, len(obtained_points_ts1))
            self.assertTrue(np.allclose(
                expected_points_ts1.astype(float),
                obtained_points_ts1.astype(float)
            ))

            self.assertEqual(generated_nb_points_2, len(obtained_points_ts2))
            self.assertTrue(np.allclose(
                expected_points_ts2.astype(float),
                obtained_points_ts2.astype(float)
            ))

            self.assertEqual(generated_nb_points_3, len(obtained_points_ts3))
            self.assertTrue(np.allclose(
                expected_points_ts3.astype(float),
                obtained_points_ts3.astype(float)
            ))

        except AssertionError:
            raise
        finally:
            # Clear Data

            # Delete the TS
            IkatsApi.ts.delete(tsuid=tsuid_1, no_exception=True)
            IkatsApi.ts.delete(tsuid=tsuid_2, no_exception=True)
            IkatsApi.ts.delete(tsuid=tsuid_3, no_exception=True)
            # Delete the result TS
            IkatsApi.ts.delete(tsuid=IkatsApi.fid.tsuid(fid=fid_pattern % ({'fid': fid_ts1})), no_exception=True)
            IkatsApi.ts.delete(tsuid=IkatsApi.fid.tsuid(fid=fid_pattern % ({'fid': fid_ts2})), no_exception=True)
            IkatsApi.ts.delete(tsuid=IkatsApi.fid.tsuid(fid=fid_pattern % ({'fid': fid_ts3})), no_exception=True)
