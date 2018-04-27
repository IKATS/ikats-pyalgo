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
import unittest

import numpy as np

from ikats.algo.core.resampling.resampling_computation import downsampling_ts, LOGGER
from ikats.core.resource.api import IkatsApi


def log_to_stdout(logger_to_use):
    """
    Allow to print some loggers to stdout
    :param logger_to_use: the LOGGER object to redirect to stdout
    """

    logger_to_use.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger_to_use.addHandler(stream_handler)


# Prints the logger to display
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
    fid = "UNIT_TEST_Downsampling_%s" % ts_id

    if ts_id == 1:
        ts_content = [
            [1e12, 5.0],
            [1e12 + 1000, 6.0],
            [1e12 + 2000, 6.0],
            [1e12 + 3000, 8.0],
            [1e12 + 4000, -15.0],
            [1e12 + 5000, 2.0],
            [1e12 + 6000, 6.0],
            [1e12 + 7000, 3.0],
            [1e12 + 8000, 2.0],
            [1e12 + 9000, 42.0],
            [1e12 + 10000, 8.0],
            [1e12 + 11000, 8.0],
            [1e12 + 12000, 8.0],
            [1e12 + 13000, 8.0]
        ]
    elif ts_id == 2:
        ts_content = [
            [1e12, 5.0],
            [1e12 + 1000, 6.0],
            [1e12 + 2000, 6.0],
            [1e12 + 3600, 8.0],
            [1e12 + 4000, -15.0],
            [1e12 + 5000, 2.0],
            [1e12 + 6000, 6.0],
            [1e12 + 7200, 3.0],
            [1e12 + 8000, 2.0],
            [1e12 + 9000, 5.0],
            [1e12 + 13000, 10.0]
        ]
    elif ts_id == 3:
        ts_content = [
            [1e12, 5.0],
            [1e12 + 1000, 6.0],
            [1e12 + 2000, 6.0],
            [1e12 + 3000, 8.0],
            [1e12 + 100000, 5.0],
            [1e12 + 101000, 9.0],
            [1e12 + 102000, 5.0],
            [1e12 + 103000, 10.0]
        ]
    elif ts_id == 4:
        ts_content = [
            [1e12, 42.0]
        ]
    else:
        raise NotImplementedError

    # Remove former potential story having this name
    try:
        tsuid = IkatsApi.fid.tsuid(fid=fid)
        IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
    except ValueError:
        # No TS to delete
        pass

    # Create the timeseries
    result = IkatsApi.ts.create(fid=fid, data=np.array(ts_content))
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_ref_period", value=1000, force_update=True)
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_nb_points", value=len(ts_content), force_update=True)
    if not result['status']:
        raise SystemError("Error while creating TS %s" % ts_id)

    return {"tsuid": result['tsuid'], "funcId": fid}


class TestDownsampling(unittest.TestCase):
    """
    Test of Downsampling computation
    """

    def _check_results(self, ts_list, result, expected_data):
        """
        Check the results of the downsampling and compare it to the expected data

        :param ts_list: list of duet tsuid/funcId to match input to output
        :param result: raw result of the operator
        :param expected_data: expected data to be used as comparison reference

        :type ts_list: list of dict
        :type result: dict
        :type expected_data: dict
        """

        # Check number of results is the same
        self.assertEqual(len(ts_list), len(result))

        # Check data content
        for index, ts_item in enumerate(ts_list):
            original_tsuid = ts_item["tsuid"]
            obtained_tsuid = result[original_tsuid]["tsuid"]
            obtained_data = IkatsApi.ts.read([obtained_tsuid])[0]

            # Compare values
            try:
                self.assertTrue(np.allclose(
                    np.array(expected_data[original_tsuid], dtype=np.float64),
                    np.array(obtained_data, dtype=np.float64),
                    atol=1e-2))
            except Exception:
                print("ts_item:%s" % ts_item)
                print("Expected (%d points)" % len(expected_data[original_tsuid]))
                print(expected_data[original_tsuid])
                print("Obtained (%d points)" % len(obtained_data))
                print(obtained_data)
                raise

    @staticmethod
    def _cleanup_ts(obtained_result=None, ts_list=None):
        """
        Cleanup the time series used as inputs + resulting time series.

        :param obtained_result: raw results obtained by algorithm
        :type obtained_result: dict
        """
        if obtained_result is not None:
            for original_ts in obtained_result:
                IkatsApi.ts.delete(tsuid=obtained_result[original_ts]['tsuid'], no_exception=True)
                IkatsApi.ts.delete(tsuid=original_ts, no_exception=True)
        if ts_list is not None:
            for ts_item in ts_list:
                IkatsApi.ts.delete(tsuid=ts_item['tsuid'], no_exception=True)

    def test_nominal(self):
        """
        Compute the downsampling on a single time series without any constraint
        Check the time series is processed
        """

        # Prepare inputs
        ts_list = [gen_ts(1)]

        # Prepare expected output
        expected_results = {
            ts_list[0]['tsuid']: [
                [1e12, 6.0],
                [1e12 + 2000, 8.0],
                [1e12 + 4000, 2.0],
                [1e12 + 6000, 6.0],
                [1e12 + 8000, 42.0],
                [1e12 + 10000, 8.0],
                [1e12 + 12000, 8.0]
            ]
        }

        result = None
        try:
            # Call algorithm
            result = downsampling_ts(ts_list=ts_list,
                                     resampling_period=2000, timestamp_position="BEG", aggregation_method="MAX",
                                     nb_points_by_chunk=50000, generate_metadata=True)

            # Check the results
            self._check_results(ts_list, result, expected_results)

        finally:
            # Cleanup
            self._cleanup_ts(result)

    def test_not_aligned(self):
        """
        Compute the downsampling on a single time series with a resampling period not a multiple of the original period
        Check the time series is processed
        """

        # Prepare inputs
        ts_list = [gen_ts(1)]

        # Prepare expected output
        expected_results = {
            ts_list[0]['tsuid']: [
                [1e12, 6.0],
                [1e12 + 1400, 6.0],
                [1e12 + 2800, 8.0],
                [1e12 + 4200, 2.0],
                [1e12 + 5600, 6.0],
                [1e12 + 7000, 3.0],
                [1e12 + 8400, 42.0],
                [1e12 + 9800, 8.0],
                [1e12 + 11200, 8.0],
                [1e12 + 12600, 8.0]
            ]
        }

        result = None
        try:
            # Call algorithm
            result = downsampling_ts(ts_list=ts_list,
                                     resampling_period=1400, timestamp_position="BEG", aggregation_method="MAX",
                                     nb_points_by_chunk=4, generate_metadata=True)

            # Check the results
            self._check_results(ts_list, result, expected_results)

        finally:
            # Cleanup
            self._cleanup_ts(result)

    def test_aggregation_max(self):
        """
        Compute the downsampling on a single time series using MAX aggregation
        Check the time series is processed and result matches the aggregation method
        """

        # Prepare inputs
        ts_list = [gen_ts(1)]

        # Prepare expected output
        expected_results = {
            ts_list[0]['tsuid']: [
                [1e12, 6.0],
                [1e12 + 2000, 8.0],
                [1e12 + 4000, 2.0],
                [1e12 + 6000, 6.0],
                [1e12 + 8000, 42.0],
                [1e12 + 10000, 8.0],
                [1e12 + 12000, 8.0]
            ]
        }

        result = None
        try:
            # Call algorithm
            result = downsampling_ts(ts_list=ts_list,
                                     resampling_period=2000, timestamp_position="BEG", aggregation_method="MAX",
                                     nb_points_by_chunk=50000, generate_metadata=True)

            # Check the results
            self._check_results(ts_list, result, expected_results)
        finally:

            # Cleanup
            self._cleanup_ts(result)

    def test_aggregation_min(self):
        """
        Compute the downsampling on a single time series using MIN aggregation
        Check the time series is processed and result matches the aggregation method
        """

        # Prepare inputs
        ts_list = [gen_ts(1)]

        # Prepare expected output
        expected_results = {
            ts_list[0]['tsuid']: [
                [1e12, 5.0],
                [1e12 + 2000, 6.0],
                [1e12 + 4000, -15.0],
                [1e12 + 6000, 3.0],
                [1e12 + 8000, 2.0],
                [1e12 + 10000, 8.0],
                [1e12 + 12000, 8.0]
            ]
        }

        result = None
        try:
            # Call algorithm
            result = downsampling_ts(ts_list=ts_list,
                                     resampling_period=2000, timestamp_position="BEG", aggregation_method="MIN",
                                     nb_points_by_chunk=50000, generate_metadata=True)

            # Check the results
            self._check_results(ts_list, result, expected_results)
        finally:

            # Cleanup
            self._cleanup_ts(result)

    def test_aggregation_med(self):
        """
        Compute the downsampling on a single time series using MED aggregation (median)
        To compute the median, sort (ascending) the values in the desired period and take the middle point
        (or apply a linear interpolation in case of even values)
        Check the time series is processed and result matches the aggregation method
        """

        # Prepare inputs
        ts_list = [gen_ts(1)]

        # Prepare expected output
        expected_results = {
            ts_list[0]['tsuid']: [
                [1e12, 5.5],
                [1e12 + 10000, 8.0]
            ]
        }

        result = None
        try:
            # Call algorithm
            result = downsampling_ts(ts_list=ts_list,
                                     resampling_period=10000, timestamp_position="BEG", aggregation_method="MED",
                                     nb_points_by_chunk=50000, generate_metadata=True)

            # Check the results
            self._check_results(ts_list, result, expected_results)
        finally:

            # Cleanup
            self._cleanup_ts(result)

    def test_aggregation_first(self):
        """
        Compute the downsampling on a single time series using FIRST aggregation
        Check the time series is processed and result matches the aggregation method
        """

        # Prepare inputs
        ts_list = [gen_ts(1)]

        # Prepare expected output
        expected_results = {
            ts_list[0]['tsuid']: [
                [1e12, 5.0],
                [1e12 + 10000, 8.0]
            ]
        }

        result = None
        try:
            # Call algorithm
            result = downsampling_ts(ts_list=ts_list,
                                     resampling_period=10000, timestamp_position="BEG", aggregation_method="FIRST",
                                     nb_points_by_chunk=50000, generate_metadata=True)

            # Check the results
            self._check_results(ts_list, result, expected_results)
        finally:

            # Cleanup
            self._cleanup_ts(result)

    def test_aggregation_last(self):
        """
        Compute the downsampling on a single time series using LAST aggregation
        Check the time series is processed and result matches the aggregation method
        """

        # Prepare inputs
        ts_list = [gen_ts(1)]

        # Prepare expected output
        expected_results = {
            ts_list[0]['tsuid']: [
                [1e12, 42.0],
                [1e12 + 10000, 8.0]
            ]
        }

        result = None
        try:
            # Call algorithm
            result = downsampling_ts(ts_list=ts_list,
                                     resampling_period=10000, timestamp_position="BEG", aggregation_method="LAST",
                                     nb_points_by_chunk=50000, generate_metadata=True)

            # Check the results
            self._check_results(ts_list, result, expected_results)
        finally:

            # Cleanup
            self._cleanup_ts(result)

    def test_alignment_mid(self):
        """
        Compute the downsampling on a single time series using MID alignment
        Check the time series is processed and result matches the alignment method
        """

        # Prepare inputs
        ts_list = [gen_ts(1)]

        # Prepare expected output
        expected_results = {
            ts_list[0]['tsuid']: [
                [1e12 + 3000, 8.0],
                [1e12 + 9000, 42.0],
                [1e12 + 15000, 8.0]
            ]
        }

        result = None
        try:
            # Call algorithm
            result = downsampling_ts(ts_list=ts_list,
                                     resampling_period=6000, timestamp_position="MID", aggregation_method="MAX",
                                     nb_points_by_chunk=50000, generate_metadata=True)

            # Check the results
            self._check_results(ts_list, result, expected_results)
        finally:

            # Cleanup
            self._cleanup_ts(result)

    def test_alignment_end(self):
        """
        Compute the downsampling on a single time series using END alignment
        Check the time series is processed and result matches the alignment method
        """

        # Prepare inputs
        ts_list = [gen_ts(1)]

        # Prepare expected output
        expected_results = {
            ts_list[0]['tsuid']: [
                [1e12 + 6000, 8.0],
                [1e12 + 12000, 42.0],
                [1e12 + 18000, 8.0]
            ]
        }
        result = None
        try:
            # Call algorithm
            result = downsampling_ts(ts_list=ts_list,
                                     resampling_period=6000, timestamp_position="END", aggregation_method="MAX",
                                     nb_points_by_chunk=50000, generate_metadata=True)

            # Check the results
            self._check_results(ts_list, result, expected_results)

        finally:
            # Cleanup
            self._cleanup_ts(result)

    def test_multiple_ts(self):
        """
        Compute the downsampling on multiple time series without any constraint
        Check all time series are processed
        """

        # Prepare inputs
        ts_list = [gen_ts(1), gen_ts(2)]

        # Prepare expected output
        expected_results = {
            ts_list[0]['tsuid']: [
                [1e12, 5.5],
                [1e12 + 2000, 7.0],
                [1e12 + 4000, -6.5],
                [1e12 + 6000, 4.5],
                [1e12 + 8000, 22.0],
                [1e12 + 10000, 8.0],
                [1e12 + 12000, 8.0]
            ],
            ts_list[1]['tsuid']: [
                [1e12, 5.5],
                [1e12 + 2000, 7.0],
                [1e12 + 4000, -6.5],
                [1e12 + 6000, 4.5],
                [1e12 + 8000, 3.5],
                [1e12 + 12000, 10.0]
            ]
        }

        result = None
        try:
            # Call algorithm
            result = downsampling_ts(ts_list=ts_list,
                                     resampling_period=2000, timestamp_position="BEG", aggregation_method="AVG",
                                     nb_points_by_chunk=50000, generate_metadata=True)

            # Check the results
            self._check_results(ts_list, result, expected_results)

        finally:
            # Cleanup
            self._cleanup_ts(result)

    def test_multiple_chunks(self):
        """
        Compute the downsampling on multiple time series having many points
        Check the downsampling works with multiple chunks
        """

        # Prepare inputs
        ts_list = [gen_ts(1)]

        # Prepare expected output
        expected_results = {
            ts_list[0]['tsuid']: [
                [1e12, 5.5],
                [1e12 + 2000, 7.0],
                [1e12 + 4000, -6.5],
                [1e12 + 6000, 4.5],
                [1e12 + 8000, 22.0],
                [1e12 + 10000, 8.0],
                [1e12 + 12000, 8.0]
            ]
        }

        result = None
        try:
            # Call algorithm
            result = downsampling_ts(ts_list=ts_list,
                                     resampling_period=2000, timestamp_position="BEG", aggregation_method="AVG",
                                     nb_points_by_chunk=4, generate_metadata=True)

            # Check the results
            self._check_results(ts_list, result, expected_results)

        finally:
            # Cleanup
            self._cleanup_ts(result)

    def test_empty_chunks(self):
        """
        Compute the downsampling on multiple time series having many points not evenly distributed
        Check the downsampling works with empty chunks
        """

        # Prepare inputs
        ts_list = [gen_ts(3)]

        # Prepare expected output
        expected_results = {
            ts_list[0]['tsuid']: [
                [1e12, 5.5],
                [1e12 + 2000, 7.0],
                [1e12 + 100000, 7.0],
                [1e12 + 102000, 7.5]
            ]
        }

        result = None
        try:
            # Call algorithm
            result = downsampling_ts(ts_list=ts_list,
                                     resampling_period=2000, timestamp_position="BEG", aggregation_method="AVG",
                                     nb_points_by_chunk=50000, generate_metadata=True)

            # Check the results
            self._check_results(ts_list, result, expected_results)

        finally:
            # Cleanup
            self._cleanup_ts(result)

    def test_single_point(self):
        """
        Compute the downsampling on time series having 1 point
        Check the new time series is the same as the original one
        """
        # Prepare inputs
        ts_list = [gen_ts(4)]

        # Prepare expected output
        expected_results = {
            ts_list[0]['tsuid']: [
                [1e12, 42.0]
            ]
        }

        result = None
        try:
            # Call algorithm
            result = downsampling_ts(ts_list=ts_list,
                                     resampling_period=2000, timestamp_position="BEG", aggregation_method="MAX",
                                     nb_points_by_chunk=50000, generate_metadata=True)

            # Check the results
            self._check_results(ts_list, result, expected_results)

        finally:
            # Cleanup
            self._cleanup_ts(result)

    def test_robustness_no_ts(self):
        """
        No time series is provided
        Check that the result is empty
        """

        # Prepare inputs
        ts_list = []

        # Prepare expected output
        expected_results = {}

        result = None
        try:
            # Call algorithm
            result = downsampling_ts(ts_list=ts_list,
                                     resampling_period=2000, timestamp_position="BEG", aggregation_method="AVG",
                                     nb_points_by_chunk=4, generate_metadata=True)

            # Check the results
            self._check_results(ts_list, result, expected_results)

        finally:
            # Cleanup
            self._cleanup_ts(result)
