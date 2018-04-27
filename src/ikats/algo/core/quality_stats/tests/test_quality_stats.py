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
import json
from unittest import TestCase

import time

import logging
import numpy as np

from ikats.algo.core.quality_stats.quality_stats import calc_quality_stats
from ikats.core.resource.api import IkatsApi
from ikats.algo.core.quality_stats.quality_stats_calculators import LOGGER as LOGGER_QUALITY_STATS


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


# Prints everything from these loggers
log_to_stdout(LOGGER_QUALITY_STATS)


def gen_ts(ts_id):
    """
    Generate a TS in database having all the characteristics
    :return:
    """

    # Build TS
    identifier = "UNIT_TEST_Quality_stats_%s" % ts_id

    if ts_id == 1:
        ts_content = [[np.float64(1e12), 5.0],
                      [np.float64(1e12 + 1200), 6.2],
                      [np.float64(1e12 + 1500), 6.0],
                      [np.float64(1e12 + 2000), 8.0],
                      [np.float64(1e12 + 2500), -15.0],
                      [np.float64(1e12 + 3000), 2.0],
                      [np.float64(1e12 + 4000), 6.0],
                      [np.float64(1e12 + 5000), 3.0],
                      [np.float64(1e12 + 6000), 2.0],
                      [np.float64(1e12 + 10000), 42.0],
                      [np.float64(1e12 + 11000), 8.0],
                      [np.float64(1e12 + 11001), 8.0],
                      [np.float64(1e12 + 12000), 8.0],
                      [np.float64(1e12 + 13000), 8.0]]
    elif ts_id == 2:
        ts_content = [[np.float64(1e12), 0.0],
                      [np.float64(1e12 + 1000), 0.2],
                      [np.float64(1e12 + 2000), 7.0],
                      [np.float64(1e12 + 3000), 8.0],
                      [np.float64(1e12 + 4000), -18.0],
                      [np.float64(1e12 + 13000), 2.0],
                      [np.float64(1e12 + 14000), 6.0],
                      [np.float64(1e12 + 15000), 3.0],
                      [np.float64(1e12 + 16000), 2.0],
                      [np.float64(1e12 + 20000), 123.0],
                      [np.float64(1e12 + 21000), 8.0],
                      [np.float64(1e12 + 21001), 8.0],
                      [np.float64(1e12 + 23000), 8.0]]
    elif ts_id == 3:
        # 1 big step
        # used to cover the case where the TS are not contiguous

        ts_content = [[np.float64(1e12), 0.0],
                      [np.float64(1e12 + 1000), 0.2],
                      [np.float64(1e12 + 2000), 7.0],
                      [np.float64(1e12 + 3000), 8.0],
                      [np.float64(1e12 + 4000), -18.0],
                      [np.float64(1e12 + 13000), 2.0],
                      [np.float64(1e12 + 14000), 6.0],
                      # Big step
                      [np.float64(1e12 + 11115000), 3.0],
                      [np.float64(1e12 + 11116000), 2.0],
                      [np.float64(1e12 + 11120000), 123.0],
                      [np.float64(1e12 + 11121000), 8.0],
                      [np.float64(1e12 + 11121001), 8.0],
                      [np.float64(1e12 + 11123000), 8.0]]
    elif ts_id == 4:
        # 2 big steps separated by one value
        # used to cover the case where a single point is in a chunk
        ts_content = [[np.float64(1e12), 0.0],
                      [np.float64(1e12 + 1000), 0.2],
                      [np.float64(1e12 + 2000), 7.0],
                      [np.float64(1e12 + 3000), 8.0],
                      [np.float64(1e12 + 4000), -18.0],
                      [np.float64(1e12 + 13000), 2.0],
                      [np.float64(1e12 + 14000), 6.0],
                      # Big step
                      [np.float64(1e12 + 11115000), 3.0],
                      # Big step
                      [np.float64(1e12 + 21116000), 2.0],
                      [np.float64(1e12 + 21120000), 123.0],
                      [np.float64(1e12 + 21121000), 8.0],
                      [np.float64(1e12 + 21121001), 8.0],
                      [np.float64(1e12 + 21123000), 8.0]]
    elif ts_id == 5:
        # TS doesn't varies
        # Variance should be 0
        ts_content = [[np.float64(1e12), 0.08799999952316284],
                      [np.float64(1e12 + 1000), 0.08799999952316284],
                      [np.float64(1e12 + 2000), 0.08799999952316284],
                      [np.float64(1e12 + 3000), 0.08799999952316284],
                      [np.float64(1e12 + 4000), 0.08799999952316284],
                      [np.float64(1e12 + 13000), 0.08799999952316284],
                      [np.float64(1e12 + 14000), 0.08799999952316284],
                      [np.float64(1e12 + 15000), 0.08799999952316284],
                      [np.float64(1e12 + 16000), 0.08799999952316284],
                      [np.float64(1e12 + 20000), 0.08799999952316284],
                      [np.float64(1e12 + 21000), 0.08799999952316284],
                      [np.float64(1e12 + 21001), 0.08799999952316284],
                      [np.float64(1e12 + 23000), 0.08799999952316284],
                      [np.float64(1e12 + 33000), 0.08799999952316284],
                      [np.float64(1e12 + 34000), 0.08799999952316284],
                      [np.float64(1e12 + 35000), 0.08799999952316284],
                      [np.float64(1e12 + 36000), 0.08799999952316284],
                      [np.float64(1e12 + 40000), 0.08799999952316284],
                      [np.float64(1e12 + 41000), 0.08799999952316284],
                      [np.float64(1e12 + 41001), 0.08799999952316284],
                      [np.float64(1e12 + 43000), 0.08799999952316284],
                      [np.float64(1e12 + 53000), 0.08799999952316284],
                      [np.float64(1e12 + 64000), 0.08799999952316284],
                      [np.float64(1e12 + 65000), 0.08799999952316284],
                      [np.float64(1e12 + 70000), 0.08799999952316284],
                      [np.float64(1e12 + 80000), 0.08799999952316284],
                      [np.float64(1e12 + 81000), 0.08799999952316284],
                      [np.float64(1e12 + 81001), 0.08799999952316284],
                      [np.float64(1e12 + 83000), 0.08799999952316284],
                      [np.float64(1e12 + 93000), 0.08799999952316284],
                      [np.float64(1e12 + 94000), 0.08799999952316284],
                      [np.float64(1e12 + 95000), 0.08799999952316284],
                      [np.float64(1e12 + 96000), 0.08799999952316284],
                      [np.float64(1e12 + 100000), 0.08799999952316284],
                      [np.float64(1e12 + 101000), 0.08799999952316284],
                      [np.float64(1e12 + 121001), 0.08799999952316284],
                      [np.float64(1e12 + 123000), 0.08799999952316284]]
    else:
        # Specific values for handling histograms limit
        ts_content = []
        # Simulated hist
        hist = [
            {'period': 1000, 'count': 90},
            {'period': 1500, 'count': 80},
            {'period': 2000, 'count': 70},
            {'period': 2500, 'count': 70},
            {'period': 2000, 'count': 60},
            {'period': 3000, 'count': 60},
            {'period': 6000, 'count': 50},
            {'period': 7000, 'count': 45},
            {'period': 8000, 'count': 41},
            {'period': 9000, 'count': 39},
            {'period': 10000, 'count': 37},
            {'period': 11000, 'count': 33},
            {'period': 12000, 'count': 33},
            {'period': 13000, 'count': 33},
            {'period': 14000, 'count': 33},
            {'period': 15000, 'count': 33},
            {'period': 16000, 'count': 33},
            {'period': 17000, 'count': 33},
            {'period': 18000, 'count': 33},
            {'period': 19000, 'count': 33},
            {'period': 20000, 'count': 32},
            {'period': 21000, 'count': 31},
            {'period': 22000, 'count': 30},
            {'period': 23000, 'count': 29},
            {'period': 24000, 'count': 28},
            {'period': 25000, 'count': 27},
            {'period': 26000, 'count': 26},
            {'period': 27000, 'count': 25},
            {'period': 28000, 'count': 6},
            {'period': 29000, 'count': 5},
            {'period': 30000, 'count': 5},
            {'period': 31000, 'count': 5},
            {'period': 32000, 'count': 5},
            {'period': 33000, 'count': 5},
            {'period': 34000, 'count': 5},
            {'period': 35000, 'count': 5},
            {'period': 36000, 'count': 5},
            {'period': 37000, 'count': 5},
            {'period': 38000, 'count': 5},
            {'period': 39000, 'count': 5},
            {'period': 40000, 'count': 5},
            {'period': 41000, 'count': 5},
            {'period': 42000, 'count': 5},
            {'period': 43000, 'count': 5},
            {'period': 44000, 'count': 5},
            {'period': 45000, 'count': 5},
            {'period': 46000, 'count': 5},
            {'period': 47000, 'count': 5},
            {'period': 48000, 'count': 5},
            {'period': 49000, 'count': 5},
            {'period': 50000, 'count': 5},
        ]

        prev_period = 1000000000000
        for item in hist:
            for count in range(1, item["count"]):
                prev_period += item['period']
                ts_content.append([np.float64(prev_period), count])

    # Import
    time_start = time.time()
    result = IkatsApi.ts.create(fid=identifier, data=np.array(ts_content))
    time_end = time.time()
    if not result['status']:
        raise SystemError("Error while creating TS %s" % ts_id)

    print(
        "Time:%ss - Speed:%spts/s - TSUID:%s" %
        (round(time_end - time_start, 3), round(len(ts_content) / (time_end - time_start), 3), result['tsuid']))

    return result['tsuid']


class TestQualityStats(TestCase):
    """
    Tests the quality stats operator
    """
    ts_list = []

    @classmethod
    def setUpClass(cls):
        cls.ts_list = [gen_ts(1), gen_ts(2), gen_ts(3), gen_ts(4), gen_ts(5), gen_ts(6)]

    @classmethod
    def tearDownClass(cls):
        for ts in cls.ts_list:
            IkatsApi.ts.delete(ts, no_exception=True)

    def test_calc_stats_values(self):
        """
        Nominal test with values calculation only
        """
        _, result = calc_quality_stats(
            ts_list=self.ts_list, compute_value=True, compute_time=False)

        self.assertEqual(len(result), len(self.ts_list))

        self.assertEqual(result[self.ts_list[0]]['qual_nb_points'], 14)
        self.assertEqual(result[self.ts_list[0]]['qual_min_value'], -15.0)
        self.assertEqual(result[self.ts_list[0]]['qual_max_value'], 42.0)
        self.assertAlmostEqual(result[self.ts_list[0]]['qual_average'], 6.942857143, delta=1e-6)
        self.assertAlmostEqual(result[self.ts_list[0]]['qual_variance'], 127.613877571, delta=1e-6)

        self.assertEqual(result[self.ts_list[1]]['qual_nb_points'], 13)
        self.assertEqual(result[self.ts_list[1]]['qual_min_value'], -18.0)
        self.assertEqual(result[self.ts_list[1]]['qual_max_value'], 123.0)
        self.assertAlmostEqual(result[self.ts_list[1]]['qual_average'], 12.092307692, delta=1e-6)
        self.assertAlmostEqual(result[self.ts_list[1]]['qual_variance'], 1070.00994082, delta=1e-6)

        self.assertEqual(result[self.ts_list[2]]['qual_nb_points'], 13)
        self.assertEqual(result[self.ts_list[2]]['qual_min_value'], -18.0)
        self.assertEqual(result[self.ts_list[2]]['qual_max_value'], 123.0)
        self.assertAlmostEqual(result[self.ts_list[2]]['qual_average'], 12.092307692, delta=1e-6)
        self.assertAlmostEqual(result[self.ts_list[2]]['qual_variance'], 1070.00994082, delta=1e-6)

        self.assertEqual(result[self.ts_list[4]]['qual_variance'], 0)

    def test_values_already_exist(self):
        """
        Nominal test with values calculation only. Metadata already exists
        """

        _, results = calc_quality_stats(
            ts_list=self.ts_list, compute_value=True, compute_time=False)

        expected_variance = 127.613877571

        # First pass
        self.assertAlmostEqual(results[self.ts_list[0]]['qual_variance'], expected_variance, delta=1e-6)
        variance_first_pass = float(IkatsApi.md.read(ts_list=self.ts_list)[self.ts_list[0]]["qual_variance"])
        self.assertAlmostEqual(variance_first_pass, expected_variance, delta=1e-6)

        # Manual modification to prove the result update
        IkatsApi.md.update(tsuid=self.ts_list[0], name="qual_variance", value=42)

        # Second pass
        _, results = calc_quality_stats(
            ts_list=self.ts_list, compute_value=True, compute_time=False)
        self.assertAlmostEqual(results[self.ts_list[0]]['qual_variance'], expected_variance, delta=1e-6)
        variance_second_pass = float(IkatsApi.md.read(ts_list=self.ts_list)[self.ts_list[0]]["qual_variance"])
        self.assertAlmostEqual(variance_second_pass, expected_variance, delta=1e-6)

    def test_calc_stats_time(self):
        """
        Nominal test with time calculation only
        """

        _, result = calc_quality_stats(
            ts_list=self.ts_list, compute_value=False, compute_time=True)

        self.assertEqual(len(result), len(self.ts_list))

        self.assertEqual(result[self.ts_list[0]]['qual_min_period'], 1)
        self.assertEqual(result[self.ts_list[0]]['qual_max_period'], 4000)
        self.assertEqual(result[self.ts_list[0]]['qual_ref_period'], 1000)

        self.assertEqual(result[self.ts_list[1]]['qual_min_period'], 1)
        self.assertEqual(result[self.ts_list[1]]['qual_max_period'], 9000)
        self.assertEqual(result[self.ts_list[1]]['qual_ref_period'], 1000)
        self.assertDictEqual(json.loads(result[self.ts_list[1]]['qual_hist_period']), {
            "4000": 1,
            "9000": 1,
            "1": 1,
            "1000": 8,
            "1999": 1
        })
        self.assertDictEqual(json.loads(result[self.ts_list[1]]['qual_hist_period_percent']), {
            "4000": 0.08333333333333333,
            "9000": 0.08333333333333333,
            "1": 0.08333333333333333,
            "1000": 0.6666666666666666,
            "1999": 0.08333333333333333
        })

        self.assertEqual(result[self.ts_list[2]]['qual_min_period'], 1)
        self.assertEqual(result[self.ts_list[2]]['qual_max_period'], 11101000)
        self.assertEqual(result[self.ts_list[2]]['qual_ref_period'], 1000)
        self.assertDictEqual(json.loads(result[self.ts_list[2]]['qual_hist_period']), {
            "11101000": 1,
            "9000": 1,
            "4000": 1,
            "1": 1,
            "1000": 7,
            "1999": 1
        })
        self.assertDictEqual(json.loads(result[self.ts_list[2]]['qual_hist_period_percent']), {
            "4000": 0.08333333333333333,
            "9000": 0.08333333333333333,
            "1": 0.08333333333333333,
            "1000": 0.5833333333333334,
            "11101000": 0.08333333333333333,
            "1999": 0.08333333333333333
        })

    def test_calc_stats_chunk(self):
        """
        Nominal test with chunked Timeseries
        """

        _, result = calc_quality_stats(
            ts_list=self.ts_list, compute_value=True, compute_time=True, chunk_size=4)

        self.assertEqual(len(result), len(self.ts_list))

        self.assertEqual(result[self.ts_list[0]]['qual_nb_points'], 14)
        self.assertEqual(result[self.ts_list[0]]['qual_min_value'], -15.0)
        self.assertEqual(result[self.ts_list[0]]['qual_max_value'], 42.0)
        self.assertAlmostEqual(result[self.ts_list[0]]['qual_average'], 6.942857143, delta=1e-6)
        self.assertAlmostEqual(result[self.ts_list[0]]['qual_variance'], 127.613877571, delta=1e-6)
        self.assertEqual(result[self.ts_list[0]]['qual_min_period'], 1)
        self.assertEqual(result[self.ts_list[0]]['qual_max_period'], 4000)
        self.assertEqual(result[self.ts_list[0]]['qual_ref_period'], 1000)

        self.assertEqual(result[self.ts_list[1]]['qual_nb_points'], 13)
        self.assertEqual(result[self.ts_list[1]]['qual_min_value'], -18.0)
        self.assertEqual(result[self.ts_list[1]]['qual_max_value'], 123.0)
        self.assertAlmostEqual(result[self.ts_list[1]]['qual_average'], 12.092307692, delta=1e-6)
        self.assertAlmostEqual(result[self.ts_list[1]]['qual_variance'], 1070.00994082, delta=1e-6)
        self.assertEqual(result[self.ts_list[1]]['qual_min_period'], 1)
        self.assertEqual(result[self.ts_list[1]]['qual_max_period'], 9000)
        self.assertEqual(result[self.ts_list[1]]['qual_ref_period'], 1000)
        self.assertDictEqual(json.loads(result[self.ts_list[1]]['qual_hist_period']), {
            "4000": 1,
            "9000": 1,
            "1": 1,
            "1000": 8,
            "1999": 1
        })
        self.assertDictEqual(json.loads(result[self.ts_list[1]]['qual_hist_period_percent']), {
            "4000": 0.08333333333333333,
            "9000": 0.08333333333333333,
            "1": 0.08333333333333333,
            "1000": 0.6666666666666666,
            "1999": 0.08333333333333333
        })

        self.assertEqual(result[self.ts_list[2]]['qual_nb_points'], 13)
        self.assertEqual(result[self.ts_list[2]]['qual_min_value'], -18.0)
        self.assertEqual(result[self.ts_list[2]]['qual_max_value'], 123.0)
        self.assertAlmostEqual(result[self.ts_list[2]]['qual_average'], 12.092307692, delta=1e-6)
        self.assertAlmostEqual(result[self.ts_list[2]]['qual_variance'], 1070.00994082, delta=1e-6)

        self.assertEqual(result[self.ts_list[2]]['qual_min_period'], 1)
        self.assertEqual(result[self.ts_list[2]]['qual_max_period'], 11101000)
        self.assertEqual(result[self.ts_list[2]]['qual_ref_period'], 1000)
        self.assertDictEqual(json.loads(result[self.ts_list[2]]['qual_hist_period']), {
            "11101000": 1,
            "9000": 1,
            "4000": 1,
            "1": 1,
            "1000": 7,
            "1999": 1
        })
        self.assertDictEqual(json.loads(result[self.ts_list[2]]['qual_hist_period_percent']), {
            "4000": 0.08333333333333333,
            "9000": 0.08333333333333333,
            "1": 0.08333333333333333,
            "1000": 0.5833333333333334,
            "11101000": 0.08333333333333333,
            "1999": 0.08333333333333333
        })

    def test_calc_stats_chunk_specific(self):
        """
        Boundaries tests when chunk is composed of only one point and another one with no points

        With chunk_size=4 on TS id#4, we obtain 4 chunks:
        - chunk 1 : 7 points
        - chunk 2 : 1 point
        - chunk 3 : No points
        - Chunk 4 : 5 points
        """

        ts_list = self.ts_list[3:]

        _, result = calc_quality_stats(ts_list=ts_list, compute_value=False, compute_time=True,
                                       chunk_size=4)

        self.assertEqual(len(result), len(ts_list))

        self.assertEqual(result[ts_list[0]]['qual_min_period'], 1)
        self.assertEqual(result[ts_list[0]]['qual_max_period'], 11101000)
        self.assertEqual(result[ts_list[0]]['qual_ref_period'], 1000)
        self.assertDictEqual(json.loads(result[ts_list[0]]['qual_hist_period']), {
            "11101000": 1,
            "9000": 1,
            "4000": 1,
            "1": 1,
            "10001000": 1,
            "1000": 6,
            "1999": 1
        })
        self.assertDictEqual(json.loads(result[ts_list[0]]['qual_hist_period_percent']), {
            "4000": 0.08333333333333333,
            "9000": 0.08333333333333333,
            "1": 0.08333333333333333,
            "1000": 0.5,
            "10001000": 0.08333333333333333,
            "11101000": 0.08333333333333333,
            "1999": 0.08333333333333333
        })

    def test_calc_stats_hist_too_long(self):
        """
        Boundaries tests when histogram produce too many bars
        """

        ts_list = self.ts_list[4:]

        _, result = calc_quality_stats(ts_list=ts_list, compute_value=False, compute_time=True)

        self.assertEqual(len(result), len(ts_list))

        # self.assertEqual(len(result[ts_list[0]]['qual_hist_pc_period'].keys()), 1000)
        meta = IkatsApi.md.read(ts_list=ts_list)

        qual_hist_period = json.loads(meta[ts_list[0]]["qual_hist_period"])
        qual_hist_period_percent = json.loads(meta[ts_list[0]]["qual_hist_period_percent"])
        self.assertLessEqual(len(meta[ts_list[0]]["qual_hist_period"]), 255)
        self.assertLessEqual(len(meta[ts_list[0]]["qual_hist_period_percent"]), 255)
        self.assertLessEqual(len(qual_hist_period.keys()), 20)
        self.assertLessEqual(len(qual_hist_period_percent.keys()), 20)
