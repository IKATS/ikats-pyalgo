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
import os
import unittest
from unittest import TestCase

import mock
import numpy as np

from ikats.algo.core.paa import run_paa_from_tsuid, run_paa_from_ts_list, run_paa_from_ds
from ikats.core.resource.client import TemporalDataMgr

TDM = TemporalDataMgr()

LOGGER = logging.getLogger("ikats.algo.core.sax")
# Log format
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)


# noinspection PyUnusedLocal
def get_ts_mock(self, tsuid_list):
    """
    Mock of get_ts method from TemporalDataMgr
    :param self:
    :param tsuid_list:
    :return:
    """
    result = []
    if "TSUID1" in tsuid_list:
        result.append(np.array([[np.float64(0), 1.0],
                                [np.float64(1000), 2.0],
                                [np.float64(2000), 3.0],
                                [np.float64(3000), 4.0],
                                [np.float64(4000), 5.0],
                                [np.float64(5000), 6.0],
                                [np.float64(6000), 7.0],
                                [np.float64(7000), 8.0],
                                [np.float64(8000), 9.0],
                                [np.float64(9000), 10.0]]))

    if "TSUID2" in tsuid_list:
        result.append(np.array([[np.float64(0), 5.0],
                                [np.float64(1000), 6.0],
                                [np.float64(2000), 8.0],
                                [np.float64(3000), -15.0],
                                [np.float64(4000), 2.0],
                                [np.float64(5000), 6.0],
                                [np.float64(6000), 3.0],
                                [np.float64(7000), 2.0],
                                [np.float64(8000), 42.0],
                                [np.float64(9000), 8.0]]))
    if "TSUID3" in tsuid_list:
        result.append(np.array([[np.float64(500), 5.0],
                                [np.float64(1000), 6.0],
                                [np.float64(2000), 8.0],
                                [np.float64(3000), 7.0],
                                [np.float64(4000), 9.0],
                                [np.float64(7000), 9.0],
                                [np.float64(9000), 9.0],
                                [np.float64(11000), 9.0],
                                [np.float64(13000), 9.0],
                                [np.float64(14500), 10.0]]))
    if "unknown" in tsuid_list:
        result.append([])
    return result


# Local meta data database for testing purposes
META_DATA_BASE = {}


# noinspection PyUnusedLocal
def import_meta_mock(self, tsuid, name, value):
    """
    Mock of the import method ikats.core.resource.client.TemporalDataMgr.import_meta_data
    :param self:
    :param tsuid:
    :param name:
    :param value:
    """
    if tsuid not in META_DATA_BASE:
        META_DATA_BASE[tsuid] = {}

    META_DATA_BASE[tsuid][name] = value
    return True


# noinspection PyUnusedLocal,PyUnusedLocal
def get_meta_data_mock(self, ts_list):
    """
    Mock of the import method ikats.core.resource.client.TemporalDataMgr.get_meta_data
    :param ts_list:
    :param self:
    """
    return META_DATA_BASE


# noinspection PyUnusedLocal,PyTypeChecker
class TestPAA(TestCase):
    """
    Test of PAA algorithm
    """

    @classmethod
    @mock.patch('ikats.core.resource.client.TemporalDataMgr.import_meta_data', import_meta_mock)
    def setUpClass(cls):
        """
        Setup performed once
        Used to fill in our metadata database and to check the spark env is installed
        :return:
        """
        TDM.import_meta_data('TSUID1', 'qual_nb_points', 10)
        TDM.import_meta_data('TSUID2', 'qual_nb_points', 10)
        TDM.import_meta_data('TSUID3', 'qual_nb_points', 10)

        if os.getenv("PYSPARK_PYTHON") is None:
            assert "env PYSPARK_PYTHON must be defined"
        if os.getenv("SPARK_HOME") is None:
            assert "env SPARK_HOME must be defined"

    @unittest.skipIf('SKIP_LONG_TEST' in os.environ and os.environ['SKIP_LONG_TEST'],
                     "This test is too long and must not be run every time")
    def test_paa_from_ds(self):
        """
        Tests the nominal case of a PAA computation on a dataset
        """
        paa_size = 5
        ds_name = '130_dataset'
        result_1 = run_paa_from_ds(tdm=TDM, ds_name=ds_name, paa_size=paa_size, out_ts=False)

        self.assertEqual(len(result_1), 130)

        for ts in result_1:
            self.assertEqual(len(result_1[ts]), paa_size)

        result_2 = run_paa_from_ds(tdm=TDM,
                                   ds_name=ds_name,
                                   paa_size=paa_size,
                                   out_ts=False,
                                   activate_spark=True)
        self.assertEqual(len(result_2), 130)

        for ts in result_1:
            self.assertEqual(result_2[ts], result_1[ts])

    @mock.patch('ikats.core.resource.client.TemporalDataMgr.get_ts', get_ts_mock)
    @mock.patch('ikats.core.resource.client.TemporalDataMgr.get_meta_data', get_meta_data_mock)
    def test_paa_from_ts_list(self):
        """
        Tests the nominal case of a PAA computation on a ts list
        """

        paa_size = 7
        results = run_paa_from_ts_list(tdm=TDM, ts_list=['TSUID1', 'TSUID2', 'TSUID3'], paa_size=paa_size, out_ts=False)

        self.assertEqual(len(results), 3)
        for ts in results:
            self.assertEqual(len(results[ts]), paa_size)

        results = run_paa_from_ts_list(tdm=TDM, ts_list=['TSUID1', 'TSUID2', 'TSUID3'], paa_size=paa_size, out_ts=True)

        self.assertEqual(len(results), 3)
        self.assertEqual(len(results['TSUID1']), 10)
        self.assertEqual(len(results['TSUID2']), 10)
        self.assertEqual(len(results['TSUID3']), 10)

    @mock.patch('ikats.core.resource.client.TemporalDataMgr.get_ts', get_ts_mock)
    @mock.patch('ikats.core.resource.client.TemporalDataMgr.get_meta_data', get_meta_data_mock)
    def test_paa_from_tsuid(self):
        """
        Tests the nominal case of a PAA computation on a tsuid
        """

        results = run_paa_from_tsuid(tdm=TDM, tsuid="TSUID1", paa_size=2)
        self.assertEqual(results.values[0], 3)
        self.assertEqual(results.values[1], 3)
        self.assertEqual(results.values[2], 3)
        self.assertEqual(results.values[3], 3)
        self.assertEqual(results.values[4], 3)
        self.assertEqual(results.values[5], 8)
        self.assertEqual(results.values[6], 8)
        self.assertEqual(results.values[7], 8)
        self.assertEqual(results.values[8], 8)
        self.assertEqual(results.values[9], 8)

        results = run_paa_from_tsuid(tdm=TDM, tsuid="TSUID1", paa_size=2, out_ts=False)
        self.assertEqual(results, [3, 8])

        results = run_paa_from_tsuid(tdm=TDM, tsuid="TSUID1", paa_size=3)
        self.assertEqual(results.values[0], 2.5)
        self.assertEqual(results.values[1], 2.5)
        self.assertEqual(results.values[2], 2.5)
        self.assertEqual(results.values[3], 2.5)
        self.assertEqual(results.values[4], 6)
        self.assertEqual(results.values[5], 6)
        self.assertEqual(results.values[6], 6)
        self.assertEqual(results.values[7], 9)
        self.assertEqual(results.values[8], 9)
        self.assertEqual(results.values[9], 9)

        results = run_paa_from_tsuid(tdm=TDM, tsuid="TSUID1", paa_size=3, out_ts=False)
        self.assertEqual(results, [2.5, 6, 9])

        results = run_paa_from_tsuid(tdm=TDM, tsuid="TSUID1", paa_size=4)
        self.assertEqual(results.values[0], 2)
        self.assertEqual(results.values[1], 2)
        self.assertEqual(results.values[2], 2)
        self.assertEqual(results.values[3], 4.5)
        self.assertEqual(results.values[4], 4.5)
        self.assertEqual(results.values[5], 6.5)
        self.assertEqual(results.values[6], 6.5)
        self.assertEqual(results.values[7], 9)
        self.assertEqual(results.values[8], 9)
        self.assertEqual(results.values[9], 9)

        results = run_paa_from_tsuid(tdm=TDM, tsuid="TSUID1", paa_size=5)
        self.assertEqual(results.values[0], 1.5)
        self.assertEqual(results.values[1], 1.5)
        self.assertEqual(results.values[2], 3.5)
        self.assertEqual(results.values[3], 3.5)
        self.assertEqual(results.values[4], 5.5)
        self.assertEqual(results.values[5], 5.5)
        self.assertEqual(results.values[6], 7.5)
        self.assertEqual(results.values[7], 7.5)
        self.assertEqual(results.values[8], 9.5)
        self.assertEqual(results.values[9], 9.5)

        results = run_paa_from_tsuid(tdm=TDM, tsuid="TSUID1", paa_size=6)
        self.assertEqual(results.values[0], 1.5)
        self.assertEqual(results.values[1], 1.5)
        self.assertEqual(results.values[2], 3.5)
        self.assertEqual(results.values[3], 3.5)
        self.assertEqual(results.values[4], 5)
        self.assertEqual(results.values[5], 6.5)
        self.assertEqual(results.values[6], 6.5)
        self.assertEqual(results.values[7], 8)
        self.assertEqual(results.values[8], 9.5)
        self.assertEqual(results.values[9], 9.5)

        results = run_paa_from_tsuid(tdm=TDM, tsuid="TSUID1", paa_size=7)
        self.assertEqual(results.values[0], 1.5)
        self.assertEqual(results.values[1], 1.5)
        self.assertEqual(results.values[2], 3)
        self.assertEqual(results.values[3], 4)
        self.assertEqual(results.values[4], 5.5)
        self.assertEqual(results.values[5], 5.5)
        self.assertEqual(results.values[6], 7)
        self.assertEqual(results.values[7], 8)
        self.assertEqual(results.values[8], 9.5)
        self.assertEqual(results.values[9], 9.5)

        results = run_paa_from_tsuid(tdm=TDM, tsuid="TSUID1", paa_size=8)
        self.assertEqual(results.values[0], 1.5)
        self.assertEqual(results.values[1], 1.5)
        self.assertEqual(results.values[2], 3)
        self.assertEqual(results.values[3], 4)
        self.assertEqual(results.values[4], 5)
        self.assertEqual(results.values[5], 6)
        self.assertEqual(results.values[6], 7)
        self.assertEqual(results.values[7], 8)
        self.assertEqual(results.values[8], 9.5)
        self.assertEqual(results.values[9], 9.5)

        results = run_paa_from_tsuid(tdm=TDM, tsuid="TSUID1", paa_size=9)
        self.assertEqual(results.values[0], 1.5)
        self.assertEqual(results.values[1], 1.5)
        self.assertEqual(results.values[2], 3)
        self.assertEqual(results.values[3], 4)
        self.assertEqual(results.values[4], 5)
        self.assertEqual(results.values[5], 6)
        self.assertEqual(results.values[6], 7)
        self.assertEqual(results.values[7], 8)
        self.assertEqual(results.values[8], 9)
        self.assertEqual(results.values[9], 10)

        results = run_paa_from_tsuid(tdm=TDM, tsuid="TSUID1", paa_size=10)
        self.assertEqual(results.values[0], 1)
        self.assertEqual(results.values[1], 2)
        self.assertEqual(results.values[2], 3)
        self.assertEqual(results.values[3], 4)
        self.assertEqual(results.values[4], 5)
        self.assertEqual(results.values[5], 6)
        self.assertEqual(results.values[6], 7)
        self.assertEqual(results.values[7], 8)
        self.assertEqual(results.values[8], 9)
        self.assertEqual(results.values[9], 10)

        results = run_paa_from_tsuid(tdm=TDM, tsuid="TSUID3", paa_size=4)
        self.assertEqual(results.values[0], 7)
        self.assertEqual(results.values[1], 7)
        self.assertEqual(results.values[2], 7)
        self.assertEqual(results.values[3], 7)
        self.assertEqual(results.values[4], 7)
        self.assertEqual(results.values[5], 9)
        self.assertEqual(results.values[6], 9)
        self.assertEqual(results.values[7], 9)
        self.assertEqual(results.values[8], 9.5)
        self.assertEqual(results.values[9], 9.5)

    @mock.patch('ikats.core.resource.client.TemporalDataMgr.get_ts', get_ts_mock)
    def test_paa_from_tsuid_robustness(self):
        """
        Tests the robustness cases of a PAA computation on a TSUID
        """

        # PAA size too long
        with self.assertRaises(ValueError):
            run_paa_from_tsuid(tdm=TDM, tsuid="TSUID1", paa_size=11)

        # PAA size must be >0
        with self.assertRaises(ValueError):
            run_paa_from_tsuid(tdm=TDM, tsuid="TSUID1", paa_size=0)

        # PAA size must be positive
        with self.assertRaises(ValueError):
            run_paa_from_tsuid(tdm=TDM, tsuid="TSUID1", paa_size=-1)

        # PAA size must an integer
        with self.assertRaises(TypeError):
            run_paa_from_tsuid(tdm=TDM, tsuid="TSUID1", paa_size="e")
