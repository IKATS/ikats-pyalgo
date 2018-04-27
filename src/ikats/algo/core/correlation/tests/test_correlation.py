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
import unittest

import mock

from ikats.core.resource.client import TemporalDataMgr
from ikats.core.resource.client.temporal_data_mgr import DTYPE
from ikats.algo.core.correlation import pearson_correlation_matrix, ccf
# noinspection PyProtectedMember
from ikats.algo.core.correlation import spark_ccf, _ccf
import numpy as np

LOGGER = logging.getLogger("ikats.algo.core.correlation")
# Log format
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)

# Local meta data database for testing purposes
META_DATA_BASE = {}


# noinspection PyUnusedLocal
def import_meta_mock(_, tsuid, name, value, data_type=DTYPE.string, force_update=False):
    """
    Mock of the import method ikats.core.resource.client.TemporalDataMgr.import_meta_data
    :param data_type:
    :param force_update:
    :param self:
    :param tsuid:
    :param name:
    :param value:
    """
    if tsuid not in META_DATA_BASE:
        META_DATA_BASE[tsuid].clear()

    if name not in META_DATA_BASE[tsuid] or force_update:
        META_DATA_BASE[tsuid][name] = {'value': value, 'type': data_type}
        return True
    return False


# noinspection PyUnusedLocal
def get_ts_mock(_, tsuid_list):
    """
    Mock of TemporalDataMgr.get_ts method

    Same parameters and types as the original function

    """
    if tsuid_list == ['00001', '00002', '00003', '00004']:
        return [np.array([[np.float64(1449755790000), 1.0],
                          [np.float64(1449755791000), 2.0],
                          [np.float64(1449755792000), 3.0]]),
                np.array([[np.float64(1449755790000), 4.0],
                          [np.float64(1449755791000), 5.0],
                          [np.float64(1449755792000), 7.0]]),
                np.array([[np.float64(1449755790000), 7.0],
                          [np.float64(1449755791000), 6.0],
                          [np.float64(1449755792000), 5.0]]),
                np.array([[np.float64(1449755790000), 7.0],
                          [np.float64(1449755791000), 6.0],
                          [np.float64(1449755792000), 4.0]])]

    if tsuid_list == ['00005', '00006']:
        return [np.array([[np.float64(1449755790000), 1.0],
                          [np.float64(1449755791000), 12.0],
                          [np.float64(1449755792000), 3.0],
                          [np.float64(1449755790000), 24.0]]),
                np.array([[np.float64(1449755790000), 14.0],
                          [np.float64(1449755790000), 13.0],
                          [np.float64(1449755791000), 45.0],
                          [np.float64(1449755792000), 5.5]])]

    if tsuid_list == '00001':
        return np.array([[np.float64(1449755790000), 1.0],
                         [np.float64(1449755791000), 2.0],
                         [np.float64(1449755792000), 3.0]])

    if tsuid_list == '00002':
        return np.array([[np.float64(1449755790000), 4.0],
                         [np.float64(1449755791000), 5.0],
                         [np.float64(1449755792000), 7.0]])


# noinspection PyUnusedLocal
def get_data_set_mock(self, data_set):
    """
    Mock of TemporalDataMgr.get_data_set method

    Same parameters and types as the original function

    """
    return {"description": "description of my data set",
            "ts_list": ['00001',
                        '00002',
                        '00003',
                        '00004']}


# noinspection PyUnusedLocal
def search_fid_mock(self, criterion_type, criteria_list):
    """
    Mock of TemporalDataMgr.search_functional_identifiers method

    Same parameters and types as the original function

    """
    return [{"tsuid": "00001", "funcId": "FuncId_0001"},
            {"tsuid": "00002", "funcId": "FuncId_0002"},
            {"tsuid": "00003", "funcId": "FuncId_0003"},
            {"tsuid": "00004", "funcId": "FuncId_0004"}]


# noinspection PyBroadException
class TestCorrelation(unittest.TestCase):
    """
    This class tests the correlation functions
    """

    @mock.patch('ikats.core.resource.client.TemporalDataMgr.get_ts', get_ts_mock)
    @mock.patch('ikats.core.resource.client.TemporalDataMgr.import_meta_data', import_meta_mock)
    def test_pearson_corr(self):
        """
        test of pearson correlation calculation using mocked timeseries
        """
        tdm = TemporalDataMgr()

        # Correlation matrix
        results = pearson_correlation_matrix(tdm=tdm,
                                             tsuid_list_or_dataset=['00001',
                                                                    '00002',
                                                                    '00003',
                                                                    '00004'],
                                             tsuids_out=True)

        print('Pearson Correlation matrix :\n', results)

        # The result must be a 5x5 matrix with a header line and a header column
        self.assertEqual(results.shape, (5, 5))

    @mock.patch('ikats.core.resource.client.TemporalDataMgr.get_data_set', get_data_set_mock)
    @mock.patch('ikats.core.resource.client.TemporalDataMgr.get_ts', get_ts_mock)
    @mock.patch('ikats.core.resource.client.TemporalDataMgr.import_meta_data', import_meta_mock)
    def test_pearson_corr_dataset(self):
        """
        test of pearson correlation calculation using mocked timeseries data
        and mocked dataset
        """
        tdm = TemporalDataMgr()

        # Correlation matrix
        results = pearson_correlation_matrix(tdm=tdm,
                                             tsuid_list_or_dataset='dataset_test_test',
                                             tsuids_out=True)

        print('Pearson Correlation matrix :\n', results)

        # The result must be a 5x5 matrix with a header line and a header column
        self.assertEqual(results.shape, (5, 5))

    @mock.patch('ikats.core.resource.client.TemporalDataMgr.import_meta_data', import_meta_mock)
    def test_pearson_corr_real_ts(self):
        """
        test of pearson correlation calculation using real timeseries (dataset)
        """
        tdm = TemporalDataMgr()

        # correlation calculation on benchmark
        results = pearson_correlation_matrix(
            tdm=tdm,
            tsuid_list_or_dataset='Portfolio',
            tsuids_out=True,
            cut_ts=True)

        print('Pearson Correlation matrix :\n', results)

        # The result must be a 14x14 matrix with a header line and a header column
        self.assertEqual(results.shape, (14, 14))

    @mock.patch('ikats.core.resource.client.TemporalDataMgr.get_ts', get_ts_mock)
    @mock.patch('ikats.core.resource.client.TemporalDataMgr.import_meta_data', import_meta_mock)
    @mock.patch('ikats.core.resource.client.TemporalDataMgr.search_functional_identifiers',
                search_fid_mock)
    def test_pearson_corr_fid(self):
        """
        test of pearson correlation calculation using mocked timeseries
        and mocked functional ids
        """
        tdm = TemporalDataMgr()

        # correlation calculation
        results = pearson_correlation_matrix(tdm=tdm,
                                             tsuid_list_or_dataset=['00001',
                                                                    '00002',
                                                                    '00003',
                                                                    '00004'],
                                             tsuids_out=False)

        print('Pearson Correlation matrix :\n', results)

        # The result must be a 5x5 matrix with a header line and a header column
        self.assertEqual(results.shape, (5, 5))

    @mock.patch('ikats.core.resource.client.TemporalDataMgr.import_meta_data', import_meta_mock)
    def test_spark_ccf_dataset_real_ts(self):
        """
        test of ccf calculation using real timeseries (dataset)
        calculation is distributed by spark
        """
        tdm = TemporalDataMgr()

        start_time = time.time()
        # CCF correlation calculation
        results = spark_ccf(tdm=tdm, tsuid_list_or_dataset='Portfolio', cut_ts=True)

        LOGGER.info("EXECUTION TIME : %.3f seconds", time.time() - start_time)

        print('CORRELATION CCF :\n', results)

        # The result must be a 14x14 matrix with a header line and a header column
        self.assertEqual(results.shape, (14, 14))

    @mock.patch('ikats.core.resource.client.TemporalDataMgr.import_meta_data', import_meta_mock)
    def test_ccf_dataset_real_ts(self):
        """
        test of ccf calculation using real timeseries (dataset)
        """
        tdm = TemporalDataMgr()

        start_time = time.time()
        # CCF correlation calculation
        results = ccf(tdm=tdm, tsuid_list_or_dataset='Portfolio', cut_ts=True)

        LOGGER.info("EXECUTION TIME : %.3f seconds", time.time() - start_time)

        print('CORRELATION CCF :\n', results)

        # The result must be a 14x14 matrix with a header line and a header column
        self.assertEqual(results.shape, (14, 14))

    @mock.patch('ikats.core.resource.client.TemporalDataMgr.get_ts', get_ts_mock)
    @mock.patch('ikats.core.resource.client.TemporalDataMgr.import_meta_data', import_meta_mock)
    @mock.patch('ikats.core.resource.client.TemporalDataMgr.search_functional_identifiers',
                search_fid_mock)
    def test_ccf(self):
        """
        test of ccf calculation using mocked timeseries
        result is exactly checked
        """
        tdm = TemporalDataMgr()

        # CCF correlation calculation
        results = ccf(tdm=tdm, tsuid_list_or_dataset=['00001', '00002', '00003', '00004'], cut_ts=True)

        print('CORRELATION CCF :\n', results)

        # The result must be a 5x5 matrix with a header line and a header column
        self.assertEqual(results.shape, (5, 5))
        self.assertListEqual(results.tolist(), np.array(
            [['', 'FuncId_0001', 'FuncId_0002', 'FuncId_0003', 'FuncId_0004'],
             ['FuncId_0001', '1.000000000000000', '0.981980506061966', '-1.000000000000000', '-0.981980506061966'],
             ['FuncId_0002', '0.981980506061966', '1.000000000000000', '-0.981980506061966', '-1.000000000000000'],
             ['FuncId_0003', '-1.000000000000000', '-0.981980506061966', '1.000000000000000', '0.981980506061966'],
             ['FuncId_0004', '-0.981980506061966', '-1.000000000000000', '0.981980506061966', '1.000000000000000']])
                             .tolist())

    @mock.patch('ikats.core.resource.client.TemporalDataMgr.import_meta_data', import_meta_mock)
    def test_commutativity_of_ccf(self):
        """
        test of commutativity of cross correlation calculation :
        result is not exactly the same because cross correlation function is
        just reversed when inverting input timeseries
        """
        first_array = np.array([12.0, 2.0, -3.5])

        second_array = np.array([115.0, -2.2, 33.0])

        results_1 = _ccf(first_array, second_array)
        results_2 = _ccf(second_array, first_array)[::-1]

        # The results must be equal to prove the commutativity
        self.assertListEqual(results_1.tolist(), results_2.tolist())

    @mock.patch('ikats.core.resource.client.TemporalDataMgr.import_meta_data', import_meta_mock)
    def test_use_lag_max_param_for_ccf(self):
        """
        test of using lag_max parameter when processing ccf calculation
        """
        first_array = np.array([12.0, 2.0, -3.5, 1.0, 25.6])
        second_array = np.array([115.0, -2.2, 33.0, 145.2, 1.0])

        results_1 = _ccf(first_array, second_array)
        results_2 = _ccf(first_array, second_array, lag_max=4)  # in that case lag_max should be ignored
        results_3 = _ccf(first_array, second_array, lag_max=3)
        results_4 = _ccf(first_array, second_array, lag_max=2)
        results_5 = _ccf(first_array, second_array, lag_max=1)
        results_6 = _ccf(first_array, second_array, lag_max=0)
        results_7 = _ccf(first_array, second_array, lag_max=-1)  # in that case lag_max should be ignored

        # check of lag_max effect
        self.assertListEqual(results_1.tolist(), results_2.tolist())
        self.assertListEqual(results_1.tolist(), results_7.tolist())
        self.assertEqual(len(results_2), 9)
        self.assertEqual(len(results_3), 7)
        self.assertEqual(len(results_4), 5)
        self.assertEqual(len(results_5), 3)
        self.assertEqual(len(results_6), 1)

    @mock.patch('ikats.core.resource.client.TemporalDataMgr.import_meta_data', import_meta_mock)
    def test_compare_ccf_spark(self):
        """
        test to perform a comparison between calculation with and
        without spark. Result should be the same.
        """
        tdm = TemporalDataMgr()

        # CCF correlation calculation
        results_1 = ccf(tdm=tdm,
                        tsuid_list_or_dataset="Portfolio",
                        lag_max=None,
                        tsuids_out=True,
                        cut_ts=True)

        # CCF correlation calculation
        results_2 = spark_ccf(tdm=tdm,
                              tsuid_list_or_dataset="Portfolio",
                              lag_max=None,
                              tsuids_out=True,
                              cut_ts=True)

        self.assertListEqual(results_1.tolist(), results_2.tolist())
