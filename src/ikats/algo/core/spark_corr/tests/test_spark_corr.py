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
import time
import numpy as np
import mock

from ikats.core.library.spark import ScManager
from ikats.core.resource.client.temporal_data_mgr import TemporalDataMgr
from ikats.algo.core.spark_corr import SparkCorrelation

LOGGER = logging.getLogger("ikats.algo.core.correlation")
# Log format
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)


def get_ts_mock(_, tsuid_list):
    """
    Mock of TemporalDataMgr.get_ts method
    Same parameters and types as the original function
    Used to prove the correct results of the correlation matrix
    :param self:
    :param tsuid_list:
    :param sd:
    :param ed:
    """
    if tsuid_list == '00001':
        return [np.array([[1449755790000, 1], [1449755791000, 2], [1449755792000, 3]])]
    if tsuid_list == '00002':
        return [np.array([[1449755790000, 4], [1449755791000, 5], [1449755792000, 7]])]
    if tsuid_list == '00003':
        return [np.array([[1449755790000, 7], [1449755791000, 6], [1449755792000, 5]])]
    if tsuid_list == '00004':
        return [np.array([[1449755790000, 7], [1449755791000, 6], [1449755792000, 4]])]


class Test(unittest.TestCase):
    """
    Tests the Spark Correlation function
    """

    @classmethod
    def setUpClass(cls):
        """
        needs environment vars to be explicitly set :
        SPARK_HOME
        and
        PYSPARK_PYTHON
        """
        if os.getenv("PYSPARK_PYTHON") is None:
            assert "env PYSPARK_PYTHON must be defined"
        if os.getenv("SPARK_HOME") is None:
            assert "env SPARK_HOME must be defined"

        # Create a spark Context
        ScManager.create()

    @unittest.skipIf('SKIP_LONG_TEST' in os.environ and os.environ['SKIP_LONG_TEST'],
                     "This test is too long and must not be run every time")
    def test_spark_pearson_130_ts(self):
        """
        Run the test on a big dataset (130 items)

        """

        tdm = TemporalDataMgr()
        tsuids = tdm.get_data_set('130_dataset')['ts_list']
        start_time = time.time()
        sp_corr = SparkCorrelation(tdm)
        sp_corr.run(tsuids, 'pearson')
        LOGGER.info("EXECUTION TIME (for %d TS with %d pts/ea = %d points): %.3f seconds",
                    len(tsuids),
                    sp_corr.ts_len_ref,
                    (len(tsuids) * sp_corr.ts_len_ref),
                    (time.time() - start_time))
        self.assertEqual(len(sp_corr.results), len(tsuids))
        self.assertEqual(len(sp_corr.results[0]), len(tsuids))
        # Check CSV build
        csv = sp_corr.get_csv()
        self.assertGreater(len(csv), 0)
        csv = sp_corr.get_csv()
        with open('/tmp/result_130.csv', 'w', newline='') as opened_file:
            opened_file.write(csv)
            # ntdm.add_data('/tmp/result_130.csv', "spark_correlation_130", "CSV")

    @unittest.skipIf('SKIP_LONG_TEST' in os.environ and os.environ['SKIP_LONG_TEST'],
                     "This test is too long and must not be run every time")
    def test_spark_pearson_400_ts(self):
        """
        Run the test on a big dataset (400 items)

        """
        if os.getenv("PYSPARK_PYTHON") is None:
            self.fail("env PYSPARK_PYTHON must be defined")
        if os.getenv("SPARK_HOME") is None:
            self.fail("env SPARK_HOME must be defined")
        tdm = TemporalDataMgr()
        tsuids = tdm.get_data_set('400TS_9K')['ts_list']
        start_time = time.time()
        sp_corr = SparkCorrelation(tdm)
        sp_corr.run(tsuids)
        LOGGER.info("EXECUTION TIME (for %d TS with %d pts/ea = %d points): %.3f seconds",
                    len(tsuids),
                    sp_corr.ts_len_ref,
                    (len(tsuids) * sp_corr.ts_len_ref),
                    (time.time() - start_time))
        self.assertEqual(len(sp_corr.results), len(tsuids))
        self.assertEqual(len(sp_corr.results[0]), len(tsuids))
        csv = sp_corr.get_csv()
        with open('/tmp/result_400.csv', 'w', newline='') as opened_file:
            opened_file.write(csv)

    @mock.patch('ikats.core.resource.client.TemporalDataMgr.get_ts', get_ts_mock)
    def test_spark_pearson_results(self):
        """
        Checks the result of a pearson correlation matrix based on benchmark values

        """
        if os.getenv("PYSPARK_PYTHON") is None:
            self.fail("env PYSPARK_PYTHON must be defined")
        tsuids = ['00001', '00002', '00003', '00004']
        tdm = TemporalDataMgr()
        start_time = time.time()
        sp_corr = SparkCorrelation(tdm)
        sp_corr.force_parallel_get_ts = False
        sp_corr.run(tsuids, 'pearson')
        LOGGER.info("EXECUTION TIME (for %d TS with %d pts/ea = %d points): %.3f seconds",
                    len(tsuids),
                    sp_corr.ts_len_ref,
                    (len(tsuids) * sp_corr.ts_len_ref),
                    (time.time() - start_time))
        self.assertEqual(np.allclose(sp_corr.results,
                                     np.array([[1, 0.98198051, -1, -0.98198051],
                                               [0.98198051, 1, -0.98198051, -1],
                                               [-1, -0.98198051, 1, 0.98198051],
                                               [-0.98198051, -1, 0.98198051, 1]])), True)


if __name__ == "__main__":
    unittest.main()
