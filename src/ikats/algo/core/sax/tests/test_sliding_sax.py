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

import mock
import numpy as np
from numpy.random import normal, uniform

from ikats.algo.core.pattern.random_proj import ConfigSax
from ikats.algo.core.sax.sax import SAX
from ikats.algo.core.sax.sliding_sax import sliding_windows, run_sax_on_sequences
from ikats.core.library.spark import ScManager

LOGGER = logging.getLogger(__name__)


def get_ts_mock(tsuid_list):
    """
    Mock of IkatsApi.IkatsApi.ts.read method
    Generate an RDD containing some TS.

    :param tsuid_list: the ts name choosen (str)
    :type: str

    :return [result] : list of numpy array containing [timestamps, time_serie_value]
    :rtype list of numpy.array

    Note that the 'random_projection' function call IkatsTimeseries.read(tsuid_list=ts_name)[0]
    here, mock_get_ts(ts_name)[0] return 'result'
    """
    result = np.array([])

    if "linear_time_serie" in tsuid_list:
        result = np.array([[np.float64(1000), 1.0],
                           [np.float64(2000), 2.0],
                           [np.float64(3000), 3.0],
                           [np.float64(4000), 4.0],
                           [np.float64(5000), 5.0],
                           [np.float64(6000), 6.0],
                           [np.float64(7000), 7.0],
                           [np.float64(8000), 8.0],
                           [np.float64(9000), 9.0],
                           [np.float64(10000), 10.0],
                           [np.float64(11000), 11.0],
                           [np.float64(12000), 12.0]])

    if "ts_with_constant_pattern" in tsuid_list:
        result = np.array([[np.float64(1000), 10.0],
                           [np.float64(2000), -9.0],
                           [np.float64(3000), 1.0],
                           [np.float64(4000), 1.0],
                           [np.float64(5000), 1.0],
                           [np.float64(6000), 1.0],
                           [np.float64(7000), 1.0],
                           [np.float64(8000), 1.0],
                           [np.float64(9000), 1.0],
                           [np.float64(10000), 1.0],
                           [np.float64(11000), 9.0],
                           [np.float64(12000), -8.0]])

    if "specific_time_serie" in tsuid_list:
        result = np.array([[np.float64(1000), 0],
                           [np.float64(2000), 0],
                           [np.float64(3000), 0],
                           [np.float64(4000), 0],
                           [np.float64(5000), 0],
                           [np.float64(6000), 0],
                           [np.float64(7000), 0],
                           [np.float64(8000), 0],
                           [np.float64(9000), -10.0],
                           [np.float64(10000), 10.0]])

    if "sequences_1_ts0" in tsuid_list:
        result = np.array([[np.float64(1000), 2],
                           [np.float64(2000), 0],
                           [np.float64(3000), 1],
                           [np.float64(4000), 3],
                           [np.float64(5000), 9],
                           [np.float64(6000), 0],
                           [np.float64(7000), -3],
                           [np.float64(8000), 2],
                           [np.float64(9000), -5],
                           [np.float64(10000), 0],
                           [np.float64(11000), -4],
                           [np.float64(12000), 7]])

    if "sequences_1_ts1" in tsuid_list:
        result = np.array([[np.float64(13000), 4],
                           [np.float64(14000), 9],
                           [np.float64(15000), -1],
                           [np.float64(16000), 3],
                           [np.float64(17000), -1],
                           [np.float64(18000), -8],
                           [np.float64(19000), -2],
                           [np.float64(20000), -3],
                           [np.float64(21000), -4],
                           [np.float64(22000), 3],
                           [np.float64(23000), -10],
                           [np.float64(24000), -2]])

    if "simple_sequences_ts0" in tsuid_list:
        result = np.array([[np.float64(1000), 4],
                           [np.float64(2000), 4],
                           [np.float64(3000), 0],
                           [np.float64(4000), 2]])
    if "simple_sequences_ts1" in tsuid_list:
        result = np.array([[np.float64(5000), -2],
                           [np.float64(6000), 2],
                           [np.float64(7000), -2],
                           [np.float64(8000), 0]])

    # The sliding_windows function call IkatsTimeseries.read(tsuid_list=ts_name)[0]
    # here, mock_get_ts(ts_name)[0] return 'result'
    return [result]


class TestSlidingWindow(unittest.TestCase):
    """
    Test sliding window function

    Test sliding window with different recovery values. The recovery parameter modify only the final number of
    sequences, that is why we verify this number
    """

    @mock.patch('ikats.core.resource.api.IkatsApi.ts.read', get_ts_mock)
    def test_sliding_window_recovery(self):
        """
        Testing the recovery parameter.
        """
        sax_info = ConfigSax(paa=3,
                             sequences_size=6,
                             with_mean=True,
                             with_std=True,
                             global_norm=False,
                             local_norm=False,
                             linear_filter=False,
                             recovery=0.5,
                             coefficients=[1, 1],
                             alphabet_size=6)
        ts_name = ["linear_time_serie"]
        spark_ctx = ScManager.get()
        # Test with recovery = 0.5
        result, _ = sliding_windows(ts_list=ts_name, sax_info=sax_info, spark_ctx=spark_ctx)

        result = result.collect()
        # 2 sequences in the timeseries => 3 sequences at the end
        self.assertEqual(len(result), 3)

        # Test with MAX recovery
        # recovery = 1 (the maximum : 100 % <=> the next window start one point to the right)
        sax_info.recovery = 1.0
        result, _ = sliding_windows(ts_list=ts_name, sax_info=sax_info, spark_ctx=spark_ctx)
        result = result.collect()

        # remember that in 'sliding_window' function, we call 'get_ts_mock(ts_name)[0]'
        ts = get_ts_mock(ts_name)[0]
        ts_val_0 = list(ts[0: 6][:, 1])
        ts_val_1 = list(ts[6: 12][:, 1])
        timestamp_0 = list(ts[0: 6][:, 0])
        timestamp_1 = list(ts[6: 12][:, 0])

        # Check the timestamp and the values of the two sequences
        # result[i] = (key, list([timestamps, values],[,],...))

        # check ts value
        condition = (np.all(result[i][1][:, 1] in ts_val_0 for i in range(len(result))) or
                     np.all(result[i][1][:, 1] in ts_val_1 for i in range(len(result))))

        self.assertTrue(condition)

        # check timestamps
        condition = (np.all(result[i][1][:, 0] in timestamp_0 for i in range(len(result))) or
                     np.all(result[i][1][:, 0] in timestamp_1 for i in range(len(result))))
        self.assertTrue(condition)

        # Test with MINIMUM recovery
        # recovery = 0 (no recovery)
        sax_info.recovery = 0.01
        result2, _ = sliding_windows(ts_list=ts_name, sax_info=sax_info, spark_ctx=spark_ctx)
        result2 = result2.collect()
        # 2 sequences in the timeseries => 2 sequences
        self.assertEqual(len(result2), 2)

    # Test sliding window : the filter for the linear sequences and the constant sequences
    @mock.patch('ikats.core.resource.api.IkatsApi.ts.read', get_ts_mock)
    def test_sliding_window_filter(self):
        """
        Testing linear filter.
        """
        sax_info = ConfigSax(paa=3,
                             sequences_size=6,
                             with_mean=True,
                             with_std=True,
                             global_norm=False,
                             local_norm=False,
                             linear_filter=True,
                             recovery=0.5,
                             coefficients=[1, 0.5],
                             alphabet_size=6)

        spark_ctx = ScManager.get()
        # Test for linear sequences
        result, _ = sliding_windows(ts_list=["linear_time_serie"], sax_info=sax_info, spark_ctx=spark_ctx)

        result = result.collect()
        # all sequences are linear => no sequence
        self.assertEqual(len(result), 0)

        # Test for constant sequences with a maximum recovery (= 0 => no overlap between sequences)
        sax_info.coefficients = [0, 1]
        sax_info.recovery = 0
        result, _ = sliding_windows(ts_list=["ts_with_constant_pattern"], sax_info=sax_info, spark_ctx=spark_ctx)
        result = result.collect()
        LOGGER.info("result=%s", result)
        LOGGER.info("ts_init=%s", get_ts_mock("ts_with_constant_pattern"))
        # Sequence of 12 pts, recovery = 0 (no recovery) -> 2 sequences
        self.assertEqual(len(result), 2)

    # Test sliding window : the global and local normalization
    @mock.patch('ikats.core.resource.api.IkatsApi.ts.read', get_ts_mock)
    def test_sliding_window_norm(self):
        """
        Testing global and local norm.
        """
        epsilon = 1.0e-10
        # recovery = 1 (no recovery) -> 3 seq of 4 points (nb_points = 12)
        sax_info = ConfigSax(paa=3,
                             sequences_size=4,
                             with_mean=True,
                             with_std=True,
                             global_norm=True,
                             local_norm=False,
                             linear_filter=False,
                             recovery=0,
                             coefficients=[0.1, 1],
                             alphabet_size=6)

        spark_ctx = ScManager.get()
        # Test with global normalization : the timeseries is normalized
        result, coeff = sliding_windows(ts_list=["linear_time_serie"], sax_info=sax_info, spark_ctx=spark_ctx)

        result = result.collect()
        coeff = coeff.collect()
        # Check coeff : coeff is the mean and variance of each sequence

        # 12 points no recovery (recovery=0) -> 3 seq of 4 points
        self.assertEqual(len(coeff), 3)

        # ts_value is an array with the sequences values
        ts_value = np.array([])
        for i, _ in enumerate(result):
            # result[i] = (key, list([timestamps, values],[,],...))
            ts_value = np.concatenate((result[i][1][:, 1], ts_value))

        LOGGER.info("result=%s", result)
        # no recovery => 2 seq * 6 points = 12 values = npoints
        self.assertEqual(len(ts_value), 12)

        LOGGER.info("ts_std=%s", (ts_value.std()))
        LOGGER.info("ts_mean=%s", np.mean(ts_value))
        # global normalisation => ts_value have a standard deviation of 1 and mean if 0
        self.assertTrue(1 - epsilon < np.std(ts_value) < 1 + epsilon)
        self.assertTrue(- epsilon < np.mean(ts_value) < epsilon)

        # Test with local normalization : all the sequences are normalized
        sax_info.global_norm = False
        sax_info.local_norm = True
        sax_info.linear_filter = True

        # Recovery = 1 : maximum recovery
        sax_info.recovery = 1
        result, coeff = sliding_windows(ts_list=["ts_with_constant_pattern"], sax_info=sax_info, spark_ctx=spark_ctx)
        result = result.collect()

        # Verify that each sequence are normalized
        for i, _ in enumerate(result):
            # result[i] = (key, list([timestamps, values],[,],...))
            seq_value = result[i][1][:, 1]
            self.assertTrue(1 - epsilon < np.std(seq_value) < 1 + epsilon)
            self.assertTrue(- epsilon < np.mean(seq_value) < epsilon)


class TestSlidingSAX(unittest.TestCase):
    """
    Test run_sax_on_sequences function

    We suppose that the previously tests are good, so we can use the sliding_window function
    Test sliding window and SAX algorithm on a linear timeseries
    """

    @mock.patch('ikats.core.resource.api.IkatsApi.ts.read', get_ts_mock)
    def test_sliding_window_sax_basic(self):
        """
        Test the nominal case
        """
        sax_info = ConfigSax(paa=3,
                             sequences_size=6,
                             with_mean=True,
                             with_std=True,
                             global_norm=False,
                             local_norm=False,
                             linear_filter=False,
                             recovery=0.5,
                             coefficients=[0.1, 0.9],
                             alphabet_size=3)

        spark_ctx = ScManager.get()
        result, _ = sliding_windows(ts_list=["linear_time_serie"], sax_info=sax_info, spark_ctx=spark_ctx)

        sax_result = run_sax_on_sequences(rdd_sequences_data=result, paa=sax_info.paa,
                                          alphabet_size=sax_info.alphabet_size)

        # recovery = 0.5 and word_size = 3 => sax_result = 'aab abc bcc'
        self.assertEqual(sax_result.sax_word, 'aababcbcc')

    # Test SAX algorithm
    @mock.patch('ikats.core.resource.api.IkatsApi.ts.read', get_ts_mock)
    def test_sax(self):
        """
        Test with no calculate the PAA (4 PAA for 4 points in a sequence) and the PAA are equidistants
        """
        sax_info = ConfigSax(paa=4,
                             sequences_size=4,
                             with_mean=True,
                             with_std=True,
                             global_norm=False,
                             local_norm=False,
                             linear_filter=False,
                             recovery=0.5,
                             coefficients=[0.1, 0.9],
                             alphabet_size=4)
        spark_ctx = ScManager.get()
        result, _ = sliding_windows(ts_list=["simple_sequences_ts0", "simple_sequences_ts1"], sax_info=sax_info,
                                    spark_ctx=spark_ctx)

        LOGGER.info("sliding_windows done!")

        sax_result = run_sax_on_sequences(rdd_sequences_data=result,
                                          paa=sax_info.paa,
                                          alphabet_size=sax_info.alphabet_size)

        result = result.collect()
        LOGGER.info("sax_result=%s", sax_result)
        LOGGER.info("result=%s", result)

        # the PAA : [[4, 4, 0, 2], [-2, 2, -2, 0]]
        self.assertEqual(sax_result.paa.collect(), [4, 4, 0, 2, -2, 2, -2, 0])
        # the result expected : 'ddbc acab'
        self.assertEqual(sax_result.sax_word, 'ddbcacab')

        # Test with calculate the PAA
        sax_info = ConfigSax(paa=4,
                             sequences_size=12,
                             with_mean=True,
                             with_std=True,
                             global_norm=False,
                             local_norm=False,
                             linear_filter=False,
                             recovery=0.5,
                             coefficients=[0.1, 0.9],
                             alphabet_size=4)

        result, _ = sliding_windows(ts_list=["sequences_1_ts0", "sequences_1_ts1"], sax_info=sax_info,
                                    spark_ctx=spark_ctx)

        sax_result = run_sax_on_sequences(rdd_sequences_data=result,
                                          paa=sax_info.paa,
                                          alphabet_size=sax_info.alphabet_size)

        # the PAA : [[1, 4, -2, 1], [4, -2, -3, -3]]
        self.assertEqual(sax_result.paa.collect(), [1, 4, -2, 1, 4, -2, -3, -3])
        # the result expected : 'cdbc dbaa'
        self.assertEqual(sax_result.sax_word, 'cdbcdbaa')

    def test_breakpoints(self):
        """
        To check the breakpoints, we will create a large list of PAA which have a Gaussian, Uniform distributions.
        See "http://wims.unice.fr/wims/fr_tool~stat~table.fr.html" to check the quantiles of there distributions
        """
        epsilon = 0.1

        # Uniform distribution
        paa = list(uniform(-1, 5, 5000))
        breakpoints = SAX.build_breakpoints(ts_points_value=paa, alphabet_size=6)

        # If we divide a uniform distribution in six (alphabet_size = 6), the breakpoints will be [0, 1, 2, 3, 4]
        self.assertTrue(0 + epsilon > breakpoints[0] > 0 - epsilon)
        self.assertTrue(1 + epsilon > breakpoints[1] > 1 - epsilon)
        self.assertTrue(2 + epsilon > breakpoints[2] > 2 - epsilon)
        self.assertTrue(3 + epsilon > breakpoints[3] > 3 - epsilon)
        self.assertTrue(4 + epsilon > breakpoints[4] > 4 - epsilon)

        # Gaussian distribution
        epsilon = 0.2
        paa = list(normal(0, 3, 5000))
        breakpoints = SAX.build_breakpoints(ts_points_value=paa, alphabet_size=5)

        # The quantiles of the gaussian distribution for the values [0.2, 0.4, 0.6, 0.8] (because alphabet_size = 5)
        # is [-2.5248637, -0.7600413, 0.7600413, 2.5248637]
        self.assertTrue(-2.5248637 + epsilon > breakpoints[0] > -2.5248637 - epsilon)
        self.assertTrue(-0.7600413 + epsilon > breakpoints[1] > -0.7600413 - epsilon)
        self.assertTrue(0.7600413 + epsilon > breakpoints[2] > 0.7600413 - epsilon)
        self.assertTrue(2.5248637 + epsilon > breakpoints[3] > 2.5248637 - epsilon)

    # If there is less different PAA values than the alphabet size => problem with the breakpoints
    # In this example :
    #   - PAA values in {0, -10, 10} => 3 different values
    #   - alphabet size = 5
    # => 3 < 5 => breakpoints = [0, 0, 0, 0]
    @unittest.skip("Very specific test: cancelled at the moment because no such small dataset expected.")
    @mock.patch('ikats.core.resource.api.IkatsApi.ts.read', get_ts_mock)
    def test_sw_sax_limit_constant(self):
        """
        Test sliding window and SAX on a constant timeseries with two greater values
        """
        sax_info = ConfigSax(paa=10,
                             sequences_size=10,
                             with_mean=True,
                             with_std=True,
                             global_norm=False,
                             local_norm=False,
                             linear_filter=False,
                             recovery=0.5,
                             coefficients=[0.1, 0.9],
                             alphabet_size=5)

        spark_ctx = ScManager.get()

        result, _ = sliding_windows(ts_list=["specific_time_serie"],
                                    sax_info=sax_info,
                                    spark_ctx=spark_ctx)

        print("result={}".format(result.collect()))

        sax_result = run_sax_on_sequences(rdd_sequences_data=result,
                                          paa=sax_info.paa,
                                          alphabet_size=sax_info.alphabet_size)

        print("sax_word={}".format(sax_result.sax_word))
        # PAA_value = 0 => 'c'
        # PAA_value = 10 => 'e' or 'd'
        # PAA_value = -10 => 'a' or 'b'
        self.assertTrue(sax_result.sax_word is 'ccccccccae' or sax_result.sax_word is 'ccccccccbd')


if __name__ == '__main__':
    unittest.main()
