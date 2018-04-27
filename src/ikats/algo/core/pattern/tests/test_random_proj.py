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
import random
import unittest

import mock
import numpy as np
from numpy.random import normal, exponential
from scipy.special import binom

from ikats.algo.core.pattern.collision import LOGGER as COLL_LOGGER
from ikats.algo.core.pattern.recognition import LOGGER as RECOG_LOGGER
from ikats.algo.core.sax.sliding_sax import LOGGER as SAX_LOGGER
from ikats.algo.core.pattern.random_proj import LOGGER, random_projections, regex_from_pattern_results, \
    EMPTY_REGEX_MESSAGE, ConfigSax, ConfigCollision, ConfigRecognition

LOGGER = logging.getLogger(__name__)

# Add logs to the unittest stdout
for the_logger in [SAX_LOGGER, RECOG_LOGGER, COLL_LOGGER, LOGGER, LOGGER]:
    the_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
    # Create another handler that will redirect log entries to STDOUT
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    the_logger.addHandler(stream_handler)

SAX_INFO = ConfigSax(paa=20,
                     sequences_size=1000,
                     with_mean=True,
                     with_std=True,
                     global_norm=False,
                     local_norm=True,
                     linear_filter=True,
                     recovery=0.8,
                     coefficients=[0.1, 0.9],
                     alphabet_size=10)

COLLISION_INFO = ConfigCollision(iterations=1, index=2, config_sax=SAX_INFO)

# Avoiding spark jobs here: already tested in test_recognition
RECOGNITION_INFO = ConfigRecognition(is_stopped_by_eq9=True,
                                     is_algo_method_global=True,
                                     min_value=1,
                                     iterations=10,
                                     radius=1.5,
                                     neighborhood_method=2,
                                     activate_spark=False)


def create_values(size, parameter, distribution):
    """
    Create a pattern with a gaussian or exponential distribution, or a linear pattern. The timestamps are not created.

    :param size: the number of points of the pattern
    :type size: int

    :param parameter: the variance of the gaussian distribution, or the lambda parameter of the exponential
                      distribution. Not used if the distribution parameter is 'linear'.
    :type parameter: int or float

    :param distribution: the distribution of the pattern : gaussian, exponential, or linear
    :type distribution: str

    :return: the values of the pattern corresponding to the distribution
    :rtype: numpy.ndarray
    """
    if distribution is 'gaussian':
        return normal(0, parameter, size)

    elif distribution is 'exponential':
        return exponential(1 / parameter, size)

    elif distribution is 'linear':
        # example : size = 8  =>  linear_pattern = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5]
        return np.arange(- int(size / 4), int(size / 4), 0.5)


def create_ts(pattern_positions, pattern_list, pattern_size, is_noise):
    """
    Create a time serie with a list of patterns. The size of the patterns have to be the same for all patterns.

    :param pattern_positions: the place of the patterns in the time serie.
    :type pattern_positions: list of int

    :param pattern_list: the values of the patterns
    :type pattern_list: list of numpy.ndarray

    :param pattern_size: the number of points of the patterns.
    :type pattern_size: int

    :param is_noise: if we add or not some noise which is a gaussian (mean = 0, std = std(time_serie) / 10)
    :type is_noise: bool

    :return: the time serie with the timestamps
    :rtype: numpy.ndarray

    Example : pattern_positions = [1, ..., 1]; pattern_list = [pattern_10, pattern_4, ...]
              In this example, the pattern_4 is at the beginning and the end of the time serie.
    """

    # Create an array with the time serie values
    ts_values = np.array([])
    for i in pattern_positions:
        ts_values = np.concatenate((ts_values, pattern_list[i]), axis=0)

    ts_size = len(pattern_positions) * pattern_size

    # add noise
    if is_noise is True:
        std_ts = np.std(ts_values)
        noise = normal(0, std_ts / 10, ts_size)
        ts_values = np.array(list(map(lambda x, y: x + y, ts_values, noise)))

    # add timestamps
    timestamp = range(ts_size)
    return np.array(list(map(lambda x, y: [x, y], timestamp, ts_values)))


def mock_read_ts(tsuid_list):
    """
    Mock of IkatsApi.ts.read method

    :param tsuid_list: the ts name choosen (str !)
    :type: str

    :return: list of numpy.array containing data ([[timestamp, time_serie_values],...]
    :rtype : list of numpy.array

    The 'random_projection' function call IkatsTimeseries.read(tsuid_list=tsuid_list)[0]
    here, mock_get_ts(tsuid_list)[0] return 'result'
    """
    result = np.array([])

    # test_random_proj_one_ts
    if tsuid_list in ["ts1"]:
        # The time serie have 10 000 points, with 10 patterns created where the pattern4 is a linear pattern. We check
        # if this pattern is delete from the sequences list by the filter.

        # Create patterns
        pattern_size = 1000
        pattern1 = create_values(pattern_size, 0.1, 'gaussian')
        pattern2 = create_values(pattern_size, 2, 'exponential')
        pattern3 = create_values(pattern_size, 200, 'gaussian')
        pattern4 = create_values(pattern_size, 0, 'linear')

        # Create the time serie
        result = create_ts(pattern_positions=[0, 1, 2, 0, 3, 1, 1, 3, 0, 2],
                           pattern_list=[pattern1, pattern2, pattern3, pattern4],
                           pattern_size=pattern_size,
                           is_noise=True)

    # test_random_proj_dataset
    if tsuid_list in ["tsa", "tsb"]:
        # The time serie have 10 000 points, with 10 patterns created where the pattern4 is a linear pattern. We check
        # if this pattern is delete from the sequences list by the filter.

        # Create patterns
        pattern_size = 1000
        pattern1 = create_values(pattern_size, 0.1, 'gaussian')
        pattern2 = create_values(pattern_size, 2, 'exponential')
        pattern3 = create_values(pattern_size, 200, 'gaussian')
        pattern4 = create_values(pattern_size, 0, 'linear')

        # Create the time serie
        if tsuid_list == "tsa":
            result = create_ts(pattern_positions=[0, 1, 2, 0],
                               pattern_list=[pattern1, pattern2, pattern3, pattern4],
                               pattern_size=pattern_size,
                               is_noise=True)
        if tsuid_list == "tsb":
            result = create_ts(pattern_positions=[3, 1, 1, 3, 0, 2],
                               pattern_list=[pattern1, pattern2, pattern3, pattern4],
                               pattern_size=pattern_size,
                               is_noise=True)
    # test_paa_values
    if tsuid_list in ["test_paa_values"]:
        # Create patterns
        pattern_size = 1000
        pattern1 = create_values(pattern_size, 0.1, 'gaussian')
        pattern2 = create_values(pattern_size, 2, 'exponential')
        pattern3 = create_values(pattern_size, 200, 'gaussian')
        pattern4 = create_values(pattern_size, 0, 'linear')

        # Create the time serie
        result = create_ts(pattern_positions=[0, 1, 2, 0, 3, 1, 1, 3, 0, 2],
                           pattern_list=[pattern1, pattern2, pattern3, pattern4],
                           pattern_size=pattern_size,
                           is_noise=True)

    if tsuid_list in ["testPatternA",
                      "tesPatternB",
                      "testPatternC",
                      "testPatternConstant",
                      "testPatternLinear",
                      "testPatternTooSmall",
                      "testPatternTrivialMatch",
                      "testPatternRealistic"]:

        if tsuid_list == "testPatternA":
            result = np.array([[np.float64(5000), -2],
                               [np.float64(6000), 2],
                               [np.float64(7000), -2],
                               [np.float64(8000), 0],
                               [np.float64(9000), 10]])

        elif tsuid_list == "testPatternB":
            result = np.array([[np.float64(1000), 4],
                               [np.float64(2000), 5],
                               [np.float64(3000), 0],
                               [np.float64(4000), 2]])

        elif tsuid_list == "testPatternC":
            result = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 6], [5, 0], [6, 1],
                               [7, 8], [8, 8], [9, 9], [10, 10]])

        elif tsuid_list == "testPatternConstant":
            result = np.array([[11, 11.5], [12, 11.75], [13, 11.85], [14, 11.95], [15, 12]])

        elif tsuid_list == "testPatternLinear":
            result = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                               [7, 8], [8, 9], [9, 10], [10, 11]])
        elif tsuid_list == "testPatternTooSmall":
            result = np.array([[0, 1], [1, 2]])

        elif tsuid_list == "testPatternTrivialMatch":
            result = np.array([[1, 1], [2, 2], [3, 2], [4, 1], [5, 2], [6, 2], [7, 1], [8, 2], [9, 2], [10, 1]]),

        elif tsuid_list == "testPatternRealistic":
            result = np.array(list(zip(range(100), random.sample(range(-5000, 5000), 100))))

    return [result]


class TestRandomProj(unittest.TestCase):
    """
    Check the random projections algorithm with one time serie and a dataset
    """

    @mock.patch('ikats.core.resource.api.IkatsApi.ts.read', mock_read_ts)
    def test_random_proj_one_ts(self):
        """
        The timeseries have 10 000 points, with 10 patterns created where the pattern 4 is a linear pattern. We check
        if this pattern is delete from the sequences list by the filter.

        The size of the sequences have to be smaller or equal to the  pattern size, unless the filter will not detect
        linear sequences.
        """
        SAX_INFO.sequences_size = 500
        # Check that linear sequences are filter, and not constant sequences.
        SAX_INFO.coefficients = [0, 0.9]

        result = random_projections(ts_list=["ts1"],
                                    sax_info=SAX_INFO,
                                    collision_info=COLLISION_INFO,
                                    recognition_info=RECOGNITION_INFO)

        # Check the length of the alphabet used, and so the breakpoints created
        self.assertEqual(len(result["disc_break_points"]), SAX_INFO.alphabet_size)
        self.assertEqual(len(result["break_points"]), SAX_INFO.alphabet_size - 1)

    @mock.patch('ikats.core.resource.api.IkatsApi.ts.read', mock_read_ts)
    def test_random_proj_dataset(self):
        """
        The timeseries have 10 000 points, with 10 patterns created where the pattern4 is a linear pattern. We check
        if this pattern is delete from the sequences list by the filter.

        the size of the sequences have to be smaller or equal to the  pattern size, unless the filter will not detect
        linear sequences.
        """

        SAX_INFO.sequences_size = 500
        # Check that linear sequences are filter, and not constant sequences.
        SAX_INFO.coefficients = [0, 0.9]

        result = random_projections(ts_list=["tsa", "tsb"],
                                    sax_info=SAX_INFO,
                                    collision_info=COLLISION_INFO,
                                    recognition_info=RECOGNITION_INFO)

        print("result={}".format(result))
        # Check the length of the alphabet used, and so the breakpoints created
        self.assertEqual(len(result["disc_break_points"]), SAX_INFO.alphabet_size)
        self.assertEqual(len(result["break_points"]), SAX_INFO.alphabet_size - 1)

    def test_regex(self):
        """
        Test the format of the regular expression
        :return:
        """
        alphabet_size = 3
        tests = [
            (['a', 'b', 'c'], '*'),
            (['a', 'a', 'a'], 'a'),
            (['a', 'b', 'b'], '[ab]'),
            (['b', 'a', 'b'], '[ab]'),
        ]

        for words, regex in tests:
            self.assertEqual(regex_from_pattern_results(words, alphabet_size), regex)

        for words, regex in tests:
            long_words = [word * 3 for word in words]
            long_regex = regex * 3

            self.assertEqual(regex_from_pattern_results(long_words, alphabet_size), long_regex)

    def test_empty_regex(self):
        """
        Tests an empty regular expression
        """
        alphabet_size = 3
        words = []

        self.assertEqual(regex_from_pattern_results(words, alphabet_size), EMPTY_REGEX_MESSAGE)

    @mock.patch('ikats.core.resource.api.IkatsApi.ts.read', mock_read_ts)
    def test_paa_values(self):
        """
        Test the length of (all !) the paa values results (sax_info.specify_paa=True).
        """

        # The size of the sequences have to be smaller or equal to the  pattern size, unless the filter will not detect
        # linear sequences.
        SAX_INFO.sequences_size = 500
        # Check that linear sequences are filter, and not constant sequences.
        SAX_INFO.coefficients = [0, 0.9]
        SAX_INFO.specify_paa = True
        COLLISION_INFO.nb_iterations = 1

        # Compute the paa values
        result = random_projections(ts_list=["test_paa_values"],
                                    sax_info=SAX_INFO,
                                    collision_info=COLLISION_INFO,
                                    recognition_info=RECOGNITION_INFO)

        LOGGER.info("result=%s", result)

        # Check the length of the paa values results:
        pattern_names = list(result["patterns"].keys())
        # For all the patterns ("Pi")
        for pattern in pattern_names:

            # All the TS names (here just "ts_1")
            ts_names = list(result["patterns"][pattern]["locations"].keys())

            # Check all the TS
            for name in ts_names:
                # Check the very internal of the *result* dict
                current_result = result["patterns"][pattern]["locations"][name]['seq']
                # current_result is a list of sequences (dict) containing paa_values (key "paa_values")

                # Check all the sequences of each pattern
                for i in range(0, len(current_result)):
                    # Check the very internal of the *result* dict (depth 7 to have access to the paa values !!)
                    # current_result[i]['paa_value']
                    self.assertEqual(len(current_result[i]['paa_value']), SAX_INFO.paa)

    @mock.patch('ikats.core.resource.api.IkatsApi.ts.read', mock_read_ts)
    def test_patterns(self):
        """
        Test the 'main' function of random projection.
        Setting all the param is quite long...
        """
        sax_info = ConfigSax(paa=4,
                             sequences_size=4,
                             with_mean=True,
                             with_std=True,
                             global_norm=True,
                             local_norm=True,
                             linear_filter=True,
                             recovery=1,
                             coefficients=[0.1, 0.9],
                             alphabet_size=4)

        ts_list = [{'tsuid': 'testPatternA'},
                   {'tsuid': 'tesPatternB'},
                   {'tsuid': 'testPatternC'},
                   {'tsuid': 'testPatternConstant'},
                   {'tsuid': 'testPatternLinear'},
                   {'tsuid': 'testPatternTooSmall'},
                   {'tsuid': 'testPatternTrivialMatch'},
                   {'tsuid': 'testPatternRealistic'}]

        # get all the tsuid (necessary for the format of the result)
        tsuid_list = []
        for ts_ref in ts_list:
            tsuid_list.append(ts_ref['tsuid'])

        COLLISION_INFO.nb_iterations = 10
        # set the recognition_info.min_value as done in 'main_random_projection'
        max_iterations = binom(sax_info.paa, COLLISION_INFO.index)
        RECOGNITION_INFO.min_value = int(0.05 * max_iterations)

        RECOGNITION_INFO.activate_spark = None

        # We have set the values

        result = random_projections(ts_list=tsuid_list,
                                    sax_info=sax_info,
                                    collision_info=COLLISION_INFO,
                                    recognition_info=RECOGNITION_INFO)

        LOGGER.info(len(result["patterns"]))
        self.assertTrue(len(result["patterns"]) > 0)
