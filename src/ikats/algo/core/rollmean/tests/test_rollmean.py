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

from ikats.algo.core.rollmean.rollmean_computation import rollmean, Alignment, rollmean_tsuid, rollmean_ts_list, \
    rollmean_ds
from ikats.core.resource.api import IkatsApi
from ikats.core.resource.client import TemporalDataMgr

LOGGER = logging.getLogger()
# Log format
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)

# TS used for calculation
TS1_DATA = np.array([[np.float64(14879031000), 1.0],
                     [np.float64(14879032000), 2.0],
                     [np.float64(14879033000), 10.0],
                     [np.float64(14879034000), 3.0],
                     [np.float64(14879035000), 4.0],
                     [np.float64(14879036000), 5.0],
                     [np.float64(14879037000), 6.0],
                     [np.float64(14879038000), 7.0],
                     [np.float64(14879039000), 8.0],
                     [np.float64(14879040000), 9.0],
                     [np.float64(14879041000), 10.0]])


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
        ts_content = TS1_DATA
    elif ts_id == 2:
        ts_content = [
            [14879030000, 5.0],
            [14879031000, 6.0],
            [14879032000, 8.0],
            [14879033000, -15.0],
            [14879034000, 2.0],
            [14879035000, 6.0],
            [14879036000, 3.0],
            [14879037000, 2.0],
            [14879038000, 42.0],
            [14879039000, 8.0]
        ]
    elif ts_id == 3:
        ts_content = [
            [14879030500, 5.0],
            [14879031000, 6.0],
            [14879032000, 8.0],
            [14879033000, 7.0],
            [14879034000, 9.0],
            [14879037000, 9.0],
            [14879039000, 9.0],
            [14879041000, 9.0],
            [14879043000, 9.0],
            [14879044500, 10.0]
        ]
    else:
        raise NotImplementedError

    # Create the timeseries
    result = IkatsApi.ts.create(fid=fid, data=np.array(ts_content))
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_ref_period", value=1000, force_update=True)
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_nb_points", value=len(ts_content), force_update=True)
    IkatsApi.md.create(tsuid=result['tsuid'], name="metric", value="metric_%s" % ts_id, force_update=True)
    IkatsApi.md.create(tsuid=result['tsuid'], name="funcId", value="fid_%s" % ts_id, force_update=True)
    if not result['status']:
        raise SystemError("Error while creating TS %s" % ts_id)

    return {"tsuid": result['tsuid'], "funcId": fid}


# Temporal data manager
TDM = TemporalDataMgr()


class TestRollmean(unittest.TestCase):
    """
    Test the rollmean algorithm
    """

    def test_rollmean_value(self):
        """
        Testing the result values of the rollmean algorithm
        """
        window_size = 2
        results = rollmean(TS1_DATA, window_size=window_size, alignment=Alignment.left)
        result = results.data[:, 1]
        self.assertEqual(result[0], 1.5)
        self.assertEqual(result[1], 6.0)
        self.assertEqual(result[2], 6.5)
        self.assertEqual(result[3], 3.5)
        self.assertEqual(result[4], 4.5)
        self.assertEqual(result[5], 5.5)
        self.assertEqual(result[6], 6.5)
        self.assertEqual(result[7], 7.5)
        self.assertEqual(result[8], 8.5)
        self.assertEqual(result[9], 9.5)

    def test_rollmean_window(self):
        """
        Testing the window size and period options of the rollmean algorithm
        """

        ts_info = gen_ts(1)

        try:

            window_size = 6
            results_1 = rollmean_tsuid(tdm=TDM, tsuid=ts_info['tsuid'], window_size=window_size,
                                       alignment=Alignment.left, save=False)
            self.assertEqual(len(results_1), len(TS1_DATA) - window_size + 1)

            window_size = 3
            results_2 = rollmean_tsuid(tdm=TDM, tsuid=ts_info['tsuid'], window_size=window_size,
                                       alignment=Alignment.left, save=False)
            self.assertEqual(len(results_2), len(TS1_DATA) - window_size + 1)

            window_period = 3000
            results_3 = rollmean_tsuid(tdm=TDM, tsuid=ts_info['tsuid'], window_period=window_period,
                                       alignment=Alignment.left, save=False)
            self.assertEqual(len(results_3), len(TS1_DATA) - window_period / 1000 + 1)

            window_period = 6000
            results_4 = rollmean_tsuid(tdm=TDM, tsuid=ts_info['tsuid'], window_period=window_period,
                                       alignment=Alignment.left, save=False)
            self.assertEqual(len(results_4), len(TS1_DATA) - window_period / 1000 + 1)

            self.assertTrue(np.allclose(results_1.data, results_4.data))

        finally:
            # Clean up database
            self.clean_up_db([ts_info])

    @staticmethod
    def clean_up_db(ts_info):
        """
        Clean up the database by removing created TS
        :param ts_info: list of TS to remove
        """
        for ts_item in ts_info:
            # Delete created TS
            IkatsApi.ts.delete(tsuid=ts_item['tsuid'], no_exception=True)

    def test_rollmean_alignment(self):
        """
        Testing the alignments (left,center,right) of the rollmean algorithm
        """
        for window_size in [1, 2, 5, 6]:
            results_1 = rollmean(TS1_DATA, window_size=window_size, alignment=Alignment.left)
            self.assertEqual(len(results_1), len(TS1_DATA) - window_size + 1)
            self.assertEqual(results_1.timestamps[0], TS1_DATA[0][0])

            results_2 = rollmean(TS1_DATA, window_size=window_size, alignment=Alignment.right)
            self.assertEqual(len(results_2), len(TS1_DATA) - window_size + 1)
            self.assertEqual(results_2.timestamps[-1], TS1_DATA[-1][0])

            results_3 = rollmean(TS1_DATA, window_size=window_size, alignment=Alignment.center)
            self.assertEqual(len(results_3), len(TS1_DATA) - window_size + 1)
            self.assertEqual(results_3.timestamps[0], TS1_DATA[int(window_size / 2)][0])

            results_4 = rollmean(TS1_DATA, window_size=window_size, alignment=2)
            self.assertEqual(len(results_4), len(TS1_DATA) - window_size + 1)
            self.assertEqual(results_4.timestamps[0], TS1_DATA[int(window_size / 2)][0])

            self.assertTrue(np.allclose(results_1.values, results_2.values))
            self.assertTrue(np.allclose(results_1.values, results_3.values))
            self.assertTrue(np.allclose(results_3.values, results_4.values))
            self.assertTrue(np.allclose(results_3.timestamps, results_4.timestamps))

    # noinspection PyTypeChecker
    def test_rollmean_robustness(self):
        """
        Testing the robustness cases of the rollmean algorithm
        """

        ts_info = gen_ts(1)

        try:

            # Period and size set
            with self.assertRaises(ValueError):
                rollmean_tsuid(tdm=TDM, tsuid=ts_info['tsuid'], window_size=5, window_period=5,
                               alignment=Alignment.left)

            # size = 0
            with self.assertRaises(ValueError):
                rollmean(TS1_DATA, window_size=0, alignment=Alignment.left)

            # size too big
            with self.assertRaises(ValueError):
                rollmean(TS1_DATA, window_size=len(TS1_DATA), alignment=Alignment.left)

            # size too big
            with self.assertRaises(ValueError):
                rollmean_tsuid(tdm=TDM, tsuid=ts_info['tsuid'], window_period=(TS1_DATA[-1][0] - TS1_DATA[0][0]),
                               alignment=Alignment.left)

            # wrong type
            with self.assertRaises(TypeError):
                rollmean_tsuid(tdm=TDM, tsuid=ts_info['tsuid'], window_size='ttrtr', alignment=Alignment.left)

            # wrong type
            with self.assertRaises(TypeError):
                rollmean_tsuid(tdm=TDM, tsuid=ts_info['tsuid'], window_period='ttrtr', alignment=Alignment.left)

            # wrong type
            with self.assertRaises(TypeError):
                rollmean_tsuid(tdm=TDM, tsuid=ts_info['tsuid'], window_period=500, alignment="wrong")

            # wrong value
            with self.assertRaises(TypeError):
                rollmean(TS1_DATA, window_size=5, alignment=0)

            # wrong value
            with self.assertRaises(TypeError):
                rollmean(TS1_DATA, window_size=5, alignment=4)

        finally:
            # Clean up database
            self.clean_up_db([ts_info])

    def test_rollmean_ts_list(self):
        """
        Testing the the rollmean algorithm from ts list
        """

        ts_info = gen_ts(1)
        try:
            window_size = 10
            results = rollmean_ts_list(tdm=TDM, ts_list=[ts_info['tsuid']], window_size=window_size,
                                       alignment=Alignment.left, save=False)
            self.assertEqual(len(results), 1)

        finally:
            # Clean up database
            self.clean_up_db([ts_info])

    def test_rollmean_ds(self):
        """
        Testing the the rollmean algorithm from dataset
        """
        window_size = 10
        results = rollmean_ds(tdm=TDM, ds_name='Portfolio', window_size=window_size,
                              alignment=Alignment.left, save=False)
        self.assertEqual(len(results), 13)
