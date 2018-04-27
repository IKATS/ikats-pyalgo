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
import time
import numpy as np
import mock

from ikats.algo.core.discretization import discretize_dataset
from ikats.core.resource.api import IkatsApi
from ikats.core.resource.client.temporal_data_mgr import DTYPE
from ikats.core.library.exception import IkatsException, IkatsNotFoundError

LOGGER = logging.getLogger("ikats.algo.core.discretization")
# Log format
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)

# List of mocked tables
TABLES = {}


# noinspection PyUnusedLocal
def create_table(data, name=None):
    """
    Mock of IkatsApi.table.create method
    """
    if name is None:
        name = data["table_desc"]["name"]
    TABLES[name] = data


def read_table(name):
    """
    Mock of IkatsApi.table.read method
    """
    return TABLES[name]


class TestDiscretization(unittest.TestCase):
    """
    Tests the discretization module
    """

    def setUp(self):
        self.ds_name = "dataset_test_discretization"
        self.fid1 = 'Timeseries1_For_Discretization_Unit_Testing'
        self.tsuid1 = IkatsApi.ts.create_ref(self.fid1)
        self.fid2 = 'Timeseries2_For_Discretization_Unit_Testing'
        self.tsuid2 = IkatsApi.ts.create_ref(self.fid2)

    def tearDown(self):
        try:
            IkatsApi.ds.delete(ds_name=self.ds_name, deep=True)
        except (TypeError, IkatsNotFoundError, SystemError, ValueError):
            pass
        IkatsApi.ts.delete(tsuid=self.tsuid1, no_exception=True)
        IkatsApi.ts.delete(tsuid=self.tsuid2, no_exception=True)

    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    def test_nominal_discretization(self):
        """
        case : NOMINAL
        nb time series : 2
        nb buckets : 2
        operators : "MIN", "MAX", "AVG", "STD" (all)
        """
        # Preparing 1st TS
        tsuid1 = IkatsApi.ts.create(fid=self.fid1,
                                    data=np.array([
                                        [1449759331000, 3.0],
                                        [1449759332000, 15.0],
                                        [1449759333000, 8.0],
                                        [1449759336000, 25.89],
                                        [1449759338000, 3.0],
                                        [1449759339000, 21.2],
                                        [1449759340000, 18],
                                        [1449759343000, 15.0],
                                        [1449759343500, 12.0],
                                        [1449759344000, 7.5],
                                        [1449759349000, 35.0]]))['tsuid']
        IkatsApi.md.create(tsuid=tsuid1, name='qual_ref_period', value=1000, data_type=DTYPE.number)

        # Preparing 2nd TS
        tsuid2 = IkatsApi.ts.create(fid=self.fid2,
                                    data=np.array([
                                        [1449759331000, -500.0],
                                        [1449759331800, 500.0],
                                        [1449759333042, 1501.5]]))['tsuid']
        IkatsApi.md.create(tsuid=tsuid2, name='qual_ref_period', value=800, data_type=DTYPE.number)

        IkatsApi.ds.create(self.ds_name, "", [self.tsuid1, self.tsuid2])

        table_name = discretize_dataset(ds_name=self.ds_name,
                                        nb_buckets=2,
                                        table_name=str(int(time.time())),
                                        operators_list=["MIN", "MAX", "AVG", "STD"],
                                        nb_points_by_chunk=5)

        result = IkatsApi.table.read(name=table_name)

        self.assertAlmostEqual(float(result['content']['cells'][0][0]), 3, delta=1e-2)
        self.assertAlmostEqual(float(result['content']['cells'][0][1]), 25.89, delta=1e-2)
        self.assertAlmostEqual(float(result['content']['cells'][0][2]), 13.44, delta=1e-2)
        self.assertAlmostEqual(float(result['content']['cells'][0][3]), 8.33, delta=1e-2)

        self.assertAlmostEqual(float(result['content']['cells'][0][4]), 7.5, delta=1e-2)
        self.assertAlmostEqual(float(result['content']['cells'][0][5]), 35, delta=1e-2)
        self.assertAlmostEqual(float(result['content']['cells'][0][6]), 17.37, delta=1e-2)
        self.assertAlmostEqual(float(result['content']['cells'][0][7]), 10.52, delta=1e-2)

        self.assertAlmostEqual(float(result['content']['cells'][1][0]), -500, delta=1e-2)
        self.assertAlmostEqual(float(result['content']['cells'][1][1]), 500, delta=1e-2)
        self.assertAlmostEqual(float(result['content']['cells'][1][2]), 0, delta=1e-2)
        self.assertAlmostEqual(float(result['content']['cells'][1][3]), 500, delta=1e-2)

        self.assertAlmostEqual(float(result['content']['cells'][1][4]), 1501.5, delta=1e-2)
        self.assertAlmostEqual(float(result['content']['cells'][1][5]), 1501.5, delta=1e-2)
        self.assertAlmostEqual(float(result['content']['cells'][1][6]), 1501.5, delta=1e-2)
        self.assertAlmostEqual(float(result['content']['cells'][1][7]), 0, delta=1e-2)

    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    def test_too_much_buckets(self):
        """
        case : DEGRADED (number of buckets > number of points) - Exception raised
        nb timeseries : 1
        nb buckets : 12
        nb points : 11
        """
        # Preparing  TS
        IkatsApi.ts.create(fid=self.fid1,
                           data=np.array([
                               [1449759331000, 3.0],
                               [1449759332000, 15.0],
                               [1449759333000, 8.0],
                               [1449759336000, 25.89],
                               [1449759338000, 3.0],
                               [1449759339000, 21.2],
                               [1449759340000, 18],
                               [1449759343000, 15.0],
                               [1449759343500, 12.0],
                               [1449759344000, 7.5],
                               [1449759349000, 35.0]]))

        IkatsApi.md.create(tsuid=self.tsuid1, name='qual_ref_period', value=1000, data_type=DTYPE.number)

        IkatsApi.ds.create(self.ds_name, "", [self.tsuid1])

        with self.assertRaises(IkatsException):
            discretize_dataset(ds_name=self.ds_name,
                               nb_buckets=12,
                               table_name=str(int(time.time())),
                               operators_list=["MIN", "MAX", "AVG", "STD"],
                               nb_points_by_chunk=5)

            # self.fail("Unexpected result for discretization testing : this test should have raised an exception ")

    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    def test_empty_buckets(self):
        """
        case : empty buckets is filled with none
        nb timeseries : 1
        nb buckets : 11
        nb points : 11
        """
        # Preparing 1st TS
        fid1 = 'Timeseries1_For_Discretization_Unit_Testing'
        IkatsApi.ts.create(fid=fid1,
                           data=np.array([
                               [1449759331000, 3.0],
                               [1449759332000, 15.0],
                               [1449759333000, 8.0],
                               # here empty bucket no 3 (sd,ed) = (1449759334275, 1449759335911)
                               [1449759336000, 25.89],
                               [1449759338000, 3.0],
                               [1449759339000, 21.2],
                               [1449759340000, 18],
                               [1449759343000, 15.0],
                               [1449759343500, 12.0],
                               [1449759344000, 7.5],
                               [1449759349000, 35.0]]))
        IkatsApi.md.create(tsuid=self.tsuid1, name='qual_ref_period', value=1000, data_type=DTYPE.number)

        IkatsApi.ds.create(self.ds_name, "", [self.tsuid1])

        table_name = discretize_dataset(ds_name=self.ds_name,
                                        nb_buckets=11,
                                        table_name=str(int(time.time())),
                                        operators_list=["MIN", "MAX", "AVG", "STD"])

        result = IkatsApi.table.read(name=table_name)

        # check empty chunk is filled with None
        # bucket 1 : indexes 0 to 3
        # bucket 2 : indexes 4 to 7
        # bucket 3 : indexes 8 to 11
        self.assertEqual(result['content']['cells'][0][8], None)
        self.assertEqual(result['content']['cells'][0][9], None)
        self.assertEqual(result['content']['cells'][0][10], None)
        self.assertEqual(result['content']['cells'][0][11], None)
