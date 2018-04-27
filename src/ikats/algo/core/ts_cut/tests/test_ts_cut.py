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

import mock

from ikats.algo.core.ts_cut import TsCut, cut_ts, cut_ds


def ts_fid_mock(tsuid, *args, **kwargs):
    """
    Mock of IkatsApi.ts.fid method
    """

    return "FuncId_%s" % tsuid


# noinspection PyUnusedLocal
def ds_read_mock(data_set, *args, **kwargs):
    """
    Mock of IkatsApi.ts.fid method

    Same parameters and types as the original function

    """
    return {"description": "description of my data set",
            "ts_list": ['00001',
                        '00002',
                        '00003',
                        '00004']}


def md_read_mock(ts_list, *args, **kwargs):
    """
    Mock of the get_meta_data
    :param ts_list:
    :return:
    """
    if ts_list == 'MOCK_TSUID':
        return {
            'MOCK_TSUID': {
                'nb_points': 48,
                'ikats_start_date': 1000,
                'ikats_end_date': 48000
            }
        }
    if ts_list == ['00001', '00002', '00003', '00004']:
        return {
            'MOCK_TSUID': {
                'nb_points': 48,
                'ikats_start_date': 1000,
                'ikats_end_date': 48000
            }
        }
    return {}


def ts_create_mock(fid, data=None, *args, **kwargs):
    """
    Mock for IkatsApi.ts.create
    :param fid:
    :param data:
    :param generate_metadata:
    :param args:
    :param kwargs:
    :return:
    """
    return {
        'status': True,
        'errors': 0,
        'numberOfSuccess': len(data),
        'summary': "%s points imported" % len(data),
        'tsuid': "TSUID_NEW",
        'funcId': fid,
        'responseStatus': 200
    }


# noinspection PyUnusedLocal
def ts_read_mock(tsuid_list, sd=None, ed=None, *args, **kwargs):
    """
    Mock for IkatsApi.ts.read
    :param tsuid_list:
    :param sd:
    :param ed:
    :return:
    """
    if tsuid_list == 'MOCK_TSUID':
        results = [[1000, 5.18856434e-02],
                   [2000, 6.94607040e-03],
                   [3000, -5.26332467e-02],
                   [4000, -4.40156312e-02],
                   [5000, 3.75013936e-03],
                   [6000, -6.35807976e-02],
                   [7000, 1.45687443e-01],
                   [8000, -5.03118675e-02],
                   [9000, 6.32878506e-03],
                   [10000, -4.24221316e-02],
                   [11000, -2.94103443e-02],
                   [12000, -3.95461882e-02],
                   [13000, 1.13109744e-01],
                   [14000, 2.48367673e-02],
                   [15000, -2.42348519e-02],
                   [16000, 3.12426792e-02],
                   [17000, -7.05937076e-02],
                   [18000, -1.67566779e-02],
                   [19000, 7.18532271e-02],
                   [20000, -8.74254005e-02],
                   [21000, 1.29627439e-01],
                   [22000, 1.88644107e-02],
                   [23000, -6.24884331e-03],
                   [24000, 5.26333883e-02],
                   [25000, 3.59958211e-02],
                   [26000, 6.99251845e-02],
                   [27000, 7.51970063e-02],
                   [28000, 9.01660091e-02],
                   [29000, -1.59407775e-01],
                   [30000, -1.98599134e-02],
                   [31000, 0.00000000e+00],
                   [32000, -7.92749040e-02],
                   [33000, -6.06196677e-03],
                   [34000, 3.12572355e-02],
                   [35000, -6.21260930e-03],
                   [36000, 8.05300714e-02],
                   [37000, 2.20697541e-02],
                   [38000, -3.92221490e-02],
                   [39000, -1.25713737e-01],
                   [40000, 4.79124959e-02],
                   [41000, -9.73034094e-02],
                   [42000, -5.37224108e-03],
                   [43000, 3.33284911e-02],
                   [44000, 2.27311317e-02],
                   [45000, 2.92392551e-02],
                   [46000, -5.80953415e-03],
                   [47000, 2.38023769e-02],
                   [48000, -4.54516512e-02]]

        if ed:
            match = [a for a in results if a[0] <= ed]
            results = match
        return [results]
    if tsuid_list == ['00001', '00002', '00003', '00004']:
        results = [[[1, 5.18856434e-02],
                    [2, 6.94607040e-03],
                    [3, -5.26332467e-02],
                    [4, -4.40156312e-02],
                    [5, 3.75013936e-03],
                    [6, -6.35807976e-02],
                    [7, 1.45687443e-01],
                    [8, -5.03118675e-02],
                    [9, 6.32878506e-03]],
                   [[2, 6.94607040e-03],
                    [3, -5.26332467e-02],
                    [4, -4.40156312e-02],
                    [5, 3.75013936e-03],
                    [6, -6.35807976e-02],
                    [7, 1.45687443e-01],
                    [8, -5.03118675e-02],
                    [9, 6.32878506e-03],
                    [10, 5.18856434e-02]],
                   [[3, -5.26332467e-02],
                    [4, -4.40156312e-02],
                    [5, 3.75013936e-03],
                    [6, -6.35807976e-02],
                    [7, 1.45687443e-01],
                    [8, -5.03118675e-02],
                    [9, 6.32878506e-03],
                    [10, 5.18856434e-02],
                    [11, 6.94607040e-03]],
                   [[4, -4.40156312e-02],
                    [5, 3.75013936e-03],
                    [6, -6.35807976e-02],
                    [7, 1.45687443e-01],
                    [8, -5.03118675e-02],
                    [9, 6.32878506e-03],
                    [10, 5.18856434e-02],
                    [11, 6.94607040e-03],
                    [12, -5.26332467e-02]]]
        return [results]
    return []


class TestTsCut(TestCase):
    """
    Test of the TS cut operator
    """

    @mock.patch("ikats.core.resource.api.IkatsApi.ts.read", ts_read_mock)
    @mock.patch("ikats.core.resource.api.IkatsApi.md.read", md_read_mock)
    @mock.patch("ikats.core.resource.api.IkatsApi.ts.fid", ts_fid_mock)
    def test_cut_nb_points(self):
        """
        Test of a nominal cut function
        """

        cut_method = TsCut()
        results = cut_method.cut(tsuid='MOCK_TSUID', nb_points=5)
        self.assertEqual(len(results), 5)

        results = cut_method.cut(tsuid='MOCK_TSUID', nb_points='4')
        self.assertEqual(len(results), 4)

    @mock.patch("ikats.core.resource.api.IkatsApi.ts.read", ts_read_mock)
    @mock.patch("ikats.core.resource.api.IkatsApi.ts.create", ts_create_mock)
    @mock.patch("ikats.core.resource.api.IkatsApi.md.read", md_read_mock)
    @mock.patch("ikats.core.resource.api.IkatsApi.ts.fid", ts_fid_mock)
    def test_cut_custom_fid(self):
        """
        Test of a nominal cut function
        """
        fid = "TEST_CUT_FID"

        results = cut_ts(tsuid='MOCK_TSUID', nb_points=5, fid=fid, save=True)
        self.assertEqual(results['funcId'], fid)

    @mock.patch("ikats.core.resource.api.IkatsApi.ts.read", ts_read_mock)
    @mock.patch("ikats.core.resource.api.IkatsApi.md.read", md_read_mock)
    @mock.patch("ikats.core.resource.api.IkatsApi.ts.fid", ts_fid_mock)
    def test_cut_end_date(self):
        """
        Test of a nominal cut function

        """

        cut_method = TsCut()
        results = cut_method.cut(tsuid='MOCK_TSUID', ed=10000)
        self.assertEqual(len(results), 10)

    @mock.patch("ikats.core.resource.api.IkatsApi.ts.read", ts_read_mock)
    @mock.patch("ikats.core.resource.api.IkatsApi.md.read", md_read_mock)
    @mock.patch("ikats.core.resource.api.IkatsApi.ts.fid", ts_fid_mock)
    def test_cut_no_length(self):
        """
        Test of a nominal cut function

        """

        cut_method = TsCut()
        results = cut_method.cut(tsuid='MOCK_TSUID', nb_points=0)
        self.assertEqual(len(results), 0)

    @mock.patch("ikats.core.resource.api.IkatsApi.ts.read", ts_read_mock)
    @mock.patch("ikats.core.resource.api.IkatsApi.md.read", md_read_mock)
    @mock.patch("ikats.core.resource.api.IkatsApi.ts.fid", ts_fid_mock)
    def test_cut_robustness(self):
        """
        Test some robustness cases for cut function
        """

        cut_method = TsCut()
        with self.assertRaises(ValueError):
            # ed < sd
            cut_method.cut(tsuid='MOCK_TSUID', sd=1, ed=0)
        with self.assertRaises(ValueError):
            # ed AND nb_points defined
            cut_method.cut(tsuid='MOCK_TSUID', sd=1, nb_points=20, ed=1449755780000)
        with self.assertRaises(ValueError):
            # no ed nor nb_points defined
            cut_method.cut(tsuid='MOCK_TSUID', sd=1)
        with self.assertRaises(ValueError):
            # TS not found
            cut_method.cut(tsuid='no_ts', sd=1, nb_points=20)
        with self.assertRaises(ValueError):
            # TS not found
            cut_method.cut(tsuid='TS_without_md', nb_points=20)
        with self.assertRaises(TypeError):
            # nb_points invalid
            cut_method.cut(tsuid='MOCK_TSUID', sd=1, nb_points="wrong_value")

    @mock.patch("ikats.core.resource.api.IkatsApi.ts.read", ts_read_mock)
    @mock.patch("ikats.core.resource.api.IkatsApi.md.read", md_read_mock)
    @mock.patch("ikats.core.resource.api.IkatsApi.ts.fid", ts_fid_mock)
    def test_cut_entry_point(self):
        """
        Test of a nominal cut function
        """
        results = cut_ts(tsuid='MOCK_TSUID', nb_points=5, save=False)
        self.assertEqual(len(results), 5)

    # No mock possible due to multiprocessing
    def test_cut_ds_nb_points(self):
        """
        Test of a nominal cut function by providing number of points
        """
        results = cut_ds(ds_name="Portfolio", sd=1449755766000, nb_points=17, save=False)

        self.assertEqual(len(results), 13)

        for i in results:
            self.assertEqual(len(i), 17)

    # No mock possible due to multiprocessing
    def test_cut_ds_end_date(self):
        """
        Test of a nominal cut function by providing end date
        """
        results = cut_ds(ds_name="Portfolio", sd=1449755766000, ed=1449755780000, save=False)
        self.assertEqual(len(results), 13)

        for i in results:
            self.assertEqual(len(i), 15)

    def test_cut_ds_no_pt_in_interval(self):
        """
        Test of a cut function with no point in interval
        """
        results = cut_ds(ds_name="Portfolio", sd=1449755766001, ed=1449755766002, save=False)
        self.assertEqual(len(results), 13)

        for i in results:
            self.assertEqual(len(i), 0)
