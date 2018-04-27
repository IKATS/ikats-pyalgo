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
from unittest import TestCase
import numpy as np

from ikats.algo.core.ts_cut.ds_cut_from_metric import _spark_get_cut_ranges

from ikats.algo.core.ts_cut import cut_ds_from_metric
from ikats.core.library.exception import IkatsNotFoundError
from ikats.core.resource.api import IkatsApi

LOGGER = logging.getLogger('ikats.algo.core.ds_cut_from_metric')
# Log format
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')

# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)


class TestTsCut(TestCase):
    """
    Tests the ds_cut_from_metric method

    """

    def test_get_cut_ranges(self):
        """
        Unit test about the range finder
        """
        data = np.array([[1e12, -5.0],
                         [1e12 + 1000, 6.2],
                         [1e12 + 2000, 6.0],
                         [1e12 + 3600, 8.0],
                         [1e12 + 4000, -15.0],
                         [1e12 + 5000, 2.0],
                         [1e12 + 6000, 6.0],
                         [1e12 + 7000, 3.0],
                         [1e12 + 8000, -2.0],
                         [1e12 + 9000, -42.0],
                         [1e12 + 10000, 8.0],
                         [1e12 + 11000, 8.0],
                         [1e12 + 12000, 8.0],
                         [1e12 + 13000, -8.0]])

        result = _spark_get_cut_ranges(data=data, lambda_expr=lambda x: x > 0)

        self.assertEqual(result, [
            [1e12 + 1000, 1e12 + 3600],
            [1e12 + 5000, 1e12 + 7000],
            [1e12 + 10000, 1e12 + 12000]
        ])

    def test_ds_cut_metric_nominal(self):
        """
        Cut from metric nominal
        """

        try:
            # Create Test data

            # Source used to cut
            result_ts_1 = IkatsApi.ts.create(
                fid="source_cut",
                data=np.array([[1e12, -1],
                               [1e12 + 1000, 1],
                               [1e12 + 2000, 2],
                               [1e12 + 3000, 3],
                               [1e12 + 4000, -3.0],
                               [1e12 + 5000, 3],
                               [1e12 + 6000, 4],
                               [1e12 + 7000, 2],
                               [1e12 + 8000, -2.0],
                               [1e12 + 9000, -10],
                               [1e12 + 10000, 10.0],
                               [1e12 + 11000, 10.0],
                               [1e12 + 12000, 10.0],
                               [1e12 + 13000, -10.0]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_1['tsuid'], name="metric", value="cut_metric")
            IkatsApi.md.create(tsuid=result_ts_1['tsuid'], name="qual_ref_period", value="1000")

            # TS to cut having several points cut (nominal)
            result_ts_2 = IkatsApi.ts.create(
                fid="to_cut_1",
                data=np.array([[1e12, 1],
                               [1e12 + 500, 2],
                               [1e12 + 1000, 3],
                               [1e12 + 2000, 4],
                               [1e12 + 3000, 5],
                               [1e12 + 3500, 6],
                               [1e12 + 4000, 7],
                               [1e12 + 4500, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 6000, 10],
                               [1e12 + 7000, 11],
                               [1e12 + 8000, 12],
                               [1e12 + 8900, 13],
                               [1e12 + 9900, 14],
                               [1e12 + 11000, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_2['tsuid'], name="metric", value="cut_dest1")
            IkatsApi.md.create(tsuid=result_ts_2['tsuid'], name="qual_ref_period", value="1000")

            # TS to cut having single point cut
            # This point is located at 1e12 + 5000, this proves:
            # * there is no interpolation about the value to compare with (otherwise, 1e12+4800 should be matched)
            # * a single point is matched properly (reduced range)
            result_ts_3 = IkatsApi.ts.create(
                fid="to_cut_2",
                data=np.array([[1e12 + 4800, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 12500, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_3['tsuid'], name="metric", value="cut_dest2")
            IkatsApi.md.create(tsuid=result_ts_3['tsuid'], name="qual_ref_period", value="200")

            # TS to cut having no point as result
            result_ts_4 = IkatsApi.ts.create(
                fid="to_cut_3",
                data=np.array([[1e12 + 8100, 8],
                               [1e12 + 8200, 9],
                               [1e12 + 8300, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_4['tsuid'], name="metric", value="cut_dest3")
            IkatsApi.md.create(tsuid=result_ts_4['tsuid'], name="qual_ref_period", value="100")

            # TS to cut having all points in result
            result_ts_5 = IkatsApi.ts.create(
                fid="to_cut_4",
                data=np.array([[1e12 + 1100, 1],
                               [1e12 + 2000, 2],
                               [1e12 + 2900, 3]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_5['tsuid'], name="metric", value="cut_dest4")
            IkatsApi.md.create(tsuid=result_ts_5['tsuid'], name="qual_ref_period", value="900")

            # Creation of dataset
            ds_name = "TEST_cut_ds_156940"
            IkatsApi.ds.delete(ds_name=ds_name, deep=True)
            tsuid_list = [result_ts_1['tsuid'], result_ts_2['tsuid'], result_ts_3['tsuid'], result_ts_4['tsuid'],
                          result_ts_5['tsuid']]
            IkatsApi.ds.create(ds_name=ds_name,
                               description="Test for cut ds from metric",
                               tsuid_list=tsuid_list)
            ds_length = len(tsuid_list)

            # Run the test
            fid_pattern = "%(fid)s_%(M)s_cut"
            metric = "cut_metric"
            result = cut_ds_from_metric(ds_name=ds_name,
                                        metric=metric,
                                        criteria="M>0",
                                        fid_pattern=fid_pattern)

            # Check results
            # "-2" because Ts_ref is not cut (only used as reference) and 3rd TS has no matching point
            self.assertEqual(len(result), ds_length - 2)

            # Check filtered content from first TS
            ts1_data = IkatsApi.ts.read(tsuid_list=result[0]['tsuid'])[0]
            self.assertTrue((ts1_data == np.array([[1e12 + 1000, 3],
                                                   [1e12 + 2000, 4],
                                                   [1e12 + 3000, 5],
                                                   [1e12 + 5000, 9],
                                                   [1e12 + 6000, 10],
                                                   [1e12 + 7000, 11],
                                                   [1e12 + 11000, 15]])).all())

            # Check filtered content from second TS
            ts2_data = IkatsApi.ts.read(tsuid_list=result[1]['tsuid'])[0]
            self.assertTrue((ts2_data == np.array([[1e12 + 5000, 9]])).all())

            # Check unfiltered content from fourth TS
            ts4_data = IkatsApi.ts.read(tsuid_list=result[2]['tsuid'])[0]
            self.assertTrue((ts4_data == np.array([[1e12 + 1100, 1],
                                                   [1e12 + 2000, 2],
                                                   [1e12 + 2900, 3]])).all())
        finally:

            # Clean up database

            for fid in ['source_cut', 'to_cut_1', 'to_cut_2', 'to_cut_3', 'to_cut_4']:
                # Delete created TS
                try:
                    tsuid = IkatsApi.fid.tsuid(fid_pattern % {'M': metric, 'fid': fid})
                    IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
                except ValueError:
                    # No TS to delete
                    pass
            try:
                # Delete test dataset
                IkatsApi.ds.delete(ds_name=ds_name, deep=True)
            except (TypeError, IkatsNotFoundError, SystemError, ValueError):
                # Don't try to delete Dataset that is not yet created (crashed before creating it)
                pass

    def test_ds_cut_metric_with_chunks(self):
        """
        Cut from metric nominal
        """

        # Forcing chunk_size=3 to split into several chunks
        chunk_size = 5

        try:
            # Create Test data

            # Source used to cut
            result_ts_1 = IkatsApi.ts.create(
                fid="source_cut",
                data=np.array([[1e12, -1],
                               [1e12 + 1000, 1],
                               [1e12 + 2000, 2],
                               [1e12 + 3000, 3],
                               [1e12 + 4000, -3.0],
                               [1e12 + 5000, 3],
                               [1e12 + 6000, 4],
                               [1e12 + 7000, 2],
                               [1e12 + 8000, -2.0],
                               [1e12 + 9000, -10],
                               [1e12 + 10000, 10.0],
                               [1e12 + 11000, 10.0],
                               [1e12 + 12000, 10.0],
                               [1e12 + 13000, -10.0]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_1['tsuid'], name="metric", value="cut_metric")
            IkatsApi.md.create(tsuid=result_ts_1['tsuid'], name="qual_ref_period", value="1000")

            # TS to cut having several points cut (nominal)
            result_ts_2 = IkatsApi.ts.create(
                fid="to_cut_1",
                data=np.array([[1e12, 1],
                               [1e12 + 500, 2],
                               [1e12 + 1000, 3],
                               [1e12 + 2000, 4],
                               [1e12 + 3000, 5],
                               [1e12 + 3500, 6],
                               [1e12 + 4000, 7],
                               [1e12 + 4500, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 6000, 10],
                               [1e12 + 7000, 11],
                               [1e12 + 8000, 12],
                               [1e12 + 8900, 13],
                               [1e12 + 9900, 14],
                               [1e12 + 11000, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_2['tsuid'], name="metric", value="cut_dest1")
            IkatsApi.md.create(tsuid=result_ts_2['tsuid'], name="qual_ref_period", value="1000")

            # TS to cut having single point cut
            result_ts_3 = IkatsApi.ts.create(
                fid="to_cut_2",
                data=np.array([[1e12 + 4800, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 12500, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_3['tsuid'], name="metric", value="cut_dest2")
            IkatsApi.md.create(tsuid=result_ts_3['tsuid'], name="qual_ref_period", value="200")

            # TS to cut having no points as result
            result_ts_4 = IkatsApi.ts.create(
                fid="to_cut_3",
                data=np.array([[1e12 + 8100, 8],
                               [1e12 + 8200, 9],
                               [1e12 + 8300, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_4['tsuid'], name="metric", value="cut_dest3")
            IkatsApi.md.create(tsuid=result_ts_4['tsuid'], name="qual_ref_period", value="100")

            # Creation of dataset
            ds_name = "TEST_cut_ds_156940"
            IkatsApi.ds.delete(ds_name=ds_name, deep=True)
            tsuid_list = [result_ts_1['tsuid'], result_ts_2['tsuid'], result_ts_3['tsuid'], result_ts_4['tsuid']]
            IkatsApi.ds.create(ds_name=ds_name,
                               description="Test for cut ds from metric",
                               tsuid_list=tsuid_list)
            ds_length = len(tsuid_list)

            # Run the test (forcing chunk_sizeto split into several chunks)
            fid_pattern = "%(fid)s_%(M)s_cut"
            metric = "cut_metric"
            result = cut_ds_from_metric(ds_name=ds_name,
                                        metric=metric,
                                        criteria="M>0",
                                        fid_pattern=fid_pattern,
                                        chunk_size=chunk_size)

            # Check results
            # "-2" because Ts_ref is not cut (only used as reference) and 3rd TS has no matching point
            self.assertEqual(len(result), ds_length - 2)

            # Check filtered content from first TS
            ts1_data = IkatsApi.ts.read(tsuid_list=result[0]['tsuid'])[0]
            self.assertTrue((ts1_data == np.array([[1e12 + 1000, 3],
                                                   [1e12 + 2000, 4],
                                                   [1e12 + 3000, 5],
                                                   [1e12 + 5000, 9],
                                                   [1e12 + 6000, 10],
                                                   [1e12 + 7000, 11],
                                                   [1e12 + 11000, 15]])).all())

            # Check filtered content from second TS
            ts2_data = IkatsApi.ts.read(tsuid_list=result[1]['tsuid'])[0]
            self.assertTrue((ts2_data == np.array([[1e12 + 5000, 9]])).all())

        finally:

            # Clean up database

            for fid in ['source_cut', 'to_cut_1', 'to_cut_2', 'to_cut_3']:
                # Delete created TS
                try:
                    tsuid = IkatsApi.fid.tsuid(fid_pattern % {'M': metric, 'fid': fid})
                    IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
                except ValueError:
                    # No TS to delete
                    pass
            try:
                # Delete test dataset
                IkatsApi.ds.delete(ds_name=ds_name, deep=True)
            except Exception:
                # Don't try to delete Dataset that is not yet created (crashed before creating it)
                pass

    def test_ds_cut_group(self):
        """
        Cut from metric nominal using group by parameter
        """

        try:
            # Create Test data
            # These data contains 2 groups having different references.
            # Each reference TS has different cut ranges.
            # This case shows each group is cut based on its own reference.

            # Source used to cut
            result_ts_1 = IkatsApi.ts.create(
                fid="G1_source_cut",
                data=np.array([[1e12, -1],
                               [1e12 + 1000, 1],
                               [1e12 + 2000, 2],
                               [1e12 + 3000, 3],
                               [1e12 + 4000, -3.0],
                               [1e12 + 5000, 3],
                               [1e12 + 6000, 4],
                               [1e12 + 7000, 2],
                               [1e12 + 8000, -2.0],
                               [1e12 + 9000, -10],
                               [1e12 + 10000, 10.0],
                               [1e12 + 11000, 10.0],
                               [1e12 + 12000, 10.0],
                               [1e12 + 13000, -10.0]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_1['tsuid'], name="metric", value="cut_metric")
            IkatsApi.md.create(tsuid=result_ts_1['tsuid'], name="my_group", value="1")
            IkatsApi.md.create(tsuid=result_ts_1['tsuid'], name="qual_ref_period", value="1000")

            # TS to cut having several points cut (nominal)
            result_ts_2 = IkatsApi.ts.create(
                fid="G1_to_cut_1",
                data=np.array([[1e12, 1],
                               [1e12 + 500, 2],
                               [1e12 + 1000, 3],
                               [1e12 + 2000, 4],
                               [1e12 + 3000, 5],
                               [1e12 + 3500, 6],
                               [1e12 + 4000, 7],
                               [1e12 + 4500, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 6000, 10],
                               [1e12 + 7000, 11],
                               [1e12 + 8000, 12],
                               [1e12 + 8900, 13],
                               [1e12 + 9900, 14],
                               [1e12 + 11000, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_2['tsuid'], name="metric", value="cut_dest1")
            IkatsApi.md.create(tsuid=result_ts_2['tsuid'], name="my_group", value="1")
            IkatsApi.md.create(tsuid=result_ts_2['tsuid'], name="qual_ref_period", value="1000")

            # TS to cut having single point cut
            result_ts_3 = IkatsApi.ts.create(
                fid="G1_to_cut_2",
                data=np.array([[1e12 + 4800, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 12500, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_3['tsuid'], name="metric", value="cut_dest2")
            IkatsApi.md.create(tsuid=result_ts_3['tsuid'], name="my_group", value="1")
            IkatsApi.md.create(tsuid=result_ts_3['tsuid'], name="qual_ref_period", value="1000")

            # TS to cut having no point as result
            result_ts_4 = IkatsApi.ts.create(
                fid="G1_to_cut_3",
                data=np.array([[1e12 + 8100, 8],
                               [1e12 + 8200, 9],
                               [1e12 + 8300, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_4['tsuid'], name="metric", value="cut_dest3")
            IkatsApi.md.create(tsuid=result_ts_4['tsuid'], name="my_group", value="1")
            IkatsApi.md.create(tsuid=result_ts_4['tsuid'], name="qual_ref_period", value="1000")

            # TS to cut having all points in result
            result_ts_5 = IkatsApi.ts.create(
                fid="G1_to_cut_4",
                data=np.array([[1e12 + 1100, 1],
                               [1e12 + 2000, 2],
                               [1e12 + 2900, 3]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_5['tsuid'], name="metric", value="cut_dest4")
            IkatsApi.md.create(tsuid=result_ts_5['tsuid'], name="my_group", value="1")
            IkatsApi.md.create(tsuid=result_ts_5['tsuid'], name="qual_ref_period", value="900")

            # 2nd group is the exact opposite of the first group (source cut values are multiplied by -1)
            result_ts_6 = IkatsApi.ts.create(
                fid="G2_source_cut",
                data=np.array([[1e12, -1],
                               [1e12 + 1000, -1],
                               [1e12 + 2000, -2],
                               [1e12 + 3000, -3],
                               [1e12 + 4000, 3.0],
                               [1e12 + 5000, -3],
                               [1e12 + 6000, -4],
                               [1e12 + 7000, -2],
                               [1e12 + 8000, 2.0],
                               [1e12 + 9000, 10],
                               [1e12 + 10000, -10.0],
                               [1e12 + 11000, -10.0],
                               [1e12 + 12000, -10.0],
                               [1e12 + 13000, 10.0]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_6['tsuid'], name="metric", value="cut_metric")
            IkatsApi.md.create(tsuid=result_ts_6['tsuid'], name="my_group", value="2")
            IkatsApi.md.create(tsuid=result_ts_6['tsuid'], name="qual_ref_period", value="1000")

            result_ts_7 = IkatsApi.ts.create(
                fid="G2_to_cut_1",
                data=np.array([[1e12, 1],
                               [1e12 + 500, 2],
                               [1e12 + 1000, 3],
                               [1e12 + 2000, 4],
                               [1e12 + 3000, 5],
                               [1e12 + 3500, 6],
                               [1e12 + 4000, 7],
                               [1e12 + 4500, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 6000, 10],
                               [1e12 + 7000, 11],
                               [1e12 + 8000, 12],
                               [1e12 + 8900, 13],
                               [1e12 + 9900, 14],
                               [1e12 + 11000, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_7['tsuid'], name="metric", value="cut_dest1")
            IkatsApi.md.create(tsuid=result_ts_7['tsuid'], name="my_group", value="2")
            IkatsApi.md.create(tsuid=result_ts_7['tsuid'], name="qual_ref_period", value="1000")

            result_ts_8 = IkatsApi.ts.create(
                fid="G2_to_cut_2",
                data=np.array([[1e12 + 4800, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 12500, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_8['tsuid'], name="metric", value="cut_dest2")
            IkatsApi.md.create(tsuid=result_ts_8['tsuid'], name="my_group", value="2")
            IkatsApi.md.create(tsuid=result_ts_8['tsuid'], name="qual_ref_period", value="1000")

            result_ts_9 = IkatsApi.ts.create(
                fid="G2_to_cut_3",
                data=np.array([[1e12 + 8100, 8],
                               [1e12 + 8200, 9],
                               [1e12 + 8300, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_9['tsuid'], name="metric", value="cut_dest3")
            IkatsApi.md.create(tsuid=result_ts_9['tsuid'], name="my_group", value="2")
            IkatsApi.md.create(tsuid=result_ts_9['tsuid'], name="qual_ref_period", value="1000")

            result_ts_10 = IkatsApi.ts.create(
                fid="G2_to_cut_4",
                data=np.array([[1e12 + 1100, 1],
                               [1e12 + 2000, 2],
                               [1e12 + 2900, 3]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_10['tsuid'], name="metric", value="cut_dest4")
            IkatsApi.md.create(tsuid=result_ts_10['tsuid'], name="my_group", value="2")
            IkatsApi.md.create(tsuid=result_ts_10['tsuid'], name="qual_ref_period", value="900")

            # Creation of dataset
            ds_name = "TEST_cut_ds_157215"
            IkatsApi.ds.delete(ds_name=ds_name, deep=True)
            tsuid_list = [
                result_ts_1['tsuid'],
                result_ts_2['tsuid'],
                result_ts_3['tsuid'],
                result_ts_4['tsuid'],
                result_ts_5['tsuid'],
                result_ts_6['tsuid'],
                result_ts_7['tsuid'],
                result_ts_8['tsuid'],
                result_ts_9['tsuid'],
                result_ts_10['tsuid']
            ]
            IkatsApi.ds.create(ds_name=ds_name,
                               description="Test for cut ds from metric",
                               tsuid_list=tsuid_list)

            # Run the test
            fid_pattern = "%(fid)s__%(M)s__cut"
            metric = "cut_metric"
            result = cut_ds_from_metric(ds_name=ds_name,
                                        metric=metric,
                                        criteria="M>0",
                                        fid_pattern=fid_pattern,
                                        group_by="my_group")

            # Expected TS result:
            ev_list = {
                "G1_to_cut_1__cut_metric__cut": np.array([[1e12 + 1000, 3.0],
                                                          [1e12 + 2000, 4.0],
                                                          [1e12 + 3000, 5.0],
                                                          [1e12 + 5000, 9.0],
                                                          [1e12 + 6000, 10.0],
                                                          [1e12 + 7000, 11.0],
                                                          [1e12 + 11000, 15.0]]),
                "G1_to_cut_2__cut_metric__cut": np.array([[1e12 + 5000, 9.0]]),
                "G1_to_cut_4__cut_metric__cut": np.array([[1e12 + 1100, 1.0],
                                                          [1e12 + 2000, 2.0],
                                                          [1e12 + 2900, 3.0]]),
                "G2_to_cut_1__cut_metric__cut": np.array([[1e12 + 4000, 7.0],
                                                          [1e12 + 8000, 12.0],
                                                          [1e12 + 8900, 13.0]]),
                "G2_to_cut_3__cut_metric__cut": np.array([[1e12 + 8100, 8.0],
                                                          [1e12 + 8200, 9.0],
                                                          [1e12 + 8300, 15.0]])
            }

            # Check results
            self.assertEqual(len(result), len(ev_list.keys()))

            # Check TS content
            for item in result:
                ts_data = IkatsApi.ts.read(tsuid_list=item['tsuid'])[0]
                self.assertTrue((ts_data == ev_list[item['funcId']]).all(),
                                msg="%s doesn't match expected value" % item['funcId'])

        finally:

            # Clean up database

            for fid in ['G1_source_cut', 'G1_to_cut_1', 'G1_to_cut_2', 'G1_to_cut_3', 'G1_to_cut_4',
                        'G2_source_cut', 'G2_to_cut_1', 'G2_to_cut_2', 'G2_to_cut_3', 'G2_to_cut_4']:
                # Delete created TS
                try:
                    tsuid = IkatsApi.fid.tsuid(fid_pattern % {'M': metric, 'fid': fid})
                    IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
                except ValueError:
                    # No TS to delete
                    pass
            try:
                # Delete test dataset
                IkatsApi.ds.delete(ds_name=ds_name, deep=True)
            except Exception:
                # Don't try to delete Dataset that is not yet created (crashed before creating it)
                pass

    def test_ds_cut_metric_input_errors(self):
        """
        Cut from metric Robustness tests
        """

        try:
            # Create Test data
            # Source used to cut
            result_ts_1 = IkatsApi.ts.create(
                fid="G1_source_cut",
                data=np.array([[1e12, -1],
                               [1e12 + 1000, 1],
                               [1e12 + 2000, 2],
                               [1e12 + 3000, 3],
                               [1e12 + 4000, -3.0],
                               [1e12 + 5000, 3],
                               [1e12 + 6000, 4],
                               [1e12 + 7000, 2],
                               [1e12 + 8000, -2.0],
                               [1e12 + 9000, -10],
                               [1e12 + 10000, 10.0],
                               [1e12 + 11000, 10.0],
                               [1e12 + 12000, 10.0],
                               [1e12 + 13000, -10.0]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_1['tsuid'], name="metric", value="cut_metric")
            IkatsApi.md.create(tsuid=result_ts_1['tsuid'], name="my_group", value="1")
            IkatsApi.md.create(tsuid=result_ts_1['tsuid'], name="qual_ref_period", value="1000")

            # TS to cut having several points cut (nominal)
            result_ts_2 = IkatsApi.ts.create(
                fid="G1_to_cut_1",
                data=np.array([[1e12, 1],
                               [1e12 + 500, 2],
                               [1e12 + 1000, 3],
                               [1e12 + 2000, 4],
                               [1e12 + 3000, 5],
                               [1e12 + 3500, 6],
                               [1e12 + 4000, 7],
                               [1e12 + 4500, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 6000, 10],
                               [1e12 + 7000, 11],
                               [1e12 + 8000, 12],
                               [1e12 + 8900, 13],
                               [1e12 + 9900, 14],
                               [1e12 + 11000, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_2['tsuid'], name="metric", value="cut_dest1")
            IkatsApi.md.create(tsuid=result_ts_2['tsuid'], name="my_group", value="1")
            IkatsApi.md.create(tsuid=result_ts_2['tsuid'], name="qual_ref_period", value="1000")

            # TS to cut having single point cut
            result_ts_3 = IkatsApi.ts.create(
                fid="G1_to_cut_2",
                data=np.array([[1e12 + 4800, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 12500, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_3['tsuid'], name="metric", value="cut_dest2")
            IkatsApi.md.create(tsuid=result_ts_3['tsuid'], name="my_group", value="1")
            IkatsApi.md.create(tsuid=result_ts_3['tsuid'], name="qual_ref_period", value="200")

            # TS to cut having no point as result
            result_ts_4 = IkatsApi.ts.create(
                fid="G1_to_cut_3",
                data=np.array([[1e12 + 8100, 8],
                               [1e12 + 8200, 9],
                               [1e12 + 8300, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_4['tsuid'], name="metric", value="cut_dest3")
            IkatsApi.md.create(tsuid=result_ts_4['tsuid'], name="my_group", value="1")
            IkatsApi.md.create(tsuid=result_ts_4['tsuid'], name="qual_ref_period", value="100")

            # TS to cut having all points in result
            result_ts_5 = IkatsApi.ts.create(
                fid="G1_to_cut_4",
                data=np.array([[1e12 + 1100, 1],
                               [1e12 + 2000, 2],
                               [1e12 + 2900, 3]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_5['tsuid'], name="metric", value="cut_dest4")
            IkatsApi.md.create(tsuid=result_ts_5['tsuid'], name="my_group", value="1")
            IkatsApi.md.create(tsuid=result_ts_5['tsuid'], name="qual_ref_period", value="900")

            # 2nd group is the exact opposite of the first group (source cut values are multiplied by -1)
            result_ts_6 = IkatsApi.ts.create(
                fid="G2_source_cut",
                data=np.array([[1e12, -1],
                               [1e12 + 1000, -1],
                               [1e12 + 2000, -2],
                               [1e12 + 3000, -3],
                               [1e12 + 4000, 3.0],
                               [1e12 + 5000, -3],
                               [1e12 + 6000, -4],
                               [1e12 + 7000, -2],
                               [1e12 + 8000, 2.0],
                               [1e12 + 9000, 10],
                               [1e12 + 10000, -10.0],
                               [1e12 + 11000, -10.0],
                               [1e12 + 12000, -10.0],
                               [1e12 + 13000, 10.0]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_6['tsuid'], name="metric", value="cut_metric")
            IkatsApi.md.create(tsuid=result_ts_6['tsuid'], name="my_group", value="2")
            IkatsApi.md.create(tsuid=result_ts_6['tsuid'], name="qual_ref_period", value="1000")

            result_ts_7 = IkatsApi.ts.create(
                fid="G2_to_cut_1",
                data=np.array([[1e12, 1],
                               [1e12 + 500, 2],
                               [1e12 + 1000, 3],
                               [1e12 + 2000, 4],
                               [1e12 + 3000, 5],
                               [1e12 + 3500, 6],
                               [1e12 + 4000, 7],
                               [1e12 + 4500, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 6000, 10],
                               [1e12 + 7000, 11],
                               [1e12 + 8000, 12],
                               [1e12 + 8900, 13],
                               [1e12 + 9900, 14],
                               [1e12 + 11000, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_7['tsuid'], name="metric", value="cut_dest1")
            IkatsApi.md.create(tsuid=result_ts_7['tsuid'], name="my_group", value="2")
            IkatsApi.md.create(tsuid=result_ts_7['tsuid'], name="qual_ref_period", value="1000")

            result_ts_8 = IkatsApi.ts.create(
                fid="G2_to_cut_2",
                data=np.array([[1e12 + 4800, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 12500, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_8['tsuid'], name="metric", value="cut_dest2")
            IkatsApi.md.create(tsuid=result_ts_8['tsuid'], name="my_group", value="2")
            IkatsApi.md.create(tsuid=result_ts_8['tsuid'], name="qual_ref_period", value="1000")

            result_ts_9 = IkatsApi.ts.create(
                fid="G2_to_cut_3",
                data=np.array([[1e12 + 8100, 8],
                               [1e12 + 8200, 9],
                               [1e12 + 8300, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_9['tsuid'], name="metric", value="cut_dest3")
            IkatsApi.md.create(tsuid=result_ts_9['tsuid'], name="my_group", value="2")
            IkatsApi.md.create(tsuid=result_ts_9['tsuid'], name="qual_ref_period", value="1000")

            result_ts_10 = IkatsApi.ts.create(
                fid="G2_to_cut_4",
                data=np.array([[1e12 + 1100, 1],
                               [1e12 + 2000, 2],
                               [1e12 + 2900, 3]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_10['tsuid'], name="metric", value="cut_dest4")
            IkatsApi.md.create(tsuid=result_ts_10['tsuid'], name="my_group", value="2")
            IkatsApi.md.create(tsuid=result_ts_10['tsuid'], name="qual_ref_period", value="900")

            # Creation of dataset
            ds_name = "TEST_cut_ds_157215_KO"
            IkatsApi.ds.delete(ds_name=ds_name, deep=True)
            tsuid_list = [
                result_ts_1['tsuid'],
                result_ts_2['tsuid'],
                result_ts_3['tsuid'],
                result_ts_4['tsuid'],
                result_ts_5['tsuid'],
                result_ts_6['tsuid'],
                result_ts_7['tsuid'],
                result_ts_8['tsuid'],
                result_ts_9['tsuid'],
                result_ts_10['tsuid']
            ]
            IkatsApi.ds.create(ds_name=ds_name,
                               description="Test for cut ds from metric",
                               tsuid_list=tsuid_list)

            # Run the test
            fid_pattern = "%(fid)s__%(M)s__cut"
            metric = "cut_metric"

            with self.assertRaises(ValueError):
                # Dataset is unknown
                cut_ds_from_metric(ds_name="Unknown_dataset",
                                   metric=metric,
                                   criteria="M>0",
                                   fid_pattern=fid_pattern,
                                   group_by="my_group")

            with self.assertRaises(ValueError):
                # Metric is unknown
                cut_ds_from_metric(ds_name=ds_name,
                                   metric="unknown_metric",
                                   criteria="M>0",
                                   fid_pattern=fid_pattern,
                                   group_by="my_group")

            with self.assertRaises(ValueError):
                # Group by is unknown
                cut_ds_from_metric(ds_name=ds_name,
                                   metric=metric,
                                   criteria="M>0",
                                   fid_pattern=fid_pattern,
                                   group_by="unknown_group")

            with self.assertRaises(ValueError):
                # Group by is wrong
                cut_ds_from_metric(ds_name=ds_name,
                                   metric=metric,
                                   criteria="M>0",
                                   fid_pattern=fid_pattern,
                                   group_by=" ")

            with self.assertRaises(KeyError):
                # fid_pattern is wrong
                cut_ds_from_metric(ds_name=ds_name,
                                   metric=metric,
                                   criteria="M>0",
                                   fid_pattern="%(wrong_replacement)s",
                                   group_by="my_group")

        finally:

            # Clean up database

            for fid in ['G1_source_cut', 'G1_to_cut_1', 'G1_to_cut_2', 'G1_to_cut_3', 'G1_to_cut_4',
                        'G2_source_cut', 'G2_to_cut_1', 'G2_to_cut_2', 'G2_to_cut_3', 'G2_to_cut_4']:
                # Delete created TS
                try:
                    tsuid = IkatsApi.fid.tsuid(fid_pattern % {'M': metric, 'fid': fid})
                    IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
                except ValueError:
                    # No TS to delete
                    pass

            try:
                # Delete test dataset
                IkatsApi.ds.delete(ds_name=ds_name, deep=True)
            except Exception:
                # Don't try to delete Dataset that is not yet created (crashed before creating it)
                pass

    def test_ds_cut_no_group_match(self):
        """
        Cut from metric Robustness tests
        """

        try:
            # Create Test data
            # Source used to cut
            result_ts_1 = IkatsApi.ts.create(
                fid="G1_source_cut",
                data=np.array([[1e12, -1],
                               [1e12 + 1000, 1],
                               [1e12 + 2000, 2],
                               [1e12 + 3000, 3],
                               [1e12 + 4000, -3.0],
                               [1e12 + 5000, 3],
                               [1e12 + 6000, 4],
                               [1e12 + 7000, 2],
                               [1e12 + 8000, -2.0],
                               [1e12 + 9000, -10],
                               [1e12 + 10000, 10.0],
                               [1e12 + 11000, 10.0],
                               [1e12 + 12000, 10.0],
                               [1e12 + 13000, -10.0]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_1['tsuid'], name="metric", value="cut_metric")
            IkatsApi.md.create(tsuid=result_ts_1['tsuid'], name="my_group", value="1")
            IkatsApi.md.create(tsuid=result_ts_1['tsuid'], name="qual_ref_period", value="1000")

            # TS to cut having several points cut (nominal)
            result_ts_2 = IkatsApi.ts.create(
                fid="G1_to_cut_1",
                data=np.array([[1e12, 1],
                               [1e12 + 500, 2],
                               [1e12 + 1000, 3],
                               [1e12 + 2000, 4],
                               [1e12 + 3000, 5],
                               [1e12 + 3500, 6],
                               [1e12 + 4000, 7],
                               [1e12 + 4500, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 6000, 10],
                               [1e12 + 7000, 11],
                               [1e12 + 8000, 12],
                               [1e12 + 8900, 13],
                               [1e12 + 9900, 14],
                               [1e12 + 11000, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_2['tsuid'], name="metric", value="cut_dest1")
            IkatsApi.md.create(tsuid=result_ts_2['tsuid'], name="my_group", value="1")
            IkatsApi.md.create(tsuid=result_ts_2['tsuid'], name="qual_ref_period", value="1000")

            # TS to cut having single point cut
            result_ts_3 = IkatsApi.ts.create(
                fid="G1_to_cut_2",
                data=np.array([[1e12 + 4800, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 12500, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_3['tsuid'], name="metric", value="cut_dest2")
            IkatsApi.md.create(tsuid=result_ts_3['tsuid'], name="my_group", value="1")
            IkatsApi.md.create(tsuid=result_ts_3['tsuid'], name="qual_ref_period", value="200")

            # TS to cut having no point as result
            result_ts_4 = IkatsApi.ts.create(
                fid="G1_to_cut_3",
                data=np.array([[1e12 + 8100, 8],
                               [1e12 + 8200, 9],
                               [1e12 + 8300, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_4['tsuid'], name="metric", value="cut_dest3")
            IkatsApi.md.create(tsuid=result_ts_4['tsuid'], name="my_group", value="1")
            IkatsApi.md.create(tsuid=result_ts_4['tsuid'], name="qual_ref_period", value="100")

            # TS to cut having all points in result
            result_ts_5 = IkatsApi.ts.create(
                fid="G1_to_cut_4",
                data=np.array([[1e12 + 1100, 1],
                               [1e12 + 2000, 2],
                               [1e12 + 2900, 3]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_5['tsuid'], name="metric", value="cut_dest4")
            IkatsApi.md.create(tsuid=result_ts_5['tsuid'], name="my_group", value="1")
            IkatsApi.md.create(tsuid=result_ts_5['tsuid'], name="qual_ref_period", value="900")

            # 2nd group is the exact opposite of the first group (source cut values are multiplied by -1)
            result_ts_6 = IkatsApi.ts.create(
                fid="G2_source_cut",
                data=np.array([[1e12, -1],
                               [1e12 + 1000, -1],
                               [1e12 + 2000, -2],
                               [1e12 + 3000, -3],
                               [1e12 + 4000, 3.0],
                               [1e12 + 5000, -3],
                               [1e12 + 6000, -4],
                               [1e12 + 7000, -2],
                               [1e12 + 8000, 2.0],
                               [1e12 + 9000, 10],
                               [1e12 + 10000, -10.0],
                               [1e12 + 11000, -10.0],
                               [1e12 + 12000, -10.0],
                               [1e12 + 13000, 10.0]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_6['tsuid'], name="metric", value="cut_metric")
            IkatsApi.md.create(tsuid=result_ts_6['tsuid'], name="my_group", value="2")
            IkatsApi.md.create(tsuid=result_ts_6['tsuid'], name="qual_ref_period", value="1000")

            result_ts_7 = IkatsApi.ts.create(
                fid="G2_to_cut_1",
                data=np.array([[1e12, 1],
                               [1e12 + 500, 2],
                               [1e12 + 1000, 3],
                               [1e12 + 2000, 4],
                               [1e12 + 3000, 5],
                               [1e12 + 3500, 6],
                               [1e12 + 4000, 7],
                               [1e12 + 4500, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 6000, 10],
                               [1e12 + 7000, 11],
                               [1e12 + 8000, 12],
                               [1e12 + 8900, 13],
                               [1e12 + 9900, 14],
                               [1e12 + 11000, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_7['tsuid'], name="metric", value="cut_dest1")
            IkatsApi.md.create(tsuid=result_ts_7['tsuid'], name="my_group", value="2")
            IkatsApi.md.create(tsuid=result_ts_7['tsuid'], name="qual_ref_period", value="1000")

            result_ts_8 = IkatsApi.ts.create(
                fid="G2_to_cut_2",
                data=np.array([[1e12 + 4800, 8],
                               [1e12 + 5000, 9],
                               [1e12 + 12500, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_8['tsuid'], name="metric", value="cut_dest2")
            IkatsApi.md.create(tsuid=result_ts_8['tsuid'], name="my_group", value="2")
            IkatsApi.md.create(tsuid=result_ts_8['tsuid'], name="qual_ref_period", value="1000")

            result_ts_9 = IkatsApi.ts.create(
                fid="G2_to_cut_3",
                data=np.array([[1e12 + 8100, 8],
                               [1e12 + 8200, 9],
                               [1e12 + 8300, 15]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_9['tsuid'], name="metric", value="cut_dest3")
            IkatsApi.md.create(tsuid=result_ts_9['tsuid'], name="my_group", value="2")
            IkatsApi.md.create(tsuid=result_ts_9['tsuid'], name="qual_ref_period", value="1000")

            result_ts_10 = IkatsApi.ts.create(
                fid="G2_to_cut_4",
                data=np.array([[1e12 + 1100, 1],
                               [1e12 + 2000, 2],
                               [1e12 + 2900, 3]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_10['tsuid'], name="metric", value="cut_dest4")
            IkatsApi.md.create(tsuid=result_ts_10['tsuid'], name="my_group", value="2")
            IkatsApi.md.create(tsuid=result_ts_10['tsuid'], name="qual_ref_period", value="900")

            # No reference for Group 3
            result_ts_11 = IkatsApi.ts.create(
                fid="G3_to_cut_1",
                data=np.array([[1e12 + 1100, 1],
                               [1e12 + 2000, 2],
                               [1e12 + 2900, 3]]),
                generate_metadata=True)
            IkatsApi.md.create(tsuid=result_ts_10['tsuid'], name="metric", value="to_cut_1")
            IkatsApi.md.create(tsuid=result_ts_10['tsuid'], name="my_group", value="3")
            IkatsApi.md.create(tsuid=result_ts_10['tsuid'], name="qual_ref_period", value="900")

            # Creation of dataset
            ds_name = "TEST_cut_ds_157215_KO"
            IkatsApi.ds.delete(ds_name=ds_name, deep=True)
            tsuid_list = [
                result_ts_1['tsuid'],
                result_ts_2['tsuid'],
                result_ts_3['tsuid'],
                result_ts_4['tsuid'],
                result_ts_5['tsuid'],
                result_ts_6['tsuid'],
                result_ts_7['tsuid'],
                result_ts_8['tsuid'],
                result_ts_9['tsuid'],
                result_ts_10['tsuid'],
                result_ts_11['tsuid']
            ]
            IkatsApi.ds.create(ds_name=ds_name,
                               description="Test for cut ds from metric",
                               tsuid_list=tsuid_list)

            # Run the test
            fid_pattern = "%(fid)s__%(M)s__cut"
            metric = "cut_metric"

            with self.assertRaises(ValueError):
                # Group 3 is not found
                cut_ds_from_metric(ds_name=ds_name,
                                   metric=metric,
                                   criteria="M>0",
                                   fid_pattern=fid_pattern,
                                   group_by="my_group")

        finally:

            # Clean up database

            for fid in ['G1_source_cut', 'G1_to_cut_1', 'G1_to_cut_2', 'G1_to_cut_3', 'G1_to_cut_4',
                        'G2_source_cut', 'G2_to_cut_1', 'G2_to_cut_2', 'G2_to_cut_3', 'G2_to_cut_4',
                        'G3_to_cut_1']:
                # Delete created TS
                try:
                    tsuid = IkatsApi.fid.tsuid(fid_pattern % {'M': metric, 'fid': fid})
                    IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
                except ValueError:
                    # No TS to delete
                    pass
            try:
                # Delete test dataset
                IkatsApi.ds.delete(ds_name=ds_name, deep=True)
            except Exception:
                # Don't try to delete Dataset that is not yet created (crashed before creating it)
                pass
