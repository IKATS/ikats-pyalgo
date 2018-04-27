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

from ikats.algo.core.resampling import resampling_ts
from ikats.core.resource.api import IkatsApi
from ikats.core.resource.client.temporal_data_mgr import DTYPE

LOGGER = logging.getLogger("ikats.algo.core.resampling")
# Log format
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)


class TestResamp(unittest.TestCase):
    """
    Test of the resampling operator
    """

    def test_downsampl_via_resampling(self):
        """
        case : NOMINAL
        """
        fid = 'Timeseries1_For_Resampling_Unit_Testing'
        tsuid = IkatsApi.ts.create(fid=fid,
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
                                       [1449759352000, 35.0]]))['tsuid']
        IkatsApi.md.create(tsuid=tsuid, name='qual_ref_period', value=1000, data_type=DTYPE.number)

        expected_result = np.array([
            [1449759331000, 3.0],
            [1449759332500, 8.0],
            # [1449759334000, None],
            [1449759335500, 25.89],
            [1449759337000, 3.0],
            [1449759338500, 21.2],
            [1449759340000, 18],
            # [1449759341500, None],
            [1449759343000, 7.5],
            # [1449759344500, None],
            # [1449759346000, None],
            # [1449759347500, None],
            # [1449759349000, None],
            # [1449759350500, None],
            [1449759352000, 35.0]
        ], dtype=object)

        resampled_tsuid = None

        try:
            tsuid = IkatsApi.fid.tsuid('Timeseries1_For_Resampling_Unit_Testing_resampled_to_1500ms_BEG_MIN')
            IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
        except ValueError:
            pass

        try:
            list_result = resampling_ts(ts_list=[{"tsuid": tsuid, "funcId": fid}],
                                        resampling_period=1500, aggregation_method="MIN", timestamp_position="BEG",
                                        generate_metadata=False)
            resampled_tsuid = list_result[0]['tsuid']
            obtained_result = IkatsApi.ts.read(tsuid_list=[resampled_tsuid])[0]
            self.assertTrue(np.allclose(
                np.array(expected_result, dtype=np.float64),
                np.array(obtained_result, dtype=np.float64),
                atol=1e-3))
        finally:
            IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
            if resampled_tsuid:
                IkatsApi.ts.delete(tsuid=resampled_tsuid, no_exception=True)
