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
import os
import logging
import timeit
import unittest
import numpy as np

from ikats.core.resource.api import IkatsApi
from ikats.algo.core.resampling.resampling_computation import resampling_ts, LOGGER
from ikats.core.resource.client.temporal_data_mgr import DTYPE


def generate_random_timeseries(start_date, nb_points, period):
    """
    Generating a randomized periodic timeseries of nb points (values between 0 and 100)
    first timestamp is start_date
    """
    end_date = start_date + (nb_points - 1) * period

    # Generating timestamps (ms)
    timestamps = np.linspace(start_date, end_date, nb_points, dtype=np.int64)

    # Generating values between 0 and 100
    values = np.random.sample(nb_points) * 100

    ts = np.empty(shape=(len(values), 2), dtype=object)
    ts[:, 0] = timestamps
    ts[:, 1] = values

    # Original timeseries
    return ts


def log_to_stdout(logger_to_use):
    """
    Allow to print some loggers to stdout
    :param logger_to_use: the LOGGER object to redirect to stdout
    """

    logger_to_use.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger_to_use.addHandler(stream_handler)


# Prints the logger to display
log_to_stdout(LOGGER)


class TestUpsampling(unittest.TestCase):
    """
    Test of Upsampling computation
    """

    def test_upsampling_lin_interp(self):
        """
        case : LINEAR_INTERPOLATION
        nb of points by chunk set to 4 to check algorithm behavior at chunks limits :
            - with no point in chunk
            - with only one point in chunk
            - with no point at the beginning of chunk
            - with no point at the end of chunk
            - with several points in the chunk
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

        expected_result = np.array([[1449759331000, 3.0],
                                    [1449759331500, 9.0],
                                    [1449759332000, 15.0],
                                    [1449759332500, 11.5],
                                    [1449759333000, 8.0],
                                    [1449759333500, 10.981],
                                    [1449759334000, 13.963],
                                    [1449759334500, 16.944],
                                    [1449759335000, 19.926],
                                    [1449759335500, 22.908],
                                    [1449759336000, 25.889],
                                    [1449759336500, 20.167],
                                    [1449759337000, 14.444],
                                    [1449759337500, 8.722],
                                    [1449759338000, 3.0],
                                    [1449759338500, 12.1],
                                    [1449759339000, 21.2],
                                    [1449759339500, 19.6],
                                    [1449759340000, 18.0],
                                    [1449759340500, 17.5],
                                    [1449759341000, 17.0],
                                    [1449759341500, 16.5],
                                    [1449759342000, 16.0],
                                    [1449759342500, 15.5],
                                    [1449759343000, 15.0],
                                    [1449759343500, 12.0],
                                    [1449759344000, 7.5],
                                    [1449759344500, 9.218],
                                    [1449759345000, 10.937],
                                    [1449759345500, 12.656],
                                    [1449759346000, 14.375],
                                    [1449759346500, 16.093],
                                    [1449759347000, 17.812],
                                    [1449759347500, 19.531],
                                    [1449759348000, 21.25],
                                    [1449759348500, 22.968],
                                    [1449759349000, 24.687],
                                    [1449759349500, 26.406],
                                    [1449759350000, 28.125],
                                    [1449759350500, 29.843],
                                    [1449759351000, 31.562],
                                    [1449759351500, 33.281],
                                    [1449759352000, 35.0]], dtype=object)
        resampled_tsuid = None
        try:
            list_result = resampling_ts(ts_list=[{"tsuid": tsuid, "funcId": fid}],
                                        resampling_period=500,
                                        adding_method='LINEAR_INTERPOLATION',
                                        nb_points_by_chunk=4)
            resampled_tsuid = list_result[0]['tsuid']
            obtained_result = IkatsApi.ts.read(tsuid_list=[resampled_tsuid])[0]
            self.assertTrue(
                np.allclose(
                    np.array(expected_result, dtype=np.float64),
                    np.array(obtained_result, dtype=np.float64),
                    atol=1e-3)
            )
        finally:
            IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
            if resampled_tsuid:
                IkatsApi.ts.delete(tsuid=resampled_tsuid, no_exception=True)

    def test_upsampling_value_before(self):
        """
        case : VALUE_BEFORE
        nb of points by chunk set to 4 to check algorithm behavior at chunks limits :
            - with no point in chunk
            - with only one point in chunk
            - with no point at the beginning of chunk
            - with no point at the end of chunk
            - with several points in the chunk
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

        expected_result = np.array([[1449759331000, 3.0],
                                    [1449759331500, 3.0],
                                    [1449759332000, 15.0],
                                    [1449759332500, 15.0],
                                    [1449759333000, 8.0],
                                    [1449759333500, 8.0],
                                    [1449759334000, 8.0],
                                    [1449759334500, 8.0],
                                    [1449759335000, 8.0],
                                    [1449759335500, 8.0],
                                    [1449759336000, 25.889],
                                    [1449759336500, 25.889],
                                    [1449759337000, 25.889],
                                    [1449759337500, 25.889],
                                    [1449759338000, 3.0],
                                    [1449759338500, 3.0],
                                    [1449759339000, 21.2],
                                    [1449759339500, 21.2],
                                    [1449759340000, 18.0],
                                    [1449759340500, 18.0],
                                    [1449759341000, 18.0],
                                    [1449759341500, 18.0],
                                    [1449759342000, 18.0],
                                    [1449759342500, 18.0],
                                    [1449759343000, 15.0],
                                    [1449759343500, 12.0],
                                    [1449759344000, 7.5],
                                    [1449759344500, 7.5],
                                    [1449759345000, 7.5],
                                    [1449759345500, 7.5],
                                    [1449759346000, 7.5],
                                    [1449759346500, 7.5],
                                    [1449759347000, 7.5],
                                    [1449759347500, 7.5],
                                    [1449759348000, 7.5],
                                    [1449759348500, 7.5],
                                    [1449759349000, 7.5],
                                    [1449759349500, 7.5],
                                    [1449759350000, 7.5],
                                    [1449759350500, 7.5],
                                    [1449759351000, 7.5],
                                    [1449759351500, 7.5],
                                    [1449759352000, 35.0]], dtype=object)
        resampled_tsuid = None
        try:
            list_result = resampling_ts(ts_list=[{"tsuid": tsuid, "funcId": fid}],
                                        resampling_period=500,
                                        adding_method='VALUE_BEFORE',
                                        nb_points_by_chunk=4)
            resampled_tsuid = list_result[0]['tsuid']
            obtained_result = IkatsApi.ts.read(tsuid_list=[resampled_tsuid])[0]
            self.assertTrue(
                np.allclose(
                    np.array(expected_result, dtype=np.float64),
                    np.array(obtained_result, dtype=np.float64),
                    atol=1e-3)
            )
        finally:
            IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
            if resampled_tsuid:
                IkatsApi.ts.delete(tsuid=resampled_tsuid, no_exception=True)

    def test_upsampling_value_after(self):
        """
        case : VALUE_AFTER
        nb of points by chunk set to 4 to check algorithm behavior at chunks limits :
            - with no point in chunk
            - with only one point in chunk
            - with no point at the beginning of chunk
            - with no point at the end of chunk
            - with several points in the chunk
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

        expected_result = np.array([[1449759331000, 3.0],
                                    [1449759331500, 15.0],
                                    [1449759332000, 15.0],
                                    [1449759332500, 8.0],
                                    [1449759333000, 8.0],
                                    [1449759333500, 25.889],
                                    [1449759334000, 25.889],
                                    [1449759334500, 25.889],
                                    [1449759335000, 25.889],
                                    [1449759335500, 25.889],
                                    [1449759336000, 25.889],
                                    [1449759336500, 3.0],
                                    [1449759337000, 3.0],
                                    [1449759337500, 3.0],
                                    [1449759338000, 3.0],
                                    [1449759338500, 21.2],
                                    [1449759339000, 21.2],
                                    [1449759339500, 18.0],
                                    [1449759340000, 18.0],
                                    [1449759340500, 15.0],
                                    [1449759341000, 15.0],
                                    [1449759341500, 15.0],
                                    [1449759342000, 15.0],
                                    [1449759342500, 15.0],
                                    [1449759343000, 15.0],
                                    [1449759343500, 12.0],
                                    [1449759344000, 7.5],
                                    [1449759344500, 35.0],
                                    [1449759345000, 35.0],
                                    [1449759345500, 35.0],
                                    [1449759346000, 35.0],
                                    [1449759346500, 35.0],
                                    [1449759347000, 35.0],
                                    [1449759347500, 35.0],
                                    [1449759348000, 35.0],
                                    [1449759348500, 35.0],
                                    [1449759349000, 35.0],
                                    [1449759349500, 35.0],
                                    [1449759350000, 35.0],
                                    [1449759350500, 35.0],
                                    [1449759351000, 35.0],
                                    [1449759351500, 35.0],
                                    [1449759352000, 35.0]], dtype=object)

        resampled_tsuid = None
        try:
            list_result = resampling_ts(ts_list=[{"tsuid": tsuid, "funcId": fid}],
                                        resampling_period=500,
                                        adding_method='VALUE_AFTER',
                                        nb_points_by_chunk=4)
            resampled_tsuid = list_result[0]['tsuid']
            obtained_result = IkatsApi.ts.read(tsuid_list=[resampled_tsuid])[0]
            self.assertTrue(
                np.allclose(
                    np.array(expected_result, dtype=np.float64),
                    np.array(obtained_result, dtype=np.float64),
                    atol=1e-3)
            )
        finally:
            IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
            if resampled_tsuid:
                IkatsApi.ts.delete(tsuid=resampled_tsuid, no_exception=True)

    def test_upsampling_multi_ts(self):
        """
        Multiple timeseries to resample
        """

        # Preparing 1st TS
        fid1 = 'Timeseries1_For_Resampling_Unit_Testing'
        tsuid1 = IkatsApi.ts.create(fid=fid1,
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
        IkatsApi.md.create(tsuid=tsuid1, name='qual_ref_period', value=1000, data_type=DTYPE.number)

        # Preparing 2nd TS
        fid2 = 'Timeseries2_For_Resampling_Unit_Testing'
        tsuid2 = IkatsApi.ts.create(fid=fid2,
                                    data=np.array([
                                        [1449759331000, -500.0],
                                        [1449759331800, 500.0],
                                        [1449759333042, 1501.5]]))['tsuid']
        IkatsApi.md.create(tsuid=tsuid2, name='qual_ref_period', value=800, data_type=DTYPE.number)

        expected_result1 = np.array([[1449759331000, 3.0],
                                     [1449759331500, 9.0],
                                     [1449759332000, 15.0],
                                     [1449759332500, 11.5],
                                     [1449759333000, 8.0],
                                     [1449759333500, 10.981],
                                     [1449759334000, 13.963],
                                     [1449759334500, 16.944],
                                     [1449759335000, 19.926],
                                     [1449759335500, 22.908],
                                     [1449759336000, 25.889],
                                     [1449759336500, 20.167],
                                     [1449759337000, 14.444],
                                     [1449759337500, 8.722],
                                     [1449759338000, 3.0],
                                     [1449759338500, 12.1],
                                     [1449759339000, 21.2],
                                     [1449759339500, 19.6],
                                     [1449759340000, 18.0],
                                     [1449759340500, 17.5],
                                     [1449759341000, 17.0],
                                     [1449759341500, 16.5],
                                     [1449759342000, 16.0],
                                     [1449759342500, 15.5],
                                     [1449759343000, 15.0],
                                     [1449759343500, 12.0],
                                     [1449759344000, 7.5],
                                     [1449759344500, 9.218],
                                     [1449759345000, 10.937],
                                     [1449759345500, 12.656],
                                     [1449759346000, 14.375],
                                     [1449759346500, 16.093],
                                     [1449759347000, 17.812],
                                     [1449759347500, 19.531],
                                     [1449759348000, 21.25],
                                     [1449759348500, 22.968],
                                     [1449759349000, 24.687],
                                     [1449759349500, 26.406],
                                     [1449759350000, 28.125],
                                     [1449759350500, 29.843],
                                     [1449759351000, 31.562],
                                     [1449759351500, 33.281],
                                     [1449759352000, 35.0]], dtype=object)

        expected_result2 = np.array([[1449759331000, -500.0],
                                     [1449759331500, 125.0],
                                     [1449759332000, 661.27],
                                     [1449759332500, 1064.45],
                                     [1449759333000, 1467.63]], dtype=object)
        resampled_tsuid1 = None
        resampled_tsuid2 = None
        try:
            list_result = resampling_ts(
                ts_list=[
                    {"tsuid": tsuid1, "funcId": fid1},
                    {"tsuid": tsuid2, "funcId": fid2}
                ],
                resampling_period=500,
                adding_method='LINEAR_INTERPOLATION',
                nb_points_by_chunk=4)

            resampled_tsuid1 = list_result[0]['tsuid']
            resampled_tsuid2 = list_result[1]['tsuid']

            obtained_result1 = IkatsApi.ts.read(tsuid_list=[resampled_tsuid1])[0]
            self.assertTrue(
                np.allclose(
                    np.array(expected_result1, dtype=np.float64),
                    np.array(obtained_result1, dtype=np.float64),
                    atol=1e-3)
            )
            obtained_result2 = IkatsApi.ts.read(tsuid_list=[resampled_tsuid2])[0]
            self.assertTrue(
                np.allclose(
                    np.array(expected_result2, dtype=np.float64),
                    np.array(obtained_result2, dtype=np.float64),
                    atol=1e-3)
            )

        finally:
            IkatsApi.ts.delete(tsuid=tsuid1, no_exception=True)
            IkatsApi.ts.delete(tsuid=tsuid2, no_exception=True)
            if resampled_tsuid1:
                IkatsApi.ts.delete(tsuid=resampled_tsuid1, no_exception=True)
            if resampled_tsuid2:
                IkatsApi.ts.delete(tsuid=resampled_tsuid2, no_exception=True)

    @unittest.skipIf('SKIP_LONG_TEST' in os.environ and os.environ['SKIP_LONG_TEST'],
                     "This test is too long and must not be run every time")
    def test_upsamp_chk_number_of_pts(self):
        """
        Resampling of various sized timeseries for checking number of generated points
        """
        nb_points = [1245, 13566]
        period = 1200
        period_resamp = 900

        for nb_points in nb_points:
            # Generation of a timeseries for test
            ts = generate_random_timeseries(start_date=1000000000000, nb_points=nb_points, period=period)
            fid = 'Timeseries_TEST_' + str(nb_points) + 'pts'
            tsuid = IkatsApi.ts.create(fid=fid, data=ts)['tsuid']

            IkatsApi.md.create(tsuid=tsuid, name='qual_ref_period', value=str(period),
                               data_type=DTYPE.number, force_update=True)
            IkatsApi.md.create(tsuid=tsuid, name='ikats_start_date', value=str(ts[0][0]),
                               data_type=DTYPE.date, force_update=True)
            IkatsApi.md.create(tsuid=tsuid, name='ikats_end_date', value=str(ts[-1][0]),
                               data_type=DTYPE.date, force_update=True)

            LOGGER.info("Randomized timeseries [size = %s points] generated and created in opentsdb", nb_points)

            tsuid_res = []
            try:
                start_time = timeit.default_timer()
                tsuid_res = resampling_ts(ts_list=[{"tsuid": tsuid, "funcId": fid}], resampling_period=period_resamp,
                                          timestamp_position='MID', adding_method="VALUE_BEFORE",
                                          nb_points_by_chunk=int(nb_points / 10))
                end_time = str(timeit.default_timer() - start_time)
                LOGGER.info('Resampling execution time %s : %s', tsuid, end_time)

                nb_points_result = int(IkatsApi.ts.nb_points(tsuid=tsuid_res[0]['tsuid']))
            finally:
                # Removing timeseries
                IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
                for ts in tsuid_res:
                    IkatsApi.ts.delete(tsuid=ts['tsuid'], no_exception=True)

            # Check number of points generated
            LOGGER.info("Initial number of points : %s", nb_points)
            LOGGER.info("Number of point after resampling : %s", str(nb_points_result))
            LOGGER.info("Expected number of points : %s", str(int((nb_points - 1) * (period / period_resamp)) + 1))
            self.assertTrue(nb_points_result == int((nb_points - 1) * (period / period_resamp)) + 1)
