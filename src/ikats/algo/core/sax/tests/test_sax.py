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

from ikats.algo.core.sax.sax import run_sax_from_tsuid, run_sax_from_ds, run_sax_from_ts_list, LOGGER
from ikats.core.resource.api import IkatsApi
from ikats.core.resource.client import TemporalDataMgr

TDM = TemporalDataMgr()

# Log format
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER.setFormatter(FORMATTER)

LOGGER.addHandler(STREAM_HANDLER)


def gen_ts(ts_id):
    """
    Generate a TS in database used for test bench where id is defined

    :param ts_id: Identifier of the TS to generate (see content below for the structure)
    :type ts_id: int

    :return: the TSUID and funcId
    :rtype: dict
    """

    # Build TS identifier
    fid = "UNIT_TEST_Sax_%s" % ts_id

    if ts_id == 1:
        ts_content = [
            [1e12 + 0, 1.0],
            [1e12 + 1000, 2.0],
            [1e12 + 2000, 3.0],
            [1e12 + 3000, 4.0],
            [1e12 + 4000, 5.0],
            [1e12 + 5000, 6.0],
            [1e12 + 6000, 7.0],
            [1e12 + 7000, 8.0],
            [1e12 + 8000, 9.0],
            [1e12 + 9000, 10.0]
        ]
    elif ts_id == 2:
        ts_content = [
            [1e12 + 0, 5.0],
            [1e12 + 1000, 6.0],
            [1e12 + 2000, 8.0],
            [1e12 + 3000, -15.0],
            [1e12 + 4000, 2.0],
            [1e12 + 5000, 6.0],
            [1e12 + 6000, 3.0],
            [1e12 + 7000, 2.0],
            [1e12 + 8000, 42.0],
            [1e12 + 9000, 8.0]
        ]
    elif ts_id == 3:
        ts_content = [
            [1e12 + 500, 5.0],
            [1e12 + 1000, 6.0],
            [1e12 + 2000, 8.0],
            [1e12 + 3000, 7.0],
            [1e12 + 4000, 9.0],
            [1e12 + 7000, 9.0],
            [1e12 + 9000, 9.0],
            [1e12 + 11000, 9.0],
            [1e12 + 13000, 9.0],
            [1e12 + 14500, 10.0]
        ]
    else:
        raise NotImplementedError
    try:
        tsuid = IkatsApi.fid.tsuid(fid=fid)
        IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
    except ValueError:
        # No TS to delete
        pass

    # Create the timeseries
    result = IkatsApi.ts.create(fid=fid, data=np.array(ts_content))
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_ref_period", value=1000, force_update=True)
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_nb_points", value=len(ts_content), force_update=True)
    if not result['status']:
        raise SystemError("Error while creating TS %s" % ts_id)

    return {"tsuid": result['tsuid'], "funcId": fid}


# noinspection PyTypeChecker
class TestSAX(TestCase):
    """
    Test of SAX algorithm
    """

    def test_sax(self):
        """
        Nominal cases of building SAX from a TSUID
        """

        ts_info = gen_ts(1)

        try:

            # Simple test
            results = run_sax_from_tsuid(tdm=TDM, tsuid=ts_info['tsuid'], word_size=5, alphabet_size=5)
            self.assertEqual(results['sax_string'], 'abcde')

            # Simple Test but with double length to see the letter duplication
            results = run_sax_from_tsuid(tdm=TDM, tsuid=ts_info['tsuid'], word_size=10, alphabet_size=5)
            self.assertEqual(results['sax_string'], 'aabbccddee')

            # See another full length (TSUID has length=10)
            results = run_sax_from_tsuid(tdm=TDM, tsuid=ts_info['tsuid'], word_size=10, alphabet_size=10)
            self.assertEqual(results['sax_string'], 'abcdefghij')

            # One out of 2 elements
            results = run_sax_from_tsuid(tdm=TDM, tsuid=ts_info['tsuid'], word_size=5, alphabet_size=10)
            self.assertEqual(results['sax_string'], 'acehj')

        finally:
            # Clean up database
            for ts_item in [ts_info]:
                # Delete created TS
                IkatsApi.ts.delete(tsuid=ts_item['tsuid'], no_exception=True)

    def test_sax_from_ts_list(self):
        """
        Nominal cases of building SAX from a TS list without Spark
        """
        ts_list = [gen_ts(1), gen_ts(2), gen_ts(3)]
        tsuid_list = [x['tsuid'] for x in ts_list]
        word_size = 5

        try:

            # Run a calculation to see all TS have their SAX word computed
            results_1 = run_sax_from_ts_list(tdm=TDM, ts_list=tsuid_list, alphabet_size=10, word_size=word_size)
            self.assertEqual(len(results_1), len(ts_list))
            for ts in results_1:
                self.assertEqual(len(results_1[ts]['sax_string']), word_size)

            # Force the usage of local mode (instead of spark mode) to see the results are the same
            results_2 = run_sax_from_ts_list(tdm=TDM, ts_list=tsuid_list, alphabet_size=10, word_size=word_size,
                                             activate_spark=False)

            for ts in results_2:
                self.assertEqual(len(results_2[ts]['sax_string']), len(results_1[ts]['sax_string']))

        finally:
            # Clean up database
            for ts_item in ts_list:
                # Delete created TS
                IkatsApi.ts.delete(tsuid=ts_item['tsuid'], no_exception=True)

    def test_sax_from_ds(self):
        """
        Nominal cases of building SAX from a valid dataset with Spark
        """
        word_size = 5
        ds_name = 'Portfolio'
        len_ds = 13

        # Run a calculation to see all TS of the dataset have their SAX word computed
        results_1 = run_sax_from_ds(tdm=TDM, ds_name=ds_name, alphabet_size=10, word_size=word_size,
                                    activate_spark=False)
        self.assertEqual(len(results_1), len_ds)
        for ts in results_1:
            self.assertEqual(len(results_1[ts]['sax_string']), word_size)

        # Run the same calculation with spark to prove the spark usage produces the same result
        results_2 = run_sax_from_ds(tdm=TDM, ds_name=ds_name, alphabet_size=10, word_size=word_size,
                                    activate_spark=True)
        for ts in results_2:
            self.assertEqual(len(results_2[ts]['sax_string']), len(results_1[ts]['sax_string']))

    def test_sax_robustness(self):
        """
        Robustness cases of building SAX from a TSUID
        (*from_ts_list and *from_ds are just encapsulation of this method so assertions are tested at lower level)
        """

        ts_list = [gen_ts(1), gen_ts(2), gen_ts(3)]

        try:

            # Alphabet_size too long
            with self.assertRaises(ValueError):
                run_sax_from_tsuid(tdm=TDM, tsuid=ts_list[0]['tsuid'], word_size=1, alphabet_size=27)

            # Invalid alphabet_size (positive expected)
            with self.assertRaises(ValueError):
                run_sax_from_tsuid(tdm=TDM, tsuid=ts_list[0]['tsuid'], word_size=1, alphabet_size=-1)

            # Invalid alphabet_size (int expected)
            with self.assertRaises(ValueError):
                run_sax_from_tsuid(tdm=TDM, tsuid=ts_list[0]['tsuid'], word_size=1, alphabet_size='a')

            # Invalid TSUID
            with self.assertRaises(TypeError):
                run_sax_from_tsuid(tdm=TDM, tsuid=42, word_size=1, alphabet_size=3)

            # Invalid word_size (int expected)
            with self.assertRaises(ValueError):
                run_sax_from_tsuid(tdm=TDM, tsuid=ts_list[0]['tsuid'], word_size='a', alphabet_size=5)

            # Invalid word_size (positive expected)
            with self.assertRaises(ValueError):
                run_sax_from_tsuid(tdm=TDM, tsuid=ts_list[0]['tsuid'], word_size=-1, alphabet_size=5)

        finally:
            # Clean up database
            for ts_item in ts_list:
                # Delete created TS
                IkatsApi.ts.delete(tsuid=ts_item['tsuid'], no_exception=True)
