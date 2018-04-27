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

from ikats.algo.core.cut_y import cut_y, LOGGER
from ikats.core.resource.api import IkatsApi


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


log_to_stdout(LOGGER)


def gen_ts(ts_id):
    """
    Generate a TS in database used for test bench where id is defined

    :param ts_id: Identifier of the TS to generate (see content below for the structure)
    :type ts_id: int

    :return: the TSUID and funcId
    :rtype: dict
    """

    # Build TS identifier
    fid = "UNIT_TEST_cutY_%s" % ts_id

    if ts_id == 1:
        ts_content = [
            [1e12, 5.0],
            [1e12 + 1000, 6.2],
            [1e12 + 2000, 1.0],
            [1e12 + 3000, 8.0],
            [1e12 + 4000, -15.0],
            [1e12 + 5000, 2.0],
            [1e12 + 6000, 6.0],
            [1e12 + 7000, 3.0],
            [1e12 + 8000, 2.0],
            [1e12 + 9000, 42.0],
            [1e12 + 10000, 8.0],
            [1e12 + 11000, 8.0],
            [1e12 + 12000, 8.0],
            [1e12 + 13000, 8.0]
        ]
    elif ts_id == 2:
        ts_content = [
            [1e12, -1.0],
            [1e12 + 1000, -15.0],
            [1e12 + 2000, 100.0],
            [1e12 + 3000, -8.0],
            [1e12 + 4000, -1.0],
            [1e12 + 5000, 2.0],
            [1e12 + 6000, 8.0],
            [1e12 + 7000, 2.6],
            [1e12 + 8000, 8.009],
            [1e12 + 9000, 7.0],
            [1e12 + 10000, 0.0],
            [1e12 + 11000, 0.0],
            [1e12 + 12000, 12.0],
            [1e12 + 13000, 3.0]
        ]
    else:
        raise NotImplementedError

    # Remove former TS if exists
    try:
        tsuid_to_remove = IkatsApi.fid.tsuid(fid=fid)
        IkatsApi.ts.delete(tsuid=tsuid_to_remove, no_exception=True)
    except ValueError:
        # No TS to remove
        pass

    # Create the timeseries
    result = IkatsApi.ts.create(fid=fid, data=np.array(ts_content))
    IkatsApi.md.create(tsuid=result['tsuid'], name="metric", value=fid, force_update=True)
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_ref_period", value=1000, force_update=True)
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_nb_points", value=len(ts_content), force_update=True)
    if not result['status']:
        raise SystemError("Error while creating TS %s" % ts_id)

    return {"tsuid": result['tsuid'], "funcId": fid}


class TestCutY(TestCase):
    """
    Test of Cut-Y algorithm
    """

    @staticmethod
    def _delete_if_exist(ts=None, fid=None):
        """
        Try to delete the TSUID if it exists.
        If a FID is provided, try to get the corresponding TSUID before deleting it

        :param ts: TSUID/FuncId duet to delete
        :param fid: Functional identifier to delete

        :type ts: dict or None
        :type fid:  str or None
        """

        if ts is None and fid is None:
            # Input parameters are mutually exclusive
            raise RuntimeError("Can't have both TS and FID set to None")

        if ts is None and fid is not None:
            # Get the TSUID from the functional Identifier
            try:
                tsuid = IkatsApi.fid.tsuid(fid=fid)
                ts = {'tsuid': tsuid, 'funcId': fid}
            except ValueError:
                return

        IkatsApi.ts.delete(tsuid=ts['tsuid'], no_exception=True)

    def _check_cut_y_results(self, expect_matching, expect_not_matching,
                             obtained_matching, obtained_not_matching):
        """
        Check the results of the algorithm by comparing expected and obtained results for matching and not-matching data

        :param expect_matching: list of expected data points matching the criterion in the expected output order
        :param expect_not_matching: list of expected data points not-matching the criterion in the expected output order
        :param obtained_matching: list of obtained ts list matching the criteria
        :param obtained_not_matching: list of obtained ts list non-matching the criteria

        :type expect_matching: list
        :type expect_not_matching: list
        :type obtained_matching: list
        :type obtained_not_matching: list
        """

        for i, item in enumerate(expect_matching):
            self._compare_results(expected_data_points=expect_matching[i],
                                  obtained_tsuid=obtained_matching[i]['tsuid'],
                                  msg="Matching TS %s" % (i + 1))
        self.assertEqual(len(expect_matching),
                         len(obtained_matching))
        for i, item in enumerate(expect_not_matching):
            self._compare_results(expected_data_points=item,
                                  obtained_tsuid=obtained_not_matching[i]['tsuid'],
                                  msg="Not matching TS %s" % (i + 1))
        self.assertEqual(len(expect_not_matching),
                         len(obtained_not_matching))

    def _compare_results(self, expected_data_points, obtained_tsuid, msg=None):
        """
        Function used to compare the expected timeseries content with the content of the obtained tsuid reference.
        The comparison is about the value of every point.
        The value comparison is OK if the absolute difference is lesser than 1e-2

        :param expected_data_points: list of list of date and values
        :param obtained_tsuid: TSUID reference
        :param msg: optional message

        :type expected_data_points: list
        :type obtained_tsuid: str
        :type msg: str or None

        :raises ValueError: if the values differ
        """

        # Get the points corresponding to the TSUID returned by algorithm
        obtained_data_points = IkatsApi.ts.read(tsuid_list=[obtained_tsuid])[0]

        try:
            # Compare content (date and values)
            self.assertTrue(np.allclose(
                np.array(expected_data_points, dtype=np.float64),
                np.array(obtained_data_points, dtype=np.float64),
                atol=1e-2))
        except ValueError:
            if msg is not None:
                print(msg)
            print("expected_data_points (%d points)" % len(expected_data_points))
            print(expected_data_points)
            print("obtained_data_points (%d points)" % len(obtained_data_points))
            print(obtained_data_points)
            raise

    def _cleanup_former_results(self, fid_pattern, input_ts_list):
        """
        Cleanup former existing results (if exist) by building the expected FID of the generated timeseries

        :param fid_pattern: pattern to use to build the FID of generated timeseries
        :param input_ts_list: information about inputs to build the FID

        :type fid_pattern: str
        :type input_ts_list: list
        """

        # For every input timeseries
        for _, item in enumerate(input_ts_list):
            # For both matching and not matching generated result
            for compl in ['', '_compl']:
                replacement_keys = {'fid': item['funcId'], 'metric': item['funcId'], 'compl': compl}
                fid_to_delete = fid_pattern.format(**replacement_keys)
                self._delete_if_exist(fid=fid_to_delete)

    def _cleanup_results(self, input_ts_list, obtained_match, obtained_no_match):
        """
        Cleanup results of the current test after having checked them.

        :param input_ts_list: list of timeseries used as input of the algorithm build especially for this test
        :param obtained_match: list of matching timeseries obtained as result of algorithm
        :param obtained_no_match: list of non-matching timeseries obtained as result of algorithm

        :type input_ts_list: list
        :type obtained_match: list
        :type obtained_no_match: list
        """
        if input_ts_list is not None:
            for item in input_ts_list:
                self._delete_if_exist(ts=item)
        if obtained_match is not None:
            for timeseries in obtained_match:
                self._delete_if_exist(ts=timeseries)
        if obtained_no_match is not None:
            for timeseries in obtained_no_match:
                self._delete_if_exist(ts=timeseries)

    def test_cut_y_nominal(self):
        """
        Compute the nominal cut-Y of 2 timeseries.
        """

        # Matching condition
        condition = "1<Y<=8.009"

        # Expected TS
        expected_ts_list_matching = [[
            [1e12, 5.0],
            [1e12 + 1000, 6.2],
            [1e12 + 3000, 8.0],
            [1e12 + 5000, 2.0],
            [1e12 + 6000, 6.0],
            [1e12 + 7000, 3.0],
            [1e12 + 8000, 2.0],
            [1e12 + 10000, 8.0],
            [1e12 + 11000, 8.0],
            [1e12 + 12000, 8.0],
            [1e12 + 13000, 8.0]
        ], [
            [1e12 + 5000, 2.0],
            [1e12 + 6000, 8.0],
            [1e12 + 7000, 2.6],
            [1e12 + 8000, 8.009],
            [1e12 + 9000, 7.0],
            [1e12 + 13000, 3.0]
        ]]

        expected_ts_list_not_matching = [[
            [1e12 + 2000, 1.0],
            [1e12 + 4000, -15.0],
            [1e12 + 9000, 42.0]
        ], [
            [1e12, -1.0],
            [1e12 + 1000, -15.0],
            [1e12 + 2000, 100.0],
            [1e12 + 3000, -8.0],
            [1e12 + 4000, -1.0],
            [1e12 + 10000, 0.0],
            [1e12 + 11000, 0.0],
            [1e12 + 12000, 12.0]
        ]]

        fid_pattern = "TEST_{fid}_cutY{compl}"

        input_ts_list = None
        obtained_ts_list_matching = None
        obtained_ts_list_not_matching = None

        try:
            # Prepare list of TS
            input_ts_list = [gen_ts(1), gen_ts(2)]

            # Cleanup former results
            self._cleanup_former_results(fid_pattern=fid_pattern, input_ts_list=input_ts_list)

            # Call the algorithm
            obtained_ts_list_matching, obtained_ts_list_not_matching = cut_y(original_ts_list=input_ts_list,
                                                                             criterion=condition,
                                                                             fid_pattern=fid_pattern)

            # Check content
            self._check_cut_y_results(expect_matching=expected_ts_list_matching,
                                      expect_not_matching=expected_ts_list_not_matching,
                                      obtained_matching=obtained_ts_list_matching,
                                      obtained_not_matching=obtained_ts_list_not_matching)

        finally:

            # Cleanup results
            self._cleanup_results(input_ts_list=input_ts_list,
                                  obtained_match=obtained_ts_list_matching,
                                  obtained_no_match=obtained_ts_list_not_matching)

    def test_cut_y_inherit(self):
        """
        Check the created TS inherit from the original one.
        """

        # Matching condition
        condition = "1<Y<=8.009"

        fid_pattern = "TEST_{fid}_cutY{compl}"

        input_ts_list = None
        obtained_ts_list_matching = None
        obtained_ts_list_not_matching = None

        try:
            # Prepare list of TS
            input_ts_list = [gen_ts(1), gen_ts(2)]

            for item in input_ts_list:
                IkatsApi.md.create(item['tsuid'], "MyMetaToKeep", "hello")

            # Cleanup former results
            self._cleanup_former_results(fid_pattern=fid_pattern, input_ts_list=input_ts_list)

            # Call the algorithm
            obtained_ts_list_matching, obtained_ts_list_not_matching = cut_y(original_ts_list=input_ts_list,
                                                                             criterion=condition,
                                                                             fid_pattern=fid_pattern)

            # Check metadata inheritance
            md_to_check = IkatsApi.md.read([x['tsuid'] for x in obtained_ts_list_matching])
            for _, item in enumerate(obtained_ts_list_matching):
                self.assertFalse("MyMetaToKeep" not in md_to_check[item['tsuid']])
            md_to_check = IkatsApi.md.read([x['tsuid'] for x in obtained_ts_list_not_matching])
            for _, item in enumerate(obtained_ts_list_not_matching):
                self.assertFalse("MyMetaToKeep" not in md_to_check[item['tsuid']])

        finally:

            # Cleanup results
            self._cleanup_results(input_ts_list=input_ts_list,
                                  obtained_match=obtained_ts_list_matching,
                                  obtained_no_match=obtained_ts_list_not_matching)

    def test_cut_y_full_match(self):
        """
        Compute the cut-Y on 2 timeseries with full match criteria.
        """

        # Matching condition
        condition = "True"

        # Expected TS
        expected_ts_list_matching = [[
            [1e12, 5.0],
            [1e12 + 1000, 6.2],
            [1e12 + 2000, 1.0],
            [1e12 + 3000, 8.0],
            [1e12 + 4000, -15.0],
            [1e12 + 5000, 2.0],
            [1e12 + 6000, 6.0],
            [1e12 + 7000, 3.0],
            [1e12 + 8000, 2.0],
            [1e12 + 9000, 42.0],
            [1e12 + 10000, 8.0],
            [1e12 + 11000, 8.0],
            [1e12 + 12000, 8.0],
            [1e12 + 13000, 8.0]
        ], [
            [1e12, -1.0],
            [1e12 + 1000, -15.0],
            [1e12 + 2000, 100.0],
            [1e12 + 3000, -8.0],
            [1e12 + 4000, -1.0],
            [1e12 + 5000, 2.0],
            [1e12 + 6000, 8.0],
            [1e12 + 7000, 2.6],
            [1e12 + 8000, 8.009],
            [1e12 + 9000, 7.0],
            [1e12 + 10000, 0.0],
            [1e12 + 11000, 0.0],
            [1e12 + 12000, 12.0],
            [1e12 + 13000, 3.0]
        ]]

        fid_pattern = "TEST_{fid}_cutY{compl}"

        input_ts_list = None
        obtained_ts_list_matching = None
        obtained_ts_list_not_matching = None

        try:
            # Prepare list of TS
            input_ts_list = [gen_ts(1), gen_ts(2)]

            # Cleanup former results
            self._cleanup_former_results(fid_pattern=fid_pattern, input_ts_list=input_ts_list)

            # Call the algorithm
            obtained_ts_list_matching, obtained_ts_list_not_matching = cut_y(original_ts_list=input_ts_list,
                                                                             criterion=condition,
                                                                             fid_pattern=fid_pattern)

            # Check content
            self._check_cut_y_results(expect_matching=expected_ts_list_matching,
                                      expect_not_matching=[],
                                      obtained_matching=obtained_ts_list_matching,
                                      obtained_not_matching=obtained_ts_list_not_matching)

        finally:

            # Cleanup results
            self._cleanup_results(input_ts_list=input_ts_list,
                                  obtained_match=obtained_ts_list_matching,
                                  obtained_no_match=obtained_ts_list_not_matching)

    def test_cut_y_no_match(self):
        """
        Compute the cut-Y on 2 timeseries with no match criteria.
        """

        # Matching condition
        condition = "False"

        # Expected TS
        expected_ts_list_not_matching = [[
            [1e12, 5.0],
            [1e12 + 1000, 6.2],
            [1e12 + 2000, 1.0],
            [1e12 + 3000, 8.0],
            [1e12 + 4000, -15.0],
            [1e12 + 5000, 2.0],
            [1e12 + 6000, 6.0],
            [1e12 + 7000, 3.0],
            [1e12 + 8000, 2.0],
            [1e12 + 9000, 42.0],
            [1e12 + 10000, 8.0],
            [1e12 + 11000, 8.0],
            [1e12 + 12000, 8.0],
            [1e12 + 13000, 8.0]
        ], [
            [1e12, -1.0],
            [1e12 + 1000, -15.0],
            [1e12 + 2000, 100.0],
            [1e12 + 3000, -8.0],
            [1e12 + 4000, -1.0],
            [1e12 + 5000, 2.0],
            [1e12 + 6000, 8.0],
            [1e12 + 7000, 2.6],
            [1e12 + 8000, 8.009],
            [1e12 + 9000, 7.0],
            [1e12 + 10000, 0.0],
            [1e12 + 11000, 0.0],
            [1e12 + 12000, 12.0],
            [1e12 + 13000, 3.0]
        ]]

        fid_pattern = "TEST_{fid}_cutY{compl}"

        input_ts_list = None
        obtained_ts_list_matching = None
        obtained_ts_list_not_matching = None

        try:
            # Prepare list of TS
            input_ts_list = [gen_ts(1), gen_ts(2)]

            # Cleanup former results
            self._cleanup_former_results(fid_pattern=fid_pattern, input_ts_list=input_ts_list)

            # Call the algorithm
            obtained_ts_list_matching, obtained_ts_list_not_matching = cut_y(original_ts_list=input_ts_list,
                                                                             criterion=condition,
                                                                             fid_pattern=fid_pattern)

            # Check content
            self._check_cut_y_results(expect_matching=[],
                                      expect_not_matching=expected_ts_list_not_matching,
                                      obtained_matching=obtained_ts_list_matching,
                                      obtained_not_matching=obtained_ts_list_not_matching)

        finally:

            # Cleanup results
            self._cleanup_results(input_ts_list=input_ts_list,
                                  obtained_match=obtained_ts_list_matching,
                                  obtained_no_match=obtained_ts_list_not_matching)

    def test_cut_y_multi_chunks(self):
        """
        Compute the nominal cut-Y of 2 timeseries with more than one chunk in data.
        """

        # Matching condition
        condition = "1<Y<=8.009"

        # Expected TS
        expected_ts_list_matching = [[
            [1e12, 5.0],
            [1e12 + 1000, 6.2],
            [1e12 + 3000, 8.0],
            [1e12 + 5000, 2.0],
            [1e12 + 6000, 6.0],
            [1e12 + 7000, 3.0],
            [1e12 + 8000, 2.0],
            [1e12 + 10000, 8.0],
            [1e12 + 11000, 8.0],
            [1e12 + 12000, 8.0],
            [1e12 + 13000, 8.0]
        ], [
            [1e12 + 5000, 2.0],
            [1e12 + 6000, 8.0],
            [1e12 + 7000, 2.6],
            [1e12 + 8000, 8.009],
            [1e12 + 9000, 7.0],
            [1e12 + 13000, 3.0]
        ]]

        expected_ts_list_not_matching = [[
            [1e12 + 2000, 1.0],
            [1e12 + 4000, -15.0],
            [1e12 + 9000, 42.0]
        ], [
            [1e12, -1.0],
            [1e12 + 1000, -15.0],
            [1e12 + 2000, 100.0],
            [1e12 + 3000, -8.0],
            [1e12 + 4000, -1.0],
            [1e12 + 10000, 0.0],
            [1e12 + 11000, 0.0],
            [1e12 + 12000, 12.0]
        ]]

        fid_pattern = "TEST_{fid}_cutY{compl}"

        input_ts_list = None
        obtained_ts_list_matching = None
        obtained_ts_list_not_matching = None

        try:
            # Prepare list of TS
            input_ts_list = [gen_ts(1), gen_ts(2)]

            # Cleanup former results
            self._cleanup_former_results(fid_pattern=fid_pattern, input_ts_list=input_ts_list)

            # Call the algorithm
            obtained_ts_list_matching, obtained_ts_list_not_matching = cut_y(original_ts_list=input_ts_list,
                                                                             criterion=condition,
                                                                             fid_pattern=fid_pattern,
                                                                             chunk_size=4)

            # Check content
            self._check_cut_y_results(expect_matching=expected_ts_list_matching,
                                      expect_not_matching=expected_ts_list_not_matching,
                                      obtained_matching=obtained_ts_list_matching,
                                      obtained_not_matching=obtained_ts_list_not_matching)

        finally:

            # Cleanup results
            self._cleanup_results(input_ts_list=input_ts_list,
                                  obtained_match=obtained_ts_list_matching,
                                  obtained_no_match=obtained_ts_list_not_matching)

    def test_cut_y_single(self):
        """
        Compute the nominal cut-Y of 1 timeseries.
        """

        # Matching condition
        condition = "1<Y<=8.009"

        # Expected TS
        expected_ts_list_matching = [[
            [1e12, 5.0],
            [1e12 + 1000, 6.2],
            [1e12 + 3000, 8.0],
            [1e12 + 5000, 2.0],
            [1e12 + 6000, 6.0],
            [1e12 + 7000, 3.0],
            [1e12 + 8000, 2.0],
            [1e12 + 10000, 8.0],
            [1e12 + 11000, 8.0],
            [1e12 + 12000, 8.0],
            [1e12 + 13000, 8.0]
        ]]

        expected_ts_list_not_matching = [[
            [1e12 + 2000, 1.0],
            [1e12 + 4000, -15.0],
            [1e12 + 9000, 42.0]
        ]]

        fid_pattern = "TEST_{fid}_cutY{compl}"

        input_ts_list = None
        obtained_ts_list_matching = None
        obtained_ts_list_not_matching = None

        try:
            # Prepare list of TS
            input_ts_list = [gen_ts(1)]

            # Cleanup former results
            self._cleanup_former_results(fid_pattern=fid_pattern, input_ts_list=input_ts_list)

            # Call the algorithm
            obtained_ts_list_matching, obtained_ts_list_not_matching = cut_y(original_ts_list=input_ts_list,
                                                                             criterion=condition,
                                                                             fid_pattern=fid_pattern)

            # Check content
            self._check_cut_y_results(expect_matching=expected_ts_list_matching,
                                      expect_not_matching=expected_ts_list_not_matching,
                                      obtained_matching=obtained_ts_list_matching,
                                      obtained_not_matching=obtained_ts_list_not_matching)

        finally:

            # Cleanup results
            self._cleanup_results(input_ts_list=input_ts_list,
                                  obtained_match=obtained_ts_list_matching,
                                  obtained_no_match=obtained_ts_list_not_matching)

    def test_cut_y_no_ts(self):
        """
        Robustness when no TS are provided (empty TS list).
        """

        # Matching condition
        condition = "True"

        fid_pattern = "TEST_{fid}_cutY{compl}"

        # Prepare list of TS
        input_ts_list = []

        # ValueError indicates the number of items in ts_list shall be >0
        with self.assertRaises(ValueError):
            # Call the algorithm
            cut_y(original_ts_list=input_ts_list,
                  criterion=condition,
                  fid_pattern=fid_pattern)
