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
import time
from _functools import reduce as freduce
import logging
from math import sqrt
import sys
from unittest.case import TestCase, skipIf

from ikats.core.library.exception import IkatsException, IkatsNotFoundError, IkatsConflictError
from ikats.core.library.spark import ScManager
from ikats.core.resource.api import IkatsApi
from ikats.core.resource.client.temporal_data_mgr import DTYPE
from ikats.core.resource.interface import ResourceLocator

from ikats.algo.core.correlation import loop
from ikats.algo.core.correlation.data import is_triangular_matrix, get_triangular_matrix
import numpy as np

# Context to use for every test
CONTEXT = "FlightIdentifier"

# Variable to use for every test
VARIABLE = "metric"

# PRECISION: Accepted absolute tolerance for floating point check
#   - important: the PRECISION depends on the configured loop.ConfigCorrelationLoop.the_digits_number
#     => see default value used by loop.correlation_ts_list_loop
PRECISION = 1e-3


class TestCorrelationLoop(TestCase):
    """
    Unittest class of the implementation correlation.loop
    """
    log = logging.getLogger("TestCorrelationLoop")

    @staticmethod
    def __save_dataset(dataset_definition,
                       variable_identifier=VARIABLE,
                       context_identifier=CONTEXT,
                       var_type=DTYPE.string,
                       ctx_type=DTYPE.number):
        """
        Saves the unittest dataset and returns the result
        :param dataset_definition: details about the dataset
        :type dataset_definition: dict
        :return: the result: list of TSUIDS
        :rtype: list
        """
        tsuids = []
        for funcid, [meta_dict, ts] in dataset_definition.items():

            tsuid = IkatsApi.ts.create(fid=funcid, data=ts)['tsuid']

            tsuids.append(tsuid)

            IkatsApi.md.create(tsuid=tsuid, name='ikats_start_date', value=str(ts[0][0]),
                               data_type=DTYPE.date, force_update=True)

            IkatsApi.md.create(tsuid=tsuid, name='ikats_end_date', value=str(ts[-1][0]),
                               data_type=DTYPE.date, force_update=True)

            IkatsApi.md.create(tsuid=tsuid, name="qual_nb_points", value=str(len(ts)),
                               data_type=DTYPE.number, force_update=True)

            period = (ts[-1][0] - ts[0][0]) / len(ts)
            IkatsApi.md.create(tsuid=tsuid, name='qual_ref_period', value=str(period),
                               data_type=DTYPE.number, force_update=True)

            if variable_identifier in meta_dict:
                IkatsApi.md.create(tsuid=tsuid, name=variable_identifier, value=meta_dict[variable_identifier],
                                   data_type=var_type, force_update=True)
            if context_identifier in meta_dict:
                IkatsApi.md.create(tsuid=tsuid, name=context_identifier, value=meta_dict[context_identifier],
                                   data_type=ctx_type, force_update=True)
            if len(tsuids) % 20 == 0:
                TestCorrelationLoop.log.info("%s TS created so far", len(tsuids))
        TestCorrelationLoop.log.info("%s TS created", len(tsuids))
        return tsuids

    @staticmethod
    def __remove_dataset(tsuids):
        """
        Remove the timeseries defined
        :param tsuids: list of tsuids
        :type tsuids: list of str
        :raise exception: error deleting some timeseries
        """
        failed = []
        for tsuid in tsuids:
            try:
                IkatsApi.ts.delete(tsuid)
            except (TypeError, IkatsNotFoundError, IkatsConflictError, SystemError):
                failed.append(tsuid)
        if len(failed) > 0:
            raise Exception("Failed to clean the timeseries: TSUIDS={}".format(failed))

    @classmethod
    def setUpClass(cls):
        """
        Sets up the unittest:
          * Prepares the log
          * Prepares the required input data: ts and metadata

        :param cls: The TestCorrelationLoop class
        :type cls: class
        """
        cls.log.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        cls.log.addHandler(stream_handler)

        ScManager.log.setLevel(logging.INFO)
        ScManager.log.addHandler(stream_handler)

    @staticmethod
    def __my_pearson(x_val, y_val):
        """
        Local pearson computation used to check the coefficient value computed by algorithm.

        :param x_val: first set of data to use for computation
        :type x_val: np.array
        :param y_val: second set of data to use for computation
        :type y_val: np.array
        :return: the pearson correlation coefficient
        :rtype: float
        """

        x_mean = (sum(x_val) + 0.0) / len(x_val)
        y_mean = (sum(y_val) + 0.0) / len(y_val)
        x_val_2 = x_val - x_mean
        y_val_2 = y_val - y_mean
        x_var = sum(x_val_2 ** 2) / len(x_val)
        y_var = sum(y_val_2 ** 2) / len(y_val)

        corr = sum(x_val_2 * y_val_2) / sqrt(x_var * y_var) / len(x_val)

        return corr

    def __pearson_from_dataset(self, dataset, context, var_one, var_two):
        """
        Computes the pearson correlation coeff for the test, with the function TestCorrelationLoop.__my_pearson

        :param dataset: the unit test dataset must be a dictionary of
          - keys: funcIDS
          - values: [ <metadata dict>, <Timeseries numpy array> ]
        :type dataset: dict
        :param context: the context value
        :type context: int or str
        :param var_one: the name of the variable one
        :type var_one: str
        :param var_two: the name of the variable two
        :type var_two: str

        :return: the pearson correlation coeff computed by TestCorrelationLoop.__my_pearson
        :rtype: float
        :raise IkatsException: Test preparation error when piece of data is missing
        """
        ts_selection = [value[1] for value in dataset.values() if
                        (value[0].get(CONTEXT, None) == context) and
                        (value[0].get(VARIABLE, None) in [var_one, var_two])]

        if len(ts_selection) != 2:
            msg = "Test preparation error: expects 2 TS defined for cts={} vars=[{},{}]"
            raise IkatsException(msg.format(context, var_one, var_two))

        # Read values of timeseries, ignoring the timestamps
        x = ts_selection[0][:, 1]
        y = ts_selection[1][:, 1]
        the_min = min(x.size, y.size)
        x_val = x[:the_min]
        y_val = y[:the_min]

        return self.__my_pearson(x_val, y_val)

    @staticmethod
    def __retrieve_funcids(tsuids):
        """
        Internal purpose: retrieves funcIds from a list of tsuids
        :param tsuids: TSUIDS list to get functional identifier from
        :type tsuids: list
        :return: list of funcids
        :rtype: list of str
        :raise exception: error occurred
        """
        try:
            return [IkatsApi.ts.fid(tsuid=tsuid) for tsuid in tsuids]
        except Exception:
            raise IkatsException("Failed to convert tsuids={} to funcids".format(tsuids))

    def __get_matrices(self, computed_result, ref_dim):
        """
        Factorized code: read the 4 matrices from their correlation loop result
          - Mean matrix
          - Variance matrix
          - OID matrix: links to process-data
          - Extracted matrix: links replaced by json content from process-data
        :param computed_result: result from correlation loop as returned
               by CorrelationDataset::get_json_friendly_dict()
        :type computed_result: dict
        :param ref_dim: expected size of matrices
        :type ref_dim: int
        :return: the matrices prepared for the tests are:
           - Mean matrix
           - Variance matrix
           - OID matrix
           - Extracted matrix
        :rtype: list, list, list, list
        """
        # Step 1: gets the Mean matrix
        self.assertIsNotNone(computed_result['matrices'][0], "Mean Correlation object exists")
        mean_obj = computed_result['matrices'][0]
        self.assertEqual(mean_obj['desc']['label'], "Mean Correlation", "Description of mean correlation")
        self.assertIsNotNone(mean_obj['data'], "Mean Correlation data exists")
        mean_matrix = mean_obj['data']
        self.assertTrue(is_triangular_matrix(mean_matrix, ref_dim), "Mean matrix is triangular")

        # Step 2: gets the Variance matrix
        self.assertIsNotNone(computed_result['matrices'][1], "Variance Correlation object exists")
        variance_obj = computed_result['matrices'][1]
        self.assertEqual(variance_obj['desc']['label'], "Variance", "Description of the variance correlation")
        self.assertIsNotNone(variance_obj['data'], "Variance data exists")
        variance_matrix = variance_obj['data']
        self.assertTrue(is_triangular_matrix(variance_matrix, ref_dim), "Variance matrix is triangular")

        # Step 3: gets the OID matrix (object identifiers pointing to the process_data table)
        oid_matrix = computed_result['matrices'][2]['data']
        self.assertTrue(is_triangular_matrix(oid_matrix, ref_dim),
                        "OID links: matrix is triangular, dim={}".format(ref_dim))

        # Step 4: gets the extracted matrix from OID matrix: loading process_data from their OID.
        extracted_matrix = get_triangular_matrix(dim=ref_dim,
                                                 default_value_diag=None,
                                                 default_value_other=None)

        local_ntdm = ResourceLocator().ntdm
        for r_idx, row in enumerate(oid_matrix):
            # if item is None
            #   => extracted is not applied and lambda returns None
            extracted_matrix[r_idx] = [identifier and local_ntdm.download_data(identifier).json for identifier in row]

        return mean_matrix, variance_matrix, oid_matrix, extracted_matrix

    def __test_corrs_by_context(self, json_corr_by_context, ref_contexts, ref_var_one, ref_var_two):
        """
        Tests the json content that was produced by correlation.data.CorrelationsByContext

            - the presence and lengths of the following lists:
              - contexts,
              - correlation,
              - tsuid pairs,
            - the context values are matching ref_contexts,
            - the variable names are matching sorted(ref_var_one, ref_var_two).

        And finally returns the parsed contexts, correlations and tsuid pairs.

        :param json_corr_by_context: should be a well-formed dict.
        :type json_corr_by_context: dict
        :param ref_contexts: expected context values that should be contained by json_corr_by_context["x_value"]
        :type ref_contexts: list
        :param ref_var_one: expected first variable name
        :type ref_var_one: str
        :param ref_var_two: expected second variable name
        :type ref_var_two: str
        :return: contexts, correlations and tsuid pairs
        :rtype: list, list, list
        """
        self.assertTrue(isinstance(json_corr_by_context, dict))

        tested_keys = ["variables", 'x_value', "y_values", "ts_lists"]

        self.assertTrue([True, True, True, True] ==
                        [key in json_corr_by_context for key in tested_keys])

        self.assertTrue(json_corr_by_context["variables"] == sorted([ref_var_one, ref_var_two]))

        y_values = json_corr_by_context['y_values']

        self.assertTrue(isinstance(y_values, list) and len(y_values) == 1)

        contexts = json_corr_by_context["x_value"]
        corrs = y_values[0]
        tsuid_pairs = json_corr_by_context["ts_lists"]

        # Testing that all the lists are well defined and sized
        for label, tested_vect in zip(["ctx", "corrs", "tsuid_pairs"],
                                      [contexts['data'], corrs['data'], tsuid_pairs]):
            self.assertTrue(isinstance(tested_vect, list) and len(tested_vect) == len(ref_contexts),
                            "Testing {}".format(label))

        self.assertEqual(contexts['data'], ref_contexts)

        # Testing that defined pairs are well-formed
        for idx_pair, pair in enumerate(tsuid_pairs):
            if pair is not None:
                self.assertTrue(
                    isinstance(pair, list)
                    and len(pair) == 2
                    and isinstance(pair[0], str)
                    and isinstance(pair[1], str))
            else:
                # Pair is None means that piece of data is not available
                # => corresponding correlation is None
                self.assertIsNone(corrs['data'][idx_pair])

        return contexts, corrs, tsuid_pairs

    def test_my_pearson(self):
        """
        Test the pearson calculation used for tests

        Provided by
        http://stackoverflow.com/questions/3949226/calculating-pearson-correlation-and-significance-in-python
        """
        corr = self.__my_pearson(x_val=np.array([1.0, 2.0, 3.0]), y_val=np.array([1.0, 5.0, 7.0]))
        self.assertTrue(abs(corr - 0.981980506062) <= PRECISION)

    def test_init_config_from_meta_ref0(self):
        """
        Tests that initialized config from metadata is ok with the cases of ignored TS
          - ignored TS (variable+context): there is no other TS sharing the same context
          - ignored TS: TS without context (ie without metadata providing the context)
          - ignored TS: TS without variable (ie without metadata order_by providing the variable)
          - ignored variable: when all associated TS are ignored

        Note: this is the unique white box unittest on loop._initialize_config_from_meta
        """

        # Dataset with incomplete meta
        # ref0_TS7 cannot be correlated to any other TS for context=5
        #    => context 5 is without any correlations
        #    => var WS2 will be ignored
        #
        # ref0_TS11 is without context (flight)
        #
        # ref0_TS10 is without variable (metric)

        # Remaining variables on consistent data:
        loaded_meta = {
            'ref0_TS1': {CONTEXT: 1, VARIABLE: "HEADING", "qual_nb_points": 80},
            'ref0_TS2': {CONTEXT: 1, VARIABLE: "GS"},
            'ref0_TS3': {CONTEXT: 1, VARIABLE: "WS1"},
            'ref0_TS4': {CONTEXT: 2, VARIABLE: "HEADING", "funcId": "ignored"},
            'ref0_TS5': {CONTEXT: 2, VARIABLE: "GS"},
            'ref0_TS6': {CONTEXT: 2, VARIABLE: "WS1"},
            'ref0_TS7': {CONTEXT: 5, VARIABLE: "WS2"},
            'ref0_TS8': {CONTEXT: 4, VARIABLE: "WS1"},
            'ref0_TS9': {CONTEXT: 4, VARIABLE: "WS3"},
            'ref0_TS10': {CONTEXT: 4},
            'ref0_TS11': {VARIABLE: "WS1"}
        }

        corr_loop_config, contexts, variables = \
            loop._initialize_config_from_meta(ts_metadata_dict=loaded_meta,
                                              context_meta=CONTEXT,
                                              variable_meta=VARIABLE)

        self.assertListEqual([1, 2, 4], contexts, msg="Test applicable contexts.")
        self.assertListEqual(["GS", "HEADING", "WS1", "WS3"], variables, msg="Test evaluated variables.")

        ref_corr_loop_config = [(0, [(0, 'ref0_TS2'), (1, 'ref0_TS1'), (2, 'ref0_TS3')]),
                                (1, [(0, 'ref0_TS5'), (1, 'ref0_TS4'), (2, 'ref0_TS6')]),
                                (2, [(2, 'ref0_TS8'), (3, 'ref0_TS9')])]

        for ref, computed in zip(ref_corr_loop_config, sorted(corr_loop_config, key=lambda x: x[0])):
            self.assertEqual(ref[0], computed[0], "Test computed config: contexts")
            self.assertListEqual(ref[1], computed[1], "Test computed config: (var,tsuid) pairs")

    def test_nominal_ref1(self):
        """
        Tests matrices without null elements: all variables defined in each context
        """

        # Prepare REF1
        # --------------
        ts_ramp_up = np.array([[1101889318000 + x * 1000, 10.0 * x] for x in range(10)], dtype=np.dtype('O'))

        ts_ramp_up_longer = np.array([[1101889318000 + x * 1000, 10.0 * x - 5.0]
                                      for x in range(15)], dtype=np.dtype('O'))

        ts_up_and_down = np.array([[1101889318000 + x * 1050, 10.0 * (5 - abs(5 - x))]
                                   for x in range(10)], dtype=np.dtype('O'))

        ts_down_and_up = np.array([[1101889318500 + x * 1000, 10.0 * (abs(5 - x))]
                                   for x in range(10)], dtype=np.dtype('O'))

        dataset = {
            'REF1_TS1': [{CONTEXT: '1', VARIABLE: "WS1"}, ts_ramp_up],
            'REF1_TS2': [{CONTEXT: '1', VARIABLE: "WS2"}, ts_up_and_down],
            'REF1_TS3': [{CONTEXT: '1', VARIABLE: "HEADING"}, ts_up_and_down],
            'REF1_TS4': [{CONTEXT: '2', VARIABLE: "WS1"}, ts_ramp_up_longer],
            'REF1_TS5': [{CONTEXT: '2', VARIABLE: "WS2"}, ts_down_and_up],
            'REF1_TS6': [{CONTEXT: '2', VARIABLE: "HEADING"}, ts_ramp_up]
        }

        ts_selection_ref1 = None
        try:
            ts_selection_ref1 = TestCorrelationLoop.__save_dataset(dataset)

            computed_result = loop.correlation_ts_list_loop(ts_list=ts_selection_ref1,
                                                            corr_method=loop.PEARSON,
                                                            context_meta=CONTEXT)

            self.assertListEqual(['HEADING', 'WS1', 'WS2'], computed_result['variables'], "Test sorted variables")

            mean_matrix, variance_matrix, oid_matrix, extracted_matrix = self.__get_matrices(computed_result, ref_dim=3)

            # Testing the linked correlation results
            # ----------------------------------------

            # Ts internal content: CorrelationsByContext for variables indexes [0,0]
            #   variables indexes [0,0] <=> variables ['HEADING', 'HEADING']
            # ----------------------------------------------------------------------

            obj_heading_heading_by_flights = extracted_matrix[0][0]

            contexts, corrs, tsuid_pairs = \
                self.__test_corrs_by_context(json_corr_by_context=obj_heading_heading_by_flights,
                                             ref_contexts=['1', '2'],
                                             ref_var_one="HEADING",
                                             ref_var_two="HEADING")

            # Pearson Correlation == 1 because Corr(x,x) is 1.0 if Var(x) is not zero
            self.assertEqual(1.0, corrs['data'][0], "Pearson Corr(HEADING,HEADING) for flight 1")
            self.assertEqual(1.0, corrs['data'][1], "Pearson Corr(HEADING,HEADING) for flight 2")

            # Tests internal content: CorrelationsByContext for variables indexes [0,1]
            #   variables indexes [0,1] <=> variables ['HEADING', 'WS1']
            # -------------------------------------------------------------------------
            obj_heading_ws1_by_flights = extracted_matrix[0][1]

            # - Computes the expected correlation for context=1, according the tested dataset
            ref_heading_ws1_fl_1 = self.__pearson_from_dataset(dataset=dataset,
                                                               context='1',
                                                               var_one="HEADING",
                                                               var_two="WS1")
            # - Computes the expected correlation for context=2, according the tested dataset
            ref_heading_ws1_fl_2 = self.__pearson_from_dataset(dataset=dataset,
                                                               context='2',
                                                               var_one="HEADING",
                                                               var_two="WS1")

            contexts, heading_ws1_corrs, tsuid_pairs = \
                self.__test_corrs_by_context(json_corr_by_context=obj_heading_ws1_by_flights,
                                             ref_contexts=['1', '2'],
                                             ref_var_one="HEADING",
                                             ref_var_two="WS1")

            # Checking that tsuid are consistent with expected funcID
            # flight 1 + HEADING => funcID is REF1_TS3
            # flight 1 + WS1     => funcID is REF1_TS1
            # flight 2 + HEADING => funcID is REF1_TS6
            # flight 2 + WS1     => funcID is REF1_TS4
            for ref_funcid, tsuid in zip(["REF1_TS3", "REF1_TS1", "REF1_TS6", "REF1_TS4"],
                                         [tsuid_pairs[0][0], tsuid_pairs[0][1],
                                          tsuid_pairs[1][0], tsuid_pairs[1][1]]):
                # Just checks that the actual funcId is the same as the expected one in dataset definition
                #
                actual_funcid = IkatsApi.md.read(tsuid)[tsuid]["funcId"]

                self.assertEqual(ref_funcid, actual_funcid,
                                 "Testing tsuid={}: equality between funcId={} and ref={}".format(tsuid,
                                                                                                  actual_funcid,
                                                                                                  ref_funcid))

        finally:
            self.__remove_dataset(ts_selection_ref1)

        self.assertTrue(abs(ref_heading_ws1_fl_1 - heading_ws1_corrs['data'][0]) <= PRECISION,
                        "Pearson Corr(HEADING,WS1) for flight 1")
        self.assertTrue(abs(ref_heading_ws1_fl_2 - heading_ws1_corrs['data'][1]) <= PRECISION,
                        "Pearson Corr(HEADING,WS1) for flight 2")

        # Testing the mean correlation results
        # ------------------------------------

        # Diagonal means: always 1.0 when defined
        for i in range(3):
            self.assertTrue(mean_matrix[i][0] == 1.0)

        # The other means: just one case tested: the HEADING + WS1 correlation pair
        values = heading_ws1_corrs['data']
        self.assertTrue(abs(mean_matrix[0][1] - freduce(lambda x, y: x + y,
                                                        values, 0.0) / len(values)) <= PRECISION)

        # Testing the variance correlation results
        # ----------------------------------------

        # Diagonal variance: always 0.0 as the correlations are 1.0, 1.0, ...
        # => There is no case of constant TS producing NaN correlations
        for i in range(3):
            self.assertTrue(abs(variance_matrix[i][0]) <= PRECISION)

        # The other variances: just one case tested: Var(Corr(HEADING, WS1))
        mean = mean_matrix[0][1]
        values = heading_ws1_corrs['data']
        # recomputing the variance ...
        self.assertTrue(abs(variance_matrix[0][1] - freduce(lambda x, y: x + (y - mean) ** 2,
                                                            values, 0.0) / len(values)) <= PRECISION)

    def test_incomplete_ref2(self):
        """
        Tests initializations with incomplete data:
          - matrices with null elements:
            null elements means that there is no pair of variables (v1, v2) in any of the found contexts.
          - ignored TS (variable + context): there is no other TS sharing the same context
          - ignored TS: TS without context (ie without metadata providing the context)
          - ignored TS: TS without variable (ie without metadata order_by providing the variable)
          - ignored variable: when all associated TS are ignored
        """

        # Prepare ref2
        # ------------
        ts_ramp_up = np.array([[1101889318000 + x * 1000, 8.0 * x] for x in range(10)], dtype=np.dtype('O'))

        ts_ramp_up_longer = np.array([[1101889318000 + x * 1000, 13.0 * x - 5.0]
                                      for x in range(15)], dtype=np.dtype('O'))

        ts_ramp_down = np.array([[1101889318000 + x * 1000, - 10.0 * x] for x in range(10)], dtype=np.dtype('O'))

        ts_ramp_down_shorter = np.array([[1101889318000 + x * 1000, 2.0 - 10.0 * x]
                                         for x in range(7)], dtype=np.dtype('O'))

        ts_up_and_down = np.array([[1101889318000 + x * 1050, 10.0 * (5 - abs(5 - x))]
                                   for x in range(10)], dtype=np.dtype('O'))

        # Dataset with incomplete meta
        dataset = {
            'ref2_TS1': [{CONTEXT: '1', VARIABLE: "GS"}, ts_ramp_down],
            'ref2_TS2': [{CONTEXT: '1', VARIABLE: "HEADING"}, ts_up_and_down],
            'ref2_TS3': [{CONTEXT: '1', VARIABLE: "WS1"}, ts_ramp_up],
            'ref2_TS4': [{CONTEXT: '2', VARIABLE: "GS"}, ts_ramp_up],
            'ref2_TS5': [{CONTEXT: '2', VARIABLE: "HEADING"}, ts_ramp_down_shorter],
            'ref2_TS6': [{CONTEXT: '2', VARIABLE: "WS1"}, ts_ramp_up],
            'ref2_TS7': [{CONTEXT: '5', VARIABLE: "WS2"}, ts_ramp_up_longer],
            'ref2_TS8': [{CONTEXT: '4', VARIABLE: "WS1"}, ts_ramp_down],
            'ref2_TS9': [{CONTEXT: '4', VARIABLE: "WS3"}, ts_ramp_up_longer],
            'ref2_TS10': [{CONTEXT: '4'}, ts_ramp_up],
            'ref2_TS11': [{VARIABLE: "HEADING"}, ts_ramp_up_longer]
        }

        # Test purposes: tests cases where
        #    some TS / variable / contexts are ignored
        #
        # ref2_TS7 cannot be correlated to any other TS for context = 5
        #    => context 5 is without any correlations
        #    => var WS2 will be ignored
        #
        # ref2_TS11 is without context (flight)
        #
        # ref2_TS10 is without variable (metric)

        # TEST_REF2_1
        # Expected: TS funcIDs pointed by TSUIDS in the correlation loop results:
        #   - ITS7, ITS10, ITS11 ignored
        ref_funcids_set = {"ref2_TS1", "ref2_TS2", "ref2_TS3", "ref2_TS4", "ref2_TS5", "ref2_TS6",
                           "ref2_TS8", "ref2_TS9"}
        ref_specific_fid_test = {"1_HEADING_WS1": ["ref2_TS2", "ref2_TS3"], "4_WS1_WS3": ["ref2_TS8", "ref2_TS9"]}

        # TEST_REF2_2
        # Expected: variables applicable to correlation loop:
        #   - WS2 ignored
        ref_vars = ["GS", "HEADING", "WS1", "WS3"]

        # TEST_REF2_3
        # Expected: contexts applicable to correlation loop:
        #   - Context 5 is ignored
        ref_contexts = ['1', '2', '4']

        # TEST_REF2_4
        # Expected: some cells have undefined value
        #  - HEADING + WS3 has no context => None cells in CorrelationDataset matrices (null in json)
        #  - GS + WS3      has no context => None cells in CorrelationDataset matrices (null in json)
        ref_undef_corr_indexes = [[0, 3], [1, 3]]

        ts_selection_ref2 = None
        computed_funcids = set()
        try:
            ts_selection_ref2 = self.__save_dataset(dataset)

            computed_result = loop.correlation_ts_list_loop(ts_list=ts_selection_ref2,
                                                            corr_method=loop.PEARSON,
                                                            context_meta=CONTEXT)

            # Getting the matrices before testing specific cases
            mean_matrix, variance_matrix, oid_matrix, extracted_matrix = self.__get_matrices(computed_result, ref_dim=4)

            # Step 1:
            # Iterate inside the OID matrix and its extracted content:
            #   - TEST_REF2_1: testing the TS funcIDs pointed by TSUIDS in the correlation loop results:
            #   - TEST_REF2_3: testing the contexts described in each defined cell

            for i, row in enumerate(extracted_matrix):
                for j, def_cell in enumerate(row):
                    # Squared-matrix column = i +j
                    col = i + j
                    if def_cell is None:
                        continue
                    try:
                        print("Testing incomplete corr({},{}): cell={}".format(i, j, def_cell))
                        contexts, corrs, tsuid_pairs = self.__test_corrs_by_context(def_cell,
                                                                                    ref_contexts=ref_contexts,
                                                                                    ref_var_one=ref_vars[i],
                                                                                    ref_var_two=ref_vars[col])
                        self.assertListEqual(ref_contexts, contexts['data'], "TEST_REF2_3: tests contexts coverage")
                    except Exception:
                        self.fail("Unexpected error around TEST_REF2_1 or TEST_REF2_3")

                    for ctx, pair in enumerate(tsuid_pairs):
                        specific_test = "{}_{}_{}".format(ref_contexts[ctx], ref_vars[i], ref_vars[col])
                        if pair is None:
                            continue
                        fid_pair = self.__retrieve_funcids(pair)
                        computed_funcids.add(fid_pair[0])
                        computed_funcids.add(fid_pair[1])
                        if specific_test in ref_specific_fid_test:
                            msg = "TEST_REF2_1 tsuids={} => fids={} for corr({},{}) for context={}"
                            self.assertListEqual(fid_pair,
                                                 ref_specific_fid_test[specific_test],
                                                 msg.format(pair, fid_pair, ref_vars[i], ref_vars[col],
                                                            ref_contexts[ctx]))

            # Finally tests that all expected fids are pointed
            self.assertEqual(computed_funcids, ref_funcids_set,
                             "TEST_REF2_1 tests coverage of the ref_funcIds_set={}".format(ref_funcids_set))

            # Step 2:
            # Testing TEST_REF2_2: testing the variable defined in the results
            self.assertListEqual(ref_vars, computed_result['variables'], "TEST_REF2_2: Test sorted variables")

            # Step 3:
            # Iterates on matrices Mean, Variance, OID:
            #   - testing TEST_REF2_4: Expected: some cells have undefined value
            for info, matrix in zip(["OID", "Mean", "Variance"], [oid_matrix, mean_matrix, variance_matrix]):

                for [i, j] in ref_undef_corr_indexes:
                    # Triangular-matrix column = j - i
                    col = j - i
                    msg = "TEST_REF2_4: tests {} matrix at [{},{}] is None for [{},{}]"
                    self.assertIsNone(matrix[i][col], msg.format(info, i, col, ref_vars[i], ref_vars[col]))

        finally:
            self.__remove_dataset(ts_selection_ref2)

    def test_nan_ref3(self):
        """
        Tests nominal case when at least one NaN correlation is computed:
        it is due too the fact that one of the timeseries variance is zero: in the case
        of Pearson correlation.
        """
        l_nan = loop.NAN

        ts_ramp_up = np.array([[1101889318000 + x * 1000, 10.0 * x] for x in range(10)], dtype=np.dtype('O'))

        ts_ramp_down = np.array([[1101889318000 + x * 1000, - 10.0 * x] for x in range(10)], dtype=np.dtype('O'))

        ts_constant = np.array([[1101889318500 + x * 1000, - 10.0] for x in range(10)], dtype=np.dtype('O'))

        ts_constant_2 = np.array([[1101889318500 + x * 900, - 5.0] for x in range(7)], dtype=np.dtype('O'))

        ts_constant_3 = np.array([[1101889318300 + x * 1000, 10.0] for x in range(17)], dtype=np.dtype('O'))

        # Prepare REF3
        # ------------
        dataset = {
            'REF3_TS1': [{CONTEXT: "1", VARIABLE: "WS1"}, ts_ramp_down],
            'REF3_TS2': [{CONTEXT: "1", VARIABLE: "WS2"}, ts_ramp_up],
            'REF3_TS3': [{CONTEXT: "1", VARIABLE: "HEADING"}, ts_constant],
            'REF3_TS4': [{CONTEXT: "2", VARIABLE: "WS1"}, ts_ramp_up],
            'REF3_TS5': [{CONTEXT: "2", VARIABLE: "WS2"}, ts_constant_2],
            'REF3_TS6': [{CONTEXT: "2", VARIABLE: "HEADING"}, ts_constant_3]
        }
        # Dataset producing NaN correlations:
        #
        # - context 1: the REF3_TS3 is constant
        #    Corr(HEADING, WS1) is NaN
        #    Corr(HEADING, WS2) is NaN
        #
        # - context 2: both REF3_TS5(WS2) and REF3_TS6(HEADING) are constant:
        #    Corr(HEADING, WS1) is NaN
        #    Corr(HEADING, WS2) is NaN
        #    Corr(WS1,WS2) is NaN
        #
        # Here are the different tests planned on matrices:
        #
        # TEST_MEAN_REF3: on Mean(Corr): indexed by HEADING, WS1, WS2:
        #
        #    [ "Nan", "Nan", "Nan" ]
        #              [ 1.0, -1.0 ]
        #                    [ 1.0 ]
        ref_mean = [[l_nan, l_nan, l_nan],
                    [1.0, -1.0],
                    [1.0]]

        # TEST_VAR_REF3: on Variance(Corr): indexed by HEADING, WS1, WS2:
        #
        #    [ "Nan", "Nan", "Nan" ]
        #               [ 0.0, 0.0 ]
        #                    [ 0.0 ]
        ref_variance = [[l_nan, l_nan, l_nan],
                        [0.0, 0.0],
                        [0.0]]

        # Tests on OID indexed by  HEADING, WS1, WS2:
        # 3 tests: following the linked process-data:
        #
        # TEST_LINK_REF3_1: testing OID[0,0] for Corr(Heading,Heading)
        #              => x= [ 1, 2 ]
        #                 y= [ "Nan", "Nan" ]
        #                 tsuids=[
        #                      [<tsuid(REF3_TS3)>,<tsuid(REF3_TS3)>],
        #                      [<tsuid(REF3_TS6)>,<tsuid(REF3_TS6)>]
        #                 ]
        #
        # TEST_LINK_REF3_2: testing OID[0,1] for Corr(HEADING,WS1)
        #              => x= [ 1, 2 ]
        #                 y= [ "Nan", "Nan" ]
        #                 tsuids=[
        #                      [<tsuid(REF3_TS3)>,<tsuid(REF3_TS1)>],
        #                      [<tsuid(REF3_TS6)>,<tsuid(REF3_TS4)>]
        #                 ]
        #
        # TEST_LINK_REF3_3: testing OID[1,2] for Corr(WS1,WS2)
        #              => x= [ 1, 2 ]
        #                 y= [-1, "Nan"]
        #                 tsuids=[
        #                      [<tsuid(REF3_TS1)>,<tsuid(REF3_TS2)>],
        #                      [<tsuid(REF3_TS4)>,<tsuid(REF3_TS5)>]
        #                 ]
        #
        # =====================================================================================================
        #                     TEST name    | list of | list of | list of    | list
        #                                  |variable | context | corr       | of pointed
        #                                  | indexes | values  | values     | fid pairs
        #                                  |         |         |            |
        ref_extracted = [["TEST_LINK_REF3_1", [0, 0], ['1', '2'], [l_nan, l_nan], [["REF3_TS3", "REF3_TS3"],
                                                                                   ["REF3_TS6", "REF3_TS6"]]],
                         ["TEST_LINK_REF3_2", [0, 1], ['1', '2'], [l_nan, l_nan], [["REF3_TS3", "REF3_TS1"],
                                                                                   ["REF3_TS6", "REF3_TS4"]]],
                         ["TEST_LINK_REF3_3", [1, 2], ['1', '2'], [-1.0, l_nan], [["REF3_TS1", "REF3_TS2"],
                                                                                  ["REF3_TS4", "REF3_TS5"]]]]

        ts_selection_ref3 = None
        try:
            ts_selection_ref3 = self.__save_dataset(dataset, ctx_type=DTYPE.string)

            # Launch ...
            computed_result = loop.correlation_ts_list_loop(ts_list=ts_selection_ref3,
                                                            corr_method=loop.PEARSON,
                                                            context_meta=CONTEXT)

            computed_vars = computed_result['variables']
            self.assertListEqual(["HEADING", "WS1", "WS2"], computed_vars, "Testing computed variables")

            # Getting the matrices before testing specific cases
            mean_matrix, variance_matrix, oid_matrix, extracted_matrix = self.__get_matrices(computed_result, ref_dim=3)

            def compare_corrs(corr1, corr2):
                """
                Compare correlations provided

                :param corr1: First correlation value
                :param corr2: Second correlation value

                :return: True if they match, False otherwise
                """
                if corr1 is None:
                    return corr2 is None
                elif isinstance(corr1, str):
                    return corr1 == corr2
                else:
                    # Required for numeric: test the equality according to PRECISION
                    return abs(corr1 - corr2) <= PRECISION

            for info, [mat, ref] in zip(["Mean", "Variance"],
                                        [[mean_matrix, ref_mean], [variance_matrix, ref_variance]]):

                msg = "TEST_MEAN_REF3: Check consistency of {} matrix with NaN".format(info)

                for row_i, row in enumerate(mat):
                    for col_i, col in enumerate(row):
                        self.assertTrue(compare_corrs(col, ref[row_i][col_i]), msg)

            # Tests TEST_LINK_REF3_XXX
            for test, [i, j], contexts, corrs, pairs in ref_extracted:
                # Triangular-matrix col = j -i
                col = j - i
                extracted_obj = extracted_matrix[i][col]

                self.assertListEqual(contexts, extracted_obj['x_value']['data'],
                                     "{}: tests the contexts  (for {},{})".format(test,
                                                                                  computed_vars[i],
                                                                                  computed_vars[j]))
                for corr_i, corr in enumerate(corrs):
                    self.assertTrue(compare_corrs(corr, extracted_obj['y_values'][0]['data'][corr_i]),
                                    "{}: tests the Corrs({},{})".format(test, computed_vars[i], computed_vars[j]))
                tsuid_pairs = extracted_obj['ts_lists']
                fid_pairs = [self.__retrieve_funcids(pair) for pair in tsuid_pairs]

                msg = "{}: tsuids={} are matching: fids={} == ref={} (for {},{})"
                self.assertListEqual(pairs, fid_pairs,
                                     msg.format(test, tsuid_pairs, fid_pairs, pairs,
                                                computed_vars[i], computed_vars[j]))
        finally:
            self.__remove_dataset(ts_selection_ref3)

    def test_inconsistent_ref4(self):
        """
        Tests degraded case when the choice of context + variable is inconsistent:
          - the system detects more than one TS for a fixed pair of (variable + context) => raises an error

        Ex: the user entered "AircraftIdentifier" instead of "FlightIdentifier" for the context;
        and metric is selected as order_by variable.
        """
        ts_ramp_up = np.array([[1101889318000 + x * 1000, 10.0 * x] for x in range(10)], dtype=np.dtype('O'))

        ts_ramp_down = np.array([[1101889318000 + x * 1000, - 10.0 * x] for x in range(10)], dtype=np.dtype('O'))

        # Prepare REF4
        # ------------
        dataset = {
            'REF4_TS1': [{"Flight": "1", "AircraftIdentifier": "A1", VARIABLE: "WS1"}, ts_ramp_down],
            'REF4_TS2': [{"Flight": "2", "AircraftIdentifier": "A1", VARIABLE: "WS1"}, ts_ramp_up],
            'REF4_TS3': [{"Flight": "2", "AircraftIdentifier": "A1", VARIABLE: "HEADING"}, ts_ramp_down],
            'REF4_TS4': [{"Flight": "3", "AircraftIdentifier": "A2", VARIABLE: "WS1"}, ts_ramp_up],
            'REF4_TS6': [{"Flight": "3", "AircraftIdentifier": "A2", VARIABLE: "HEADING"}, ts_ramp_down]
        }

        ts_selection_ref4 = None
        try:
            ts_selection_ref4 = self.__save_dataset(dataset,
                                                    context_identifier="AircraftIdentifier",
                                                    ctx_type=DTYPE.string)

            # Launch ...
            msg = "the system ought to raise inconsistency: more than one TS for a fixed pair of (variable + context)"
            with self.assertRaises(IkatsException, msg=msg):
                loop.correlation_ts_list_loop(ts_list=ts_selection_ref4,
                                              corr_method=loop.PEARSON,
                                              context_meta="AircraftIdentifier",
                                              variable_meta=VARIABLE)

        finally:
            self.__remove_dataset(ts_selection_ref4)

    def test_no_computable_corrs_ref5(self):
        """
        Tests that correlation loop is robust when case leads to empty result:
          - tested here: case when there is no computable Corr(X, Y)
            where variables X and Y are different for the same context
        """
        ts_ramp_up = np.array([[1101889318000 + x * 1000, 10.0 * x] for x in range(10)], dtype=np.dtype('O'))

        ts_ramp_down = np.array([[1101889318000 + x * 1000, - 10.0 * x] for x in range(10)], dtype=np.dtype('O'))

        # Prepare REF5
        # ------------
        dataset = {
            'REF5_TS1': [{CONTEXT: 1, VARIABLE: "WS1"}, ts_ramp_down],
            'REF5_TS2': [{CONTEXT: 2, VARIABLE: "WS2"}, ts_ramp_up],
            'REF5_TS3': [{CONTEXT: 3, VARIABLE: "HEADING"}, ts_ramp_down],
            'REF5_TS4': [{CONTEXT: 4, VARIABLE: "WS3"}, ts_ramp_up],
            'REF5_TS5': [{CONTEXT: 5, VARIABLE: "WS4"}, ts_ramp_down],
            'REF5_TS6': [{CONTEXT: 6, VARIABLE: "GS"}, ts_ramp_down]
        }
        ts_selection_ref5 = None
        try:
            ts_selection_ref5 = self.__save_dataset(dataset)

            test_info = "TEST_REF5_1"
            description = "case when there is no computable Corr(X, Y) with X != Y"
            try:

                # Launch ...
                computed_result = loop.correlation_ts_list_loop(ts_list=ts_selection_ref5,
                                                                corr_method=loop.PEARSON,
                                                                context_meta=CONTEXT,
                                                                variable_meta=VARIABLE)

                self.assertTrue(computed_result['variables'] == [])
                self.assertTrue(computed_result['context']["number_of_contexts"] == 0)
                self.assertTrue(computed_result['context']["label"] == CONTEXT)
                for mat in computed_result['matrices']:
                    self.assertTrue(is_triangular_matrix(matrix=mat['data'], expected_dim=0),
                                    "Matrix data {} should be empty !".format(mat))
            except Exception as err:
                msg = "{}: {}: Unexpected error or assert failure={}"
                self.fail(msg.format(test_info, description, err))

        finally:
            self.__remove_dataset(ts_selection_ref5)

    def test_no_context_no_var_ref6(self):
        """
        Tests that correlation loop is robust when case leads to empty result:
          - missing metadata for context => no contexts
          - missing metadata for ordering variables => no variables

        """
        ts_ramp_up = np.array([[1101889318000 + x * 1000, 10.0 * x] for x in range(10)], dtype=np.dtype('O'))

        # Prepare REF6
        # ------------
        dataset = {
            'REF6_TS1': [{CONTEXT: 1, VARIABLE: "WS1"}, ts_ramp_up],
            'REF6_TS2': [{CONTEXT: 1, VARIABLE: "WS2"}, ts_ramp_up],
        }
        ts_selection_ref6 = None
        try:
            ts_selection_ref6 = self.__save_dataset(dataset)

            def_robust_cases = [("TEST_REF6_1", CONTEXT, "FAKE",
                                 "case when there ordering metadata is missing => no variables defined"),
                                ("TEST_REF6_2", "FAKE", VARIABLE,
                                 "case when there context metadata is missing => no contexts defined")]

            for (test_info, context_meta, variable_meta, description) in def_robust_cases:
                try:
                    # Launch ...
                    computed_result = loop.correlation_ts_list_loop(ts_list=ts_selection_ref6,
                                                                    corr_method=loop.PEARSON,
                                                                    context_meta=context_meta,
                                                                    variable_meta=variable_meta)
                    self.assertTrue(computed_result['variables'] == [])
                    self.assertTrue(computed_result['context']["number_of_contexts"] == 0)
                    self.assertTrue(computed_result['context']["label"] == context_meta)
                    for mat in computed_result['matrices']:
                        self.assertTrue(is_triangular_matrix(matrix=mat['data'], expected_dim=0),
                                        "Matrix data {} should be empty !".format(mat))

                except Exception as err:
                    msg = "{}: {}: Unexpected error or assert failure={}"
                    self.fail(msg.format(test_info, description, err))

        finally:
            self.__remove_dataset(ts_selection_ref6)

    @skipIf('SKIP_LONG_TEST' in os.environ and os.environ['SKIP_LONG_TEST'],
            "This test is too long and must not be run every time")
    def test_perf_ref7(self):
        """
        Integration test / Perf test
        """
        tsuids = self.generate_perf_dataset(nb_contexts=400,
                                            nb_vars=16,
                                            nb_points=1000,
                                            reload_existing_dataset=True)
        try:
            start_time = time.time()

            computed_result = loop.correlation_ts_list_loop(ts_list=tsuids,
                                                            corr_method=loop.PEARSON,
                                                            context_meta=CONTEXT,
                                                            variable_meta=VARIABLE)

            end_time = time.time()
            self.log.info(computed_result)

            self.log.info("Ran correlation_ts_list_loop in %.5fs", end_time - start_time)
        except Exception as err:
            self.fail("IT test_perf_ref7 got unexpected error={}".format(err))

    def generate_perf_dataset(self, nb_contexts, nb_vars, nb_points, reload_existing_dataset):
        """
        Dataset generator for performance test
        Warning, this call is quite long and shall be limited
        :param nb_contexts: Number of contexts to generate
        :param nb_vars: Number of variables to generate
        :param nb_points: Number of points per timeseries
        :param reload_existing_dataset: Don't overwrite previous matching dataset if True
        :type nb_contexts: int
        :type nb_vars: int
        :type nb_points: int
        :type reload_existing_dataset: bool
        :return: the list of created TSUIDs
        :rtype: list
        """

        # Templates for dataset name and TS funcIds
        #
        #   in this test case: record a dataset parent of TS
        #   => make easier the data cleaning !!!
        #
        # Dataset name: REF7_CORR_LOOP_<nb contexts>_<nb_vars>_<test nb_points>
        template_dataset_name = "REF7_CORR_LOOP_{:03d}_{:03d}_{}"
        #
        # funcId: REF7_CORR_LOOP_TS<increment>_<context value>_<variable value>_<nb_points>
        #
        template_ts_funcid = "REF7_CORR_LOOP_TS{}_{}_{}_{}"

        # TS definitions
        #
        half = nb_points / 2.0
        ts_ramp_up = np.array([[1101889318000 + x * 1000, 2.0 * x] for x in range(nb_points)], dtype=np.dtype('O'))

        ts_ramp_up_longer = np.array([[1101889318000 + x * 1000, 10.0 * x - 5.0]
                                      for x in range(nb_points)], dtype=np.dtype('O'))

        ts_up_and_down = np.array([[1101889318000 + x * 1050, 10.0 * (half - abs(half - x))]
                                   for x in range(nb_points)], dtype=np.dtype('O'))

        ts_down_and_up = np.array([[1101889318500 + x * 1000, 10.0 * (abs(half - x))]
                                   for x in range(nb_points)], dtype=np.dtype('O'))

        ts_set = [ts_ramp_up, ts_ramp_up_longer, ts_up_and_down, ts_down_and_up]

        dataset_name = template_dataset_name.format(nb_contexts, nb_vars, nb_points)
        dataset = {}
        increment = 0
        for context in [x + 1 for x in range(nb_contexts)]:
            for var_index, variable in enumerate(["VAR{:03d}".format(x + 1) for x in range(nb_vars)]):
                increment = increment + 1
                funcid = template_ts_funcid.format(increment, context, variable, nb_points)
                dataset[funcid] = [{CONTEXT: context, VARIABLE: variable},
                                   ts_set[(var_index + context) % len(ts_set)]]

        # 1 - Cleaning previous dataset if demanded
        # 2 - Create dataset when required
        # 3 - Gets the tsuid list
        existing_dataset = IkatsApi.ds.read(ds_name=dataset_name)
        if (not reload_existing_dataset) and len(existing_dataset['ts_list']) != 0:
            self.log.info("Deleting previous dataset=%s ...", dataset_name)
            IkatsApi.ds.delete(ds_name=dataset_name, deep=True)
            self.log.info("... deleted")

        if (not reload_existing_dataset) or len(existing_dataset['ts_list']) == 0:
            self.log.info("Creating TS for dataset=%s ...", dataset_name)
            created_tsuids = self.__save_dataset(dataset)
            self.log.info("... TS created for dataset=%s", dataset_name)
            self.log.info("Creating dataset=%s ...", dataset_name)
            IkatsApi.ds.create(ds_name=dataset_name,
                               description="Dataset created by correlation.tests.test_loop.test_perf_ref7",
                               tsuid_list=created_tsuids)
            self.log.info("...created dataset=%s", dataset_name)

        else:
            self.log.info("Reading the TS for existing  dataset=%s", dataset_name)
            existing_dataset = IkatsApi.ds.read(ds_name=dataset_name)
            created_tsuids = existing_dataset['ts_list']

        return created_tsuids
