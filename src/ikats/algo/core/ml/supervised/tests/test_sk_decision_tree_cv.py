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
import json
import os
import random
from collections import defaultdict
from unittest import TestCase

import mock
import numpy as np

from ikats.algo.core.ml.supervised.sk_decision_tree_cv import fit
from ikats.core.library.exception import IkatsException, IkatsInputTypeError
from ikats.core.resource.api import IkatsApi

np.random.seed(777)
random.seed(777)

TU_IRIS_POPULATION_FILE = os.path.dirname(os.path.realpath(__file__)) + "/Population_Iris.json"

# List of mocked tables
TABLES = {}


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


def get_table():
    """
    This function returns the table given by TU_IRIS_POPULATION_FILE
    :return: the table given by TU_IRIS_POPULATION_FILE
    :rtype: dict
    """

    with open(TU_IRIS_POPULATION_FILE, 'r') as mock_file:
        tested_pop = json.load(mock_file)

    return tested_pop


# noinspection PyIncorrectDocstring
def mock_fit_ko():
    """
    Simulates error raised from sk-learn fit method sklearn.tree.DecisionTreeClassifier.fit
    :param X:Unused in stub
    :type X:Unused in stub
    :param y:Unused in stub
    :type y:Unused in stub
    :param sample_weight:Unused in stub
    :type sample_weight:Unused in stub
    :param check_input:Unused in stub
    :type check_input:Unused in stub
    :param X_idx_sorted:Unused in stub
    :type X_idx_sorted:Unused in stub
    """

    raise Exception("Mock: simulated error from fit method")


class TestSkDecisionTreeCV(TestCase):
    """
    Test of the Decision Tree Cross Validation operator
    """

    def test_determinism(self):
        """
        Tests if the random is deterministic
        :return:
        """
        self.assertEqual(random.random(), 0.22933408950153078, "Not the expected value in a deterministic environment")

    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    def test_fit_nominal_from_id(self):
        """
        Tests the nominal execution based upon mock data: IRIS data
        """

        mdl, dot, best_params, table_name = fit(population=get_table(), target_column_name='"Species"',
                                                table_name="my_table",
                                                identifier_column_name="Id")

        cv_res = IkatsApi.table.read(name=table_name)

        # trained: 71,5.9,3.2,4.8,1.8,I. versicolor
        # => test below should obtain same class:
        #
        # mdl.predict([[5.9, 3.2, 4.8, 1.8]]) returns a numpy.ndarray
        # so we have to convert predicted value to str:
        for value, ref_predict in [[[5.9, 3.2, 4.8, 1.8], "I. versicolor"],
                                   [[5.7, 2.8, 4.1, 1.3, ], "I. versicolor"],
                                   [[5.9, 3.0, 5.1, 1.8], "I. virginica"]]:
            predicted = "{}".format(mdl.predict([value])[0])
            self.assertEqual(predicted, ref_predict, "Failed to use the computed model: predict x={}".format(value))

        self.assertTrue(type(dot) == str, "Bad type for the dot returned")
        self.assertTrue(type(best_params) == str, "Bad type for the best parameters returned")
        self.assertTrue(type(cv_res) == defaultdict, "Bad type for the cross validation results returned")

    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    def test_fit_from_id_with_param(self):
        """
        Tests the nominal execution based upon mock data: IRIS data
        """

        mdl, dot, best_params, table_name = fit(population=get_table(), target_column_name='"Species"',
                                                identifier_column_name="Id", depth_parameters="0;2;5;3",
                                                balanced_parameters="True;False", table_name="my_table", folds=5)

        cv_res = IkatsApi.table.read(name=table_name)

        # trained: 71,5.9,3.2,4.8,1.8,I. versicolor
        # => test below should obtain same class:
        #
        # mdl.predict([[5.9, 3.2, 4.8, 1.8]]) returns a numpy.ndarray
        # so we have to convert predicted value to str:
        predicted = "{}".format(mdl.predict([[5.9, 3.0, 5.1, 1.8]])[0])
        self.assertEqual(predicted, "I. virginica", "Failed to use the computed model")

        self.assertTrue(type(dot) == str, "Bad type for the dot returned")
        self.assertTrue(type(best_params) == str, "Bad type for the best parameters returned")
        self.assertTrue(type(cv_res) == defaultdict, "Bad type for the cross validation results returned")

    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    def test_fit_ko_from_bad_depth(self):
        """
        Tests the execution when the user gives bad depth parameters
        """
        with self.assertRaises(IkatsException):
            fit(population=get_table(), target_column_name='"Species"',
                identifier_column_name="Id", depth_parameters="0.4;toto;5;3,14",
                balanced_parameters="True;False", table_name="my_table", folds=3)

    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    def test_fit_ko_from_bad_balancing(self):
        """
        Tests the execution when the user gives bad balancing parameters
        """
        with self.assertRaises(IkatsException):
            fit(population=get_table(), target_column_name='"Species"',
                identifier_column_name="Id", depth_parameters="0;3;5;4",
                balanced_parameters="true;false", table_name="my_table", folds=3)

    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    def test_fit_ko_from_bad_folds(self):
        """
        Tests the execution when the user gives bad number of folds
        """
        with self.assertRaises(IkatsInputTypeError):
            fit(population="id", target_column_name='"Species"',
                identifier_column_name="Id", depth_parameters="0;3;5;4",
                balanced_parameters="True;False", table_name="my_table", folds=2.4)

    @mock.patch('sklearn.tree.DecisionTreeClassifier.fit', mock_fit_ko)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    def test_fit_population_sklearn_ko(self):
        """
        Tests the unexpected error, raised by algorithm
        """
        with self.assertRaises(IkatsException):
            fit(population=get_table(), target_column_name='"Species"',
                identifier_column_name="Id", depth_parameters="0;2;5;3",
                table_name="my_table",
                balanced_parameters="True;False")
