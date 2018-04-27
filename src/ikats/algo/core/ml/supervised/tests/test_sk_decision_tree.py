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
import unittest

import mock

from ikats.algo.core.ml.supervised.sk_decision_tree import fit, predict
from ikats.core.library.exception import IkatsNotFoundError, IkatsException
from ikats.core.resource.api import IkatsApi

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
    try:
        return TABLES[name]
    except KeyError:
        raise IkatsNotFoundError("Table %s not found" % name)


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


class TestSkDecisionTree(unittest.TestCase):
    """
    Tests about the Decision Tree implemented with Scikit Learn
    """

    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    def setUp(self):
        # Prepare data
        with open(TU_IRIS_POPULATION_FILE, 'r') as mock_file:
            IkatsApi.table.create(data=json.load(mock_file), name="iris")

    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    def test_fit_nominal_from_id(self):
        """
        Tests the nominal execution based upon mock data: IRIS data
        """

        mdl, dot = fit(population="iris",
                       target_column_name='"Species"',
                       identifier_column_name="Id",
                       max_depth=0,
                       balanced_class_weight=False)

        # trained: 71,5.9,3.2,4.8,1.8,I. versicolor
        # => test below should obtain same class:
        #
        # mdl.predict([[5.9, 3.2, 4.8, 1.8]]) returns a numpy.ndarray
        # so we have to convert predicted value to str:
        for x_val, ref_predict in [[[5.9, 3.2, 4.8, 1.8], "I. versicolor"],
                                   [[5.7, 2.8, 4.1, 1.3, ], "I. versicolor"],
                                   [[5.9, 3.0, 5.1, 1.8], "I. virginica"]]:
            predicted = "{}".format(mdl.predict([x_val])[0])
            self.assertEqual(predicted, ref_predict, "Failed to use the computed model: predict x={}".format(x_val))

        self.assertEqual(type(dot), str, "Bad type for the dot returned")

    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    def test_fit_with_depth(self):
        """
        Tests the nominal execution based upon mock data: IRIS data
        """

        mdl, dot = fit(population="iris",
                       target_column_name='"Species"',
                       identifier_column_name="Id",
                       max_depth=4,
                       balanced_class_weight=True)

        # trained: 71,5.9,3.2,4.8,1.8,I. versicolor
        # => test below should obtain same class:
        #
        # mdl.predict([[5.9, 3.2, 4.8, 1.8]]) retuns a numpy.ndarray
        # so we have to convert predicted value to str:
        predicted = "{}".format(mdl.predict([[5.9, 3.2, 4.8, 1.8]])[0])
        self.assertEqual(predicted, "I. versicolor", "Failed to use the computed model")

        self.assertEqual(type(dot), str, "Bad type for the dot returned")

    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    def test_fit_ko_not_found(self):
        """
        Tests the execution failure when the population is not readable from the processdata in DB.
        """
        with self.assertRaises(IkatsNotFoundError):
            fit(population="no_table",
                target_column_name='"Species"',
                identifier_column_name="Id",
                max_depth=0,
                balanced_class_weight=False)

    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    @mock.patch('sklearn.tree.DecisionTreeClassifier.fit', mock_fit_ko)
    def test_fit_population_sklearn_ko(self):
        """
        Tests the unexpected error, raised by algorithm
        """
        with self.assertRaises(IkatsException):
            fit(population="iris",
                target_column_name='"Species"',
                identifier_column_name="Id",
                max_depth=0,
                balanced_class_weight=False)

    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    def test_predict_nominal(self):
        """
        Tests the nominal execution based upon mock data: IRIS data
        tests data originated from mocked iris dataset => accuracy = 100%
        """

        mdl, _ = fit(population="iris",
                     target_column_name='"Species"',
                     identifier_column_name="Id",
                     max_depth=0,
                     balanced_class_weight=False)

        test_table = {
            "table_desc": {},
            "headers": {
                "col": {
                    "data": ["Id", "Sepal length", "Sepal width", "Petal length", "Petal width", "Species"]
                },
                "row": {
                    "data": [None, "1", "5", "8", "128"]
                }
            },
            "content": {
                "cells": [["5.9", "3.2", "4.8", "1.8", "I. versicolor"],
                          ["5.0", "3.5", "1.6", "0.6", "I. setosa"],
                          ["5.9", "3.0", "5.1", "1.8", "I. virginica"],
                          ["6.8", "3.0", "5.5", "2.1", "I. virginica"]]
            }
        }

        attempted_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 2]]

        # Create the table used for this test
        create_table(name="pop_table", data=test_table)

        table_name, accuracy = predict(model=mdl,
                                       population_name="pop_table",
                                       target_column_name='Species',
                                       identifier_column_name='Id',
                                       table_name='my_table')

        matrix = IkatsApi.table.read(table_name)

        self.assertListEqual(matrix['content']['cells'], attempted_matrix)
        self.assertEqual(accuracy, 1)

    @mock.patch("ikats.core.resource.api.IkatsApi.table.read", read_table)
    @mock.patch("ikats.core.resource.api.IkatsApi.table.create", create_table)
    def test_predict_pop_wo_row_head(self):
        """
        Tests the nominal execution based upon mock data: IRIS data
        tests data altered (one record modified) => accuracy = 75%
        """

        mdl, _ = fit(population="iris",
                     target_column_name='"Species"',
                     identifier_column_name="Id",
                     max_depth=0,
                     balanced_class_weight=False)

        test_table = {
            "table_desc": {},
            "headers": {
                "col": {
                    "data": ["Id", "Sepal length", "Sepal width", "Petal length", "Petal width", "Species"]
                }
            },
            "content": {
                "cells": [["5", "5.0", "3.5", "1.6", "0.6", "I. versicolor"],
                          ["5", "5.0", "3.5", "1.6", "0.6", "I. setosa"],
                          ["8", "5.9", "3.0", "5.1", "1.8", "I. virginica"],
                          ["128", "6.8", "3.0", "5.5", "2.1", "I. virginica"]]
            }
        }

        attempted_matrix = [[0, 1, 0], [0, 1, 0], [0, 0, 2]]

        # Create the table used for this test
        create_table(name="pop_table", data=test_table)

        table_name, accuracy = predict(model=mdl,
                                       population_name="pop_table",
                                       target_column_name='Species',
                                       identifier_column_name='Id',
                                       table_name='my_table')

        matrix = IkatsApi.table.read(table_name)

        self.assertListEqual(matrix['content']['cells'], attempted_matrix)
        self.assertEqual(accuracy, 0.75)
