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
from collections import defaultdict

import logging

import re
from sklearn.externals.six import StringIO
from sklearn import tree
from sklearn.metrics import confusion_matrix

from ikats.core.resource.api import IkatsApi
from ikats.core.library.exception import IkatsException, IkatsInputTypeError, IkatsInputError

"""
Assumed: population is described with functional type 'table'
  - property table_desc:
    - title: name of the population
    - desc: uploaded filename

  - property headers.col.data is a list sized M: column headers naming the population features

  - property headers.row.data is a list sized N+1:
       headers.row.data[0] is undefined/unused
       headers.row.data[i] defines first feature of i-th object described (N objects in population)

  - property content.cells is a matrix:
       content.cells[i-1][...] describes the M-1 following features of the i-th object
"""

LOGGER = logging.getLogger(__name__)


def fit(population, target_column_name, identifier_column_name, max_depth=None, balanced_class_weight=False):
    """
    The IKATS wrapper published in catalog of fit algorithm for DecisionTree of scikit-learn

    :param population: the population data is functional type 'table' in the the DB table.
        - the table object (dict)
        - otherwise its name is accepted (str)
    :type population: str or dict
    :param target_column_name: the name of the attribute providing the class label of the observed subject.
       Must match one of the available population attributes.
    :type target_column_name: str
    :param identifier_column_name: the name of the attribute identifying each observed subject.
       Must match one of the available population attributes.
    :type identifier_column_name: str
    :param max_depth: optional, default 0: the maximum depth of the tree. When set to 0 (default value),
                        there is no constraint on the depth.
    :type max_depth: int
    :param balanced_class_weight: optional, default False: when True: apply a weight balancing on the classes,
        inversely proportional to class frequencies in the input data:
        weight(label) = total_samples / (nb_classes * count_samples(label)).
    :type balanced_class_weight: bool
    :return: tuple of results: model, dot:
      - model: is the model computed by the internal library Scikit-learn
      - dot: is the textual definition of the tree, based upon dot format: content parsable by graph viewers.
    :rtype:  DecisionTreeClassifier, str
    :raises IkatsInputTypeError: if an argument has an unexpected type
    :raises IkatsNotFoundError: if the population cannot be found
    :raises IkatsException: another error occurred
    """

    if type(target_column_name) is not str:
        msg = "target_column_name has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(target_column_name).__name__))

    if type(identifier_column_name) is not str:
        msg = "identifier_column_name has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(identifier_column_name).__name__))

    if (type(max_depth) is not int) and (max_depth is not None):
        msg = "max_depth has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(max_depth).__name__))
    if type(max_depth) is int and max_depth == 0:
        # front: default value is zero => translated into None in order to be compatible with scikit-learn
        max_depth = None

    if type(balanced_class_weight) is not bool:
        msg = "balanced_class_weight has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(balanced_class_weight).__name__))

    if type(population) is str:
        obj_population = IkatsApi.table.read(population)
        LOGGER.info("Population used %s", population)
    elif type(population) is dict:
        obj_population = population
    else:
        msg = "population has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(population).__name__))

    return fit_population(obj_population, target_column_name, identifier_column_name, max_depth, balanced_class_weight)


def fit_population(population, target_column_name, identifier_column_name, max_depth=None, balanced_class_weight=False):
    """
    The internal wrapper of fit algorithm for DecisionTree of scikit-learn.

    :param population: the population data whose functional type is 'table'
    :type population: dict
    :param target_column_name:
    :type target_column_name:
    :param identifier_column_name:
    :type identifier_column_name:
    :param max_depth:
    :type max_depth:
    :param balanced_class_weight:
    :type balanced_class_weight:
    :raises IkatsException: error occurred.
    """

    LOGGER.info("Starting Decision Tree Fit with scikit-learn")
    try:
        desc_population = population.get('table_desc', None)

        LOGGER.info("with Population table_desc= %s", desc_population)

        # 1/ prepare the learning set
        #
        feature_vectors, target, class_names, column_names = split_population(population, target_column_name,
                                                                              identifier_column_name)

        # 2/ prepare the DecisionTree
        #
        my_class_weight = None
        if balanced_class_weight:
            my_class_weight = 'balanced'

        mdl = tree.DecisionTreeClassifier(max_depth=max_depth, class_weight=my_class_weight)

        mdl.fit(X=feature_vectors, y=target)

        LOGGER.info("   ... finished  fitting the Decision Tree to data")
        LOGGER.info(" - Exporting Decision Tree to dot format")
        dot_io = StringIO()
        tree.export_graphviz(mdl, out_file=dot_io, feature_names=column_names,
                             class_names=class_names, filled=True, label='all')
        dot = dot_io.getvalue()
        LOGGER.info("  ... finished exporting the Decision Tree to dot format")
        LOGGER.info("... ended  Decision Tree Fit with scikit-learn")
        return mdl, dot
    except IkatsException as ike:
        raise ike
    except Exception:
        msg = "Unexpected error: fit_population(..., {}, {}, {}, {})"
        raise IkatsException(msg.format(target_column_name,
                                        identifier_column_name,
                                        max_depth,
                                        balanced_class_weight))


def predict(model, population_name, target_column_name, identifier_column_name, table_name):
    """
    The IKATS wrapper published in catalog of predict algorithm for DecisionTree of scikit-learn

    :param model: model processed by the sklearn fit function
    :type model: sklearn.tree.DecisionTreeClassifier

    :param population_name: the name of the table containing the population data.
    :type population_name: str

    :param target_column_name: the name of the attribute providing the class label of the observed subject.
           Must match one of the available population attributes (column names).
    :type target_column_name: str

    :param identifier_column_name: the name of the attribute identifying each observed subject.
           Must match one of the available population attributes (column names).
    :type identifier_column_name: str

    :param table_name: name of the table to create
    :type table_name:  str

    :return: tuple of results: confusion matrix, mean accuracy
        confusion_matrix : name of the table containing the following:
                           Each row of the matrix represents the instances in an actual class while each column
                           represents the instances in a predicted class
        mean_accuracy : ratio between true predictions and samples number (population rows without header)
    :rtype:  str, float

    :raises IkatsInputTypeError: if an argument has an unexpected type or content
    :raises IkatsInputError: if population has an unexpected content
    """
    if table_name is None or re.match('^[a-zA-Z0-9-_]+$', table_name) is None:
        raise ValueError("Error in table name")
    if type(target_column_name) is not str:
        msg = "target_column_name has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(target_column_name).__name__))

    if type(identifier_column_name) is not str:
        msg = "identifier_column_name has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(identifier_column_name).__name__))

    if type(population_name) is not str:
        msg = "population has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(population_name).__name__))

    # Read the table
    population = IkatsApi.table.read(population_name)

    LOGGER.info("Starting Decision Tree Predict with scikit-learn")

    desc_population = population.get('table_desc', None)
    LOGGER.info("with Population table_desc= %s", desc_population)
    LOGGER.info("with target_column_name = %s", target_column_name)
    LOGGER.info("with identifier_column_name = %s", identifier_column_name)

    # 1/ Prepare the learning set
    #
    feature_vectors, target, class_names, _ = split_population(population, target_column_name,
                                                               identifier_column_name)

    # 2/ Compute the classes
    #
    result = model.predict(feature_vectors)

    # 3/ Compute the mean accuracy
    #
    mean_accuracy = model.score(feature_vectors, target)

    # 4/ Compute the confusion matrix and format it
    #
    matrix = confusion_matrix(target, result, class_names)
    formatted_matrix = _fill_table_confusion_matrix(matrix=matrix, class_names=class_names, table_name=table_name)

    # Save the table
    IkatsApi.table.create(data=formatted_matrix)

    return table_name, mean_accuracy


def split_population(population, target_column_name, identifier_column_name):
    """
    Parse population in order to extract following data :
        - feature vectors : 2D array
        - target vector : 1D array
        - list of classes names
        - list of column names

    :param population: the population data is functional type 'table' in the the processdata DB table.
    :type population: dict

    :param target_column_name: name of the target column in population table
    :type target_column_name: string

    :param identifier_column_name: name of the identifier column in population table
    :type identifier_column_name: string

    :return: tuple composed of :
        - feature vectors
        - target vector
        - list of classes names
        - list of column names
    :rtype: tuple
    :raises IkatsInputError: if population has an unexpected content
    """
    try:
        col_headers = population['headers']['col']['data']
    except Exception:
        raise IkatsInputError("No column headers found in population")
    try:
        row_headers = population['headers']['row']['data']
    except Exception:
        row_headers = None
    try:
        feature_vectors = population['content']['cells']
    except Exception:
        raise IkatsInputError("No cells content found in population")
    try:
        index_target = col_headers.index(target_column_name)
    except Exception:
        raise IkatsInputError("Not found: target column name={}".format(identifier_column_name))
    try:
        index_identifier = col_headers.index(identifier_column_name)
    except Exception:
        raise IkatsInputError("Not found: ID column name={}".format(identifier_column_name))

    column_names = list(col_headers)
    column_names.remove(target_column_name)
    column_names.remove(identifier_column_name)

    LOGGER.debug(" - Fitting Decision Tree to data")
    # - Id is not included in the learning data
    # - Target is provided separately
    if row_headers:
        first_features = row_headers[1:]
        # For each line:
        #  - append first column to the rest of columns into the raw feature_vector
        feature_vectors = [[x] + y for x, y in zip(first_features, feature_vectors)]

    # For each feature_vector: delete the features not taken into account by learning step of
    # DecisionTreeClassifier
    target = [v[index_target] for v in feature_vectors]
    class_names = []
    for feature_vector in feature_vectors:
        if feature_vector[index_target] not in class_names:
            class_names.append(feature_vector[index_target])
        for index in sorted([index_target, index_identifier], reverse=True):
            del feature_vector[index]

    return feature_vectors, target, class_names, column_names


def _fill_table_confusion_matrix(matrix, class_names, table_name):
    """
    Fill an ikats table structure with confusion matrix computed

    :param matrix: array containing values of confusion matrix
    :type matrix: numpy array

    :param : class_names: list of classes used for table headers
    :type : class_names: list

    :param : table_name: Name of the table to save
    :type : table_name: str

    :return: dict formatted as awaited by functional type table
    :rtype: dict
    """

    # Initializing table structure
    table = defaultdict(dict)

    # Filling title
    table['table_desc']['title'] = 'Confusion Matrix'
    table['table_desc']['name'] = table_name
    table['table_desc']['desc'] = "Result of Decision Tree Predict operator"

    # Filling headers columns
    table['headers']['col'] = dict()
    table['headers']['col']['data'] = ["Predicted Class"]
    table['headers']['col']['data'].extend(class_names)

    # Filling headers rows
    table['headers']['row'] = dict()
    table['headers']['row']['data'] = ["Actual Class"]
    table['headers']['row']['data'].extend(class_names)

    # Filling cells content
    table['content']['cells'] = matrix.tolist()

    return table
