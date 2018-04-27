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
import logging
import re
from collections import defaultdict

from sklearn.externals.six import StringIO
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV

from ikats.algo.core.ml.supervised.sk_decision_tree import split_population
from ikats.core.library.exception import IkatsException, IkatsInputTypeError
from ikats.core.resource.api import IkatsApi

"""
This IKATS operator implements the cross-validation procedure to find the best combination of parameters, giving a more
 accurate model, here the model is a decision tree.
"""

LOGGER = logging.getLogger(__name__)


def fit(population, target_column_name, identifier_column_name, table_name, folds=3, depth_parameters=None,
        balanced_parameters=None):
    """
    The IKATS wrapper published in catalog of GridSearchCV fit algorithm for DecisionTree of scikit-learn.

    :param population: the population data is functional type 'table' in the the processdata DB table.
        - the table object (dict)
        - otherwise its name is accepted (str)
    :type population: str or dict

    :param target_column_name: the name of the attribute providing the class label of the observed subject.
       Must match one of the available population attributes.
    :type target_column_name: str

    :param identifier_column_name: the name of the attribute identifying each observed subject.
       Must match one of the available population attributes.
    :type identifier_column_name: str

    :param depth_parameters: optional, default None : string containing the list of values of max_depth we want to test,
    separated by a ";" or in the form "range(n,p)"
    :type depth_parameters: str

    :param balanced_parameters: optional, default None : string containing the list of values of balancing we want to
            test, separated by a ";"
    :type balanced_parameters: str

    :param table_name: name of the table to create
    :type table_name:  str

    :param folds: optional, default 3 : number of folds used for the cross validation, ie. in how many parts we split
    our population
    :type folds: int

    :return: tuple of results: model, dot, best_params, cv_results:
      - model: is the model computed by the internal library Scikit-learn
      - dot: is the textual definition of the tree, based upon dot format: content parsable by graph viewers.
      - best_params: is the best combination of parameters found
      - cv_results: is the complete results of the cross validation algorithm, it's a table with the performance
                    (mean accuracy across folds and standard deviation) of every parameters combination that were
                    tested, ordered by the mean accuracy
    :rtype:  DecisionTreeClassifier, str, str, dict

    :raises IkatsInputTypeError: if an argument has an unexpected type
    :raises IkatsException: another error occurred
    """

    if table_name is None or re.match('^[a-zA-Z0-9-_]+$', table_name) is None:
        raise ValueError("Error in table name")

    if type(target_column_name) is not str:
        msg = "target_column_name has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(target_column_name).__name__))

    if type(identifier_column_name) is not str:
        msg = "identifier_column_name has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(identifier_column_name).__name__))

    if (type(folds) is not int) and (folds is not None):
        msg = "depth_parameters has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(folds).__name__))

    depth_list = []
    parsing_regexp = r'range\((?P<start>\d+),?(?P<end>\d+)?\)'

    if depth_parameters is None:
        depth_list.append(None)

    elif type(depth_parameters) is not str:
        msg = "depth_parameters has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(depth_parameters).__name__))

    else:
        if re.match(parsing_regexp, depth_parameters) is None:  # a list of values is given
            try:
                depth_list = [int(val) for val in depth_parameters.split(sep=';')]
            except ValueError:
                msg = "depth_parameters must be integers"
                raise IkatsException(msg)
        else:  # a range is given
            first_arg, second_arg = re.match(parsing_regexp, depth_parameters).groups()
            try:
                if second_arg is None:  # when the user inputs a single arg 'range(n)'
                    depth_list = range(int(first_arg))
                elif int(first_arg) > int(second_arg):  # otherwise i correct the case of a bad order of args
                    depth_list = range(int(second_arg), int(first_arg))
                elif int(second_arg) > int(first_arg):  # already ordered args
                    depth_list = range(int(first_arg), int(second_arg))
                else:
                    depth_list = [int(first_arg)]
            except ValueError:
                msg = "depth_parameters must be integers"
                raise IkatsException(msg)

    # Finally turning the result in a list on a format expected by GridSearchCV turning the 0 to a None
    depth_list = [i if i != 0 else None for i in depth_list]

    # Parsing from the input string to a list
    parsed_balancing = []

    # Defaults to no balancing
    if balanced_parameters is None:
        parsed_balancing.append(None)

    elif type(balanced_parameters) is not str:
        msg = "balanced_parameters has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(balanced_parameters).__name__))

    else:
        vals_balance = balanced_parameters.split(';')
        if len(vals_balance) > 2:
            raise IkatsException('Only two possible values for balance parameters')
        if all([v in ['True', 'False'] for v in vals_balance]):
            parsed_balancing = ['balanced' if val == 'True' else None for val in vals_balance]
        else:
            raise IkatsException('Only two possible values are True and False')

    if type(population) is str:
        obj_population = IkatsApi.table.read(population)
        LOGGER.info("Population used %s", population)
    elif type(population) is dict:
        obj_population = population
    else:
        msg = "population has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(population).__name__))

    parameters = {'max_depth': depth_list, 'class_weight': parsed_balancing}
    return fit_population_cv(population=obj_population,
                             target_column_name=target_column_name,
                             identifier_column_name=identifier_column_name,
                             table_name=table_name,
                             folds=folds,
                             parameters=parameters)


def fit_population_cv(population, target_column_name, identifier_column_name, table_name, folds=3,
                      parameters=None):
    """
    The internal wrapper of GridSearchCV fit algorithm for DecisionTree of scikit-learn.

    :param population: the population data whose functional type is 'table'
    :type population: dict

    :param target_column_name: the name of the attribute providing the class label of the observed subject.
       Must match one of the available population attributes.
    :type target_column_name: str

    :param identifier_column_name: the name of the attribute identifying each observed subject.
       Must match one of the available population attributes.
    :type identifier_column_name: str

    :param parameters: a dictionary containing the list of values to be tested that were parsed
    :type parameters: dict

    :param table_name: name of the table to create
    :type table_name:  str

    :param folds: number of folds used for the cross validation
    :type folds: int

    :raises IkatsException: error occurred.
    """

    if table_name is None or re.match('^[a-zA-Z0-9-_]+$', table_name) is None:
        raise ValueError("Error in table name")

    LOGGER.info("Starting Decision Tree CV Fit with scikit-learn")
    # To avoid having a dict as default arg of a function
    if parameters is None:
        parameters = {'max_depth': None, 'class_weight': False}

    try:
        desc_population = population.get('table_desc', None)

        LOGGER.info("with Population table_desc= %s", desc_population)

        # 1/ prepare the learning set
        #
        feature_vectors, target, class_names, column_names = split_population(population, target_column_name,
                                                                              identifier_column_name)

        # 2/ prepare the DecisionTree and CrossValidation procedure
        #
        mdl = tree.DecisionTreeClassifier()
        gcv = GridSearchCV(mdl, param_grid=parameters, cv=folds)
        gcv.fit(X=feature_vectors, y=target)

        LOGGER.info("   ... finished  fitting the Decision Tree CV to data")
        LOGGER.info(" - Exporting Decision Tree CV to dot format")
        dot_io = StringIO()
        tree.export_graphviz(gcv.best_estimator_, out_file=dot_io, feature_names=column_names,
                             class_names=class_names, filled=True, label='all')
        dot = dot_io.getvalue()
        LOGGER.info("  ... finished exporting the Decision Tree CV to dot format")

        # Formatting the result dictionary to an IKATS table
        formatted_results = _fill_table_cv_results(gcv.cv_results_)
        best_params = gcv.best_params_
        best_params['balancing'] = best_params.pop('class_weight')
        best_params['max_depth'] = 0 if best_params['max_depth'] is None else best_params['max_depth']
        best_params['balancing'] = best_params['balancing'] is not None
        formatted_best_params = json.dumps(best_params)
        LOGGER.info("... ended  Decision Tree CV Fit with scikit-learn")

        # Save the table
        description = "Result of Decision Tree Cross Validation operator"

        formatted_results['table_desc']['name'] = table_name
        formatted_results['table_desc']['desc'] = description
        IkatsApi.table.create(data=formatted_results)

        return gcv.best_estimator_, dot, formatted_best_params, table_name
    except IkatsException:
        raise
    except Exception:
        msg = "Unexpected error: fit_population(..., {}, {}, {})"
        raise IkatsException(msg.format(target_column_name,
                                        identifier_column_name,
                                        parameters))


def _fill_table_cv_results(dic_result):
    """
    Fill an ikats table structure with the results of the cv algorithm

    :param dic_result: output dictionary of GridSearchCV
    :type dic_result: dict

    :return: dict formatted as awaited by functional type table
    :rtype: dict
    """

    # Initializing table structure
    table = defaultdict(dict)

    # Filling title
    table['table_desc']['title'] = 'Cross-Validation results'

    # Preparing the cell content
    cell_content = [[i + 1 for i in range(len(dic_result['mean_fit_time']))],
                    dic_result['rank_test_score'].tolist(), [val['max_depth'] for val in dic_result['params']],
                    [val['class_weight'] is not None for val in dic_result['params']],
                    dic_result['mean_test_score'].tolist(), dic_result['std_test_score'].tolist()]
    # Turning into a numpy as array to perform transposition and sorting by rank
    cell_array = np.array(cell_content).transpose()
    cell_array = cell_array[cell_array[:, 1].argsort()]
    sorted_runs = cell_array[:, 0].astype(int).tolist()
    sorted_cells = cell_array[:, 1:].tolist()
    for cell in sorted_cells:
        cell[2] = str(cell[2] == 1)  # formatting the bool into a str that can be displayed by the table
        cell[1] = 0 if cell[1] is None else cell[1]  # formatting the None from the backend to the 0 of the front
    # Filling headers columns
    table['headers']['col'] = dict()
    table['headers']['col']['data'] = ['Score', 'rank', 'max_depth', 'balancing', 'mean_score', 'std_score']

    # Filling rows header
    table['headers']['row'] = dict()
    table['headers']['row']['data'] = ['# Run'] + sorted_runs

    # Filling cells
    table['content']['cells'] = sorted_cells

    return table
