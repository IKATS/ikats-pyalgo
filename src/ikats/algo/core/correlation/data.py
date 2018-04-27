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
from abc import abstractmethod
import time

from ikats.core.library.exception import IkatsException


# Use this module as the data provider for the correlation computing results.
#   - Created for the correlation.loop module.
#
# See the class CorrelationDataset for the ikats functional type 'correlation_dataset'.
# See the class CorrelationBycontext for the ikats functional type 'correlation_by_context'.

def get_triangular_matrix(dim, default_value_diag, default_value_other):
    """
    Gets standard matrix for the Correlation: triangular matrix encoded with lists.

    :param dim: size of the matrix
    :type dim: int
    :param default_value_diag: value initialized on the diagonal
    :type default_value_diag: number or str or None, or any type
    :param default_value_other: value initialized except on the diagonal
    :type default_value_other: number or str or None, or any type
    :return: encoded triangular matrix
    :rtype: list of lists
    """
    return [[default_value_diag] + [default_value_other] * (i - 1) for i in range(dim, 0, -1)]


def is_triangular_matrix(matrix, expected_dim):
    """
    Checks that matrix is actually triangular and well-encoded.

    :param matrix: the checked matrix
    :type matrix: list of list
    :param expected_dim: the expected length of the matrix == number of rows == diagonal size
    :type expected_dim:
    """
    if type(matrix) is not list:
        return False

    current_dim = expected_dim
    for row in matrix:
        if (type(row) is not list) or (len(row) != current_dim):
            return False
        current_dim = current_dim - 1

    # finally testing that number of rows is expected_dim:
    return current_dim == 0


class CorrelationData(object):
    """
    The CorrelationData is a correlation result that can be saved as JSON in IKATS database:
    the abstract method get_json_friendly_dict() is encoding the python object compatible with
    json library.

    Note: this abstract class is the superclass of CorrelationsBycontext and CorrelationDataset.
    """

    @classmethod
    def get_functional_type(cls):
        """
        The getter of the functional type associated to this class.

        :param cls: the class
        :type cls: class
        :return: the functional type, if defined. Default value is None.
        :rtype: str or None
        """
        return None

    @classmethod
    def generate_id(cls):
        """
        Generates a unique identifier <classname> + <timestamp> .

        This can be used in object constructor, in order to generate unique identifier of object,
        beginning with the classname.

        :param cls:
        :type cls:
        :return: the generated unique ID
        :rtype: str
        """
        return cls.__name__ + "{}".format(time.time())

    @abstractmethod
    def get_json_friendly_dict(self):
        """
        Gets the data content
        :return: the encoded content which is json-friendly, ie compliant with the json library
        :rtype: dict
        """
        pass


class CorrelationsBycontext(CorrelationData):
    """
    This class helps to generate the content of functional type CORRELATION_BY_CONTEXT:
    this data is explored as a second level from the CorrelationDataset:
    used in order to view for specific (var1, var2) the correlation curve(s) according to the contexts.

    For example with Airbus: the correlations curves for the selected variables (GS / WS1 )
    for the selected contexts: flight numbers 10 to  50.

    About the JSON content: see *self.get_json_friendly_dict* documentation.
    """
    CORRELATION_BY_CONTEXT = "correlation_by_context"

    @classmethod
    def get_functional_type(cls):
        """
        Gets the functional type of content handled by this class
        :param cls:
        :type cls:
        :return: the functional type of content handled by this class
        :rtype: str
        """
        return cls.CORRELATION_BY_CONTEXT

    def __init__(self):
        """
        Constructor of CorrelationsBycontext
        """
        self.__json_friendly_dict = {"variables": [None, None],
                                     "x_value": {"desc": {"label": None},
                                                 "data": None},
                                     "y_values": [],
                                     "ts_lists": []}

    def set_variables(self, variable_one, variable_two):
        """
        Sets the pair of variables.

        :param variable_one: the first variable name or label
        :type variable_one: str or int
        :param variable_two: the second variable name or label
        :type variable_two: str or int
        """
        self.__json_friendly_dict["variables"][0] = variable_one
        self.__json_friendly_dict["variables"][1] = variable_two

    def set_contexts(self, desc_x_label, ctx_values):
        """
        Sets the contexts definition.

        :param desc_x_label: label describing the context
        :type desc_x_label: str
        :param ctx_values: list of the context values
        :type ctx_values: list
        """
        self.__json_friendly_dict["x_value"]["desc"] = {'label': desc_x_label}
        self.__json_friendly_dict["x_value"]["data"] = ctx_values

    def add_curve(self, desc_y_label, y_values):
        """
        Adds a new serie of values, ordered by context.

        :param desc_y_label:
        :type desc_y_label:
        :param y_values:
        :type y_values:
        """
        curv = {"desc": {'label': desc_y_label},
                "data": y_values}
        self.__json_friendly_dict["y_values"].append(curv)

    def add_ts_links(self, tsuid_pairs):
        """
        Adds each tsuid pairs for each context: [ <tsuid1>, <tsuid2> ] where:

          - <tsuid1> stands for the first variable definition
          - <tsuid2> stands for the second variable definition

        Required:
          - the pairs are ordered by context set by *self.set_contexts*,
          - each pair is ordered according to the variables defined by *self.set_variables*.

        :param tsuid_pairs: list of pairs of tsuids
        :type tsuid_pairs: list of list of str
        """
        self.__json_friendly_dict["ts_lists"].extend(tsuid_pairs)

    def get_json_friendly_dict(self):
        """
         Gets the JSON encoding for this data:

            Example for Pearson:

                {
                    "variables": [ "var1", "var2" ],

                    "x_value": { "desc": { "label": "FlightIdentifier"},
                                 "data": [10,20,30,40]
                               },

                    "y_values": [ { "desc": { "label": "Pearson correlation"},
                                    "data": [correl10, correl20, correl30, correl40]
                                  }
                                ],

                    "ts_lists": [ [tsuid_var1_10, tsuid_var2_10],
                                  [tsuid_var1_20, tsuid_var2_20],
                                  [tsuidx_var1_30, tsuidy_var2_30],
                                  [tsuidx_var1_40, tsuidy_var2_40]
                                ]
                  }

        :return: the CorrelationsBycontext as a dict json-friendly
        :rtype: dict
        """
        if (self.__json_friendly_dict["variables"][0] is None) or \
                (self.__json_friendly_dict["variables"][1] is None):
            msg = "Inconsistency error: section variables is incomplete: {}"
            raise IkatsException(msg.format(self.__json_friendly_dict["variables"]))

        size_x = len(self.__json_friendly_dict["x_value"]["data"])
        for index, y_value in enumerate(self.__json_friendly_dict["y_values"]):
            if len(y_value["data"]) != size_x:
                msg = "Inconsistency: len(y_values[{}]['data']) != len(x_value) with content={}"
                raise IkatsException(msg.format(index, self.__json_friendly_dict))

        if len(self.__json_friendly_dict["ts_lists"]) != size_x:
            msg = "Inconsistency: len(ts_lists) != len(x_value) with content={}"
            raise IkatsException(msg.format(self.__json_friendly_dict))

        return self.__json_friendly_dict


class CorrelationDataset(CorrelationData):
    """
    This class helps to generate the content of functional type CORRELATION_DATASET.

    About the JSON content: see *self.get_json_friendly_dict* documentation.
    """
    CORRELATION_DATASET = "correlation_dataset"

    @classmethod
    def get_functional_type(cls):
        """
        Gets the functional type of content handled by this class
        :param cls:
        :type cls:
        :return: the functional type of content handled by this class
        :rtype: str
        """
        return cls.CORRELATION_DATASET

    def __init__(self):
        self.__json_friendly_dict = {"matrices": []}
        self.__json_friendly_dict['variables'] = None
        self.__json_friendly_dict['context'] = None

        self.__id = CorrelationDataset.generate_id()

    def get_id(self):
        """
        Gets the generated ID of self: CorrelationDataset + <timestamp>.

        This ID may be saved in the database in order to retrieve all the results
        attached to the CorrelationDataset.
        :return: the generated ID
        :rtype: str
        """
        return self.__id

    def set_variables(self, labels):
        """
        Sets the list of variables: defines labels[i] defines the label associated to the matrix index i.

        :param labels: list of labels
        :type labels: list
        """
        if type(labels) is not list:
            raise IkatsException("Inconsistency error detected: variables should be a list")
        self.__json_friendly_dict['variables'] = labels

    def set_contexts(self, contexts, meta_identifier):
        """
        Sets the contexts:
           - defines the number of contexts
           - defines the identifier

          .. note:: in this version, only the contexts length is used. It may be completed later.

        :param contexts: ordered list of values of contexts: context[c] is the context for index c
        :type contexts: list
        :param meta_identifier: name of the metadata used as the context descriptor.
        :type meta_identifier: str
        """
        if type(contexts) is not list:
            msg = "Inconsistency error detected: context should be a list: got value={}"

            raise IkatsException(msg.format(contexts))

        self.__json_friendly_dict['context'] = {"identifier": meta_identifier,
                                                "number_of_contexts": len(contexts),
                                                "label": meta_identifier}

    def add_matrix(self, matrix, desc_label):
        """
        Adds a new matrix with its description.

          .. note:: the matrix should be structured as triangular matrix structured like
                    *get_triangular_matrix* result,
                    with same size as variables defined by *self.set_variables*:
                    consistency check is performed by get_json_friendly_dict().

        :param matrix: the added triangular matrix
        :type matrix: dict
        :param desc_label: the description label for that matrix
        :type desc_label: str
        """
        self.__json_friendly_dict["matrices"].append({"desc": {"label": desc_label},
                                                      "data": matrix})

    def add_rid_matrix(self, matrix):
        """
        Adds the RID matrix as specified in front/back interface:
        the matrix of links (RID: Result IDentifier) towards to the CorrelationByContext JSON content
        saved in the database.

          .. note:: the matrix should be structured as triangular matrix, like
                    *get_triangular_matrix* result,
                    with same size as variables defined by *self.set_variables*:
                    consistency check is performed by get_json_friendly_dict().

        :param matrix: the added triangular matrix
        :type matrix: dict
        """
        self.__json_friendly_dict["matrices"].append({"desc": {"is_rid_link": True},
                                                      "data": matrix})

    def get_json_friendly_dict(self):
        """
        Gets the json-friendly dict coding the structure saved in process_data resources
        for the CorrelationDataset.FUNCTIONAL_TYPE.
        Error is raised in case of consistency check failure.

        Example for Pearson:

          {
           "variables":['variable1','variable2','variable3'],

           "context": { "identifier": "FlightIdentifier",
                        "number_of_contexts": 10,
                        "label":  "Flight number"
                    },
           "matrices": [
                         {
                            "desc": { "label": "Mean Correlation" },
                            "data": [
                                     [1,x,x],
                                     [1,x],
                                     [1]
                                    ],
                         },
                         {
                            "desc": { "label": "Variance" },
                            "data": [
                                     [var1,var2,var3],
                                     [var4,var5],
                                     [var6]
                                    ],
                         },
                         {
                            "desc": { "is_rid_link": true },

                            "data": [
                                     [@rid,@rid,@rid],
                                     [@rid,@rid],
                                     [@rid]
                                   ],
                         }
                      ]
           }

        :return: the CorrelationDataset as a dict json-friendly
        :rtype: dict
        """
        if self.__json_friendly_dict['variables'] is None:
            raise IkatsException("Inconsistency error detected: undefined variables section")

        if self.__json_friendly_dict['context'] is None:
            raise IkatsException("Inconsistency error detected: undefined context section")

        if len(self.__json_friendly_dict['matrices']) == 0:
            raise IkatsException("Inconsistency error detected: matrices section is empty")

        checked_size = len(self.__json_friendly_dict['variables'])

        for index, obj_mat in enumerate(self.__json_friendly_dict['matrices']):

            if not is_triangular_matrix(matrix=obj_mat['data'], expected_dim=checked_size):
                msg = "Inconsistency detected for len(variables)={} with matrices[{}]: desc={}"
                raise IkatsException(msg.format(checked_size, index, obj_mat['desc']))

        return self.__json_friendly_dict
