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
from unittest import TestCase
import logging
# produce a typical KMeans output
from sklearn.cluster import KMeans
from ikats.algo.core.kmeans.kmeans_pattern_group import fit_kmeans_pattern_group, LOG_SK_KMEANS
from ikats.core.library.exception import IkatsInputTypeError, IkatsInputContentError

"""
Unit test for k-means algorithm on random_projection results (type "pattern_group")

Note that the result are not reproducible (due to the sklearn::KMeans gestion of *random_projection*)
"""

# Add logs to the unittest stdout
LOG_SK_KMEANS.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER.setFormatter(FORMATTER)
LOG_SK_KMEANS.addHandler(STREAM_HANDLER)


# 2 function to create pseudo random_projection outputs


def content(pattern_group, paa, num_pattern, tsuid):
    """
    Generate the content of a pseudo "random_projection" output dict for each TSUID
    (every time the same except "paa" field)

    :param pattern_group: the pattern_group to change.
    :type pattern_group: dict

    :param paa: A list of paa values for one TS
    :type paa: list

    :param num_pattern: a numero of pattern (if None, the function don't create a new "pattern")
    :type num_pattern: int or NoneType

    :param tsuid: a tsuid str (if None, the function don't create a new "TS")
    :type tsuid: str or NoneType

    :return pattern_group: A simplified random_projection pattern_group
    :rtype pattern_group: dict
    """

    # Example of result :
    # result = {
    #     'patterns': {
    #         "P"+str(num_pattern): {
    #             'locations': {
    #                 "ts1": {
    #                     'seq': [
    #                         {
    #                             'paa_value': list(paa),
    #                         },  # {...} other sequences
    #                     ]
    #                 },  # "ts2": {...} other TS
    #             }
    #
    #         },  # "P2": {...} other patterns
    #     }
    # }

    # We break the code into 3 cases to avoid errors with dict keys creations.

    # Test the existence of the "P*num_pattern*" field
    pattern_key = pattern_group["patterns"].keys()  # ex: ["P1", "P2"]

    # Case 1/ Add a pattern "P*num_pattern*"
    #

    # If the *num_pattern* choosen don't exist:
    if "P" + str(num_pattern) not in pattern_key:

        # Add an entire "pattern" field
        pattern_group["patterns"]["P" + str(num_pattern)] = {
            'locations': {
                tsuid: {
                    'seq': [
                        {
                            'paa_value': list(paa),
                        }
                    ]
                }
            }
        }

    # If the *num_pattern* choosen already exist:
    else:

        # Case 2/ Add a TS : tsuid (pattern not specified -> first pattern ("P1")
        #

        # Test the existence of the "*tsuid*" field
        tslist = list(pattern_group["patterns"]["P1"]["locations"].keys())

        # if the *tsuid* choosen don't exist:
        if tsuid not in tslist:

            # Add an entire "ts" field
            pattern_group["patterns"]["P" + str(num_pattern)]["locations"][str(tsuid)] = {
                'seq': [
                    {
                        'paa_value': list(paa),
                    }
                ]
            }

        else:
            # Case 3/ Add a sequence (paa) (pattern and TSUID not specified -> first pattern ("P1"), first TSUID ("ts1")
            #

            # If the *num_pattern* AND the *tsuid* choosen already exist:
            # Add an entry to the field "seq":
            pattern_group["patterns"]["P" + str(num_pattern)]["locations"][tsuid]["seq"].append({
                'paa_value': list(paa),
            })

    return pattern_group


def gen_pattern_group(ts_id):
    """
    Generate a pseudo random_projection output.

    :param ts_id: Identifier of a case to generate (see content below for the structure)
    :type ts_id: int

    :return pattern_group, result: tuple composed by:
        - pattern_group: a typical (simplified) random_projection output
        - result: the expected result : list of patterns name (ex: [ ["P1","P2"], ["P3","P4"] ]
                  for a clustering into 2 groups.
    :rtype pattern_group, result: dict and list of list
    """

    # The result : a typical (simplified) "pattern_group" type
    pattern_group = {'patterns': {}}

    # 1/ Choose a case
    #
    if ts_id == 0:
        # Case 1 : 2 obvious groups
        pattern_group = content(pattern_group=pattern_group, paa=[1, 2], num_pattern=1, tsuid="a")
        pattern_group = content(pattern_group=pattern_group, paa=[2, 3], num_pattern=1, tsuid="a")
        pattern_group = content(pattern_group=pattern_group, paa=[10, 20], num_pattern=2, tsuid="a")
        pattern_group = content(pattern_group=pattern_group, paa=[15, 30], num_pattern=2, tsuid="a")
        pattern_group = content(pattern_group=pattern_group, paa=[150, 300], num_pattern=3, tsuid="a")
        pattern_group = content(pattern_group=pattern_group, paa=[150, 300], num_pattern=3, tsuid="a")
        # One TS, 2 patterns of 2 sequences (4 data) in low dimension (2)

        # the expected result
        result = [["P1", "P2", "centroid"], ["P3", "centroid"]]

    elif ts_id == 1:
        # Case 2: same case in hight dim (10)
        pattern_group = content(pattern_group=pattern_group, paa=[1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                num_pattern=1, tsuid="a")
        pattern_group = content(pattern_group=pattern_group, paa=[2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
                                num_pattern=1, tsuid="a")
        pattern_group = content(pattern_group=pattern_group, paa=[10, 20, 10, 20, 10, 20, 10, 20, 10, 20],
                                num_pattern=2, tsuid="a")
        pattern_group = content(pattern_group=pattern_group, paa=[20, 10, 20, 10, 20, 10, 20, 10, 20, 10],
                                num_pattern=2, tsuid="a")
        pattern_group = content(pattern_group=pattern_group,
                                paa=[-200, -100, -200, -100, -200, -100, -200, -100, -200, -100], num_pattern=3,
                                tsuid="a")
        # 2 TS, 4 sequences in 4 patterns (4 data) but in high dimension (10)

        # the expected result
        result = [["P1", "P2", "centroid"], ["P3", "centroid"]]

    elif ts_id == 2:
        # Case 3: Pattern groups don't bring information (27 sequences for 27 patterns)
        paa = [
            [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
            [100, 200], [101, 201], [102, 202], [103, 203], [104, 204], [105, 205], [106, 206],
            [107, 207], [108, 208],
            [-100, -200], [-101, -201], [-102, -202], [-103, -203], [-104, -204],
            [-105, -205], [-106, -206], [-107, -207], [-108, -208]
        ]

        result = []

        for i in range(0, len(paa)):
            pattern_group = content(pattern_group=pattern_group, paa=paa[i], num_pattern=i, tsuid="a")
            # 27 TS, 27 sequences for 27 patterns (3 obvious groups)

            # expected results : "Pi"
            result.append("P" + str(i))

        # Obvious groups : [ paa[0:10] , paa[10:19], paa[19:28]
        result = [result[0:10], result[10:19], result[19:28]]

    else:
        raise NotImplementedError

    return pattern_group, result


class TestKMeans(TestCase):
    """
    Test of K-Means algorithm
    """

    def test_kmeans(self):
        """
        Test the 'type' of the results.
        """

        # The same *random_state* is used for reproducible results
        random_state = 1
        pattern_group, _ = gen_pattern_group(0)
        # a typical K-Means model (for 'type' comparison)
        ref_model = KMeans(n_clusters=2, random_state=random_state)

        # 1/ Simple test (type)
        #
        result = fit_kmeans_pattern_group(pattern_group=pattern_group, n_cluster=2, random_state=random_state)
        self.assertEqual(type(result[1]), dict, msg="Error, the output is not a dict.")
        self.assertTrue((type(result[0]), type(ref_model)),
                        msg="Error, the type of the model output is not *sklearn.cluster.k_means_* .")

    def test_trivial_kmeans(self):
        """
        Test the k-means algorithm with trivial data-sets.
        """
        # The same *random_state* is used for reproducible results
        random_state = 1

        # 1/ Small data sets (for pattern_group(0) or pattern_group(1))
        # -----------------------------------------------------------

        pattern_group, expected\
            = gen_pattern_group(0)

        result = fit_kmeans_pattern_group(pattern_group=pattern_group, n_cluster=2, random_state=random_state)

        # We want the clustering {(a,b) ; (c,d)}

        # the result of the clustering for group 1
        tsuid_group = sorted(result[1].get("C1").keys())  # ['P1','P2','centroid']

        # test the exact matching between expected result and result of the algorithm.
        # Note that the algo can switch the results in some case (pb of reproducibility)
        condition = (tsuid_group == expected[0]) or (tsuid_group == expected[1])

        LOG_SK_KMEANS.info("expected=%s", expected[0])
        LOG_SK_KMEANS.info("tsuid=%s", tsuid_group)

        self.assertTrue(condition, msg="Error, the clustering is not efficient in trivial situations")

        # idem on group #2
        tsuid_group = sorted(result[1].get("C2").keys())  # ['P3', 'centroid']
        condition = (tsuid_group == expected[0]) or (tsuid_group == expected[1])

        self.assertTrue(condition, msg="Error, the clustering is not efficient in trivial situations")

        # 2/ Test with a (huge) trivial data-set (for gen_pattern_group(2))
        # -----------------------------------------------------------
        # We want the clustering [expected[0:10] , expected[10:19], expected[19:28]]

        pattern_group, expected = gen_pattern_group(2)
        n_cluster = 3

        result = fit_kmeans_pattern_group(pattern_group=pattern_group, n_cluster=n_cluster, random_state=random_state)

        # For each group
        for group in range(1, n_cluster):
            # List of the TSUID in the current group
            tsuid_group = list(result[1].get("C" + str(group)).keys())  # ex :['centroid', expected[0:10] ]

            # The group is the same than expected ?
            condition = (all(x in tsuid_group for x in expected[0:10]) or
                         all(x in tsuid_group for x in expected[10:19]) or
                         all(x in tsuid_group for x in expected[19:28]))

            self.assertTrue(condition,
                            msg="Error, the clustering is not efficient in trivial situations (case n_cluster=3)")

    # noinspection PyTypeChecker
    def test_kmeans_robustness(self):
        """
         Robustness cases for the Ikats kmeans algorithm.
        """
        # The same *random_state* is used for reproducible results
        random_state = 1
        pattern_group, _ = gen_pattern_group(1)

        # invalid sax type
        with self.assertRaises(IkatsInputTypeError, msg="Error, invalid sax type."):
            fit_kmeans_pattern_group(pattern_group=[1, 2], n_cluster=2, random_state=random_state)
            fit_kmeans_pattern_group(pattern_group={"a": [1, 2], "b": [1, 2]}, n_cluster=2, random_state=random_state)
            fit_kmeans_pattern_group(pattern_group={"a": {"paa": [1, 2]}, "b": [2, 3]}, n_cluster=2,
                                     random_state=random_state)
            fit_kmeans_pattern_group(pattern_group={"a": {"paa": [1, 2]}, "b": {"paa": [2, 3, 3]}}, n_cluster=2,
                                     random_state=random_state)
            fit_kmeans_pattern_group(pattern_group="paa", n_cluster=2, random_state=random_state)

        # invalid n_cluster type
        with self.assertRaises(IkatsInputTypeError, msg="Error, invalid n_cluster type."):
            fit_kmeans_pattern_group(pattern_group=pattern_group, n_cluster="2", random_state=random_state)
            fit_kmeans_pattern_group(pattern_group=pattern_group, n_cluster=[2, 3, 4], random_state=random_state)

        # invalid n_cluster value
        with self.assertRaises(IkatsInputContentError, msg="Error, invalid n_cluster value."):
            fit_kmeans_pattern_group(pattern_group=pattern_group, n_cluster=-2, random_state=random_state)

        # invalid random_state type
        with self.assertRaises(IkatsInputTypeError, msg="Error, invalid random_state type"):
            fit_kmeans_pattern_group(pattern_group=pattern_group, n_cluster=2, random_state="random_state")
            fit_kmeans_pattern_group(pattern_group=pattern_group, n_cluster=2, random_state=[1, 3])
