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

# produce a typical KMeans output
from sklearn.cluster import KMeans

from ikats.algo.core.kmeans.kmeans import fit_kmeans
from ikats.core.library.exception import IkatsInputTypeError


# 2 function to create pseudo SAX outputs

def content(paa):
    """
    Generate the content of the SAX dict for each TSUID (every time the same except "paa" field)

    :param paa: A list of data from a TS
    :type paa: list

    :return result: A typical SAX result for ONE TSUID
    :rtype result: dict
    """
    return {
        "paa": list(paa),
        "sax_breakpoints": [  # every time the same
            9.713135394220128,
            0.219751200339992,
            0.902016137770907,
            1.754928818168576
        ],
        "sax_string": "bdcdbaeeca"  # every time the same
    }


def gen_sax(ts_id):
    """
    Generate a pseudo SAX output.

    :param ts_id: Identifier of a case to generate (see content below for the structure)
    :type ts_id: int

    :return result: a typical SAX output
    :rtype result: dict
    """

    # 1/ Choose a case
    #
    if ts_id == 0:
        # simple test (dim 2) with obvious groups (4 data)
        paa = [[1, 2], [2, 3], [10, 20], [15, 30]]

    elif ts_id == 1:
        # Simple test with obvious groups (4 data) but in high dimension (10)
        paa = [
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            [10, 20, 10, 20, 10, 20, 10, 20, 10, 20],
            [20, 10, 20, 10, 20, 10, 20, 10, 20, 10]
        ]

    elif ts_id == 2:
        # simple test with obvious groups but with a lot of data (27) in low dimension (2)
        paa = [
            [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
            [100, 200], [101, 201], [102, 202], [103, 203], [104, 204], [105, 205], [106, 206],
            [107, 207], [108, 208],
            [-100, -200], [-101, -201], [-102, -202], [-103, -203], [-104, -204],
            [-105, -205], [-106, -206], [-107, -207], [-108, -208]
        ]
    else:
        raise NotImplementedError

    # For the functional ids
    alphabet = list("abcdefghijklmnopqrstuvwxyz1")

    # (pseudo) functional id : single letters
    func_ids = alphabet[0:len(paa)]

    # 2/ Build the result
    #
    result = {}

    # Build the dict (SAX output format) using *content* function (see above).
    for i in range(0, len(func_ids)):
        result[func_ids[i]] = content(paa[i])

    # Example of a result:
    # { tsuid1 : {
    #               "sax_string":"bdcdbaeeca",
    #               "paa":[...],
    #               "sax_breakpoints":[...]
    #               },
    # tsuid2 :{...
    # }}
    #

    return result


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
        sax = gen_sax(0)
        # a typical K-Means model (for 'type' comparison)
        ref_model = KMeans(n_clusters=2, random_state=random_state)

        # 1/ Simple test (type)
        #
        result = fit_kmeans(sax=sax, n_cluster=2, random_state=random_state)
        self.assertEqual(type(result[1]), dict, msg="Error, the output is not a dict.")
        self.assertTrue((type(result[0]), type(ref_model)),
                        msg="Error, the type of the model output is not *sklearn.cluster.k_means_* .")

    def test_trivial_kmeans(self):
        """
        Test the k-means algorithm with trivial data-sets.
        """
        # The same *random_state* is used for reproducible results
        random_state = 1

        # 1/ Small data sets (for gen_sax(0) or gen_sax(1))
        # -----------------------------------------------------------

        sax = gen_sax(0)
        result = fit_kmeans(sax=sax, n_cluster=2, random_state=random_state)

        # We want the clustering {(a,b) ; (c,d)}

        # the result of the clustering for group 1
        tsuid_group = result[1].get("C1").keys()  # ex: ['centroid_1', 'b', 'a']
        # condition : the current group is (a,b) or (c,d)
        condition = (("a" in tsuid_group) and ("b" in tsuid_group)) or (("c" in tsuid_group) and ("d" in tsuid_group))

        self.assertTrue(condition, msg="Error, the clustering is not efficient in trivial situations")

        # idem on group #2
        tsuid_group = result[1].get("C2").keys()  # ex :['centroid_2', 'd', 'c']
        condition = (("a" in tsuid_group) and ("b" in tsuid_group)) or (("c" in tsuid_group) and ("d" in tsuid_group))

        self.assertTrue(condition, msg="Error, the clustering is not efficient in trivial situations")

        # 2/ Test with a (huge) trivial data-set (for gen_sax(2))
        # -----------------------------------------------------------
        # We want the clustering {(a:i) ; (j:r) ; (s:1)}
        # {alphabet[0:9] ; alphabet[9:18] ; alphabet[18:27] }

        sax = gen_sax(2)
        n_cluster = 3
        alphabet = list("abcdefghijklmnopqrstuvwxyz1")

        result = fit_kmeans(sax=sax, n_cluster=n_cluster, random_state=random_state)

        # For each group
        for group in range(1, n_cluster):
            # List of the TSUID in the current group
            tsuid_group = result[1].get("C" + str(group)).keys()  # ex :['centroid_1', 'a', 'b',...,"i"]

            # The group is the same than expected ?
            condition = (
                all(x in tsuid_group for x in alphabet[0:9]) or
                all(x in tsuid_group for x in alphabet[9:18]) or
                all(x in tsuid_group for x in alphabet[18:27])
            )

            self.assertTrue(condition,
                            msg="Error, the clustering is not efficient in trivial situations (case n_cluster=3)")

    # noinspection PyTypeChecker
    def test_kmeans_robustness(self):
        """
         Robustness cases for the Ikats kmeans algorithm.
        """
        # The same *random_state* is used for reproducible results
        random_state = 1
        sax = gen_sax(1)

        # invalid sax type
        with self.assertRaises(IkatsInputTypeError, msg="Error, invalid sax type."):
            fit_kmeans(sax=[1, 2], n_cluster=2, random_state=random_state)
            fit_kmeans(sax={"a": [1, 2], "b": [1, 2]}, n_cluster=2, random_state=random_state)
            fit_kmeans(sax={"a": {"paa": [1, 2]}, "b": [2, 3]}, n_cluster=2, random_state=random_state)
            fit_kmeans(sax={"a": {"paa": [1, 2]}, "b": {"paa": [2, 3, 3]}}, n_cluster=2, random_state=random_state)
            fit_kmeans(sax="paa", n_cluster=2, random_state=random_state)

        # invalid n_cluster type
        with self.assertRaises(IkatsInputTypeError, msg="Error, invalid n_cluster type."):
            fit_kmeans(sax=sax, n_cluster="2", random_state=random_state)
            fit_kmeans(sax=sax, n_cluster=[2, 3, 4], random_state=random_state)

        # invalid n_cluster value
        with self.assertRaises(IkatsInputTypeError, msg="Error, invalid n_cluster value."):
            fit_kmeans(sax=sax, n_cluster=-2, random_state=random_state)
            # (referenced in the code as a TypeError)

        # invalid random_state type
        with self.assertRaises(IkatsInputTypeError, msg="Error, invalid random_state type"):
            fit_kmeans(sax=sax, n_cluster=2, random_state="random_state")
            fit_kmeans(sax=sax, n_cluster=2, random_state=[1, 3])
