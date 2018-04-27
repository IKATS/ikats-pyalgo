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
from ikats.algo.core.kmeans import fit_kmeans_internal, mds_representation_kmeans, format_kmeans
from ikats.core.library.exception import IkatsInputTypeError, IkatsInputContentError

"""
    The K-Means Algorithm designed for *pattern_groups" type
    ===================================================================

    This algorithm is designed for type : "pattern_groups" (output of *random_projection* for example).

    Here, the algorithm is divided into 4 parts (same process than kmeans.py):
        * extract the data from a "pattern_groups", and summarize (mean) each "pattern" into a single list of
          paa_values (same dim than ONE sequence)
        * the k-means algorithm (kmeans.fit_kmeans_internal() )
        * the MDS (multidimensional scaling) method (2 dimensional representation for the visualisation)
          (kmeans.mds_representation_kmeans() )
        * build an output ( kmeans.format_kmeans() )


    .. note::
        The inputs must be normalised !

    .. note::
        This code is very close to ikats.algo.core.kmeans.kmeans !
"""
# Help page : http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


# Logger definition
LOG_SK_KMEANS = logging.getLogger(__name__)


def extract_paa_from_pattern_groups(pattern_group):
    """
    Extract all the paa values, and the pattern names from a type "pattern_group"
    (random projection result for example).
    And summarize the patterns by their mean (mean of the paa values)

    :param pattern_group: a huge dict of sequenced data classified into patterns.
    :type pattern_group: dict

    :returns result: The combinaison key : value of the results :
    - key : the pattern name ("P1", "P2", etc...)
    - value: the MEAN of the pattern paa values (paa values of sequences for the current pattern)

    :rtype result: dict
    """

    LOG_SK_KMEANS.info("Starting paa_values extraction from pattern group type ...")

    # 1/ Input check
    #
    if type(pattern_group) is not dict:
        msg = "pattern_group has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(pattern_group).__name__))

    # 2/ Build the results (mean of the paa values of each pattern)
    #
    result = {}

    # For each patterns ("Pi") in pattern_names list
    for pattern in pattern_group["patterns"].keys():

        # SUM all the paa_values
        # ----------------------
        # need to get the very internal of the *pattern_group* dict (depth 7 to have access to the values !!)
        mean_paa = []

        # Number of sequences in the pattern (pattern size)
        num_seq = 1

        # All the TSuid from which the pattern is originated
        ts_names = pattern_group["patterns"][pattern]["locations"].keys()

        # For each TS
        for name in ts_names:
            # For each sequence of each pattern
            for i in range(len(pattern_group["patterns"][pattern]["locations"][name]['seq'])):

                # paa values of the current sequence
                paa = pattern_group["patterns"][pattern]["locations"][name]['seq'][i]["paa_value"]

                # sum the paa values by iteration
                if mean_paa == []:  # if mean_paa is empty (first iteration)
                    mean_paa = paa
                else:
                    # sum(mean_paa, paa of the current sequence)
                    mean_paa = [mean_paa + paa for mean_paa, paa in zip(mean_paa, paa)]
                    num_seq += 1

        # The pattern is summarized by the mean of these elements (paa values of each sequences)
        mean_paa = [x / num_seq for x in mean_paa]
        result.update({pattern: mean_paa})

    LOG_SK_KMEANS.info("... Extraction done.")
    return result


def fit_kmeans_pattern_group(pattern_group, n_cluster=3, random_state=None):
    """
    The IKATS wrapper published in catalog of fit algorithm for K-Means of scikit-learn.

    Perform the k-means algorithm on *pattern_group* type.

    .. note:: The random state is not specified.

    :param pattern_group: a huge dict of sequenced data classified into patterns.
    :type pattern_group: dict

    :param n_cluster: The number of clusters to form as well as the number of centroids to generate.
    :type n_cluster: int

    :param random_state: The seed used by the random number generator (if int).
    If None, the random number generator is the RandomState instance used by np.random.
    :type random_state: int or NoneType

    .. note :: specify `random_state` to make the results reproducible.

    :return tuple of results : model, km
        model: The K-Means model used.
        km: results summarized into a dict (see format_kmeans() ).
    :rtype : sklearn.cluster.k_means_.KMeans, dict

    """
    # 1/ Check of the inputs
    #

    if type(pattern_group) is not dict:
        msg = "pattern_group has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(pattern_group).__name__))

    if (type(random_state) is not int) and (random_state is not None):
        msg = "random_state has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(random_state).__name__))

    if type(n_cluster) is not int:
        msg = "n_cluster has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(n_cluster).__name__))

    elif n_cluster < 1:
        msg = "unexpected n_cluster ({} < 1)"
        raise IkatsInputContentError(msg.format(n_cluster))
    ##

    # 2/ Prepare the learning set
    #
    paa = extract_paa_from_pattern_groups(pattern_group=pattern_group)

    # 3/ Compute the model (centroids and classes)
    #
    model = fit_kmeans_internal(data=paa, n_cluster=n_cluster, random_state=random_state)

    # 4/ Compute the MDS (Multidimensional scaling) (purpose : 2 dimensional visualisation)
    # Note that the seed (random_state_mds) is the same
    _, pos, centers_pos = mds_representation_kmeans(fit_model=model, data=paa,
                                                    random_state_mds=random_state)

    # 5/ Prepare the outputs
    #
    k_means = format_kmeans(centers_pos=centers_pos, pos=pos, tsuid_list=list(paa.keys()), model=model)

    return model, k_means
