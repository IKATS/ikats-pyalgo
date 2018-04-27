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
import numpy as np

from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances

from ikats.core.library.exception import IkatsException, IkatsInputTypeError

"""
    The K-Means Algorithm
    =====================

    This algorithm is designed for SAX (words) outputs : the TS are supposed to be resumed into words.

    Here, the algorithm is divided into 4 parts :
        * the "back transformation" [SAX word] -> [list of numbers] (pseudo-TS) (back_transformation_sax() )
        * the k-means algorithm (fit_kmeans_internal() )
        * the MDS (multidimensional scaling) method (2 dimensional representation for the visualisation)
          (mds_representation_kmeans() )
        * build an output ( format_kmeans() )



    .. note::
        This algorithm is designed for working on SAX (words) outputs

    .. note::
        If the input is normalized, the output of this algorithm is normalized too.

    .. note::
        Warning : all TS are named by their TSUID

    :example:
        .. code-block:: python
            # command

"""
# Help page : http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


# Logger definition
LOGGER = logging.getLogger(__name__)


def _back_transform_sax(sax_output):
    """
    Back transformation of the SAX transformation. Each letter of a word is changed into a number.

    .. note:: Here, the back transformation is just the paa !

    :param sax_output: The output of the SAX algorithm (the definition of PAA breakpoints values (paa :
    dimension n),the word (string), and the breakpoints interval (dimension length alphabet - 1))
    :type sax_output: dict

    :return: The "back transformation" of a SAX word into a list of number : key (tsuid) : value (paa)
    :rtype: dict
    """

    LOGGER.info("Starting back transformation of SAX words...")

    # 1/ Input checkout
    #
    if type(sax_output) is not dict:
        msg = "sax_output has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(sax_output).__name__))

    else:
        # List of all the TSUID (identifier of TS) of the SAX output
        tsuid_list = list(sax_output.keys())

        # each TSUID has its own "paa"
        for i in range(0, len(tsuid_list)):
            # Not a SAX output format
            if type(sax_output.get(tsuid_list[i])) is not dict:
                msg = "The TSUID {} is not a dict (type={})"
                raise IkatsException(msg.format(i, type(sax_output.get(tsuid_list[i]))))

            elif "paa" not in list(sax_output.get(tsuid_list[i]).keys()):
                msg = "The TSUID {} has no attribute *paa*"
                raise IkatsException(msg.format(i))

        # 2/ Build the results
        #

        result = {}

        # the SAX output : {TSUID : paa}
        for ts in range(0, len(tsuid_list)):
            # List of all the TSUID of the SAX output
            f_id = tsuid_list[ts]
            # paa (sub-dict)
            paa = sax_output.get(f_id).get("paa")

            result.update({f_id: paa})

        LOGGER.info("   ... back transformation finished")

        return result


def fit_kmeans_internal(data, n_cluster=3, random_state=None):
    """
    The internal wrapper of fit algorithm for k-means of scikit-learn.

    Perform the k-means algorithm on list of numbers (floats), not SAX words !

    :param data: The data-set : key (TS id), values (list of floats)
    :type data : dict

    :param n_cluster:The number of clusters to form as well as the number of centroids to generate.
    :type n_cluster: int

    :param random_state: The seed used by the random number generator (if int).
    If None, the random number generator is the RandomState instance used by np.random.
    :type random_state: int or NoneType

    .. note:: specify `random_state` to make the results reproducible.

    :return model: The KMeans model fitted on the input data-set.
    :rtype model: sklearn.cluster.k_means_.KMeans

    :raises IkatsException: error occurred.
    """
    LOGGER.info("Starting K-Means fit with scikit-learn ...")

    try:

        # Fit the algorithm
        model = KMeans(n_clusters=n_cluster, random_state=random_state)

        model.fit(list(data.values()))

        LOGGER.info("   ... finished fitting K-Means to data")
        LOGGER.info(" - Exporting results to sklearn.cluster.KMeans format")

        return model

    except IkatsException as ike:
        raise ike
    except Exception:
        msg = "Unexpected error: fit_kmeans_internal(..., {}, {}, {})"
        raise IkatsException(msg.format(data, n_cluster, random_state))


def mds_representation_kmeans(fit_model, data, random_state_mds=None):
    """
    Compute the MultiDimensional Scaling (MDS) transformation to the K-Means results.
    Purpose: a two dimensional representation of the clustering.

    :param fit_model: The K-Means fitted model.
    :type fit_model: sklearn.cluster.k_means_.KMeans

    :param data: The initial data-set (ex: the paa obtained after *back_transform_sax*
    :type data: table

    :param random_state_mds: The seed used by the random number generator (if int).
    If None, the random number generator is the RandomState instance used by np.random.
    :type random_state_mds: int or NoneType

    .. note:: specify `random_state_mds` to make the results reproducible.

    :return tuple of results :
        mds: multidimensional scaling algorithm result (2 dimensional representation for the visualisation)
        pos: the position (x,y) of the initial data-set after an mds transformation
        centers_pos: the position (x,y) of the centroids after an mds transformation
    :rtype : sklearn.manifold.mds.MDS, numpy array, numpy array
    """

    LOGGER.info("Starting MultiDimensional Scaling (MDS) transformation ...")

    try:

        # 1/ Concatenate data and centroids
        #
        # get the values of the data
        data = list(data.values())

        # data to transform (data-set + centroids) must be in the same array
        data = np.concatenate((data, fit_model.cluster_centers_))

        # 2/ Compute the euclidean distance (n*n table)
        #
        # Note that *dist* contains also the distance of each point to its centroid (*n_cluster* last rows/columns)
        dist = euclidean_distances(data)

        # 3/ Compute the MDS algorithm
        #
        # We can specify `random_state` to make the results reproducible.
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state_mds)

        # Compute
        pos = mds.fit_transform(dist)

        # 4/ sort the results (position of the data and position of the centroids in different tables)
        #

        # The positions of the centroids are in the *n_clusters* last lines of the table
        # (corresponding to the position of the centers in the MDS 2 dimensional representation)
        centers_pos = pos[range(len(pos) - fit_model.n_clusters, len(pos))]

        # the position of each point of the initial data-set (without the position of the centroids)
        pos = pos[range(0, len(pos) - fit_model.n_clusters)]

        LOGGER.info("   ... finished MDS transformation")

        return mds, pos, centers_pos

    except IkatsException:
        raise
    except Exception:
        msg = "Unexpected error: mds_representation_kmeans(..., {}, {}, {})"
        raise IkatsException(msg.format(fit_model, data, random_state_mds))


def format_kmeans(centers_pos, pos, tsuid_list, model):
    """
    Build the output of the algorithm (dict) according to the catalog.

    :param centers_pos: The position (x,y) of all the centroids after an mds transformation (list of pairs).
    :type centers_pos: list

    :param pos: The position of all the points (x, y) of the data-set after an mds transformation (n * 2 matrix).
    :type pos: list

    :param model: The K-Means model used.
    :type model: sklearn.cluster.k_means_.KMeans

    :param tsuid_list: List of all the TSUID
    :type tsuid_list: list

    :return: dict formatted as awaited (tsuid, mds new coords of each point, centroid coords)
    :rtype: dict
    """
    LOGGER.info("Exporting results to the K-Means format ...")

    # Example of the dict *result*:
    # {
    #   "C1": {
    #    "centroid": [x,y],
    #    "*tsuid1*": [x,y],
    #    "*tsuid2*": [x,y]
    #   },
    #   "C2" : ...
    # }

    # Initializing result structure
    result = dict()

    # For each cluster
    for center in range(1, model.n_clusters + 1):

        center_label = "C" + str(center)  # "C1", "C2",...

        result[center_label] = {}  # result["C1"] = ...

        # position of the centroid : {"C1": {"centroid": [x,y]}}
        #
        result[center_label]["centroid"] = list(centers_pos[center - 1])

        # position of each point of the data-set : {"C1": {"*tsuid1*": [x,y], ... }}
        #

        # For each points of the data-set:
        for index in range(0, len(tsuid_list)):
            # if the data is in the current cluster
            if model.labels_[index] == center - 1:
                # position of the data : "*tsuid1*": [x,y]
                result[center_label][tsuid_list[index]] = list(pos[index])

    return result


def fit_kmeans(sax, n_cluster=3, random_state=None):
    """
    The IKATS wrapper published in catalog of fit algorithm for K-Means of scikit-learn.

    Perform the k-means algorithm on sax words.

    .. note:: The random state is not specified

    :param sax: The output of the SAX algorithm over some TS
    :type sax: dict

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

    if type(sax) is not dict:
        msg = "sax has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(sax).__name__))

    if (type(random_state) is not int) and (random_state is not None):
        msg = "random_state has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(random_state).__name__))

    if type(n_cluster) is not int:
        msg = "n_cluster has unexpected type={}"
        raise IkatsInputTypeError(msg.format(type(n_cluster).__name__))

    elif n_cluster < 1:
        msg = "unexpected n_cluster ({} < 1)"
        raise IkatsInputTypeError(msg.format(n_cluster))

    ##

    # 2/ Prepare the learning set
    #
    paa = _back_transform_sax(sax_output=sax)

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
