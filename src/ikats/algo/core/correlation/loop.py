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
from _collections import defaultdict
from functools import lru_cache
import itertools
import json
import logging
from math import ceil, floor

from ikats.core.library.exception import IkatsException, IkatsInputContentError
from ikats.core.library.spark import ScManager
from ikats.core.resource.api import IkatsApi
from ikats.core.resource.interface import ResourceLocator
from natsort.natsort import natsorted
from scipy.stats.stats import pearsonr

from ikats.algo.core.correlation.data import CorrelationDataset, \
    CorrelationsBycontext, get_triangular_matrix
import numpy as np

"""
This module integrates the 'correlation loop' algorithm, industrialized:
  - see the top level function correlation_ts_loop() below, configured into IKATS catalogue.
"""

# LOGGER:
LOGGER = logging.getLogger(__name__)

PEARSON = "Pearson"
CORRELATION_METHODS = [PEARSON]

# 2 is chosen because it is impossible to compute this value
# => used initializing
NOT_YET_COMPUTED_CORR = 2

# Value encoded in JSON result for np.nan values
NAN = "NaN"


def the_pearson(ts_1, ts_2):
    """
    This function performs the Pearson correlation between two time series
    If a time series has a variance = 0, the corr value is equal to NaN
    :param ts_1: first time series
    :type ts_1: np.array
    :param ts_2: second time series
    :type ts_2: np.array
    :return: Pierson correlation of x and ts_2
    :rtype: float or NaN
    """
    # Ugly and critical: time series should be resampled
    the_min = min(ts_1.size, ts_2.size)
    return pearsonr(ts_1[:the_min], ts_2[:the_min])[0]


CORRELATION_FUNCTIONS = {PEARSON: the_pearson}


class ConfigCorrelationLoop(object):
    """
    Instance of ConfigCorrelationLoop groups some internal parameters:
      - initialized for instance according to the cluster for the Spark/Yarn computing.
      - or precision of JSON results according the PostGres database capacity ...
    """

    def __init__(self, the_num_partitions, the_point_cache_size, the_digits_number):
        """
        Constructor
        :param the_num_partitions: recommended number of spark partitions on the cluster
        :type the_num_partitions: int
        :param the_point_cache_size: the maximum number of points memorized into the timeseries cache:
                                     one cache is managed by one spark task.
                                     The cache may prevent the task to reload several times the same
                                     timeseries, while computing one correlation matrix chunk
        :type the_point_cache_size: int
        :param the_digits_number: number of significant digits saved in the json results, for the float numbers
            This is useful to reduce the volume of results in the PostGres database
        :type the_digits_number: int
        """
        self.num_partitions = the_num_partitions
        self.the_point_cache_size = the_point_cache_size
        self.the_digits_number = the_digits_number


def _encode_corr_json(correlation):
    """
    Encodes the computed correlation for the json content:
      - About handling the Nan value: it has been decided with front story to encode
        the specific NaN into a string
      - None is correctly translated into JSon null value
    :param correlation: the computed correlation
    :type correlation: float64 (including NaN) or None
    """
    if correlation is None:
        return None
    elif np.isnan(correlation) or np.isinf(correlation):
        return NAN
    else:
        return correlation


def _round(correlation, ndigits):
    """
    Round the float value according to the maximum number of digits:
      - this preserve a lot of memory, saving the JSON data in the database

    Note: method robust to str or None values, which are unchanged
    :param correlation: the computed correlation
    :type correlation: float or str or None
    :param ndigits: the maximum number of digits taken into account
    :type ndigits: int
    """
    # this tests is applicable to float or numpy.float64 (float64 extends float)
    if isinstance(correlation, float):
        return round(float(correlation), ndigits)
    else:
        return correlation


def _initialize_config_from_meta(ts_metadata_dict, context_meta, variable_meta):
    """
    Prepares the correlation loop configuration from the uploaded metadata of selected timeseries.

    :param ts_metadata_dict: uploaded metadata
    :type ts_metadata_dict: dict
    :param context_meta: name of metadata providing the context.

    Example with Airbus datasets: "FlightIdentifier"

    :type context_meta: str
    :param variable_meta: name of the metadata providing the variable name: variables are sorted by alphanumeric order.

    Example with Airbus datasets: "metric"

    :type variable_meta: str
    :return: computed config is multiple result:
      - config_corr_loop: list of ( <context index>, [ (<var index 1> , <tsuid 1>), ..., (<var index N> , <tsuid N>) ] )
      - contexts: ordered list of contexts:  <context> = contexts[<context_index>]
      - variables: ordered list of variables: <variable name> = variables[ <var index> ]
    :rtype: list, list list
    :raise exception: IkatsInputContentError when an inconsistency cancels the correlations computing
    """
    ts_metadata_accepted = defaultdict(dict)
    ts_variables_accepted_set = set()
    for tsuid, meta in ts_metadata_dict.items():
        if context_meta not in meta:
            LOGGER.info("- Ignored: TS without context (meta %s): %s", context_meta, tsuid)
        elif variable_meta not in meta:
            LOGGER.info("- Ignored: TS without defined variable (meta %s): %s", variable_meta, tsuid)
        else:
            context_value = meta[context_meta]
            variable_name = meta[variable_meta]
            if variable_name in ts_metadata_accepted[context_value]:
                msg = "Inconsistency: context={} variable={} should match 1 TS: got at least 2 TS {} {}"
                raise IkatsInputContentError(msg.format(tsuid,
                                                        ts_metadata_accepted[context_value][variable_name]))

            ts_metadata_accepted[context_value][variable_name] = tsuid

    def ignore_unique_ts(the_context, tsuid_by_var):
        """
        - removes context with one single ts => useless to compute correlation
        - or else completes the set ts_variables_accepted_set

        :param the_context:
        :param tsuid_by_var:
        :return:
        """

        if len(tsuid_by_var) == 1:
            LOGGER.info("- Ignored: unique TS in context %s=%s: %s", context_meta,
                        the_context,
                        list(tsuid_by_var.values())[0])
        else:
            for var in tsuid_by_var:
                ts_variables_accepted_set.add(var)
        return len(tsuid_by_var) == 1

    ts_metadata_accepted = {ctx: tsuid_by_var for ctx, tsuid_by_var in ts_metadata_accepted.items()
                            if ignore_unique_ts(ctx, tsuid_by_var) == False}

    # provides translation indexes => value on contexts
    contexts = natsorted(ts_metadata_accepted.keys())
    # provides translation indexes => value on variables
    variables = natsorted(ts_variables_accepted_set)

    # computes the corr_loop_config
    # ( <context index>, [ (<var index 1> , <tsuid 1>), ..., (<var index N> , <tsuid N>) ] )
    #
    # Note: sorted( [ (2, "TS2"), (1, "TS1"), (0, "TS0"), ] )
    #       returns [(0, 'TS0'), (1, 'TS1'), (2, 'TS2')]
    #
    corr_loop_config = [(contexts.index(ctx), sorted([(variables.index(var), tsuid)
                                                      for var, tsuid in tsuid_by_var.items()]))
                        for ctx, tsuid_by_var in ts_metadata_accepted.items()]

    return corr_loop_config, contexts, variables


def _spark_combine_pairs(context, variables, nb_corr_matrix_blocks):
    """
    Prepare the lists of comparable pairs of variables, for specified context.
    :param context: the context where pairs of variable are evaluated
    :type context: int or str
    :param variables: the list of variables
    :type variables: list
    :param nb_corr_matrix_blocks: the number of matrix blocks: when superior to 1: this will
      split the list pairs in several lists. This aims at improving the spark distribution over the cluster
    :type nb_corr_matrix_blocks: int
    """
    # build pairs of variable:
    # ex: variables =  [ "A", "B", "C" ]
    #     => pairs = [ ( "A", "A"), ( "A", "B"), ( "A", "C"), ( "B", "B"), ("B", "C"), ("C", "C") ]
    pairs = list(itertools.combinations_with_replacement(variables, 2))

    nb_pairs = len(pairs)

    # the driver has demanded to split the work into correlation matrix_blocks for a fixed context
    corr_matrix_block_size = max(1, floor(nb_pairs / nb_corr_matrix_blocks))

    # split the list pairs into list of lists: partition of correlation matrix_blocks,
    # with usual size corr_matrix_block_size
    #
    # ex: with corr_matrix_block_size =2, context=1
    #  [ ( 1, [ ( "A", "A"), ( "A", "B"), ( "A", "C") ] ),
    #    ( 1, [ ( "B", "B"), ( "B", "C"), ( "C", "C") ] ) ]
    return [(context, pairs[i:i + corr_matrix_block_size]) for i in range(0, nb_pairs, corr_matrix_block_size)]


def _spark_correlate_pairs(context, var_pairs, corr_method, ts_cache_size):
    """
    Computes the correlations of variables inside one observed context.

    :param context: the context index.
    :type context: int
    :param var_pairs: list of <pair 1_1>, <pair 1_2>, <pair 1_3>, ..., <pair M_N>

       where <pair X_Y> is:
         ((<var X index>, <tsuid X> ), (<var Y index>, <tsuid Y>))

    :type var_pairs: list of tuples
    :param corr_method: label of correlation method among CORRELATION_METHODS
    :type corr_method: str
    :param ts_cache_size: maximum number of loaded timeseries in one task
    :type ts_cache_size: int
    :return: list of ( (<Var X index>, <Var Y index>), <correlation X Y>) where
       <correlation X Y> is (<context>, (<tsuid X>, <tsuid Y>), correlation)
    :rtype: list of tuples
    """

    @lru_cache(maxsize=ts_cache_size)
    def read_ts_values(tsuid):
        """
        Reads a timeseries and return its data
        :param tsuid:
        :return:
        """
        return IkatsApi.ts.read(tsuid)[0][:, 1]

    # tested in algo pre-requisites
    the_func_method = CORRELATION_FUNCTIONS[corr_method]

    # List to return
    ret = []

    for (var_a, tsuid_a), (var_b, tsuid_b) in var_pairs:
        # Load time series
        values_a = read_ts_values(tsuid_a)
        values_b = read_ts_values(tsuid_b)

        corr = the_func_method(values_a, values_b)

        ret.append(((var_a, var_b),
                    (context, (tsuid_a, tsuid_b), corr)))

    # here: we tested that read_ts_values.cache_clear()
    # was not efficient
    return ret


def _spark_build_corrs_by_context(variables,
                                  agg_ctx_ts_corr,
                                  desc_context,
                                  sorted_variables,
                                  sorted_contexts,
                                  corr_method,
                                  parent_id,
                                  ndigits):
    """
    Saves the aggregated correlation information grouped for one pair of variables:
    saved as JSON processdata in the IKATS database.

    And then, reduces this correlation information provided into:

      - the mean of computed correlations
      - the variance of computed correlations
      - the ID of saved processdata (or the full content)

      .. note::
        - the saved information is the low-level data: correlations by context for a fixed pair of variables.
          See loop.data.CorrelationsByContext.
        - the mean, variance and ID are piece of information in the high-level data: correlation matrices.
          See loop.data.CorrelationsDataset.

    :param variables: the pair of variable indexes (X, Y).
    :type variables: list of int
    :param agg_ctx_ts_corr: the correlation data for (X, Y): this is a list of tuples like
      (<context index>, (tsuid_X, tsuid_Y), <correlation result> )
    :type agg_ctx_ts_corr: list of tuples
    :param desc_context: description of the context: this text is also saved in the JSON
    :type desc_context: str
    :param sorted_variables: sorted variables: the indexes are matched by variables definitions
    :type sorted_variables: list
    :param sorted_contexts: sorted contexts: the indexes are matched by context indexes from agg_ctx_ts_corr
    :type sorted_contexts: list
    :param corr_method: the label selecting the correlation method: one of values from CORRELATION_METHODS
    :type corr_method: str
    :param parent_id: this ID will be passed as the processID required to save the processdata.
    :type parent_id: str
    :param ndigits: number of digits saved in the json for the float numbers: before saving the json,
                    floats are rounded.
    :type ndigits: int
    :return: ( [ <var X index>, <var Y index>], <JSON result ID>, <Mean correlation>, <Var correlation> )
    :rtype: tuple
    """
    # INPUT: ((<var X index>, <var Y index>), list of tuples:
    #                                  (<context index>, (tsuid_X, tsuid_Y), <correlation result> )
    #                )
    #
    # OUTPUT: ( [<var X index>, <var Y index>], <OID/object>, <Mean correlation>, <Var correlation>
    #                )
    # PROCESS: aggregates by key=(<var X index>, <var Y index>) the correlation information profiles,
    #          enhanced with tsuid pairs
    #
    if corr_method == PEARSON:

        # sort the tuples by context index: required to have ordered series
        agg_ctx_ts_corr.sort(key=lambda x: x[0])

        # result object will be completed below,
        # and finally will produce the JSON content
        res_obj = CorrelationsBycontext()

        # extract the variable indexes from the key
        var_name_1 = sorted_variables[int(variables[0])]
        var_name_2 = sorted_variables[int(variables[1])]

        # 1/ sets the pair of variables
        res_obj.set_variables(var_name_1, var_name_2)

        # defined correlations evaluated below in order to compute the mean/variance
        defined_correlations = []

        # Complete defined cells with undefined content: when one context has some missing values
        # - complete contexts: provided by sorted_contexts
        complete_correlations = [None] * len(sorted_contexts)
        complete_links = [None] * len(sorted_contexts)

        for ctx, ts_pair, corr in agg_ctx_ts_corr:

            encoded_corr = _encode_corr_json(corr)
            # apply the ndigits formatting on the float numbers
            complete_correlations[ctx] = _round(encoded_corr, ndigits)

            if type(encoded_corr) is np.float64:
                # avoid undefined values None
                defined_correlations.append(encoded_corr)

            complete_links[ctx] = ts_pair

        res_obj.set_contexts(desc_x_label=desc_context,
                             ctx_values=sorted_contexts)

        # Computes the mean aggregation, ignoring None
        mean_ignoring_nan = np.nanmean(defined_correlations)
        # Computes the variance aggregation, ignoring None
        var_ignoring_nan = np.nanvar(defined_correlations)

        # encode the numpy.nan into json value
        mean_ignoring_nan = _encode_corr_json(mean_ignoring_nan)
        var_ignoring_nan = _encode_corr_json(var_ignoring_nan)

        res_obj.add_curve(desc_y_label="Pearson correlation", y_values=complete_correlations)

        res_obj.add_ts_links(tsuid_pairs=complete_links)

        # derogation: use ResourceLocator() in order to save process-data
        # until it is not implemented by IkatsApi : IkatsApi.data.write
        # => singleton wrapper to NTDM
        encoded_json = json.dumps(obj=res_obj.get_json_friendly_dict())

        logical_reference = "CorrPearson_{}-{}_{}".format(var_name_1, var_name_2, parent_id)

        result = ResourceLocator().ntdm.add_data(data=encoded_json,
                                                 process_id=parent_id,
                                                 data_type="JSON",
                                                 name=logical_reference)

        if not result.get('status', False):
            msg = "Spark Task failed to save as process-data: CorrelationsByContext={}"
            raise IkatsException(msg.format(logical_reference))

        # returns the object identifier in DB: result['id']
        return [int(variables[0]), int(variables[1])], result['id'], mean_ignoring_nan, var_ignoring_nan

    else:
        raise IkatsException("Spark Task: not yet implemented: correlation method={}".format(corr_method))


def correlation_ts_loop(ds_name,
                        corr_method,
                        context_name):
    """
    Wrapper of correlation_ts_list_loop():

      - translates the ds_name into a flat list of TSUIDs: computed_list

      - and then calls correlation_ts_list_loop(ts_list=computed_list,
                                                corr_method=corr_method,
                                                context_meta=context_name)

    .. seealso:: full documentation in correlation_ts_list_loop()

    :param ds_name: name of the dataset grouping all the timeseries involved by the correlation loop.
    :type ds_name: str
    :param corr_method: the method computing the correlation between 2 timeseries.

      The value must be in CORRELATION_METHODS.

      Choose PEARSON to apply the pearson correlation.
    :type corr_method: str
    :param context_name: name of the metadata identifying each observed context,
      where correlations are computed.

      .. note:: this metadata shall exist for each timeseries, otherwise the
        latter will be ignored.

      With Airbus example: 'FlightIdentifier' identifies the flight as observed context.

    :type context_name: str
    :return: JSON-friendly dict: see correlation_ts_list_loop return documentation
    :rtype: dict as json-friendly structure for json library
    """
    if ds_name is None or type(ds_name) is not str:
        msg = "Unexpected arg value: defined str is expected for ds_name={}"
        raise IkatsException(msg.format(msg.format(ds_name)))

    LOGGER.info("Evaluating timeseries list from the dataset named: %s", ds_name)

    ds_content = IkatsApi.ds.read(ds_name)
    computed_list = ds_content['ts_list']

    LOGGER.info("- Found %s TS from %s to %s", len(computed_list),
                computed_list[0],
                computed_list[-1])
    return correlation_ts_list_loop(ts_list=computed_list,
                                    corr_method=corr_method,
                                    context_meta=context_name)


def correlation_ts_list_loop(ts_list,
                             corr_method,
                             context_meta,
                             variable_meta='metric',
                             config=ConfigCorrelationLoop(the_num_partitions=24,
                                                          the_point_cache_size=50e6,
                                                          the_digits_number=4)):
    """
    Computes the correlations between timeseries selected by observed variables and contexts.

    The observed contexts are defined by the context_meta argument.
    The variables are defined by variable_meta argument.

    Assumed:
      - Each context has a list of distinct variables.
      - Each timeseries is uniquely associated to one context and one variable.

    Example with Airbus data:
      - the *context* is a flight in an Airbus dataset of timeseries.
      - the *variables* could be metric 'WS1', metric 'WS2' etc.

    This algorithm is spark-distributed on the cluster.

    Spark summary
    *************

      - **step 1** The driver prepares a set of configured tuples: each tuple is configured for one context,
               and has a list of (variable, timeseries reference). Timeseries references are tsuids.

      - **step 2** A RDD is initialized from the set of cells **'configured tuples'**

      - **step 3** A new RDD is computed from step 2: each cell **'configured tuple'** is transformed into list of
        **'correlation inputs'**: this cell is prepared to be processed by the correlation method, for a
        subpart of the correlation matrice computed for one context

        At this step, each task task executes: *_spark_combine_pairs()*

      - **step 4** A new RDD is computed as set of **'correlation result'** cells from cells **'correlations inputs'**:
        each task will read timeseries pairs, compute the correlation result from selected method (Pearson, ...)

        At this step, each task task executes: *_spark_correlate_pairs()*

      - **step 5**: aggregates **'correlation result'** by variable pairs into RDD of
        **'aggregated correlations'** cells. Each task will

        1. creates and saves low-level results CorrelationsByContext into IKATS database, as JSON content.

          .. seealso:: the JSON is described in the
            ikats.algo.core.correlation.data.CorrelationDataset::get_json_friendly_dict()

        2. returns **'aggregated correlation'** cells providing

          - pair of variable indexes
          - aggregated values: Mean, Variance
          - saved reference of CorrelationsByContext

        At this step, each task executes: *_spark_build_corrs_by_context()*

      - **step 6**: the driver collects the RDD of **'aggregated correlations'**, and computes the high-level result,
        which is a CorrelationDataset.

        Finally the JSON generated by CorrelationDataset is returned.

    :param ts_list: selected timeseries list on which are computed the correlations
    :type ts_list: list
    :param corr_method: the method computing the correlation between 2 timeseries.

      The value must be in CORRELATION_METHODS.

      Choose PEARSON to apply the pearson correlation.
    :type corr_method: str
    :param context_meta: name of the metadata identifying each observed context,
      where correlations are computed.

      .. note:: this metadata shall exist for each timeseries, otherwise the
        latter will be ignored.

      With Airbus example: 'FlightIdentifier' identifies the flight as observed context.

    :type context_meta: str
    :param variable_meta: Optional, with default value 'metric',
      the name of the metadata identifying the variables.

      .. note:: this metadata shall exist for each timeseries, otherwise the
        latter will be ignored.

      The metadata values will be sorted in a list providing the effective indexes of matrices:
      the correlation matrix: the N-th index is reserved to the timeseries having the N-th value of
      this metadata in alphanumeric order.

      It is advised to keep the default value: this advanced argument must provide distinct indexes for each
      timeseries under same observed context.

    :type variable_meta: str
    :return: JSON-friendly dict grouping

      - Matrix of means of correlations (see step5)

      - Matrix of variances of correlations (see step5)

      - Matrix of references to the JSON content of CorrelationByContext (see step 5)

      .. seealso:: detailed JSON structure in
        ikats.algo.core.correlation.data.CorrelationDataset::get_json_friendly_dict()

    :rtype: dict as json-friendly structure for json library
    :raise exception: IkatsException when an error occurred while processing the correlations.
    """

    sc = None

    try:
        LOGGER.info("Starting correlation loop ...")
        LOGGER.info(" - observed contexts based on: %s", context_meta)
        LOGGER.info(" - variables ordered by: %s", variable_meta)

        # Check parameters
        corr_func = CORRELATION_FUNCTIONS.get(corr_method, None)
        if corr_func is None:
            msg = "Unknown correlation method from CORRELATION_FUNCTIONS: corr_method={}"
            raise IkatsException(msg.format(corr_method))

        if type(ts_list) is not list:
            msg = "Unexpected type: list expected for ts_list={}"
            raise IkatsException(msg.format(msg.format(ts_list)))

        if type(context_meta) is not str or len(context_meta) == 0:
            msg = "Unexpected arg value: defined str is expected for context_meta={}"
            raise IkatsException(msg.format(msg.format(context_meta)))
        if type(variable_meta) is not str or len(variable_meta) == 0:
            msg = "Unexpected arg value: defined str is expected for variable_meta={}"
            raise IkatsException(msg.format(msg.format(variable_meta)))

        # Hyp: the metadata part can be loaded from the driver

        ts_metadata_dict = IkatsApi.md.read(ts_list)

        # Note: the algorithm discards the variables X without Corr(X,Y) for Y different from X
        #       but when X is retained, the final result will present the Corr(X,X) beside the Corr(X,Y)
        corr_loop_config, sorted_contexts, sorted_variables = _initialize_config_from_meta(ts_metadata_dict,
                                                                                           context_meta=context_meta,
                                                                                           variable_meta=variable_meta)

        LOGGER.info("- sorted_contexts=%s", sorted_contexts)
        LOGGER.info("- sorted_variables=%s", sorted_variables)

        nb_contexts = len(sorted_contexts)

        if nb_contexts * len(sorted_variables) == 0:
            # Algo simply return empty result when there is no variable or no context consistent
            #
            # - case 1: case when there is no computable Corr(X, Y)
            #           where variables X and Y are different for the same context
            # - case 2: missing metadata for context_name => no context
            # - case 3: missing metadata for ordering_meta => no variable
            #
            LOGGER.warning("Empty result from selection=%s", ts_list)
            obj_empty_result = CorrelationDataset()
            obj_empty_result.set_contexts(contexts=sorted_contexts, meta_identifier=context_meta)
            obj_empty_result.set_variables(labels=sorted_variables)
            obj_empty_result.add_matrix(matrix=[], desc_label="Empty Mean correlation")
            obj_empty_result.add_matrix(matrix=[], desc_label="Empty Variance correlation")
            obj_empty_result.add_rid_matrix(matrix=[])

            return obj_empty_result.get_json_friendly_dict()

        # Computes the number of matrix chunks
        # one matrix chunk will be handled by one task at
        # -------------------------------------
        if nb_contexts < config.num_partitions:
            # Case when there are fewer contexts than recommended partitions:
            # - the computing of one matrix is split into several chunks
            nb_matrix_blocks = ceil(float(config.num_partitions) / nb_contexts)
        else:
            nb_matrix_blocks = 1

        LOGGER.info("- number of matrix blocks by context=%s", nb_matrix_blocks)

        # Computes the timeseries LRU cache size used by one task
        # -------------------------------------------------------
        # 1/ retrieve nb points for each TS, default value is assumed to be 1e6 in order to be robust
        # in case 'qual_nb_points' is not available, (should not happen ...)
        defined_nb_points = [int(v.get('qual_nb_points', 1e6)) for v in ts_metadata_dict.values()]
        # 2/ evaluate the number of points by one task carrying one matrice chunk
        total_nb_points_by_ctx = sum(defined_nb_points) / nb_contexts / nb_matrix_blocks
        if config.the_point_cache_size >= total_nb_points_by_ctx:
            # the best condition:
            # system will memorize in the cache every loaded ts under the same matrice
            ts_cache_size = len(sorted_variables)
        else:
            # the case when it is required to limit the number TS memorized in the cache,
            # under the same row of correlation matrice
            # Note: len(sorted_variables) == max size of correlation row == dim matrice
            ts_cache_size = config.the_point_cache_size / total_nb_points_by_ctx * len(sorted_variables)
            ts_cache_size = ceil(max(2.0, ts_cache_size))
        LOGGER.info("- ts_cache_size=%s", ts_cache_size)

        # release ts_metadata_dict from memory
        ts_metadata_dict = None

        sc = ScManager.get()

        # Spark_step_1: initialize the RDD
        # ------------
        # OUTPUT: RDD of ( <context index>, [ (<var index 1> , <tsuid 1>), ..., (<var index N> , <tsuid N>) ] )

        rdd_initial_config = sc.parallelize(corr_loop_config, config.num_partitions)

        # Spark_step_2: combinate the pairs of timeseries by contexts and by chunks
        # ------------
        # INPUT:  RDD of ( <context index>, [ (<var index 1> , <tsuid 1>), ..., (<var index N> , <tsuid N>) ] )
        # OUTPUT: RDD of ( <context_index>, [ <pair 1_2>, <pair 1_3>, ..., <pair M_N> ] )
        #
        #    where <pair X_Y> is ((<var X index>, <tsuid X> ), (<var Y index>, <tsuid Y>))
        #
        # PROCESS: computes the cartesian product and split the list of pairs into smaller-sized lists
        #
        rdd_var_combinations = rdd_initial_config.flatMap(
            lambda x: _spark_combine_pairs(context=x[0],
                                           variables=x[1],
                                           nb_corr_matrix_blocks=nb_matrix_blocks))

        if nb_matrix_blocks > 1:
            # reshuffles all the data over the cluster ...
            rdd_var_combinations = rdd_var_combinations.repartition(nb_contexts * nb_matrix_blocks)

        # Spark_step_3: computes the correlations
        # ------------
        # INPUT:  RDD of ( <context_index>, [ <pair 1_2>, <pair 1_3>, ..., <pair M_N> ] )
        # OUTPUT: RDD of ( (<var X index>, <var Y index>), <computed corr X_Y> )
        #
        #  where
        #    <computed corr X_Y> is (<context>, (<tsuid X>, <tsuid Y>), correlation)
        #
        # PROCESS: computes the correlations on the timeseries associated to the variables
        #
        rdd_correlations = rdd_var_combinations.flatMap(lambda x: _spark_correlate_pairs(context=x[0],
                                                                                         var_pairs=x[1],
                                                                                         corr_method=corr_method,
                                                                                         ts_cache_size=ts_cache_size))

        # generates the parent_id:
        #   presently this identifier may be used by Postgres admin,
        #   to group the low-level results attached to the same high-level result
        #   => at the moment a label including a timestamp is generated
        obj_result = CorrelationDataset()
        parent_id = obj_result.get_id()

        def r_append(data, computed_corr):
            """
            Append computed correlation to data
            :param data:
            :param computed_corr:
            :return:
            """
            data.append(computed_corr)
            return data

        def r_merge(one, two):
            """
            Merge two to one
            :param one:
            :param two:
            :return:
            """
            one.extend(two)
            return one

        # Spark_step_4: aggregate the correlations by pair of variables
        # ------------
        # INPUT: RDD of ( (<var X index>, <var Y index>), <computed corr X_Y> ) as described previously
        #
        # OUTPUT: RDD of ( (<var X index>, <var Y index>), list of tuples:
        #                                  (<context index>, (tsuid_X, tsuid_Y), <correlation result> )
        #                )
        # PROCESS: aggregates by key=(<var X index>, <var Y index>) the correlation information profiles,
        #          enhanced with tsuid pairs
        #
        rdd_agg_correlations = rdd_correlations.aggregateByKey(zeroValue=[],
                                                               seqFunc=r_append,
                                                               combFunc=r_merge)

        # Spark_step_5:
        # ------------
        # INPUT: RDD of  ( (<var X index>, <var Y index>), list of tuples:
        #                                  (<context index>, (tsuid_X, tsuid_Y), <correlation result> )
        #                )
        #
        # OUTPUT: RDD of ( ( <var X index>, <var Y index>), <low-level Result ID>, <Mean correlation>, <Var correlation>
        #                )
        # PROCESS: - creates and saves aggregated low-level results as CorrelationsByContext
        #          - computes Mean and Variance of low-level results
        #          - returns summarized info: Mean+Variance+ result ID
        rdd_results_corr_by_context = \
            rdd_agg_correlations.map(lambda x: (_spark_build_corrs_by_context(variables=x[0],
                                                                              agg_ctx_ts_corr=x[1],
                                                                              desc_context=context_meta,
                                                                              sorted_variables=sorted_variables,
                                                                              sorted_contexts=sorted_contexts,
                                                                              corr_method=corr_method,
                                                                              parent_id=parent_id,
                                                                              ndigits=config.the_digits_number)))

        # Spark_step_6:
        # ------------
        #
        # 6.1: collects
        #
        # INPUT: RDD of  ( [ <var X index>, <var Y index>], <processdata ID>, <Mean(corr)>, <Var(corr)>
        #                )
        #
        # OUTPUT: collected list
        #
        # PROCESS:  collects high-level results
        #
        collected_results_corr = rdd_results_corr_by_context.collect()

        # 6.2: prepare the result
        #
        #  - Encodes the returned json-friendly content from the collected high-level results
        #  - returns the result
        #
        matrix_mean = get_triangular_matrix(dim=len(sorted_variables),
                                            default_value_diag=1.0, default_value_other=None)

        matrix_variance = get_triangular_matrix(dim=len(sorted_variables),
                                                default_value_diag=0.0, default_value_other=None)

        matrix_id = get_triangular_matrix(dim=len(sorted_variables),
                                          default_value_diag=None, default_value_other=None)

        for var_index_pair, data_oid, mean, variance in collected_results_corr:
            var_index_row = var_index_pair[0]
            var_index_col = var_index_pair[1]
            # required: recomputes the range of cell in its row
            # triangular matrix => cell(i,j) is at position j-i of the row triangular_matrix[i]
            matrix_mean[var_index_row][var_index_col - var_index_row] = mean
            matrix_variance[var_index_row][var_index_col - var_index_row] = variance
            matrix_id[var_index_row][var_index_col - var_index_row] = data_oid

        obj_result.set_contexts(contexts=sorted_contexts, meta_identifier=context_meta)

        obj_result.set_variables(sorted_variables)
        obj_result.add_matrix(matrix=matrix_mean, desc_label="Mean Correlation")
        obj_result.add_matrix(matrix=matrix_variance, desc_label="Variance")
        obj_result.add_rid_matrix(matrix_id)

        LOGGER.info("... ended correlation loop.")
        return obj_result.get_json_friendly_dict()

    except Exception:
        LOGGER.error("... ended correlation loop with error.")
        raise IkatsException("Failed execution: correlation_ts_loop()")
    finally:
        if sc:
            ScManager.stop()
