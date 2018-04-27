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
import time

import numpy as np

from ikats.core.library.spark import ScManager
from ikats.core.resource.client import TemporalDataMgr, ServerError
from pyspark.accumulators import AccumulatorParam
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.stat import Statistics


class SparkCorrelation(object):
    """
    Calculation of the correlation matrix on a list of tsuid.
    Available correlation methods are
    * Pearson
    * Spearman
    """

    def __init__(self, tdm):
        """
        Init the spark correlation class

        :param tdm: the temporal data manager client
        :type tdm: TemporalDataMgr

        """

        # Is the number of element beyond which they will be retrieved in parallel
        # len(tsuids)>=self.trig_spark_get_ts --> use spark to retrieve TS content
        # len(tsuids)<self.trig_spark_get_ts --> use simple loop to get them
        # According to performance measurements, using serial get_ts is better from 1 to 15 TS
        #
        # Perfo measurements:
        # Nb TS    parallel       serial
        # 1        1,983887196    0,159182787
        # 11       4,984174252    2,372197628
        # 12       1,52216959     1,324707508
        # 13       1,464684248    1,427070856
        # 14       1,645312548    1,511950254
        # 15       1,774277925    1,594658852
        # 16       1,582788467    1,622912407
        # 17       1,697391748    1,640071154
        # 18       1,745421171    2,015519857
        # 19       2,083591223    2,155070543
        # 21       1,663324833    2,240501165
        # 31       2,324476242    4,978622675
        # 41       2,303745508    3,84933424
        # 51       2,770316839    4,857459545
        # 61       2,896857023    7,419473648
        # 71       3,507226944    6,922876596
        # 81       3,538532257    7,565524578
        # 91       3,854919195    8,600610495
        # 101      3,971539736   11,03932238
        # 111      4,314357042   10,62726593
        # 121      4,592056274   11,35756373
        self.trig_spark_get_ts = 5

        # If set to True, force the usage of parallel method to get TS
        # no matter the trigger defined by trig_spark_get_ts
        self.force_parallel_get_ts = False

        # Connectors to Data manager
        self.tdm = tdm

        # TSUID list
        self.tsuids = []

        # Results
        self.results = None

        # Number of points of the smaller TS (used for performance measurements)
        self.ts_len_ref = 0

        # Logger
        self.logger = logging.getLogger(__name__)

    def run(self, tsuids, method='pearson'):
        """
        Run the Spark correlation calculation using the indicated method

        :param tsuids: tsuid list to use for calculation
        :type tsuids: list

        :param method: Pearson or Spearman correlation method ("pearson" by default)
        :type method: str

        :raise ValueError: if tsuids is not a list or is empty
        """

        # Check TSUIDS validity
        if (type(tsuids) != list) or (len(tsuids) == 0):
            raise ValueError("tsuids list must be filled")

        # Check method validity
        method = method.lower()
        if method not in ['pearson', 'spearman']:
            raise ValueError("method can only be pearson or spearman")

        self.tsuids = tsuids

        # Call the all in memory version of the calculation
        self._run_all_in_master_memory(method)

    def _get_ts(self, spark_context):
        """
        Retrieve the TS values using the best method
        """

        start_time = time.time()

        method = "serial"

        if len(self.tsuids) >= self.trig_spark_get_ts or self.force_parallel_get_ts:
            # Switch to parallel method to load TS
            self.logger.debug("Using Parallel method to get TS content")

            method = "parallel"
            broadcast = spark_context.broadcast({
                "host": self.tdm.host,
                "port": self.tdm.port
            })

            # Create an accumulator to store the results of the spark workers
            accumulator = spark_context.accumulator(dict(), ListAccumulatorParam())

            # Create an rdd containing the tsuids to get the data from
            rdd_tsuids = spark_context.parallelize(self.tsuids)

            def get_ts_to_accu(tsuid):
                """
                Function to be called to retrieve one TS content values
                :param tsuid: TS to get content from
                :return:
                """
                tdm = TemporalDataMgr(host=broadcast.value['host'], port=broadcast.value['port'])

                # The slice removes the timestamp column
                ts_points = tdm.get_ts(tsuid)[0][:, 1]

                accumulator.add({
                    tsuid: {
                        "data": ts_points,
                        "nb_points": len(ts_points)
                    }
                })

            # Get TS content using spark distribution to increase performance
            rdd_tsuids.foreach(get_ts_to_accu)

            ts_data = []
            for ts in self.tsuids:
                # Get minimum length of a TS
                if self.ts_len_ref == 0:
                    self.ts_len_ref = accumulator.value[ts]['nb_points']
                if self.ts_len_ref != min(self.ts_len_ref, accumulator.value[ts]['nb_points']):
                    self.ts_len_ref = min(self.ts_len_ref, accumulator.value[ts]['nb_points'])
                    self.logger.warning("TS don't have same number of points ! Cutting to the smallest one")
                    self.logger.warning("TS %s has %d", ts, accumulator.value[ts]['nb_points'])
                # Append data
                ts_data.append(accumulator.value[ts]['data'])

            # Cut TS to shortest one
            for index, val in enumerate(ts_data):
                ts_data[index] = val[:self.ts_len_ref]

            ts_data = Vectors.dense(np.array(ts_data).T)

        else:
            # Use serial method to get TS content
            self.logger.debug("Using Serial method to get TS content")

            ts_data = []
            for ts in self.tsuids:
                # Get the TS
                ts_content = self.tdm.get_ts(ts)[0][:, 1]

                # Get minimum length of a TS
                if self.ts_len_ref == 0:
                    self.ts_len_ref = len(ts_content)

                if self.ts_len_ref != min(self.ts_len_ref, len(ts_content)):
                    self.ts_len_ref = min(self.ts_len_ref, len(ts_content))
                    self.logger.warning("TS don't have same number of points ! Cutting to the smallest one")
                    self.logger.warning("TS %s has %d", ts, len(ts_content))

                # Append data
                ts_data.append(ts_content)

            # Cut TS to shortest one
            for index, val in enumerate(ts_data):
                ts_data[index] = val[:self.ts_len_ref]

            ts_data = Vectors.dense(np.array(ts_data).T)

        self.logger.debug("%d TS loaded (%s method) (%d pts) in %s seconds",
                          len(self.tsuids), method, self.ts_len_ref * len(self.tsuids), (time.time() - start_time))
        return spark_context.parallelize(ts_data)

    def _run_all_in_master_memory(self, method):
        """
        Run the spark pearson correlation by loading all the TS content (ie. values) in master memory

        Each coefficient will be computed by a worker (Spark decides the best choice to apply)
        """

        # Create or get a spark Context
        spark_context = ScManager.get()

        # Get TS content
        rdd_content = self._get_ts(spark_context)

        # Job distribution is made by Statistics.corr (Spark correlation matrix calculation)
        self.results = Statistics.corr(rdd_content, method=method)

        ScManager.stop()

    def get_csv(self, headers_fid=True):
        """
        Returns CSV format of the result
        :param headers_fid: fill the headers with functional identifiers if True, with TSUIDS otherwise
        :return: the CSV text
        """

        # Define headers
        headers = self.tsuids
        if headers_fid:
            headers = []
            for ts in self.tsuids:
                try:
                    fid = self.tdm.get_func_id_from_tsuid(ts)
                    headers.append(fid)
                except ValueError:
                    headers.append('NOFID_%s' % ts)
                except ServerError:
                    headers.append('NOFID_%s' % ts)
                except TypeError:
                    headers.append('NOFID_%s' % ts)

        # Fill headers
        csv = ';' + ';'.join(headers) + '\n'

        # Fill body
        for index, line in enumerate(self.results):
            csv = csv + headers[index] + ";" + ";".join([str(x) for x in line]) + "\n"

        return csv


class ListAccumulatorParam(AccumulatorParam):
    """
    Accumulator of internal type dict
    inherited from Spark, justify the PEP8 errors
    """

    def zero(self, initial_value):
        """
        Init the internal variable. initial_value is ignore here
           :param initial_value:
           :type initial_value: any
        """
        return dict()

    # noinspection PyPep8Naming
    def addInPlace(self, v1, v2):
        """
            Add two values of the accumulator's data type,
            returning a new value
            add v2 to v1

            :param v1: parameter 1 to use for addition
            :param v2: parameter 2 to use for addition
        """
        v1.update(v2)
        return v1

    # noinspection PyPep8Naming
    def getValueForTuple(self, tsuid1, tsuid2):
        """
            Return the value stored into the internal dict

            :param tsuid1: first tsuid
            :type tsuid1: str

            :param tsuid2: second tsuid
            :type tsuid2: str

            :return: the value for the couple of tsuid:
            :rtype: str
        """

        if tsuid1 == tsuid2:
            return 0
        else:
            # noinspection PyUnresolvedReferences
            return self._value[[tsuid1, tsuid2]]
