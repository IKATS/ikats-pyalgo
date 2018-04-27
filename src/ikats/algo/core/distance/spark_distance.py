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

from scipy.spatial.distance import euclidean

from ikats.core.library.spark import ScManager, ListAccumulatorParam


class SparkDistance(object):
    """
        Calculation of distance matrix using euclidean distance
        on a list of tsuid.
        The euclidean distance needs the two time series to be of the same size.
        So the biggest time serie is shrink to the size of the little one before
        applying the scipy euclidean function.

    """

    def __init__(self, tdm, ts_load_split_size=10):
        """
        init the spark distance class

        :param tdm: the temporal data manager client
        :type tdm: TemporalDataMgr

        :param ts_load_split_size: size of TS packet to load from TDM
        :type ts_load_split_size: int

        """

        self.tdm = tdm
        self.ts_load_split_size = ts_load_split_size
        self.spark_context = ScManager.get()

        self.logger = logging.getLogger(__name__)

    def run(self, tsuids):
        """
        Run the Spark Distance calculation

        create the RDD for each tsuid,
        load the TS from tdm in a broadcast dictionary (ie shared by all workers)
        map the RDD with cartesian product ( ie get RDD1,RDD1 RDD1,RDD2 RDD2,RDD1 RDD2,RDD2 with 2 RDD)
        to get the comparison couples.
        then reduce applying distance function and add the result into Accumulator (ie shared by all workers)
        distance function take the two TS from broadcast dictionary, shrink the biggest and apply euclidean

        Usage: pi [tsuid1] [tsuid2] ...[tsuidn]

        example : tsuids = ['0000110000030003F30000040003F1',
                            '0000110000030003F40000040003F1',
                            '0000110000030003F50000040003F1',
                            '0000110000030003F60000040003F1',
                            '0000110000030003F70000040003F1']

        :param tsuids: a list of tsuids (str)
        :type tsuids: list

        """

        # creation of the RDD
        rdd = self.spark_context.parallelize(tsuids)

        self.logger.info("rdd parallelized")
        self.logger.info("loading TS")
        start_time = time.time()

        j = len(tsuids) // self.ts_load_split_size
        self.logger.debug(type(tsuids))
        self.logger.info("Number of TS: %i ", len(tsuids))
        ts = list()
        for i in range(0, j + 1):
            k = (i + 1) * self.ts_load_split_size
            if k > len(tsuids):
                k = len(tsuids)
            self.logger.info("extract TS from index %i to %i ", i * self.ts_load_split_size, k)
            ts.extend(self.tdm.get_ts(tsuids[i * self.ts_load_split_size:k]))

        ts_dic = dict()
        self.logger.info("Number of TS loaded : %i ", len(ts))
        for index in range(0, len(tsuids)):
            ts_dic[tsuids[index]] = ts[index]
        # broadcast var used to get the map result
        broadcast_var = self.spark_context.broadcast(ts_dic)
        loading_end_time = time.time()
        self.logger.info("Loading Time : %s ", loading_end_time - start_time)

        # create the result accumulator
        list_accum = self.spark_context.accumulator(dict(), ListAccumulatorParam())

        def calculate_distance(tsuid_list):
            """
               :param tsuid_list: a pair of tsuids
               :type tsuid_list: list
            """
            # use py4j logger to avoid Serialization problems.
            logger = logging.getLogger('py4j')
            logger.setLevel(logging.INFO)
            logger.removeHandler(logger.handlers[0])
            # sh = logging.StreamHandler(sys.stdout)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

            # start distance calculus.
            logger.debug("tsuid1= %s", tsuid_list[0])
            logger.debug("tsuid2= %s", tsuid_list[1])
            if tsuid_list[0] != tsuid_list[1]:
                first_ts = np.array(broadcast_var.value[tsuid_list[0]][:, 1])
                second_ts = np.array(broadcast_var.value[tsuid_list[1]][:, 1])
                calculus_len = min(len(first_ts), len(second_ts))

                distance = euclidean(first_ts[0:calculus_len], second_ts[0:calculus_len])

                # logger.debug("tsuid list %s and distance %f" % (tsuid_list, distance))
                list_accum.add({tsuid_list: distance})

        __import__('ikats.algo.core.distance')
        rdd.cartesian(rdd).foreach(calculate_distance)

        ScManager.stop()

        computation_end_time = time.time()
        self.logger.info("Loading Time : %s ", loading_end_time - start_time)
        self.logger.info("Compute Time : %s ", computation_end_time - loading_end_time)
        return list_accum.value
