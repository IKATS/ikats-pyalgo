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
import os
import time

from ikats.algo.core.spark_corr import SparkCorrelation
from ikats.core.library.spark import ScManager
from ikats.core.resource.client import TemporalDataMgr


def main_test():
    """
    Functional test entry point
    """

    logger = logging.getLogger("ikats.algo.core.correlation")
    # Log format
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
    # Create another handler that will redirect log entries to STDOUT
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if os.getenv("PYSPARK_PYTHON") is None:
        os.putenv("PYSPARK_PYTHON", "/home/ikats/tools/ikats_processing/bin/python")
    if os.getenv("SPARK_HOME") is None:
        os.putenv("SPARK_HOME", "/opt/spark")

    print('Loading Spark Context')
    # Get a spark Context
    ScManager.get()

    tdm = TemporalDataMgr()

    answer = 'n'
    tsuid_list = []
    ds_name = ''
    while answer.lower() != 'y':
        ds_name = input('\nEnter dataset Name: ')
        tsuid_list = tdm.get_data_set(ds_name)['ts_list']

        print("%s TS found in dataset %s" % (len(tsuid_list), ds_name))

        if len(tsuid_list) > 0:
            answer = input("Run the correlation matrix on these dataset? [Y/n] ")

    print('Running correlation matrix on %s TS' % len(tsuid_list))

    start_time = time.time()
    sp_corr = SparkCorrelation(tdm)
    sp_corr.force_parallel_get_ts = True
    sp_corr.run(tsuid_list)

    print("EXECUTION TIME (for %d TS with %d pts/ea = %d points): %.3f seconds" % (
        len(tsuid_list),
        sp_corr.ts_len_ref,
        (len(tsuid_list) * sp_corr.ts_len_ref),
        (time.time() - start_time)))

    if os.path.isfile('/tmp/spark_correlation_result_%s.csv' % ds_name):
        os.remove('/tmp/spark_correlation_result_%s.csv' % ds_name)
    with open('/tmp/spark_correlation_result_%s.csv' % ds_name, 'w', newline='') as opened_file:
        opened_file.write(sp_corr.get_csv())

    print("Matrix in CSV format is saved at the following location:")
    print("   /tmp/spark_correlation_result_%s.csv" % ds_name)
    print("You can check the content by doing :")
    print("   cat /tmp/spark_correlation_result_%s.csv" % ds_name)
    print("   less /tmp/spark_correlation_result_%s.csv" % ds_name)
    print("   vi /tmp/spark_correlation_result_%s.csv" % ds_name)


if __name__ == '__main__':
    main_test()
