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
import csv
import logging
import os
import unittest

from ikats.algo.core.distance.spark_distance import SparkDistance
from ikats.core.resource.client.non_temporal_data_mgr import NonTemporalDataMgr
from ikats.core.resource.client.temporal_data_mgr import TemporalDataMgr

LOGGER = logging.getLogger('ikats.algo.core.distance')
# Log format
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')

# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)


class TestSparkDistance(unittest.TestCase):
    """
    Test of the spark distance module
    """

    @unittest.skipIf('SKIP_LONG_TEST' in os.environ and os.environ['SKIP_LONG_TEST'],
                     "This test is too long and must not be run every time")
    def test_launch_ws1(self):
        """
            needs environment vars to be explicitly set :
            SPARK_HOME
            and
            PYSPARK_PYTHON
        """
        tsuid_file = os.path.dirname(os.path.realpath(__file__)) + "/tsuidList.txt"
        tsuids = list()
        func_ids = list()
        if os.getenv("PYSPARK_PYTHON") is None:
            self.fail("env PYSPARK_PYTHON must be defined")

        print(tsuid_file)
        with open(tsuid_file, 'r') as opened_file:
            # Write headers
            for line in opened_file:
                print(line)
                tsuids.extend(line.split(','))
                # Write content

        print(tsuids)
        tdm = TemporalDataMgr()
        for tsuid in tsuids:
            try:
                func_ids.append(tdm.get_func_id_from_tsuid(tsuid))
            except ValueError:
                func_ids.append(tsuid)

        ntdm = NonTemporalDataMgr()

        sd = SparkDistance(tdm)

        result_map = sd.run(tsuids)
        with open('/tmp/result_400.csv', 'w', newline='') as opened_file:
            writer = csv.writer(opened_file, delimiter=';')
            writer.writerow([' '] + func_ids)
            for tsuid1 in tsuids:
                row = list()
                try:
                    func_id = tdm.get_func_id_from_tsuid(tsuid1)
                except ValueError:
                    func_id = tsuid1
                row.append(func_id)
                for tsuid2 in tsuids:
                    if tsuid1 == tsuid2:
                        row.append("0")
                    else:
                        row.append(result_map[(tsuid1, tsuid2)])
                writer.writerow(row)
            opened_file.close()
        ntdm.add_data('/tmp/result_400.csv', "spark_distance_01", "CSV")

    @unittest.skipIf('SKIP_LONG_TEST' in os.environ and os.environ['SKIP_LONG_TEST'],
                     "This test is too long and must not be run every time")
    def test_launch(self):
        """
            needs environment vars to be explicitly set :
            SPARK_HOME
            and
            PYSPARK_PYTHON

        """
        if os.getenv("PYSPARK_PYTHON") is None:
            self.fail("env PYSPARK_PYTHON must be defined")

        tsuids = ['0000110000030003F30000040003F1',
                  '0000110000030003F40000040003F1',
                  '0000110000030003F50000040003F1',
                  '0000110000030003F60000040003F1',
                  '0000110000030003F70000040003F1',
                  '0000110000030003F80000040003F1',
                  '0000110000030003F90000040003F1',
                  '0000110000030003FA0000040003F1',
                  '0000110000030003FB0000040003F1',
                  '0000110000030003FC0000040003F1',
                  '0000110000030003FD0000040003F1',
                  '0000110000030003FE0000040003F1',
                  '0000110000030003FF0000040003F1',
                  '0000110000030004000000040003F1',
                  '0000110000030004010000040003F1',
                  '0000110000030004020000040003F1',
                  '0000110000030004030000040003F1',
                  '0000110000030004040000040003F1',
                  '0000110000030004050000040003F1',
                  '0000110000030004060000040003F1']

        tdm = TemporalDataMgr()
        ntdm = NonTemporalDataMgr()

        sd = SparkDistance(tdm)

        result_map = sd.run(tsuids)
        with open('/tmp/result.csv', 'w', newline='') as opened_file:
            writer = csv.writer(opened_file, delimiter=';')
            writer.writerow([' '] + tsuids)
            for tsuid1 in tsuids:
                row = list()
                row.append(tsuid1)
                for tsuid2 in tsuids:
                    if tsuid1 == tsuid2:
                        row.append("0")
                    else:
                        row.append(result_map[(tsuid1, tsuid2)])
                writer.writerow(row)
            opened_file.close()
        ntdm.add_data('/tmp/result.csv', "spark_distance_01", "CSV")
        print("Type Enter to stop the test")
        # sd.run(tsuids20, "local[8]")


if __name__ == "__main__":
    unittest.main()
