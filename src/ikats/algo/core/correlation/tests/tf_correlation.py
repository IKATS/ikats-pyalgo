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
import time

from ikats.core.resource.client import NonTemporalDataMgr
from ikats.core.resource.client import TemporalDataMgr
from ikats.algo.core.correlation import pearson_correlation_matrix


def main_test():
    """
    Functional test entry point
    """

    tdm = TemporalDataMgr()
    ntdm = NonTemporalDataMgr()

    nb_tsuid = 0
    tsuid_list = []
    answer = input(
        'output functional Identifiers [0] or tsuids [1] (default value : 1) :') or '1'
    if answer == '0':
        tsuids_out = False
    elif answer == '1':
        tsuids_out = True
    else:
        raise ValueError(TypeError)

    answer = input(
        'input list of tsuids [0] or dataset name [1] (default value : 0) :') or '0'
    if answer == '0':
        print('Example tsuid_list 1 (benchmark) = ')
        print('[000069000009000AE3, 00006E000009000AE8, 00006A000009000AE4, 00006B000009000AE5,'
              ' 00006C000009000AE6, 00006D000009000AE7, 000066000009000AE0, 000068000009000AE2 ]')
        print('Example tsuid_list 2 (functional ids) = ')
        print('[00001F00000B000BFE, 00008300000B000BFF, 00008400000B000C08]')

        print('Enter your list of tsuids :')
        tsuid = input('tsuid no 1 [ENTER to quit]: ')
        while tsuid != '':
            if tsuid != '':
                nb_tsuid += 1
                tsuid_list.append(tsuid)
            tsuid = input(
                'tsuid no ' + str(nb_tsuid + 1) + '[ENTER to quit]: ')
    else:
        print('Example dataset : dsCorrMat')
        tsuid_list = input('dataset_name: ')

    if tsuid_list:
        sd_temp = input('start date (yyyy-mm-dd) : ')
        ed_temp = input('end date (yyyy-mm-dd): ')
        if sd_temp and ed_temp:
            sd = int(time.mktime(time.strptime(sd_temp, "%Y-%m-%d")))
            ed = int(time.mktime(time.strptime(ed_temp, "%Y-%m-%d")))

            # Correlation Matrix computation and display
            print('Correlation Matrix (pearson) = ')
            print(pearson_correlation_matrix(tdm, tsuid_list, tsuids_out))
            print('PROCESS ID name : PearsonCorrelationMatrix')
        else:
            print('Start date / end date missing')
    else:
        print('tsuids list or dataset missing')


if __name__ == '__main__':
    main_test()
