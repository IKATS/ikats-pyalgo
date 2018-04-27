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
import numpy as np


def scale(data, with_mean=True, with_std=True, copy=True):
    """
    Transforms the original 1D array into a new scaled array having
      - mean equal to zero when with_mean is True,
      - standard deviation being equal to 1  when with_std is True (except
        when standard deviation of x is zero: the output std will be still zero)

    Note: workaround before integrating the scikit-learn library in the cluster environment
      - Recoded function: simplified version of scikit-learn scale

    :param data: original array: 1D array for this simplified scale function
    :type data: numpy array
    :param with_mean: optional, default True: sets the output mean to zero
    :type with_mean: bool
    :param with_std: optional, default True: sets the output std to 1 when possible
    :type with_std: bool
    :param copy: optional default True:  set to False to perform inplace row normalization and avoid a copy
    :type copy: bool
    :return: normalized 1D array
    """
    data_copy = np.array(data, copy=copy)
    if with_mean:
        mean_ = np.mean(data_copy)
    if with_std:
        scale_ = np.std(data_copy)
    if with_mean:
        data_copy -= mean_

    if with_std:
        if scale_ == .0:
            scale_ = 1.0
        data_copy /= scale_
    return data_copy
