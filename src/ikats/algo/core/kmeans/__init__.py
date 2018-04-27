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
# Simplify the import patterns for user

from pkgutil import extend_path
from ikats.algo.core.kmeans.kmeans import fit_kmeans_internal, mds_representation_kmeans, format_kmeans, fit_kmeans
from ikats.algo.core.kmeans.kmeans_pattern_group import fit_kmeans_pattern_group
__path__ = extend_path(__path__, __name__)
