#!/bin/bash

# Fill the environment variables
source ikats.env

# Skip the long tests by default (can be overriden)
export SKIP_LONG_TEST=${SKIP_LONG_TEST:-1}

nosetests --with-xunit ikats.algo.core
