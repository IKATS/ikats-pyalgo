#!/bin/bash

# Script to call inside container to prepare test campaign

cd /ikats
chown ikats:ikats /ikats/*

# Install tests modules dependencies
pip3 install -r test_requirements.txt
