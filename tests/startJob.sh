#!/bin/bash

# This script will prepare an empty environment, will start ikats and then run the test campaign
# After the campaign is completed, the script will get the result files to feed Jenkins metrics

# Stop any running container if error occurs
trap "echo 'Stopping...';docker-compose down >/dev/null 2>&1; exit 1" INT KILL QUIT PIPE

# Prepare docker_bindings
echo "Getting fresh docker bindings"
# Get new data
export pathToDockerBindings=$(mktemp -d /tmp/testing.XXXXXX)
pathToDockerBindingsRepo=${1:-/IKATSDATA/docker_bindings/}

if [[ ! -d ${pathToDockerBindingsRepo} ]]
then
  echo "Can't get docker_bindings. Unreachable folder"
  exit 2;
fi

dockerBindingsName="docker_bindings_EDF_portfolio"
cp ${pathToDockerBindingsRepo}${dockerBindingsName}.tar.gz ${pathToDockerBindings}

# Unzip new one
pushd ${pathToDockerBindings} >/dev/null
tar xzf ${dockerBindingsName}.tar.gz && rm ${dockerBindingsName}.tar.gz
popd >/dev/null
# Change location of docker bindings for docker-compose
sed -i "s@DOCKER_BINDINGS_POSTGRES=.*@DOCKER_BINDINGS_POSTGRES=${pathToDockerBindings}/docker_bindings/postgresql@" .env
sed -i "s@DOCKER_BINDINGS_HBASE=.*@DOCKER_BINDINGS_HBASE=${pathToDockerBindings}/docker_bindings/hbase/hbase@" .env

# Clean older images to force pull of latest version
docker rmi -f $(docker images hub.ops.ikats.org/ikats-* --format "{{.Repository}}:{{.Tag}}" | grep "latest")

# Start ikats
docker-compose up --build -d

# Container name to test (should never change)
containerName=tests_pyalgo

# Ikats path inside the container
IKATS_PATH=/ikats

# Prepare test environment
docker cp assets/test_requirements.txt ${containerName}:${IKATS_PATH}/
docker cp assets/testPrepare.sh ${containerName}:${IKATS_PATH}/
docker cp assets/pylint.rc ${containerName}:${IKATS_PATH}/
docker cp assets/testRunner.sh ${containerName}:${IKATS_PATH}/
docker exec --user root ${containerName} bash ${IKATS_PATH}/testPrepare.sh

# Wait a bit to let containers to initiate communication with others
sleep 10
# Execute the test campaign inside the docker container
docker exec --user ikats ${containerName} bash ${IKATS_PATH}/testRunner.sh
EXIT_STATUS=$?

# Get the results from docker container to host
docker cp ${containerName}:${IKATS_PATH}/nosetests.xml ./junit.xml

# Stopping docker
docker-compose down > /dev/null

# Return 0 if all tests are OK, any other number indicates tests are KO
exit ${EXIT_STATUS}
