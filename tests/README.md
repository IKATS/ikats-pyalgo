# TESTS

Here is the testing environment for python part

## Usage

Just run the `.startJob.sh` script.
Results are displayed in `STDOUT`.
`junit.xml` file is generated and contains results as JUNIT format (useful to be parsed by sonarQube)

## Files

- **startJob.sh**: entry point to prepare and run the test campaign (see comments inside to know how it works)
- **docker-compose.yml**: file describing the minimum components to start to have tests running properly
- **.env**: environment variables injected in docker-compose.yml to configure easily the components
- **assets**: this folder contains every file being copied to the python container to test
  - **pylint.rc** is used as pylint rules to be used by QA
  - **test_requirements.txt**: contains the list of python modules to install for testing purposes (with version)
  - **testPrepare.sh**: script used to prepare environment (install dependencies, environment sourcing)
  - **testRunner.sh**: script that runs the test campaign and extract results from the container after it is completed.
