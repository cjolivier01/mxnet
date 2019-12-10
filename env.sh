#!/bin/bash
export PYTHONPATH=$(pwd)/python
export LD_LIBRARY_PATH="$(pwd)/cmake-build-debug:${LD_LIBRARY_PATH}"
