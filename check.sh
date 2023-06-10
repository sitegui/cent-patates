#!/bin/bash

set -e

isort cent_patates
yapf --recursive --in-place cent_patates
mypy --package cent_patates