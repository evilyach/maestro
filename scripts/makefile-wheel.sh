#!/bin/bash

VENV_DIR=".env"

source $VENV_DIR/bin/activate

python -m pep517.build -b .

rm maestro.egg-info -rf
rm build -rf

deactivate
