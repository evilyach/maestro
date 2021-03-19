#!/bin/bash

VENV_DIR=".env"

source $VENV_DIR/bin/activate

pip install ./dist/maestro-*.whl
maestro
pip uninstall -y maestro

deactivate
