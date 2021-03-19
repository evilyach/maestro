#!/bin/bash

VENV_DIR=".env"

source $VENV_DIR/bin/activate

black maestro
isort maestro

deactivate